"""UMU track, DEV-029: is the name-located unlearning edit still active when the person's own portrait is shown, and
only overridden by the face route? (link 6 of the causal chain)

Input NP = "USER: <image>\\n{name question}\\nASSISTANT:" with the person's own portrait (forget dc facts). A 2x2 on
theta_u: attention from the positions after the image to the image tokens (visible / blocked in all layers) x the
LoRA at the name tokens (on / off), plus a control that switches the LoRA off at as many non-name text tokens.
Forgetting F = lp_theta0 - lp_thetau of the correct first answer token, theta_0 under the same input and knockout.
Text-only T (and T with the name edit off) as the reference.

Reading (DEV-029; seed 0 primary):
  6a  F(NP visible) <= 1.0 nat and F(NP knockout) >= 0.5 F(T)          the face route overrides the edit
  6b  F(NP knockout + name-off) <= 0.3 F(NP knockout)                  the restored forgetting comes from the name edit
  6c  F(NP knockout + control-off) >= 0.7 F(NP knockout)               control
  closed if 6a, 6b, 6c; not closed if 6b fails; otherwise partial.

  python scripts/mech_umu_link6.py --config configs/llava/umu_npo_rt.yaml --run-id <run> --step 300 --set umu.n_retain_eval=100
Writes results/llava/mech_umu/<run>/step<N>/link6/{summary.json,items.jsonl,log.txt}.
"""
import contextlib
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np  # noqa: E402
import torch  # noqa: E402

from scripts.mech_umu_crossmodal import block_attention, prefix_positions  # noqa: E402
from scripts.mech_umu_form_sweep import facts_for  # noqa: E402
from vlmu.config import base_parser, resolve  # noqa: E402
from vlmu.mech import tools  # noqa: E402
from vlmu.mech.hooks import decoder_layers, lora_modules, mask_contributions  # noqa: E402
from vlmu.models import get_adapter  # noqa: E402
from vlmu.runs import free, model_dir, results_dir  # noqa: E402
from vlmu.umu import encode_teacher_forced, first_target, load_umu, token_groups_umu  # noqa: E402
from vlmu.utils import Logger, atomic_write_json, read_json, write_jsonl  # noqa: E402

TOL = 0.05
A_VIS, A_KO, B_MAX, C_MIN = 1.0, 0.5, 0.3, 0.7      # DEV-029


def rows_for(f):
    q = f["q_uni"]
    base = {"pair_id": f["pair_id"], "correct_text": f["answer_text"], "name": f["name"]}
    return ({**base, "uid": f"{f['pair_id']}/T", "prompt": f"USER: {q}\nASSISTANT:", "image_bytes": None},
            {**base, "uid": f"{f['pair_id']}/NP", "prompt": f"USER: <image>\n{q}\nASSISTANT:", "image_bytes": f["img"]})


@torch.no_grad()
def first_lp(adapter, model, batch, t, vanilla=False, keep=None, block=None):
    """(log p of the correct first answer token, is it the argmax) per row."""
    with contextlib.ExitStack() as st:
        if vanilla:
            st.enter_context(model.disable_adapter())
        if keep is not None:
            st.enter_context(mask_contributions(model, list(lora_modules(model)), keep))
        if block is not None:
            st.enter_context(block_attention(model, *block))
        at = tools.forward_logits(adapter, model, batch)[t["b"], t["pos"]].float()
    lp = at.log_softmax(-1).gather(-1, t["tok"][:, None]).squeeze(-1)
    return lp.cpu().tolist(), (at.argmax(-1) == t["tok"]).cpu().tolist()


def control_mask(groups, name):
    """Per row, as many template (non-name, non-image, non-last, non-answer) tokens as the row has name tokens: the
    first ones after the image (or after the start for text-only input)."""
    ctrl = torch.zeros_like(name)
    for i in range(name.shape[0]):
        k = int(name[i].sum())
        cand = groups["template"][i].nonzero().squeeze(-1)
        img = groups["image"][i].nonzero().squeeze(-1)
        if len(img):
            cand = cand[cand > img[-1]]
        ctrl[i, cand[:k]] = True
    return ctrl


def main():
    p = base_parser(__doc__)
    p.add_argument("--run-id", required=True)
    p.add_argument("--step", type=int, default=300)
    p.add_argument("--batch-img", type=int, default=4)
    args = p.parse_args()
    cfg = resolve(args)
    run_dir = model_dir(cfg) / "umu_unlearn" / args.run_id
    out = results_dir(cfg) / "mech_umu" / args.run_id / f"step{args.step}" / "link6"
    log = Logger(out / "log.txt")
    from peft import PeftModel
    dc = set(read_json(run_dir / "baseline_dc.json")["pair_ids"])
    dfs = load_umu(cfg["paths"]["root"], cfg["umu"]["n_retain_eval"], 0, log, cfg["umu"]["forget_split"])
    facts = [f for f in sorted(facts_for(dfs, dc), key=lambda f: f["pair_id"]) if f["set"] == "forget" and f["name"] in f["q_uni"]]
    adapter = get_adapter({**cfg["model"], "id": cfg["umu"]["ckpt"], "processor_id": cfg["model"]["id"]})
    model = PeftModel.from_pretrained(adapter.load_model(), str(run_dir / "ckpt" / f"step{args.step}"), is_trainable=False)
    model.eval()
    n_layers = len(decoder_layers(model))
    image_id = getattr(model.config, "image_token_index", 32000)
    log(f"{args.run_id} step {args.step}: {len(facts)} forget dc facts with the name in the question")
    recs, checked, skipped = [], False, 0
    for i in range(0, len(facts), args.batch_img):
        chunk = facts[i:i + args.batch_img]
        rows_t, rows_np = zip(*[rows_for(f) for f in chunk])
        res = {}
        for tag, rows in (("T", rows_t), ("NP", rows_np)):
            batch = encode_teacher_forced(adapter, list(rows))
            t = first_target(tools.answer_targets(batch))
            groups, mism = token_groups_umu(adapter, model, list(rows), batch, t)
            name = groups["name"]
            if mism or not bool((name.sum(1) > 0).all()):
                raise RuntimeError(f"name tokens not found in every row ({mism} re-tokenisation mismatches)")
            keep_name = (~name).float()
            keep_ctrl = (~control_mask(groups, name)).float()
            r = {"u": first_lp(adapter, model, batch, t), "0": first_lp(adapter, model, batch, t, vanilla=True),
                 "u_nameoff": first_lp(adapter, model, batch, t, keep=keep_name)}
            if not checked and tag == "NP":
                ones = first_lp(adapter, model, batch, t, keep=torch.ones_like(keep_name))
                rws, cls = prefix_positions(batch, t, "own", image_id, 0, 0)
                empty = torch.zeros(0, dtype=torch.long, device=rws.device)
                nob = first_lp(adapter, model, batch, t, block=(range(n_layers), empty, empty))
                e1 = max(abs(a - b) for a, b in zip(ones[0], r["u"][0]))
                e2 = max(abs(a - b) for a, b in zip(nob[0], r["u"][0]))
                log(f"  checks: keep=1 vs theta_u {e1:.2e}; empty attention block vs unhooked {e2:.2e} (tolerance {TOL}); "
                    f"name tokens per row {name.sum(1).tolist()}; control tokens per row {(keep_ctrl == 0).sum(1).tolist()}")
                if e1 > TOL or e2 > TOL:
                    raise RuntimeError("a hook is not neutral")
                checked = True
            if tag == "NP":
                rws, cls = prefix_positions(batch, t, "own", image_id, 0, 0)
                blk = (range(n_layers), rws, cls)
                r.update({"u_ko": first_lp(adapter, model, batch, t, block=blk),
                          "0_ko": first_lp(adapter, model, batch, t, vanilla=True, block=blk),
                          "u_ko_nameoff": first_lp(adapter, model, batch, t, keep=keep_name, block=blk),
                          "u_ko_ctrloff": first_lp(adapter, model, batch, t, keep=keep_ctrl, block=blk)})
            res[tag] = r
        for j, f in enumerate(chunk):
            rec = {"pair_id": f["pair_id"], "field": f["field"]}
            for tag, r in res.items():
                for k, (lp, top) in r.items():
                    rec[f"{tag}/{k}/lp"], rec[f"{tag}/{k}/top1"] = lp[j], bool(top[j])
            recs.append(rec)
        log(f"  {min(i + args.batch_img, len(facts))}/{len(facts)} facts")
    model = None
    free()
    write_jsonl(out / "items.jsonl", recs)
    m = lambda k: float(np.mean([r[k] for r in recs]))  # noqa: E731
    F = {"T": m("T/0/lp") - m("T/u/lp"), "T_nameoff": m("T/0/lp") - m("T/u_nameoff/lp"),
         "NP_visible": m("NP/0/lp") - m("NP/u/lp"), "NP_visible_nameoff": m("NP/0/lp") - m("NP/u_nameoff/lp"),
         "NP_ko": m("NP/0_ko/lp") - m("NP/u_ko/lp"), "NP_ko_nameoff": m("NP/0_ko/lp") - m("NP/u_ko_nameoff/lp"),
         "NP_ko_ctrloff": m("NP/0_ko/lp") - m("NP/u_ko_ctrloff/lp")}
    top1 = {k: m(k) for k in recs[0] if k.endswith("/top1")}
    a = F["NP_visible"] <= A_VIS and F["NP_ko"] >= A_KO * F["T"]
    b = F["NP_ko_nameoff"] <= B_MAX * F["NP_ko"]
    c = F["NP_ko_ctrloff"] >= C_MIN * F["NP_ko"]
    verdict = "closed" if (a and b and c) else ("not closed" if not b else "partial")
    atomic_write_json(out / "summary.json", {"run_id": args.run_id, "step": args.step, "n_facts": len(recs),
                                             "forgetting_nats": F, "top1": top1, "checks": {"6a": a, "6b": b, "6c": c},
                                             "thresholds": {"A_vis": A_VIS, "A_ko": A_KO, "B_max": B_MAX, "C_min": C_MIN},
                                             "verdict": verdict})
    for k, v in F.items():
        log(f"  F {k:22s} {v:6.2f} nats")
    for k, v in sorted(top1.items()):
        log(f"  top-1 {k:24s} {v:.2f}")
    log(f"link 6 (DEV-029): 6a {a}, 6b {b}, 6c {c} -> {verdict}")
    log(f"wrote {out / 'summary.json'}")


if __name__ == "__main__":
    main()
