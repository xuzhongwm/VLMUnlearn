"""UMU track, DEV-030 stage 3: the three core tests, model-agnostic (LLaVA, Qwen2-VL; prompts from adapter.prompt).

On the forget dc facts of an unlearned checkpoint (first answer token, teacher forced; F = lp_theta0 - lp_thetau under
the same input and knockout):
  T    name question, no image                      F(T), and F with the LoRA off at the name tokens / at as many
                                                    other text tokens (control)
  NP   name question + the person's own portrait     F visible; F with attention from the positions after the image to
                                                    the image tokens blocked in all layers (knockout); knockout + name-off
  NB   name question + a blank grey image            F (reported)
Reading (DEV-030):
  Q2  F(NP knockout) >= 0.5 F(T), theta_0 top-1 under the knockout >= 0.6
  Q3  F(T name-off) <= 0.3 F(T) and F(T control-off) >= 0.7 F(T)
  (Q1, the leak, comes from the checkpoint's own readout: forget dc img and txt+img >= 0.85.)
  DEV-029's link-6 numbers (6a/6b/6c) are reported too.
For LLaVA, adapter.prompt gives exactly the prompts of mech_umu_link6.py, so seed 0 must reproduce DEV-029.

  python scripts/mech_umu_core3.py --config <model config> --run-id <run> --step <N> --set umu.n_retain_eval=100 [umu.ckpt=...]
Writes results/<model>/mech_umu/<run>/step<N>/core3/{summary.json,items.jsonl,log.txt}.
"""
import contextlib
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np  # noqa: E402
import torch  # noqa: E402

from scripts.mech_umu_crossmodal import block_attention, prefix_positions  # noqa: E402
from scripts.mech_umu_form_sweep import blank_image_bytes, facts_for  # noqa: E402
from scripts.mech_umu_link6 import control_mask, first_lp  # noqa: E402
from vlmu.config import base_parser, resolve  # noqa: E402
from vlmu.mech import tools  # noqa: E402
from vlmu.mech.hooks import decoder_layers, lm_parts  # noqa: E402
from vlmu.models import get_adapter  # noqa: E402
from vlmu.runs import free, model_dir, results_dir  # noqa: E402
from vlmu.umu import encode_teacher_forced, first_target, load_umu, token_groups_umu  # noqa: E402
from vlmu.utils import Logger, atomic_write_json, read_json, write_jsonl  # noqa: E402

TOL = 0.05
Q2_KO, Q2_V0, Q3_OFF, Q3_CTRL = 0.5, 0.6, 0.3, 0.7          # DEV-030
A_VIS, B_MAX, C_MIN = 1.0, 0.3, 0.7                          # DEV-029 (reported)


def main():
    p = base_parser(__doc__)
    p.add_argument("--run-id", required=True)
    p.add_argument("--step", type=int, default=300)
    p.add_argument("--batch-img", type=int, default=4)
    args = p.parse_args()
    cfg = resolve(args)
    run_dir = model_dir(cfg) / "umu_unlearn" / args.run_id
    out = results_dir(cfg) / "mech_umu" / args.run_id / f"step{args.step}" / "core3"
    log = Logger(out / "log.txt")
    from peft import PeftModel
    dc = set(read_json(run_dir / "baseline_dc.json")["pair_ids"])
    dfs = load_umu(cfg["paths"]["root"], cfg["umu"]["n_retain_eval"], 0, log, cfg["umu"]["forget_split"])
    facts = [f for f in sorted(facts_for(dfs, dc), key=lambda f: f["pair_id"]) if f["set"] == "forget" and f["name"] in f["q_uni"]]
    adapter = get_adapter({**cfg["model"], "id": cfg["umu"]["ckpt"], "processor_id": cfg["model"]["id"]})
    model = PeftModel.from_pretrained(adapter.load_model(), str(run_dir / "ckpt" / f"step{args.step}"), is_trainable=False)
    model.eval()
    n_layers = len(decoder_layers(model))
    image_id = lm_parts(model)[2]
    blank = blank_image_bytes()
    log(f"{args.run_id} step {args.step} ({cfg['model']['id']}): {len(facts)} forget dc facts; image token id {image_id}")
    log("prompts: " + repr(adapter.prompt("<q>", False)) + " / " + repr(adapter.prompt("<q>", True)))

    def rows(f, form):
        img = {"T": None, "NP": f["img"], "NB": blank}[form]
        return {"pair_id": f["pair_id"], "uid": f"{f['pair_id']}/{form}", "correct_text": f["answer_text"], "name": f["name"],
                "prompt": adapter.prompt(f["q_uni"], img is not None), "image_bytes": img}

    recs, checked = [], False
    for i in range(0, len(facts), args.batch_img):
        chunk = facts[i:i + args.batch_img]
        res = {}
        for form in ("T", "NP", "NB"):
            rr = [rows(f, form) for f in chunk]
            batch = encode_teacher_forced(adapter, rr)
            t = first_target(tools.answer_targets(batch))
            r = {"u": first_lp(adapter, model, batch, t), "0": first_lp(adapter, model, batch, t, vanilla=True)}
            if form in ("T", "NP"):
                groups, mism = token_groups_umu(adapter, model, rr, batch, t)
                name = groups["name"]
                if mism or not bool((name.sum(1) > 0).all()):
                    raise RuntimeError(f"name tokens not found in every row ({mism} mismatches) for {form}")
                keep_name, keep_ctrl = (~name).float(), (~control_mask(groups, name)).float()
                r["u_nameoff"] = first_lp(adapter, model, batch, t, keep=keep_name)
                r["u_ctrloff"] = first_lp(adapter, model, batch, t, keep=keep_ctrl)
            if form == "NP":
                rws, cls = prefix_positions(batch, t, "own", image_id, 0, 0)
                if not checked:
                    empty = torch.zeros(0, dtype=torch.long, device=rws.device)
                    nob = first_lp(adapter, model, batch, t, block=(range(n_layers), empty, empty))
                    ones = first_lp(adapter, model, batch, t, keep=torch.ones_like(keep_name))
                    e1 = max(abs(a - b) for a, b in zip(ones[0], r["u"][0]))
                    e2 = max(abs(a - b) for a, b in zip(nob[0], r["u"][0]))
                    log(f"  checks: keep=1 vs theta_u {e1:.2e}; empty block vs unhooked {e2:.2e} (tolerance {TOL}); {len(cls)} image "
                        f"tokens, {len(rws)} reading positions; name tokens per row {name.sum(1).tolist()}")
                    if e1 > TOL or e2 > TOL:
                        raise RuntimeError("a hook is not neutral for this model")
                    checked = True
                blk = (range(n_layers), rws, cls)
                r.update({"u_ko": first_lp(adapter, model, batch, t, block=blk),
                          "0_ko": first_lp(adapter, model, batch, t, vanilla=True, block=blk),
                          "u_ko_nameoff": first_lp(adapter, model, batch, t, keep=keep_name, block=blk),
                          "u_ko_ctrloff": first_lp(adapter, model, batch, t, keep=keep_ctrl, block=blk)})
            res[form] = r
        for j, f in enumerate(chunk):
            rec = {"pair_id": f["pair_id"], "field": f["field"]}
            for form, r in res.items():
                for k, (lp, top) in r.items():
                    rec[f"{form}/{k}/lp"], rec[f"{form}/{k}/top1"] = lp[j], bool(top[j])
            recs.append(rec)
        log(f"  {min(i + args.batch_img, len(facts))}/{len(facts)} facts")
    model = None
    free()
    write_jsonl(out / "items.jsonl", recs)
    m = lambda k: float(np.mean([r[k] for r in recs]))  # noqa: E731
    F = {"T": m("T/0/lp") - m("T/u/lp"), "T_nameoff": m("T/0/lp") - m("T/u_nameoff/lp"),
         "T_ctrloff": m("T/0/lp") - m("T/u_ctrloff/lp"), "NB": m("NB/0/lp") - m("NB/u/lp"),
         "NP_visible": m("NP/0/lp") - m("NP/u/lp"), "NP_ko": m("NP/0_ko/lp") - m("NP/u_ko/lp"),
         "NP_ko_nameoff": m("NP/0_ko/lp") - m("NP/u_ko_nameoff/lp"), "NP_ko_ctrloff": m("NP/0_ko/lp") - m("NP/u_ko_ctrloff/lp")}
    top1 = {k: m(k) for k in recs[0] if k.endswith("/top1")}
    q2 = F["NP_ko"] >= Q2_KO * F["T"] and top1["NP/0_ko/top1"] >= Q2_V0
    q3 = F["T_nameoff"] <= Q3_OFF * F["T"] and F["T_ctrloff"] >= Q3_CTRL * F["T"]
    l6 = {"6a": F["NP_visible"] <= A_VIS and F["NP_ko"] >= 0.5 * F["T"], "6b": F["NP_ko_nameoff"] <= B_MAX * F["NP_ko"],
          "6c": F["NP_ko_ctrloff"] >= C_MIN * F["NP_ko"]}
    atomic_write_json(out / "summary.json", {"run_id": args.run_id, "step": args.step, "model": cfg["model"]["id"],
                                             "n_facts": len(recs), "forgetting_nats": F, "top1": top1,
                                             "Q2": q2, "Q3": q3, "link6": l6})
    for k, v in F.items():
        log(f"  F {k:16s} {v:6.2f} nats")
    for k, v in sorted(top1.items()):
        log(f"  top-1 {k:22s} {v:.2f}")
    log(f"DEV-030: Q2 knockout {'PASS' if q2 else 'FAIL'}; Q3 name localisation {'PASS' if q3 else 'FAIL'}; link 6 {l6}")
    log(f"wrote {out / 'summary.json'}")


if __name__ == "__main__":
    main()
