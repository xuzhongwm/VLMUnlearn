"""UMU track, DEV-032: at which depth does the text-only edit act, overall and on the name tokens, and is it triggered by
the portrait-only question? Model-agnostic (LLaVA, Qwen2-VL; prompts from adapter.prompt).

On the forget dc facts of an unlearned checkpoint (first answer token, teacher forced; F = lp_theta0 - lp_thetau under
the same input):
  T   name question, no image. F(T), and F with the LoRA switched off
        - in one 4-layer window of decoder layers, at every position ("all")
        - in one 4-layer window, at the name tokens only ("name")
        - in all layers at the name tokens (as core3's Q3), and the LoRA kept only at the name tokens ("name only")
  NI  the portrait-only question: the person's portrait, the name replaced by "this person" (the face route of
      mech_umu_convergence.py). F(NI)
Recovery R = 1 - F(off) / F(T): 1 = that switch-off removes all forgetting.
Reading (DEV-032), applied unchanged to every model:
  D1  R_all(layers 0-3) >= 0.5        D2  R_all <= 0.3 for every window starting at layer >= 12
  D3  R_name(layers 0-3) >= 0.5       early name edit if D1-D3; not early if R_all(0-3) < 0.3 and R_name(0-3) < 0.3
  N1  F(NI) <= 1.0 nat and theta_u's top-1 on NI >= theta_0's - 0.1 (the portrait-only question does not trigger it)
With --split-at L* (DEV-033; L* = the model's convergence layer on theta_0): the LoRA is also switched off at the name
tokens (and at every position) in layers 0..L*-1 ("pre") or L*..last ("post"); output folder name_depth_split<L*>.
  (a) R_pre_name >= 0.7   (b) R_post_name <= 0.1   -> "before the merge"; "not before the merge" if R_pre_name < 0.5
  or R_post_name > 0.2; otherwise partial. Secondary: (c) R_post_all <= 0.3.
Validity checks (first batch; abort otherwise): keep = 1 on every LoRA module reproduces theta_u and keep = 0 reproduces
theta_0 (<= 0.05 nats); every LoRA module sits in a decoder layer; every row has >= 1 name token.

  python scripts/mech_umu_name_depth.py --config <model config> --run-id <run> --step <N> --set umu.n_retain_eval=100 [umu.ckpt=...]
Writes results/<model>/mech_umu/<run>/step<N>/name_depth/{summary.json,items.jsonl,log.txt,name_depth.png}.
"""
import contextlib
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np  # noqa: E402
import torch  # noqa: E402

from scripts.mech_umu_form_sweep import facts_for  # noqa: E402
from vlmu.config import base_parser, resolve  # noqa: E402
from vlmu.mech import tools  # noqa: E402
from vlmu.mech.hooks import decoder_layers, layer_index, lora_modules, mask_contributions  # noqa: E402
from vlmu.models import get_adapter  # noqa: E402
from vlmu.runs import free, model_dir, results_dir  # noqa: E402
from vlmu.umu import encode_teacher_forced, first_target, load_umu, token_groups_umu  # noqa: E402
from vlmu.utils import Logger, atomic_write_json, read_json, write_jsonl  # noqa: E402

TOL = 0.05
D_EARLY, D_LATE, D_LATE_FROM, D_NOT = 0.5, 0.3, 12, 0.3      # DEV-032
N_MAX, N_TOP = 1.0, 0.1
S_PRE, S_POST, S_NOT_PRE, S_NOT_POST, S_POST_ALL = 0.7, 0.1, 0.5, 0.2, 0.3      # DEV-033
WIN = 4


@torch.no_grad()
def first_lp(adapter, model, batch, t, vanilla=False, names=None, keep=None):
    """(log p of the correct first answer token, is it the argmax) per row; keep [B, T] applied to the LoRA modules
    `names` only (the others stay on)."""
    with contextlib.ExitStack() as st:
        if vanilla:
            st.enter_context(model.disable_adapter())
        if names is not None:
            st.enter_context(mask_contributions(model, names, keep))
        at = tools.forward_logits(adapter, model, batch)[t["b"], t["pos"]].float()
    lp = at.log_softmax(-1).gather(-1, t["tok"][:, None]).squeeze(-1)
    return lp.cpu().tolist(), (at.argmax(-1) == t["tok"]).cpu().tolist()


def plot(s, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    INK2, MUTED, GRID = "#52514e", "#898781", "#e1e0d9"
    wins = list(s["R_all"])
    x = np.arange(len(wins))
    fig, ax = plt.subplots(figsize=(5.6, 3.0), dpi=200)
    ax.bar(x - 0.2, [s["R_all"][w] for w in wins], 0.4, color="#2a78d6", label="LoRA off in the window, all positions")
    ax.bar(x + 0.2, [s["R_name"][w] for w in wins], 0.4, color="#eb6834", label="LoRA off in the window, name tokens only")
    ax.axhline(s["R_name_all_layers"], color="#eb6834", lw=0.8, ls=(0, (3, 2)), label="name tokens, all layers")
    if s.get("split_at") is not None:
        ax.axvline(s["split_at"] / WIN - 0.5, color=MUTED, lw=0.8, label=f"convergence L* = {s['split_at']}")
    ax.set_xticks(x, [f"L{w}" for w in wins], fontsize=6.5)
    ax.set_xlabel("decoder layers switched off", fontsize=7, color=INK2)
    ax.set_ylabel("recovery R = 1 − F(off) / F(T)", fontsize=7, color=INK2)
    ax.set_ylim(min(-0.1, min(min(s["R_all"].values()), min(s["R_name"].values())) - 0.05), 1.05)
    ax.set_title(f"{s['model'].split('/')[-1]} {s['run_id']} step {s['step']}: depth of the edit "
                 f"(n={s['n_facts']}, F(T)={s['F']['T']:.1f} nats)", fontsize=7, loc="left")
    ax.legend(frameon=False, fontsize=5.5)
    ax.grid(axis="y", color=GRID, lw=0.5)
    ax.tick_params(labelsize=6.5, colors=MUTED)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def main():
    p = base_parser(__doc__)
    p.add_argument("--run-id", required=True)
    p.add_argument("--step", type=int, default=300)
    p.add_argument("--batch-txt", type=int, default=16)
    p.add_argument("--batch-img", type=int, default=4)
    p.add_argument("--max-facts", type=int, default=0, help="smoke test")
    p.add_argument("--split-at", type=int, default=None, help="DEV-033: the model's convergence layer L*")
    args = p.parse_args()
    cfg = resolve(args)
    run_dir = model_dir(cfg) / "umu_unlearn" / args.run_id
    out = results_dir(cfg) / "mech_umu" / args.run_id / f"step{args.step}" / (
        "name_depth" if args.split_at is None else f"name_depth_split{args.split_at}")
    log = Logger(out / "log.txt")
    from peft import PeftModel
    dc = set(read_json(run_dir / "baseline_dc.json")["pair_ids"])
    dfs = load_umu(cfg["paths"]["root"], cfg["umu"]["n_retain_eval"], 0, log, cfg["umu"]["forget_split"])
    facts = [f for f in sorted(facts_for(dfs, dc), key=lambda f: f["pair_id"]) if f["set"] == "forget" and f["name"] in f["q_uni"]]
    if args.max_facts:
        facts = facts[:args.max_facts]
    adapter = get_adapter({**cfg["model"], "id": cfg["umu"]["ckpt"], "processor_id": cfg["model"]["id"]})
    model = PeftModel.from_pretrained(adapter.load_model(), str(run_dir / "ckpt" / f"step{args.step}"), is_trainable=False)
    model.eval()
    n_layers = len(decoder_layers(model))
    mods = list(lora_modules(model))
    by_layer = {n: layer_index(n) for n in mods}
    if any(v is None or not 0 <= v < n_layers for v in by_layer.values()):
        raise RuntimeError("a LoRA module is not in a decoder layer")
    wins = [(a, min(a + WIN, n_layers) - 1) for a in range(0, n_layers, WIN)]
    win_mods = {f"{a}-{b}": [n for n in mods if a <= by_layer[n] <= b] for a, b in wins}
    split_mods = {} if args.split_at is None else {"pre": [n for n in mods if by_layer[n] < args.split_at],
                                                   "post": [n for n in mods if by_layer[n] >= args.split_at]}
    if split_mods and not (split_mods["pre"] and split_mods["post"]):
        raise RuntimeError(f"--split-at {args.split_at} leaves no LoRA module on one side")
    log(f"{args.run_id} step {args.step} ({cfg['model']['id']}): {len(facts)} forget dc facts; {n_layers} decoder layers; "
        f"{len(mods)} LoRA modules; windows {list(win_mods)}")
    log("prompts: T " + repr(adapter.prompt("<q>", False)) + " / NI " + repr(adapter.prompt("<q with 'this person'>", True)))

    recs, checked = [], False
    for form, bs in (("T", args.batch_txt), ("NI", args.batch_img)):
        for i in range(0, len(facts), bs):
            chunk = facts[i:i + bs]
            rr = [{"pair_id": f["pair_id"], "uid": f"{f['pair_id']}/{form}", "correct_text": f["answer_text"], "name": f["name"],
                   "prompt": adapter.prompt(f["q_uni"] if form == "T" else f["q_uni"].replace(f["name"], "this person"),
                                            form == "NI"),
                   "image_bytes": f["img"] if form == "NI" else None} for f in chunk]
            batch = encode_teacher_forced(adapter, rr)
            t = first_target(tools.answer_targets(batch))
            r = {"u": first_lp(adapter, model, batch, t), "0": first_lp(adapter, model, batch, t, vanilla=True)}
            if form == "T":
                groups, mism = token_groups_umu(adapter, model, rr, batch, t)
                name = groups["name"]
                if mism or not bool((name.sum(1) > 0).all()):
                    raise RuntimeError(f"name tokens not found in every row ({mism} mismatches)")
                ones, zeros, keep_name = torch.ones_like(name).float(), torch.zeros_like(name).float(), (~name).float()
                if not checked:
                    e1 = max(abs(a - b) for a, b in zip(first_lp(adapter, model, batch, t, names=mods, keep=ones)[0], r["u"][0]))
                    e0 = max(abs(a - b) for a, b in zip(first_lp(adapter, model, batch, t, names=mods, keep=zeros)[0], r["0"][0]))
                    log(f"  checks: keep=1 vs theta_u {e1:.2e}; keep=0 vs theta_0 {e0:.2e} (tolerance {TOL}); name tokens "
                        f"per row {name.sum(1).tolist()}")
                    if e1 > TOL or e0 > TOL:
                        raise RuntimeError("the LoRA mask is not exact for this model")
                    checked = True
                r["nameoff"] = first_lp(adapter, model, batch, t, names=mods, keep=keep_name)
                r["nameonly"] = first_lp(adapter, model, batch, t, names=mods, keep=name.float())
                for w, ms in {**win_mods, **split_mods}.items():
                    r[f"off_all/{w}"] = first_lp(adapter, model, batch, t, names=ms, keep=zeros)
                    r[f"off_name/{w}"] = first_lp(adapter, model, batch, t, names=ms, keep=keep_name)
            for j, f in enumerate(chunk):
                rec = {"pair_id": f["pair_id"], "field": f["field"], "form": form}
                for k, (lp, top) in r.items():
                    rec[f"{k}/lp"], rec[f"{k}/top1"] = lp[j], bool(top[j])
                recs.append(rec)
            log(f"  {form}: {min(i + bs, len(facts))}/{len(facts)} facts")
    model = None
    free()
    write_jsonl(out / "items.jsonl", recs)
    T = [r for r in recs if r["form"] == "T"]
    NI = [r for r in recs if r["form"] == "NI"]
    m = lambda rs, k: float(np.mean([r[k] for r in rs]))  # noqa: E731
    F = {"T": m(T, "0/lp") - m(T, "u/lp"), "NI": m(NI, "0/lp") - m(NI, "u/lp"),
         "T_nameoff": m(T, "0/lp") - m(T, "nameoff/lp"), "T_nameonly": m(T, "0/lp") - m(T, "nameonly/lp")}
    R = lambda key: 1 - (m(T, "0/lp") - m(T, f"{key}/lp")) / F["T"]  # noqa: E731
    R_all = {w: R(f"off_all/{w}") for w in win_mods}
    R_name = {w: R(f"off_name/{w}") for w in win_mods}
    top1 = {"T": {k: m(T, f"{k}/top1") for k in ["u", "0", "nameoff", "nameonly"]
                  + [f"off_{g}/{w}" for g in ("all", "name") for w in {**win_mods, **split_mods}]},
            "NI": {k: m(NI, f"{k}/top1") for k in ("u", "0")}}
    first_w = next(iter(win_mods))
    late = [w for w in win_mods if int(w.split("-")[0]) >= D_LATE_FROM]
    d1, d3 = R_all[first_w] >= D_EARLY, R_name[first_w] >= D_EARLY
    d2 = max(R_all[w] for w in late) <= D_LATE
    verdict = ("early name edit" if d1 and d2 and d3 else
               "not early" if R_all[first_w] < D_NOT and R_name[first_w] < D_NOT else "partial")
    n1 = F["NI"] <= N_MAX and top1["NI"]["u"] >= top1["NI"]["0"] - N_TOP
    s = {"run_id": args.run_id, "step": args.step, "model": cfg["model"]["id"], "n_layers": n_layers, "n_facts": len(T),
         "F": F, "R_all": R_all, "R_name": R_name, "R_name_all_layers": 1 - F["T_nameoff"] / F["T"],
         "R_name_only": 1 - F["T_nameonly"] / F["T"], "top1": top1,
         "D1": d1, "D2": d2, "D3": d3, "depth_verdict": verdict, "N1": n1}
    if split_mods:
        sp = {f"R_{side}_{g}": R(f"off_{g}/{side}") for side in ("pre", "post") for g in ("name", "all")}
        a, b, c = sp["R_pre_name"] >= S_PRE, sp["R_post_name"] <= S_POST, sp["R_post_all"] <= S_POST_ALL
        sv = ("before the merge" if a and b else
              "not before the merge" if sp["R_pre_name"] < S_NOT_PRE or sp["R_post_name"] > S_NOT_POST else "partial")
        s.update({"split_at": args.split_at, **sp, "a": a, "b": b, "c": c, "split_verdict": sv})
    atomic_write_json(out / "summary.json", s)
    log(f"  F(T) {F['T']:.2f} nats (top-1 θu {top1['T']['u']:.2f}, θ0 {top1['T']['0']:.2f}); F(NI) {F['NI']:.2f} "
        f"(top-1 θu {top1['NI']['u']:.2f}, θ0 {top1['NI']['0']:.2f})")
    log(f"  R name-off all layers {s['R_name_all_layers']:.2f} (top-1 {top1['T']['nameoff']:.2f}); LoRA only at the name "
        f"{s['R_name_only']:.2f}")
    for w in win_mods:
        log(f"  window L{w}: R_all {R_all[w]:+.2f} (top-1 {top1['T'][f'off_all/{w}']:.2f}) | R_name {R_name[w]:+.2f} "
            f"(top-1 {top1['T'][f'off_name/{w}']:.2f})")
    log(f"DEV-032: D1 {d1} D2 {d2} D3 {d3} -> {verdict}; N1 (portrait-only question not triggered) {n1}")
    if split_mods:
        for side in ("pre", "post"):
            log(f"  {side} L*={args.split_at}: R_name {s[f'R_{side}_name']:+.2f} (top-1 {top1['T'][f'off_name/{side}']:.2f}) | "
                f"R_all {s[f'R_{side}_all']:+.2f} (top-1 {top1['T'][f'off_all/{side}']:.2f})")
        log(f"DEV-033: (a) {s['a']} (b) {s['b']} -> {s['split_verdict']}; (c) R_post_all <= {S_POST_ALL}: {s['c']}")
    plot(s, out / "name_depth.png")
    log(f"wrote {out / 'summary.json'} and name_depth.png")


if __name__ == "__main__":
    main()
