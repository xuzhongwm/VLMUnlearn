"""UMU track, H1 vs H1' (research.md, Hypotheses & Predictions): LoRA layer masking on an unlearning checkpoint.

theta_u = UMU's vanilla + the run's LoRA at --step. Each condition keeps the unlearning LoRA on some decoder layers
and switches it off on the others (vlmu/mech/tools.py:lora_layer_mask; exact, Milestone J check J4):
  unlearned / vanilla                     all layers kept / none kept (the two reference points)
  remove_early|mid|late                   LoRA off on layers 0-9 | 10-19 | 20-31, on everywhere else
  only_early|mid|late                     LoRA on only in that band
  remove_<a>-<a+3>                        LoRA off on one 4-layer window (8 windows)
Readouts on the run's doubly-correct facts:
  teacher-forced log p of the correct option text after the UMU-protocol prompt (first answer token and the summed
  answer tokens), per condition, for forget txt / img / txt+img and retain txt / img;
  recovery = (x_cond - x_unlearned) / (x_vanilla - x_unlearned) for the name-only routes (1 = fully restored);
  generation + exact match (UMU protocol) for the key conditions on the facts the name route lost at this step.

Predictions:
  H1  (late gate)          remove_late restores the name route; only_late reproduces most of the suppression
  H1' (early name route)   remove_early restores it; only_early reproduces most of the suppression
Masks are not additive (an early edit changes the input of every later layer), so read the remove_ and only_
conditions together. The portrait routes should barely move under any mask (control).

  python scripts/mech_umu_layer_mask.py --config configs/llava/umu_npo_rt.yaml --run-id <run_id> --step <N>
Writes results/llava/mech_umu/<run_id>/step<N>/layer_mask.json, layer_mask.png and gen_<condition>.jsonl.
"""
import collections
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np  # noqa: E402
import torch  # noqa: E402

from vlmu.config import base_parser, resolve  # noqa: E402
from vlmu.mech import tools  # noqa: E402
from vlmu.mech.hooks import decoder_layers  # noqa: E402
from vlmu.models import get_adapter  # noqa: E402
from vlmu.runs import free, model_dir, results_dir  # noqa: E402
from vlmu.umu import encode_teacher_forced, first_target, generate_protocol, load_umu, protocol_items, \
    umu_run_id  # noqa: E402
from vlmu.utils import Logger, atomic_write_json, read_json, read_jsonl, tracked_run  # noqa: E402

FORMS = {"forget": ("txt", "img", "txt+img"), "retain": ("txt", "img")}
BANDS = {"early": (0, 10), "mid": (10, 20), "late": (20, None)}
GEN_CONDITIONS = ("unlearned", "vanilla", "remove_early", "remove_mid", "remove_late", "only_early", "only_late")


def conditions(n):
    out = {"unlearned": list(range(n)), "vanilla": []}
    for name, (a, b) in BANDS.items():
        b = n if b is None else b
        out[f"remove_{name}"] = [l for l in range(n) if not a <= l < b]
        out[f"only_{name}"] = list(range(a, b))
    for a in range(0, n, 4):
        out[f"remove_{a}-{min(a + 3, n - 1)}"] = [l for l in range(n) if not a <= l < a + 4]
    return out


@torch.no_grad()
def score_condition(adapter, model, batches, keep):
    """{uid: {'first': lp, 'answer': lp}} with the LoRA kept only on `keep` layers."""
    out = {}
    with tools.lora_layer_mask(model, keep):
        for rows, batch, t_all, t_first in batches:
            logits = tools.forward_logits(adapter, model, batch)
            for name, t in (("answer", t_all), ("first", t_first)):
                lp = tools.target_logprob(logits[t["b"], t["pos"]], t["tok"])
                per = tools._per_row(lp, t).float().cpu().tolist()
                for r, v in zip(rows, per):
                    out.setdefault(r["uid"], {})[name] = round(v, 4)
    return out


def group_of(r, lost):
    g = f"{r['set']}/{r['modality']}"
    if r["modality"] == "txt":
        return [g, f"{g}/{'lost' if lost.get((r['set'], r['pair_id'])) else 'kept'}"]
    return [g]


def summarize(items, scores, lost):
    """{condition: {group: {'first': mean, 'answer': mean, 'n': n}}} plus recovery for name-only groups."""
    out = {}
    for cond, sc in scores.items():
        acc = collections.defaultdict(lambda: {"first": [], "answer": []})
        for r in items:
            for g in group_of(r, lost):
                for t in ("first", "answer"):
                    acc[g][t].append(sc[r["uid"]][t])
        out[cond] = {g: {"n": len(v["first"]), **{t: float(np.mean(v[t])) for t in ("first", "answer")}} for g, v in acc.items()}
    for cond in out:
        for g, v in out[cond].items():
            if "/txt" not in g:
                continue
            for t in ("first", "answer"):
                u, z = out["unlearned"][g][t], out["vanilla"][g][t]
                v[f"recovery_{t}"] = (v[t] - u) / (z - u) if abs(z - u) > 1e-6 else None
    return out


def plot(summ, n_layers, title, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    INK2, MUTED, GRID, AXIS = "#52514e", "#898781", "#e1e0d9", "#c3c2b7"
    series = [("forget/txt/lost", "Forget, name only, lost facts", "#2a78d6", "-"),
              ("retain/txt/lost", "Retain, name only, lost facts", "#2a78d6", "--")]
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(10.5, 3.4), dpi=200, gridspec_kw={"width_ratios": [1.5, 1]})
    for ax in (a1, a2):
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
        ax.grid(axis="y", color=GRID, lw=0.6)
        ax.set_axisbelow(True)
        ax.axhline(0, color=AXIS, lw=0.9)
        ax.axhline(1, color=AXIS, lw=0.9)
        ax.tick_params(colors=MUTED)
    wins = sorted([c for c in summ if c.startswith("remove_") and "-" in c], key=lambda c: int(c.split("_")[1].split("-")[0]))
    xs = [int(c.split("_")[1].split("-")[0]) + 1.5 for c in wins]
    for g, label, color, ls in series:
        if g not in summ["unlearned"]:
            continue
        ys = [summ[c][g].get("recovery_first") for c in wins]
        a1.plot(xs, ys, color=color, ls=ls, lw=1.8, marker="o", ms=4, label=f"{label} (n={summ['unlearned'][g]['n']})")
    a1.set_xlabel("decoder layers with the LoRA switched off (4-layer window)", color=INK2)
    a1.set_ylabel("recovery of log p(first answer token)\n0 = still unlearned, 1 = back to vanilla", color=INK2)
    a1.set_xticks(xs, [c.split("_")[1] for c in wins], fontsize=7)
    a1.legend(frameon=False, fontsize=7.5)
    # left three bars: how much switching a band OFF restores (recovery); right three: how much of the suppression a band
    # produces ON ITS OWN (1 - recovery of only_<band>). Under H1 both peak at 20-31, under H1' both at 0-9.
    bands = [("remove_early", False), ("remove_mid", False), ("remove_late", False),
             ("only_early", True), ("only_mid", True), ("only_late", True)]
    x = np.array([0, 1, 2, 3.6, 4.6, 5.6])
    for k, (g, label, color, ls) in enumerate(series):
        if g not in summ["unlearned"]:
            continue
        ys = []
        for c, alone in bands:
            r = summ[c][g].get("recovery_first")
            ys.append(0 if r is None else (1 - r if alone else r))
        a2.bar(x + (k - 0.5) * 0.36, ys, width=0.32, color=color, alpha=1.0 if ls == "-" else 0.45,
               label=label.replace(", lost facts", ""))
    a2.set_xticks(x, ["0-9", "10-19", "20-31", "0-9", "10-19", "20-31"], fontsize=7.5)
    a2.text(1, -0.22, "restored when this band is OFF", ha="center", fontsize=7.5, color=INK2, transform=a2.get_xaxis_transform())
    a2.text(4.6, -0.22, "suppression from this band ALONE", ha="center", fontsize=7.5, color=INK2, transform=a2.get_xaxis_transform())
    a2.set_ylabel("fraction of the unlearning effect", color=INK2)
    a2.legend(frameon=False, fontsize=7)
    fig.suptitle(title, x=0.01, ha="left", fontsize=9, color=INK2)
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def main():
    p = base_parser(__doc__)
    p.add_argument("--step", type=int, required=True)
    p.add_argument("--run-id", default=None, help="default: derived from the config (as scripts/unlearn_umu.py)")
    p.add_argument("--batch-img", type=int, default=8)
    p.add_argument("--batch-txt", type=int, default=16)
    p.add_argument("--no-gen", action="store_true", help="skip the generation readout")
    args = p.parse_args()
    cfg = resolve(args)
    run_id = args.run_id or umu_run_id(cfg)
    run_dir = model_dir(cfg) / "umu_unlearn" / run_id
    out = results_dir(cfg) / "mech_umu" / run_id / f"step{args.step}"
    log = Logger(out / "layer_mask_log.txt")
    adapter_dir = run_dir / "ckpt" / f"step{args.step}"
    if not (adapter_dir / "adapter_config.json").exists():
        raise SystemExit(f"no adapter at {adapter_dir}")
    with tracked_run(results_dir(cfg), out, f"mech_umu_layer_mask_{run_id}_step{args.step}", "mech_umu_layer_mask"):
        from peft import PeftModel
        dc = set(read_json(run_dir / "baseline_dc.json")["pair_ids"])
        dfs = load_umu(cfg["paths"]["root"], cfg["umu"]["n_retain_eval"], 0, log, cfg["umu"]["forget_split"])
        items = [r for r in protocol_items(dfs, forms=FORMS) if r["pair_id"] in dc]
        at_step = {r["uid"]: r["correct_exact"] for r in read_jsonl(run_dir / "evals" / f"step{args.step}.jsonl")}
        lost = {(r["set"], r["pair_id"]): at_step.get(r["uid"]) is False for r in items if r["modality"] == "txt"}
        log(f"{run_id} step {args.step}: {len(items)} rows over {len(dc)} doubly-correct facts; name route lost at this "
            f"step: forget {sum(v for (s, _), v in lost.items() if s == 'forget')}, "
            f"retain {sum(v for (s, _), v in lost.items() if s == 'retain')}")

        adapter = get_adapter({**cfg["model"], "id": cfg["umu"]["ckpt"], "processor_id": cfg["model"]["id"]})
        model = PeftModel.from_pretrained(adapter.load_model(), str(adapter_dir), is_trainable=False)
        model.eval()
        n_layers = len(decoder_layers(model))
        batches = []
        for m in sorted({r["modality"] for r in items}):
            group = sorted([r for r in items if r["modality"] == m], key=lambda r: len(r["prompt"]))
            bs = args.batch_txt if group[0]["image_bytes"] is None else args.batch_img
            for i in range(0, len(group), bs):
                chunk = group[i:i + bs]
                batch = encode_teacher_forced(adapter, chunk)
                t_all = tools.answer_targets(batch)
                batches.append((chunk, batch, t_all, first_target(t_all)))
        conds = conditions(n_layers)
        scores = {}
        for name, keep in conds.items():
            scores[name] = score_condition(adapter, model, batches, keep)
            log(f"  scored {name} (LoRA on {len(keep)}/{n_layers} layers)")
        summ = summarize(items, scores, lost)

        gen = {}
        if not args.no_gen:
            gen_rows = [r for r in items if r["modality"] == "txt" and lost[(r["set"], r["pair_id"])]]
            for name in GEN_CONDITIONS:
                with tools.lora_layer_mask(model, conds[name]):
                    res = generate_protocol(model, adapter.processor, gen_rows, out / f"gen_{name}.jsonl", log,
                                            batch_size={"img": args.batch_img, "txt": 32}, dtype=adapter.dtype)
                by = collections.defaultdict(list)
                for r in res:
                    by[r["set"]].append(r["correct_exact"])
                gen[name] = {s: {"acc": float(np.mean(v)), "n": len(v)} for s, v in by.items()}
        model = None
        free()

        atomic_write_json(out / "layer_mask.json", {"run_id": run_id, "step": args.step, "n_layers": n_layers,
                                                    "conditions": {k: v for k, v in conds.items()}, "summary": summ,
                                                    "generation_on_lost_facts": gen, "per_item": scores})
        groups = ["forget/txt/lost", "forget/txt", "retain/txt/lost", "forget/img", "forget/txt+img", "retain/img"]
        log("recovery of log p(first answer token) for name-only groups (0 = unlearned, 1 = vanilla); "
            "Δ vs unlearned for portrait groups")
        log(f"{'condition':16s} " + " ".join(f"{g:>16s}" for g in groups))
        for c in conds:
            cells = []
            for g in groups:
                v = summ[c].get(g)
                if v is None:
                    cells.append(f"{'-':>16s}")
                elif "/txt" in g and "txt+img" not in g:
                    rec = v.get("recovery_first")
                    cells.append(f"{'-' if rec is None else format(rec, '+.2f'):>16s}")
                else:
                    cells.append(f"{v['first'] - summ['unlearned'][g]['first']:>+16.2f}")
            log(f"{c:16s} " + " ".join(cells))
        if gen:
            log("generation + exact match on the facts the name route lost at this step (UMU protocol):")
            for c, v in gen.items():
                log(f"  {c:14s} " + "  ".join(f"{s} {x['acc']:.2f} (n={x['n']})" for s, x in sorted(v.items())))
        plot(summ, n_layers, f"{run_id} step {args.step}: switching the unlearning LoRA off by layer", out / "layer_mask.png")
        log(f"wrote {out / 'layer_mask.json'} and layer_mask.png")


if __name__ == "__main__":
    main()
