"""Section 6 step 2: logit-lens curves of the pilot's unlearned model vs the vanilla.

Per decoder layer: summed log-prob of the correct option letter read through the logit lens,
theta_u - theta_0, averaged over the doubly-correct calibration facts, for the image and the
text route of forget facts (and retain facts as control). Reading the result (research.md,
Hypotheses & Predictions): an image-route curve that dips and recovers points to H2; one that
stays ~0 leaves H1 vs H1' to lora_layer_mask + delta_wx_norm. The script only reports; the call is
made by a person.

  python scripts/mech_layer_logit_diff.py --config configs/llava/pilot.yaml [--step N] [--debug]

--step defaults to the first pilot checkpoint that passed checkpoint B (in --debug: the last one).
Writes results/<model>/mech/layer_logit_diff_step<N>.{json,png}.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np  # noqa: E402

from vlmu.config import base_parser, resolve  # noqa: E402
from vlmu.evaluate import CHECKPOINT_SUITE, load_items  # noqa: E402
from vlmu.mech import tools  # noqa: E402
from vlmu.models import get_adapter  # noqa: E402
from vlmu.report import COLORS, GRID, INK, INK2, SURFACE  # noqa: E402
from vlmu.runs import pilot_dir, processed_dir, require_gate, results_dir, vanilla_adapter_dir  # noqa: E402
from vlmu.utils import Logger, atomic_write_json, read_json, read_jsonl  # noqa: E402


def pick_step(cfg, step):
    if step is not None:
        return step
    s = read_json(pilot_dir(cfg) / "pilot_summary.json")
    if s["checkpoint_b"]["passed"]:
        return s["checkpoint_b"]["first_passing"]["step"]
    if cfg["debug"]:
        return max(r["step"] for r in read_jsonl(pilot_dir(cfg) / "checkpoints.jsonl"))
    raise SystemExit("checkpoint B failed: no step to analyse (pass --step explicitly to override)")


def plot(curves, n_layers, title, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(7.2, 3.8), dpi=150)
    fig.patch.set_facecolor(SURFACE)
    ax.set_facecolor(SURFACE)
    x = np.arange(n_layers)
    ax.axhline(0, color=INK2, lw=1)
    for key, c in curves.items():
        set_name, m = key.split("/")
        mean, sem = np.array(c["mean"]), np.array(c["sem"])
        ls = "-" if set_name == "forget" else (0, (4, 3))
        label = f"{set_name} / {'image+text' if m == 'img' else 'text-only'} (n={c['n']})"
        ax.plot(x, mean, color=COLORS[m], lw=2, ls=ls, label=label)
        if set_name == "forget":
            ax.fill_between(x, mean - sem, mean + sem, color=COLORS[m], alpha=0.15, lw=0)
            ax.annotate("image" if m == "img" else "text", (x[-1], mean[-1]), xytext=(4, 0), textcoords="offset points",
                        va="center", fontsize=8, color=INK)
    ax.set_xlabel("decoder layer (logit lens on its output)", color=INK2, fontsize=9)
    ax.set_ylabel("log p(correct letter): unlearned − vanilla", color=INK2, fontsize=9)
    ax.set_title(title, color=INK, fontsize=10, loc="left")
    ax.grid(axis="y", color=GRID, lw=0.6)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(GRID)
    ax.tick_params(colors=INK2, labelsize=8)
    ax.legend(frameon=False, fontsize=8, labelcolor=INK)
    fig.tight_layout()
    fig.savefig(path, facecolor=SURFACE)
    plt.close(fig)


def main():
    p = base_parser(__doc__)
    p.add_argument("--step", type=int, default=None)
    args = p.parse_args()
    cfg = resolve(args)
    out = results_dir(cfg) / "mech"
    log = Logger(out / "layer_logit_diff_log.txt")
    step = pick_step(cfg, args.step)
    adapter_dir = pilot_dir(cfg) / "ckpt" / f"step{step}" / "adapter"
    processed = processed_dir(cfg, log)
    dc = require_gate(cfg, log)

    from peft import PeftModel
    adapter = get_adapter(cfg["model"])
    model = PeftModel.from_pretrained(adapter.load_model(vanilla_adapter_dir(cfg)), str(adapter_dir))
    model.eval()
    items = load_items(processed, CHECKPOINT_SUITE, cfg["eval"].get("max_per_group"), pair_ids=dc)
    curves, per_item = {}, []
    for set_name in ("forget", "retain"):
        for m in ("img", "txt"):
            rows = [r for r in items if r["set"] == set_name and r["modality"] == m]
            if not rows:  # possible in --debug (tiny doubly-correct set)
                log(f"{set_name}/{m}: no doubly-correct calibration items, skipped")
                continue
            bs =cfg["eval"]["batch_size_txt"] if m == "txt" else cfg["eval"]["batch_size_img"]
            diffs = []
            for i in range(0, len(rows), bs):
                chunk = rows[i:i + bs]
                batch = adapter.encode_mc(chunk, processed)
                d = tools.layer_logit_diff(adapter, model, batch, tools.mc_targets(adapter, chunk, batch))["diff"]
                diffs.append(d.numpy())
                per_item += [{"uid": r["uid"], "set": set_name, "modality": m, "diff": d[:, k].round(decimals=4).tolist()}
                             for k, r in enumerate(chunk)]
            D = np.concatenate(diffs, axis=1)  # [L, n]
            curves[f"{set_name}/{m}"] = {"n": int(D.shape[1]), "mean": D.mean(1).round(4).tolist(),
                                         "sem": (D.std(1, ddof=1) / np.sqrt(D.shape[1])).round(4).tolist() if D.shape[1] > 1
                                         else [0.0] * D.shape[0]}
            c = curves[f"{set_name}/{m}"]
            log(f"{set_name}/{m}: n={c['n']} min={min(c['mean']):+.3f} at layer {int(np.argmin(c['mean']))}, "
                f"last layer {c['mean'][-1]:+.3f}")
    n_layers = len(next(iter(curves.values()))["mean"])
    res = {"model": cfg["model"]["name"], "run_id": cfg["run_id"], "step": step, "adapter": str(adapter_dir),
           "items": "doubly-correct calibration facts, MC prompt, target = correct option letter at the last position",
           "n_layers": n_layers, "curves": curves}
    atomic_write_json(out / f"layer_logit_diff_step{step}.json", res)
    atomic_write_json(out / f"layer_logit_diff_step{step}_items.json", per_item)
    plot(curves, n_layers, f"{cfg['model']['name']} · {cfg['run_id']} · step {step}: logit-lens Δ log p per layer",
         out / f"layer_logit_diff_step{step}.png")
    log(f"wrote {out / f'layer_logit_diff_step{step}.json'} and .png")


if __name__ == "__main__":
    main()
