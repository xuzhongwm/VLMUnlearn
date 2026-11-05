"""Paper figure for the UMU logit lens (scripts/mech_umu_lens.py output): Δ log p (θu − θ0) by decoder layer,
mean ± SEM over doubly-correct facts, three panels:
  (a) forget facts, all          (b) forget facts the name-only route got wrong at this step   (c) retain facts
Routes: name, no image / portrait / name + portrait (categorical slots 1-3 of the dataviz reference palette).
Shaded: layers where the vanilla's mean p(first answer token) on name-only forget facts exceeds 2%.

  python scripts/plot_umu_lens.py --lens <results/llava/mech_umu/<run>/step<N>/lens.json> [--target first|answer]
Writes lens_figure_<target>.pdf / .png next to lens.json.
"""
import argparse
import json
import math
from pathlib import Path

import numpy as np

ROUTES = [("txt", "Name, no image", "#2a78d6"), ("img", "Portrait", "#eb6834"), ("txt+img", "Name + portrait", "#1baf7a")]
INK, INK2, MUTED, GRID, AXIS, SURFACE = "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7", "#fcfcfb"


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--lens", required=True)
    p.add_argument("--target", default="first", choices=("first", "answer"))
    args = p.parse_args()
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    d = json.load(open(args.lens))
    rows, L = d["rows"], d["n_layers"]
    txt_ok = {(r["set"], r["pair_id"]): r["correct_at_step"] for r in rows if r["modality"] == "txt"}
    van = np.mean([r["first"]["lp_0"] for r in rows if r["set"] == "forget" and r["modality"] == "txt"], 0)
    band = next((l for l in range(L) if van[l] > math.log(0.02)), None)
    panels = [("a  Forget facts", lambda r: r["set"] == "forget"),
              ("b  Forget facts the name-only route lost", lambda r: r["set"] == "forget" and txt_ok[(r["set"], r["pair_id"])] is False),
              ("c  Retain facts (control)", lambda r: r["set"] == "retain")]
    plt.rcParams.update({"font.family": "sans-serif", "font.size": 8.5, "axes.edgecolor": AXIS, "axes.labelcolor": INK2,
                         "xtick.color": MUTED, "ytick.color": MUTED, "axes.titlesize": 9, "axes.titleweight": "semibold"})
    fig, axes = plt.subplots(1, 3, figsize=(10.5, 3.1), dpi=200, sharey=True)
    fig.patch.set_facecolor(SURFACE)
    x = np.arange(L)
    for ax, (title, keep) in zip(axes, panels):
        ax.set_facecolor(SURFACE)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
        ax.grid(axis="y", color=GRID, lw=0.6)
        ax.set_axisbelow(True)
        if band is not None:
            ax.axvspan(band, L - 1, color="#2a78d6", alpha=0.06, lw=0)
        ax.axvline(20, color=AXIS, lw=0.8)
        ax.axhline(0, color=AXIS, lw=0.9)
        for route, label, color in ROUTES:
            g = [r for r in rows if r["modality"] == route and keep(r)]
            if not g:
                continue
            v = np.array([r[args.target]["diff"] for r in g])
            m = v.mean(0)
            se = v.std(0, ddof=1) / np.sqrt(len(v)) if len(v) > 1 else np.zeros(L)
            ax.fill_between(x, m - se, m + se, color=color, alpha=0.12, lw=0)
            ax.plot(x, m, color=color, lw=1.8, solid_capstyle="round", label=f"{label} (n={len(g)})")
            ax.plot([L - 1], [m[-1]], "o", ms=4.5, color=color, mec=SURFACE, mew=1.2)
            if route == "txt":
                ax.annotate(f"{m[-1]:+.2f}", (L - 1, m[-1]), xytext=(5, 0), textcoords="offset points", va="center",
                            fontsize=7.5, color=INK2)
        ax.set_title(title, loc="left", color=INK)
        ax.set_xlabel("decoder layer")
        ax.set_xticks(range(0, L, 8))
        ax.set_xlim(0, L - 1 + 3)
        ax.legend(frameon=False, fontsize=7, loc="lower left", labelcolor=INK2)
    what = "first answer token" if args.target == "first" else "answer tokens (sum)"
    axes[0].set_ylabel(f"Δ log p({what})\nafter − before unlearning")
    axes[0].annotate("vanilla's answer\nemerges (p > 2%)", (band + 0.5 if band else 0, axes[0].get_ylim()[1]),
                     xytext=(0, -2), textcoords="offset points", va="top", fontsize=6.5, color=INK2)
    axes[0].annotate("L20", (20, axes[0].get_ylim()[0]), xytext=(-3, 3), textcoords="offset points", ha="right",
                     fontsize=6.5, color=MUTED)
    fig.suptitle(f"Logit lens, {d['run_id']} step {d['step']} (UMU-Bench, LLaVA-1.5-7B)", x=0.01, ha="left",
                 fontsize=9, color=INK2)
    fig.tight_layout()
    out = Path(args.lens).with_name(f"lens_figure_{args.target}")
    for ext in ("pdf", "png"):
        fig.savefig(f"{out}.{ext}", facecolor=SURFACE, bbox_inches="tight")
    print(f"wrote {out}.pdf / .png")


if __name__ == "__main__":
    main()
