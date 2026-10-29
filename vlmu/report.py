"""Aggregation: 2x2 transfer tables, asymmetry with CIs, mechanical GO/NO-GO, figure."""
import csv
from pathlib import Path

import numpy as np

from .metrics import bootstrap_nd
from .runs import results_dir, vanilla_eval_dir
from .utils import atomic_write_json, read_json, read_jsonl

MOD_LABEL = {"txt": "text-only", "img": "image+text", "pose": "image+text (held-out pose)"}
COLORS = {"txt": "#2a78d6", "img": "#eb6834", "pose": "#1baf7a"}   # reference categorical slots 1-3
INK, INK2, GRID, SURFACE = "#0b0b0b", "#52514e", "#e4e3de", "#fcfcfb"


def side_metrics(summ):
    g = lambda k: round(summ[k]["acc"], 2) if k in summ else None  # noqa: E731
    return {"retain_img": g("retain/paired/img"), "retain_txt": g("retain/paired/txt"),
            "retain_official_img": g("retain/official/img"), "retain_official_txt": g("retain/official/txt"),
            "celeb_img": g("celeb/official/img"), "celeb_txt": g("celeb/official/txt"),
            "utility_mmbench": g("utility/mmbench/img"),
            "forget_official_img": g("forget/official/img"), "forget_official_txt": g("forget/official/txt"),
            "forget_p_letters_txt": round(summ["forget/paired/txt"]["p_letters"], 3),
            "forget_p_letters_img": round(summ["forget/paired/img"]["p_letters"], 3)}


def write_csv(path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    keys = list(dict.fromkeys(k for r in rows for k in r))
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)


def aggregate(cfg, log=print):
    name = cfg["model"]["name"]
    rdir = results_dir(cfg)
    sel_runs = read_json(rdir / "selected_runs.json")
    v_items = read_jsonl(vanilla_eval_dir(cfg) / "items.jsonl")
    v_summ = read_json(vanilla_eval_dir(cfg) / "summary.json")
    transfer, asym = [], []
    base = {"model": name, "method": "vanilla", "unlearn_modality": None, "seed": None, "status": None}
    for m in ("txt", "img", "pose"):
        transfer.append({**base, "eval_modality": m, "vanilla_acc": round(v_summ[f"forget/paired/{m}"]["acc"], 2),
                         "unlearned_acc": None, "nd": None, **side_metrics(v_summ)})
    for cell, info in sel_runs.items():
        method, umod = cell.split("/")
        for seed, run in sorted(info["seeds"].items()):
            items_f = Path(run["full_eval"]) / "items.jsonl"
            if not items_f.exists():
                log(f"missing full eval for {cell} seed {seed}: {items_f}")
                continue
            u_items = read_jsonl(items_f)
            u_summ = read_json(Path(run["full_eval"]) / "summary.json")
            b = bootstrap_nd(v_items, u_items, umod, seed=int(seed))
            common = {"model": name, "method": method, "unlearn_modality": umod, "seed": int(seed),
                      "status": info["status"], "lr": info["lr"], "step": info["step"], "epoch": info["epoch"]}
            for m in ("txt", "img", "pose"):
                transfer.append({**common, "eval_modality": m,
                                 "vanilla_acc": round(v_summ[f"forget/paired/{m}"]["acc"], 2),
                                 "unlearned_acc": round(u_summ[f"forget/paired/{m}"]["acc"], 2),
                                 "nd": round(b["point"][m], 4), "nd_ci_lo": round(b["ci95"][m][0], 4),
                                 "nd_ci_hi": round(b["ci95"][m][1], 4), "n_profiles": b["n_profiles"],
                                 **side_metrics(u_summ)})
            asym.append({**common, "A": round(b["point"]["asym"], 4), "A_ci_lo": round(b["ci95"]["asym"][0], 4),
                         "A_ci_hi": round(b["ci95"]["asym"][1], 4),
                         "definition": f"ND({umod}->{umod}) - ND({umod}->{'img' if umod == 'txt' else 'txt'})"})
    write_csv(rdir / "transfer_table.csv", transfer)
    write_csv(rdir / "asymmetry.csv", asym)
    grid = read_jsonl(rdir / "grid.jsonl")
    if grid:
        write_csv(rdir / "grid.csv", grid)
    verdict = decide(asym, cfg["grid"]["methods"], [cfg["grid"]["seed"]] + cfg["selection"]["extra_seeds"])
    verdict["model"] = name
    atomic_write_json(rdir / "verdict.json", verdict)
    log(f"verdict ({name}): {verdict['verdict']} — {verdict['reason']}")
    plot(transfer, name, rdir / f"transfer_{name}.png")
    return transfer, asym, verdict


def decide(asym, methods, seeds):
    """D7 GO rule for TEXT-ONLY unlearning: A = ND(txt->txt) - ND(txt->img).
    GO: A>0 with 95% CI excluding 0 for >=2 methods in every seed (matched cells only).
    UNCLEAR: A>0 in every seed for >=2 matched methods, but CIs include 0. NO-GO otherwise."""
    per = {}
    for m in methods:
        rows = [r for r in asym if r["method"] == m and r["unlearn_modality"] == "txt"]
        got = {r["seed"]: r for r in rows}
        complete = all(s in got for s in seeds) and all(r["status"] == "matched" for r in rows) and rows
        per[m] = {"matched_all_seeds": bool(complete),
                  "positive_all_seeds": bool(complete) and all(got[s]["A"] > 0 for s in seeds),
                  "ci_excludes_0_all_seeds": bool(complete) and all(got[s]["A_ci_lo"] > 0 for s in seeds),
                  "A_by_seed": {s: (got[s]["A"], got[s]["A_ci_lo"], got[s]["A_ci_hi"]) for s in got}}
    n_sig = sum(v["ci_excludes_0_all_seeds"] for v in per.values())
    n_pos = sum(v["positive_all_seeds"] for v in per.values())
    n_matched = sum(v["matched_all_seeds"] for v in per.values())
    if n_sig >= 2:
        verdict, reason = "GO", f"{n_sig} methods show A>0 with CI excluding 0 in all seeds"
    elif n_pos >= 2:
        verdict, reason = "UNCLEAR", f"{n_pos} methods have A>0 in all seeds but only {n_sig} with CI excluding 0"
    else:
        verdict, reason = "NO-GO", f"only {n_pos} methods with A>0 in all seeds ({n_matched} matched)"
    return {"verdict": verdict, "reason": reason, "per_method": per, "seeds": seeds}


def plot(transfer, name, out_path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    rows = [r for r in transfer if r["method"] != "vanilla" and r["nd"] is not None]
    methods = [m for m in ("ga", "ga_diff", "npo") if any(r["method"] == m for r in rows)]
    if not methods:
        return
    title = {"ga": "GA", "ga_diff": "GA_Diff", "npo": "NPO"}
    fig, axes = plt.subplots(1, len(methods), figsize=(4.2 * len(methods), 3.8), sharey=True, facecolor=SURFACE)
    axes = np.atleast_1d(axes)
    width = 0.26
    for ax, method in zip(axes, methods):
        ax.set_facecolor(SURFACE)
        for gi, umod in enumerate(("txt", "img")):
            for bi, emod in enumerate(("txt", "img", "pose")):
                rs = [r for r in rows if r["method"] == method and r["unlearn_modality"] == umod and r["eval_modality"] == emod]
                if not rs:
                    continue
                x = gi + (bi - 1) * (width + 0.02)
                mean = float(np.mean([r["nd"] for r in rs]))
                unmatched = any(r["status"] != "matched" for r in rs)
                ax.bar(x, mean, width, color=COLORS[emod], edgecolor=SURFACE, linewidth=2,
                       hatch="//" if unmatched else None, zorder=2)
                ax.scatter([x] * len(rs), [r["nd"] for r in rs], s=14, color=INK, zorder=3)
                ax.text(x, max(mean, max(r["nd"] for r in rs)) + 0.03, f"{mean:.2f}", ha="center", va="bottom",
                        fontsize=8, color=INK2)
        ax.axhline(0, color=INK2, linewidth=0.8, zorder=1)
        ax.axhline(1, color=INK2, linewidth=0.8, linestyle=(0, (3, 3)), zorder=1)
        ax.set_xlim(-0.55, 1.55)
        ax.set_xticks([0, 1], ["unlearned on\ntext-only", "unlearned on\nimage+text"], color=INK)
        ax.set_title(title[method], color=INK, fontsize=11, loc="left")
        ax.grid(axis="y", color=GRID, linewidth=0.8, zorder=0)
        for s in ("top", "right", "left"):
            ax.spines[s].set_visible(False)
        ax.spines["bottom"].set_color(GRID)
        ax.tick_params(colors=INK2, length=0)
    axes[0].set_ylabel("normalised forget drop\n(1 = chance, dashed)", color=INK2)
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch
    handles = [Patch(facecolor=COLORS[m], label=f"eval: {MOD_LABEL[m]}") for m in ("txt", "img", "pose")]
    handles += [Patch(facecolor=GRID, hatch="//", edgecolor=INK2, label="unmatched cell"),
                Line2D([], [], marker="o", linestyle="", color=INK, markersize=4, label="individual seed")]
    fig.legend(handles=handles, loc="lower center", ncol=5, frameon=False, fontsize=9, bbox_to_anchor=(0.5, -0.02))
    fig.suptitle(f"{name}: cross-modal transfer of unlearning at matched in-modality forgetting "
                 f"(bars = mean over seeds)", color=INK, fontsize=10, x=0.01, ha="left")
    fig.tight_layout(rect=(0, 0.08, 1, 0.95))
    fig.savefig(out_path, dpi=160, facecolor=SURFACE)
    plt.close(fig)
