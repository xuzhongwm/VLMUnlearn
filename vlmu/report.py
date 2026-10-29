"""Aggregation: transfer table + G (confirmation entities, doubly-correct facts), GO rule, figure.

GO rule (PROTOCOL §5.1 as specified 2026-10-29; see DEVIATIONS.md): for the primary method
(GA_Diff) and primary condition (text-only unlearning), GO iff in EVERY seed G >= min_G and the
paired entity-bootstrap 95% CI lower bound > 0. NO-GO iff G < min_G in every seed. UNCLEAR
otherwise. Models with role 'pipeline_check' are never confirmatory.
"""
import csv
from pathlib import Path

import numpy as np

from .metrics import g_bootstrap, normalised_drop
from .runs import dc_file, results_dir, vanilla_eval_dir
from .utils import atomic_write_json, read_json, read_jsonl

MOD_LABEL = {"txt": "text-only", "img": "image+text", "pose": "image+text (held-out pose)"}
COLORS = {"txt": "#2a78d6", "img": "#eb6834", "pose": "#1baf7a"}   # reference categorical slots 1-3
INK, INK2, GRID, SURFACE = "#0b0b0b", "#52514e", "#e4e3de", "#fcfcfb"


def side_metrics(summ):
    dc, al = summ.get("doubly_correct", {}), summ["all_items"]
    perm = summ.get("permutation_doubly_correct", {})
    g = lambda d, k, f="acc": round(d[k][f], 4) if k in d else None  # noqa: E731
    out = {"retain_conf_img": g(dc, "retain/confirmation/paired/img"), "retain_conf_txt": g(dc, "retain/confirmation/paired/txt"),
           "celeb_official_img": g(al, "celeb/all/official/img"), "celeb_official_txt": g(al, "celeb/all/official/txt"),
           "utility_mmbench": g(al, "utility/all/mmbench/img"),
           "forget_conf_official_img": g(al, "forget/confirmation/official/img"),
           "forget_conf_official_txt": g(al, "forget/confirmation/official/txt")}
    for m in ("img", "txt"):
        out[f"margin_forget_conf_{m}"] = g(dc, f"forget/confirmation/paired/{m}", "margin")
        out[f"p_letters_forget_conf_{m}"] = g(dc, f"forget/confirmation/paired/{m}", "p_letters")
        p = perm.get(f"forget/confirmation/{m}")
        out[f"perm_agreement_forget_conf_{m}"] = round(p["agreement_all_4_shifts"], 4) if p and p["agreement_all_4_shifts"] is not None else None
        out[f"perm_acc_by_shift_forget_conf_{m}"] = p["acc_by_shift"] if p else None
    return out


def write_csv(path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    keys = list(dict.fromkeys(k for r in rows for k in r))
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)


def aggregate(cfg, log=print):
    name, role = cfg["model"]["name"], cfg["model"].get("role", "unknown")
    rdir = results_dir(cfg)
    sel_runs = read_json(rdir / "selected_runs.json")
    dc = set(read_json(dc_file(cfg))["pair_ids"])
    n_boot = cfg["go"]["n_boot"]
    v_items = read_jsonl(vanilla_eval_dir(cfg) / "items.jsonl")
    v_summ = read_json(vanilla_eval_dir(cfg) / "summary.json")
    vb = g_bootstrap(v_items, dc, n_boot=200)
    transfer, gtab = [], []
    base = {"model": name, "role": role}
    for m in ("txt", "img", "pose"):
        transfer.append({**base, "method": "vanilla", "unlearn_modality": None, "seed": None, "eval_modality": m,
                         "acc": round(vb["point"][f"acc_{m}"], 4) if vb else None, **side_metrics(v_summ)})
    for cell, info in sel_runs.items():
        method, umod = cell.split("/")
        for seed, run in sorted(info["seeds"].items()):
            items_f = Path(run["full_eval"]) / "items.jsonl"
            if not items_f.exists():
                log(f"missing full eval for {cell} seed {seed}: {items_f}")
                continue
            u_items = read_jsonl(items_f)
            u_summ = read_json(Path(run["full_eval"]) / "summary.json")
            b = g_bootstrap(u_items, dc, n_boot=n_boot, seed=int(seed))
            if b is None:
                log(f"no doubly-correct confirmation items for {cell} seed {seed}")
                continue
            common = {**base, "method": method, "unlearn_modality": umod, "seed": int(seed), "lr": info["lr"],
                      "step": run.get("step"), "epoch": run.get("epoch"), "calib_acc_in": run.get("acc_in"),
                      "n_entities": b["n_entities"], "n_pairs": b["n_pairs"]}
            for m in ("txt", "img", "pose"):
                a = b["point"][f"acc_{m}"]
                transfer.append({**common, "eval_modality": m, "acc": round(a, 4),
                                 "acc_ci_lo": round(b["ci95"][f"acc_{m}"][0], 4), "acc_ci_hi": round(b["ci95"][f"acc_{m}"][1], 4),
                                 "normalised_drop": round(normalised_drop(a), 4), **side_metrics(u_summ)})
            gtab.append({**common, "G": round(b["point"]["G"], 4), "G_ci_lo": round(b["ci95"]["G"][0], 4),
                         "G_ci_hi": round(b["ci95"]["G"][1], 4), "G_reverse": round(b["point"]["G_reverse"], 4),
                         "G_reverse_ci_lo": round(b["ci95"]["G_reverse"][0], 4),
                         "G_reverse_ci_hi": round(b["ci95"]["G_reverse"][1], 4),
                         "primary": method == cfg["go"]["primary_method"] and umod == cfg["go"]["primary_modality"]})
    write_csv(rdir / "transfer_table.csv", transfer)
    write_csv(rdir / "g_table.csv", gtab)
    grid = read_jsonl(rdir / "grid.jsonl")
    if grid:
        write_csv(rdir / "grid.csv", grid)
    seeds = [int(s) for s in cfg["grid"]["seeds"]]
    go = cfg["go"]
    verdict = {"model": name, "role": role, "confirmatory": role in ("primary", "replication"),
               "primary": decide(gtab, go["primary_method"], go["primary_modality"], seeds, go["min_G"]),
               "secondary": {f"{m}/{u}": decide(gtab, m, u, seeds, go["min_G"], reverse=(u == "img"))
                             for m in cfg["grid"]["methods"] for u in cfg["grid"]["modalities"]
                             if (m, u) != (go["primary_method"], go["primary_modality"])}}
    if not verdict["confirmatory"]:
        verdict["note"] = "pipeline check only — excluded from confirmatory evidence"
    atomic_write_json(rdir / "verdict.json", verdict)
    log(f"verdict ({name}, {role}): {verdict['primary']['verdict']} — {verdict['primary']['reason']}")
    plot(transfer, gtab, cfg, rdir / f"transfer_{name}.png")
    return transfer, gtab, verdict


def decide(gtab, method, umod, seeds, min_g, reverse=False):
    """reverse=True evaluates G_reverse (image-unlearning condition: Acc_T - Acc_I)."""
    k = "G_reverse" if reverse else "G"
    got = {r["seed"]: r for r in gtab if r["method"] == method and r["unlearn_modality"] == umod}
    per = {s: {k: got[s][k], "ci95": (got[s][f"{k}_ci_lo"], got[s][f"{k}_ci_hi"]),
               "passes": got[s][k] >= min_g and got[s][f"{k}_ci_lo"] > 0} for s in seeds if s in got}
    if len(per) < len(seeds):
        return {"verdict": "INCOMPLETE", "statistic": k, "per_seed": per,
                "reason": f"no eligible/evaluated checkpoint for seeds {[s for s in seeds if s not in got]}"}
    if all(v["passes"] for v in per.values()):
        v, why = "GO", f"all seeds {k} >= {min_g} with CI lower bound > 0"
    elif all(v[k] < min_g for v in per.values()):
        v, why = "NO-GO", f"all seeds {k} < {min_g}"
    else:
        v, why = "UNCLEAR", "criterion met in some but not all seeds, or effect >= threshold with CI including 0"
    return {"verdict": v, "statistic": k, "min": min_g, "per_seed": per, "reason": why}


def plot(transfer, gtab, cfg, out_path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch

    rows = [r for r in transfer if r["method"] != "vanilla" and r.get("acc") is not None]
    methods = [m for m in cfg["grid"]["methods"] if any(r["method"] == m for r in rows)]
    if not methods:
        return
    title = {"ga": "GA", "ga_diff": "GA_Diff (primary)", "npo": "NPO"}
    s = cfg["selection"]
    fig, axes = plt.subplots(1, len(methods), figsize=(4.6 * len(methods), 3.9), sharey=True, facecolor=SURFACE)
    axes = np.atleast_1d(axes)
    width = 0.26
    for ax, method in zip(axes, methods):
        ax.set_facecolor(SURFACE)
        ax.axhspan(s["window_lo"], s["window_hi"], color=GRID, alpha=0.6, zorder=0)
        for gi, umod in enumerate(("txt", "img")):
            for bi, emod in enumerate(("txt", "img", "pose")):
                rs = [r for r in rows if r["method"] == method and r["unlearn_modality"] == umod and r["eval_modality"] == emod]
                if not rs:
                    continue
                x = gi + (bi - 1) * (width + 0.02)
                mean = float(np.mean([r["acc"] for r in rs]))
                ax.bar(x, mean, width, color=COLORS[emod], edgecolor=SURFACE, linewidth=2, zorder=2)
                ax.scatter([x] * len(rs), [r["acc"] for r in rs], s=14, color=INK, zorder=3)
                ax.text(x, max(r["acc"] for r in rs) + 0.02, f"{mean:.2f}", ha="center", va="bottom", fontsize=8, color=INK2)
            gs = sorted([r for r in gtab if r["method"] == method and r["unlearn_modality"] == umod], key=lambda r: r["seed"])
            if gs:
                key = "G" if umod == "txt" else "G_reverse"
                ax.text(gi, 1.08, f"{'G' if umod == 'txt' else 'G_rev'} = " + ", ".join(f"{r[key]:+.2f}" for r in gs),
                        ha="center", fontsize=8, color=INK)
        ax.axhline(1.0, color=INK2, linewidth=0.8, linestyle=(0, (3, 3)), zorder=1)
        ax.axhline(0.25, color=INK2, linewidth=0.8, linestyle=(0, (1, 2)), zorder=1)
        ax.set_xlim(-0.55, 1.55)
        ax.set_ylim(0, 1.18)
        ax.set_xticks([0, 1], ["unlearned on\ntext-only (primary)", "unlearned on\nimage+text (secondary)"], color=INK)
        ax.set_title(title.get(method, method), color=INK, fontsize=11, loc="left")
        ax.grid(axis="y", color=GRID, linewidth=0.8, zorder=0)
        for sp in ("top", "right", "left"):
            ax.spines[sp].set_visible(False)
        ax.spines["bottom"].set_color(GRID)
        ax.tick_params(colors=INK2, length=0)
    axes[0].set_ylabel("accuracy on doubly-correct facts\n(confirmation entities)", color=INK2)
    handles = [Patch(facecolor=COLORS[m], label=f"eval: {MOD_LABEL[m]}") for m in ("txt", "img", "pose")]
    handles += [Patch(facecolor=GRID, label="selection window (calibration)"),
                Line2D([], [], marker="o", linestyle="", color=INK, markersize=4, label="individual seed")]
    fig.legend(handles=handles, loc="lower center", ncol=5, frameon=False, fontsize=9, bbox_to_anchor=(0.5, -0.02))
    role = cfg["model"].get("role", "")
    fig.suptitle(f"{cfg['model']['name']} ({role}): accuracy after unlearning; vanilla = 1 (dashed), chance = 0.25 (dotted); "
                 "bars = mean over seeds", color=INK, fontsize=10, x=0.01, ha="left")
    fig.tight_layout(rect=(0, 0.08, 1, 0.95))
    fig.savefig(out_path, dpi=160, facecolor=SURFACE)
    plt.close(fig)
