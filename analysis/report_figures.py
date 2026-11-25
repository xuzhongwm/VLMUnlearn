"""Figures for proposal/mechanism.tex, drawn from saved outputs only (no model is loaded).

  python analysis/report_figures.py [name ...]     # writes proposal/figures/mech_<name>.{png,pdf}; no name = all

  name          source under $VLMU_ROOT                                                              entry
  schematic     none: a drawing of the account
  matrix        runs/<model>/umu_unlearn/<run>/checkpoints.jsonl (name, portrait);                   DEV-019/021/035, 043/047,
                results/<model>/mech_umu/<run>/step<N>/desc_leak/summary.json (description)          039, 053; DEV-038
  answer_level  results/<model>/mech_umu/<run>/step<N>/answer_level/summary.json                     DEV-054
  convergence   results/<model>/mech_umu/<dc run>/step0/convergence/summary_first.json               DEV-027, DEV-035
  path          results/<model>/mech_umu/<run>/step<N>/signal{,_img}/summary.json                    DEV-044/045, DEV-046
  first_token   results/<model>/mech_umu/vanilla_<ckpt>/merge_gate/items.jsonl                       DEV-052
  delta         results/<model>/mech_umu/<run>/step<N>/delta_what/summary.json, _compare/delta_what  DEV-050
  person_code   results/<model>/mech_umu/vanilla_<ckpt>/{identity_code/summary.json,merge_gate/items.jsonl}   DEV-051, DEV-052

Colours follow the entity in every panel (dataviz reference palette, slots 1-3): the name route / name tokens are
blue, the portrait route / image tokens are orange, the last prompt token is aqua. Magnitudes use the one-hue blue
ramp. Text is always ink.
"""
import collections
import json
import os
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.lines import Line2D
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Rectangle

VROOT = Path(os.environ.get("VLMU_ROOT", "/content/drive/MyDrive/vlm_unlearn"))
ROOT, RUNS = VROOT / "results", VROOT / "runs"
OUT = Path(__file__).resolve().parents[1] / "proposal" / "figures"

NAME, FACE, LAST = "#2a78d6", "#eb6834", "#1baf7a"
INK, INK2, MUTED, GRID, AXIS, SURFACE = "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7", "#fcfcfb"
RAMP = ["#fcfcfb", "#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"]   # surface, then blue 100-700
G = "gate_rw5n200"

MODELS = {   # merge layer L* and number of decoder layers (DEV-027, DEV-035)
    "llava": dict(label="LLaVA-1.5-7B", short="LLaVA", L_star=20, n_layers=32, vanilla="vanilla_llava_smu_ft",
                  dc_run=f"umu_npo_rt_txt_lr2e-04_s0_{G}",
                  txt=(f"umu_npo_rt_txt_lr2e-04_s0_{G}", 300), img=(f"umu_npo_rt_img_lr2e-04_s0_{G}", 240)),
    "qwen2vl7b": dict(label="Qwen2-VL-7B", short="Qwen", L_star=22, n_layers=28, vanilla="vanilla_qwen2vl7b_umu_vanilla_r128",
                      dc_run="umu_npo_rt_txt_lr2e-04_s0_r128",
                      txt=("umu_npo_rt_txt_lr2e-04_s1_r128", 190), img=("umu_npo_rt_img_lr4e-04_s0_r128", 180)),
}

# Checkpoints by training form (all NPO + retain; selected by the gate of the entry named in the docstring).
TRAINED = {
    "llava": collections.OrderedDict([
        ("names", [(f"umu_npo_rt_txt_lr2e-04_s{s}_{G}", 300) for s in (0, 1, 2)]),
        ("portraits", [(f"umu_npo_rt_img_lr2e-04_s0_{G}", 240), (f"umu_npo_rt_img_lr4e-04_s2_{G}", 210), (f"umu_npo_rt_img_lr4e-04_s3_{G}", 240)]),
        ("names + descriptions", [(f"umu_npo_rt_txt_lr2e-04_s0_{G}_dm", 300)]),
        ("names + portraits", [(f"umu_npo_rt_mix_lr2e-04_s0_{G}", 540)]),
    ]),
    "qwen2vl7b": collections.OrderedDict([
        ("names", [(f"umu_npo_rt_txt_lr2e-04_s{s}_r128", 190) for s in (0, 1, 2)]),
        ("portraits", [("umu_npo_rt_img_lr4e-04_s0_r128", 180), ("umu_npo_rt_img_lr4e-04_s2_r128", 180),
                       ("umu_npo_rt_img_lr4e-04_s3_r128", 210), ("umu_npo_rt_img_lr4e-04_s4_r128", 210)]),
        ("names + descriptions", [("umu_npo_rt_txt_lr2e-04_s0_r128_dm", 476)]),
        ("names + portraits", [("umu_npo_rt_mix_lr2e-04_s0_r128", 540)]),
    ]),
}
ASKED = ["name", "portrait", "description"]
COVERS = {"names": {"name"}, "portraits": {"portrait"}, "names + descriptions": {"name", "description"},
          "names + portraits": {"name", "portrait"}}

FORWARD = [("llava", f"umu_npo_rt_txt_lr2e-04_s0_{G}", 300, "LLaVA, NPO, seed 0"), ("llava", f"umu_npo_rt_txt_lr2e-04_s4_{G}", 266, "LLaVA, NPO, seed 4"),
           ("llava", f"umu_ga_diff_txt_lr2e-04_s0_{G}", 300, "LLaVA, GD, seed 0"), ("qwen2vl7b", "umu_npo_rt_txt_lr2e-04_s0_r128", 190, "Qwen, NPO, seed 0"),
           ("qwen2vl7b", "umu_npo_rt_txt_lr2e-04_s1_r128", 190, "Qwen, NPO, seed 1"), ("qwen2vl7b", "umu_npo_rt_txt_lr2e-04_s3_r128", 228, "Qwen, NPO, seed 3"),
           ("qwen2vl7b", "umu_npo_rt_txt_lr2e-04_s4_r128", 266, "Qwen, NPO, seed 4"), ("qwen2vl7b", "umu_ga_diff_txt_lr8e-04_s0_r128", 228, "Qwen, GD, seed 0")]
REVERSE = [("llava", f"umu_npo_rt_img_lr2e-04_s0_{G}", 240, "LLaVA, NPO, seed 0"), ("llava", f"umu_npo_rt_img_lr4e-04_s2_{G}", 210, "LLaVA, NPO, seed 2"),
           ("llava", f"umu_npo_rt_img_lr4e-04_s3_{G}", 240, "LLaVA, NPO, seed 3"), ("qwen2vl7b", "umu_npo_rt_img_lr4e-04_s0_r128", 180, "Qwen, NPO, seed 0"),
           ("qwen2vl7b", "umu_npo_rt_img_lr4e-04_s2_r128", 180, "Qwen, NPO, seed 2"), ("qwen2vl7b", "umu_npo_rt_img_lr4e-04_s3_r128", 210, "Qwen, NPO, seed 3"),
           ("qwen2vl7b", "umu_npo_rt_img_lr4e-04_s4_r128", 210, "Qwen, NPO, seed 4")]


def setup():
    plt.rcParams.update({"font.family": "sans-serif", "font.size": 8.5, "axes.edgecolor": AXIS,
                         "axes.labelcolor": INK2, "xtick.color": MUTED, "ytick.color": MUTED,
                         "axes.titlesize": 9, "axes.titleweight": "semibold"})


def clean(ax, axis="y"):
    ax.set_facecolor(SURFACE)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.grid(axis=axis, color=GRID, lw=0.6)
    ax.set_axisbelow(True)


def curve(d):
    """{'0': v, '2': v, ...} -> sorted (layers, values)."""
    xs = sorted(int(k) for k in d)
    return xs, [d[str(k)] for k in xs]


def merge_line(ax, L, y=1.02, side="right"):
    ax.axvline(L, color=MUTED, lw=0.8)
    ax.annotate(f"merge layer {L}", (L, y), xytext=(3 if side == "right" else -3, 0), textcoords="offset points",
                ha="left" if side == "right" else "right", va="bottom", fontsize=7, color=INK2)


def save(fig, name):
    OUT.mkdir(parents=True, exist_ok=True)
    for ext in ("png", "pdf"):
        fig.savefig(OUT / f"{name}.{ext}", dpi=220, facecolor=SURFACE, bbox_inches="tight")
    plt.close(fig)
    print("wrote", OUT / f"{name}.png")


def jl(path):
    return [json.loads(x) for x in open(path)]


def mech(key, run, step, name):
    return json.load(open(ROOT / key / "mech_umu" / run / f"step{step}" / name / "summary.json"))


# ------------------------------------------------------------------------------------------ matrix
def matrix_values(key):
    """{trained form: {asked form: (mean, n)}} on the forget facts; None where no checkpoint has the readout."""
    out = collections.OrderedDict()
    for form, ckpts in TRAINED[key].items():
        cell = {a: [] for a in ASKED}
        for run, step in ckpts:
            ck = RUNS / key / "umu_unlearn" / run / "checkpoints.jsonl"
            if ck.exists():
                rec = next((r for r in jl(ck) if r["step"] == step), None)
                if rec:
                    cell["name"].append(rec["forget_dc_txt"]), cell["portrait"].append(rec["forget_dc_img"])
            d = ROOT / key / "mech_umu" / run / f"step{step}" / "desc_leak" / "summary.json"
            if d.exists():
                cell["description"].append(json.load(open(d))["forget"]["thetau_scored"]["desc"])
        out[form] = {a: (float(np.mean(v)), len(v)) if v else None for a, v in cell.items()}
    return out


def fig_matrix():
    cmap = LinearSegmentedColormap.from_list("blue", RAMP)
    fig, axes = plt.subplots(1, 2, figsize=(9.6, 3.3))
    fig.patch.set_facecolor(SURFACE)
    for ax, (key, m) in zip(axes, MODELS.items()):
        vals = matrix_values(key)
        print(key, {f: {a: (round(v[0], 2), v[1]) if v else None for a, v in row.items()} for f, row in vals.items()})
        forms = list(vals)
        ax.set_facecolor(SURFACE)
        for i, f in enumerate(forms):
            for j, a in enumerate(ASKED):
                v = vals[f][a]
                if v is None:
                    ax.add_patch(Rectangle((j - 0.47, i - 0.44), 0.94, 0.88, fc=SURFACE, ec=GRID, lw=0.8))
                    ax.text(j, i, "not run", ha="center", va="center", fontsize=7, color=MUTED)
                    continue
                ax.add_patch(Rectangle((j - 0.47, i - 0.44), 0.94, 0.88, fc=cmap(v[0]), ec="none"))
                ax.text(j, i, f"{v[0]:.2f}", ha="center", va="center", fontsize=10, fontweight="semibold",
                        color="white" if v[0] > 0.45 else INK)
                if a in COVERS[f]:               # this form of reference was in the unlearning data
                    ax.add_patch(Rectangle((j - 0.47, i - 0.44), 0.94, 0.88, fc="none", ec=INK, lw=1.8))
        ax.set_xlim(-0.55, len(ASKED) - 0.45)
        ax.set_ylim(len(forms) - 0.5, -0.5)
        ax.set_xticks(range(len(ASKED)))
        ax.set_xticklabels([f"asked by\n{a}" for a in ASKED], color=INK2)
        ax.xaxis.tick_top()
        ax.set_yticks(range(len(forms)))
        ax.set_yticklabels([f"unlearned with\n{f}" for f in forms] if ax is axes[0] else [], color=INK2)
        ax.tick_params(length=0)
        for s in ax.spines.values():
            s.set_visible(False)
        ax.set_title(m["label"], loc="left", color=INK, pad=28)
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(0, 1))
    cb = fig.colorbar(sm, ax=axes, fraction=0.025, pad=0.02, ticks=[0, 0.5, 1.0])
    cb.outline.set_visible(False)
    cb.set_label("accuracy on the forget facts\n(1.00 before unlearning)", color=INK2, fontsize=7.5)
    fig.legend([Rectangle((0, 0), 1, 1, fc=SURFACE, ec=INK, lw=1.8)], ["this form of reference was in the unlearning data"],
               loc="lower center", frameon=False, fontsize=7.5, labelcolor=INK2, bbox_to_anchor=(0.45, -0.07))
    save(fig, "mech_matrix")


# ------------------------------------------------------------------------------------ answer level
def fig_answer_level():
    fig, axes = plt.subplots(1, 2, figsize=(9.6, 3.6))
    fig.patch.set_facecolor(SURFACE)
    panels = [(FORWARD, "answer_level", NAME, "name", "Unlearned with names, asked by name"),
              (REVERSE, "answer_level", FACE, "image", "Unlearned with portraits, asked by portrait")]
    for ax, (ckpts, folder, color, ref, title) in zip(axes, panels):
        clean(ax, axis="x")
        rows = []
        for key, run, step, label in ckpts:
            p = ROOT / key / "mech_umu" / run / f"step{step}" / folder / "summary.json"
            if p.exists():
                rows.append((label, json.load(open(p))))
        for i, (label, s) in enumerate(rows):
            a = s["accuracy"]
            ax.plot([a["u"], a["ref_off"]], [i, i], color=AXIS, lw=1.6, solid_capstyle="round", zorder=2)
            ax.plot(a["rest_off"], i, "o", ms=12.5, mfc="none", mec=INK2, mew=1.2, zorder=3)
            ax.plot(a["u"], i, "o", ms=8.5, mfc=INK2, mec=SURFACE, mew=1.5, zorder=4)
            ax.plot(a["ref_off"], i, "o", ms=8.5, mfc=color, mec=SURFACE, mew=1.5, zorder=5)
        print(title, [(lab, round(s["share_restored"]["ref_off"], 2), round(s["share_restored"]["rest_off"], 2),
                       round(s["share_restored"]["read_off"], 2), round(s["share_restored"]["ref_only"], 2)) for lab, s in rows])
        ax.axvline(1.0, color=MUTED, lw=0.8)
        ax.annotate("original\nmodel", (1.0, -0.62), ha="center", va="bottom", fontsize=7, color=INK2, annotation_clip=False)
        ax.set_yticks(range(len(rows)))
        ax.set_yticklabels([r[0] for r in rows], color=INK2)
        ax.set_ylim(len(rows) - 0.4, -0.75)
        ax.set_xlim(-0.04, 1.06)
        ax.set_xticks([0, 0.25, 0.5, 0.75, 1.0])
        ax.set_xlabel("accuracy of the generated answer on the forget facts")
        ax.set_title(title, loc="left", color=INK, pad=14)
        handles = [Line2D([0], [0], marker="o", ls="", ms=8, mfc=INK2, mec=SURFACE), Line2D([0], [0], marker="o", ls="", ms=8, mfc=color, mec=SURFACE),
                   Line2D([0], [0], marker="o", ls="", ms=11, mfc="none", mec=INK2, mew=1.2)]
        ax.legend(handles, ["unlearned model", f"adapter off at the {ref} tokens only", "adapter off at the other prompt tokens"],
                  loc="upper center", bbox_to_anchor=(0.5, -0.2), ncol=1, frameon=False, fontsize=7.5, labelcolor=INK2)
    fig.tight_layout(w_pad=2.5)
    save(fig, "mech_answer_level")


# ------------------------------------------------------------------------------------- convergence
def fig_convergence():
    fig, axes = plt.subplots(1, 2, figsize=(9.6, 3.3), sharey=True)
    fig.patch.set_facecolor(SURFACE)
    series = [("X1", FACE, "-", "portrait question $\\rightarrow$ name question"),
              ("X2", NAME, "-", "name question $\\rightarrow$ portrait question"),
              ("W2", FACE, (0, (4, 2)), "portrait $\\rightarrow$ portrait (same form)"),
              ("W1", NAME, (0, (4, 2)), "name $\\rightarrow$ name (same form)"),
              ("IMG", INK2, (0, (1, 1.6)), "image tokens of A copied instead")]
    for ax, (key, m) in zip(axes, MODELS.items()):
        d = json.load(open(ROOT / key / "mech_umu" / m["dc_run"] / "step0" / "convergence" / "summary_first.json"))
        clean(ax)
        for name, color, ls, _ in series:
            x, y = curve(d["curves"][name])
            ax.plot(x, y, color=color, lw=2 if ls == "-" else 1.6, ls=ls, solid_capstyle="round")
        ax.axhline(0.5, color=AXIS, lw=0.8)
        merge_line(ax, m["L_star"])
        ax.set_title(f"{m['label']}, before unlearning ({d['n_pairs_kept']} pairs of facts)", loc="left", color=INK)
        ax.set_xlabel("decoder layer at which the states are copied")
        ax.set_xlim(0, m["n_layers"] - 1)
        ax.set_ylim(-0.03, 1.12)
        ax.set_yticks([0, 0.25, 0.5, 0.75, 1.0])
    axes[0].set_ylabel("share of pairs whose answer\nfollows the copied person A")
    handles = [Line2D([0], [0], color=c, lw=2 if ls == "-" else 1.6, ls=ls) for _, c, ls, _ in series]
    fig.legend(handles, [s[3] for s in series], loc="lower center", ncol=3, frameon=False, fontsize=7.5,
               labelcolor=INK2, bbox_to_anchor=(0.5, -0.13), title="states of person A's question copied into person B's",
               title_fontsize=7.5)
    fig.tight_layout(w_pad=2.0)
    save(fig, "mech_convergence")


# -------------------------------------------------------------------------------------------- path
def fig_path():
    fig, axes = plt.subplots(2, 2, figsize=(9.6, 5.6), sharey=True)
    fig.patch.set_facecolor(SURFACE)
    rows = [("txt", "signal", NAME, "name tokens", "unlearned with names, asked by name"),
            ("img", "signal_img", FACE, "image tokens", "unlearned with portraits, asked by portrait")]
    for r, (form, folder, ref_color, ref_label, what) in enumerate(rows):
        for c, (key, m) in enumerate(MODELS.items()):
            ax = axes[r, c]
            run, step = m[form]
            d = mech(key, run, step, folder)
            clean(ax)
            x, y = curve(d["restore"]["all"])
            ax.plot(x, y, color=MUTED, lw=1.2, ls=(0, (4, 2)))
            x, y = curve(d["restore"]["after"])
            ax.plot(x, y, color=AXIS, lw=1.6, solid_capstyle="round")
            x, y = curve(d["restore"]["name"])          # 'name' holds the referent group: name or image tokens
            ax.plot(x, y, color=ref_color, lw=2, solid_capstyle="round")
            peak = max(range(len(y)), key=lambda i: y[i])
            ax.annotate(ref_label, (x[peak], y[peak]), xytext=(0, 5), textcoords="offset points", ha="center",
                        fontsize=7.5, color=INK)
            x, y = curve(d["restore"]["last"])
            ax.plot(x, y, color=LAST, lw=2, solid_capstyle="round")
            ax.annotate("last prompt token", (x[-1], y[-1]), xytext=(0, 5), textcoords="offset points", ha="right",
                        fontsize=7.5, color=INK)
            merge_line(ax, m["L_star"], y=1.10, side="left")
            ax.set_title(f"{m['short']}: {what}", loc="left", color=INK)
            ax.annotate(f"{d['n_forget']} facts, forgetting {d['F_T']:.1f} nats", (0.015, 0.92),
                        xycoords="axes fraction", fontsize=7, color=INK2)
            ax.set_xlim(0, m["n_layers"] - 1)
            ax.set_ylim(-0.04, 1.24)
            ax.set_yticks([0, 0.25, 0.5, 0.75, 1.0])
            if r == 1:
                ax.set_xlabel("decoder layer at which the original model's states are restored")
        axes[r, 0].set_ylabel("share of the forgetting removed\n(first answer token)")
    handles = [Line2D([0], [0], color=NAME, lw=2), Line2D([0], [0], color=FACE, lw=2), Line2D([0], [0], color=LAST, lw=2),
               Line2D([0], [0], color=AXIS, lw=1.6), Line2D([0], [0], color=MUTED, lw=1.2, ls=(0, (4, 2)))]
    labels = ["name tokens", "image tokens", "last prompt token", "prompt tokens after the name / image",
              "all prompt tokens"]
    fig.legend(handles, labels, loc="lower center", ncol=5, frameon=False, fontsize=7.5, labelcolor=INK2,
               bbox_to_anchor=(0.5, -0.045), title="states restored at", title_fontsize=7.5)
    fig.tight_layout(w_pad=2.0, h_pad=1.6)
    save(fig, "mech_path")


# ------------------------------------------------------------------------------------- first token
ATTR = {"Height": "height", "Annual_Salary": "annual salary", "Birth": "date of birth", "Medical_Conditions": "medical conditions",
        "Residence": "residence", "Education": "education", "Occupation": "occupation", "Born": "place of birth"}


def fig_first_token():
    fig, axes = plt.subplots(1, 2, figsize=(9.6, 2.9), sharex=True)
    fig.patch.set_facecolor(SURFACE)
    order = None
    for ax, (key, m) in zip(axes, MODELS.items()):
        base = ROOT / key / "mech_umu" / m["vanilla"] / "merge_gate"
        first = json.load(open(base / "summary.json"))["layer"]
        items = [x for x in jl(base / "items.jsonl") if x["cond"] == "all_deep" and x["set"] == "retain" and x["modality"] == "img"]
        by = collections.defaultdict(list)
        for x in items:
            by[x["field"]].append(x["correct"])
        acc = {k: float(np.mean(v)) for k, v in by.items()}
        if order is None:
            order = sorted(acc, key=lambda k: -acc[k])
        print(key, "all_deep retain img", round(float(np.mean([x["correct"] for x in items])), 2), {k: (round(acc[k], 2), len(by[k])) for k in order})
        clean(ax, axis="x")
        ys = range(len(order))
        ax.barh(list(ys), [acc[k] for k in order], height=0.5, color=LAST, zorder=3)
        for y, k in zip(ys, order):
            ax.text(acc[k] + 0.015, y, f"{acc[k]:.2f}", va="center", ha="left", fontsize=7.5, color=INK)
        ax.set_yticks(list(ys))
        ax.set_yticklabels([ATTR[k] for k in order] if ax is axes[0] else [""] * len(order), color=INK2)
        ax.set_ylim(len(order) - 0.5, -0.6)
        ax.set_xlim(0, 1.14)
        ax.set_xticks([0, 0.25, 0.5, 0.75, 1.0])
        ax.set_xlabel("accuracy of the generated answer (1.00 without the replacement)")
        ax.set_title(f"{m['label']} (replaced in layers {first} to {m['n_layers'] - 1})", loc="left", color=INK)
    fig.tight_layout(w_pad=2.0)
    save(fig, "mech_first_token")


# ------------------------------------------------------------------------------------------- delta
def fig_delta():
    L = "10"
    fig, axes = plt.subplots(1, 3, figsize=(9.6, 3.5), gridspec_kw={"width_ratios": [1.0, 1.0, 1.0]})
    fig.patch.set_facecolor(SURFACE)
    rows = [(k, r, s, lab, mech(k, r, s, "delta_what")) for k, r, s, lab in FORWARD]
    ys = list(range(len(rows)))
    # (a) size of the displacement, at the forget names and at retain names
    ax = axes[0]
    clean(ax, axis="x")
    for y, (_, _, _, _, d) in zip(ys, rows):
        z = d["layers"][L]
        ax.barh(y - 0.17, z["norm_delta"] / z["norm_h0_name"], height=0.3, color=NAME, zorder=3)
        ax.barh(y + 0.17, z["norm_delta_retain"] / z["norm_h0_name"], height=0.3, color=AXIS, zorder=3)
    ax.set_yticks(ys)
    ax.set_yticklabels([r[3] for r in rows], color=INK2)
    ax.set_ylim(len(rows) - 0.4, -0.6)
    ax.set_xlabel("size of the change at the name tokens,\nrelative to the size of their state")
    ax.set_title("a  Size of the change", loc="left", color=INK)
    ax.legend([Rectangle((0, 0), 1, 1, fc=NAME), Rectangle((0, 0), 1, 1, fc=AXIS)], ["forget people", "retain people"],
              loc="upper center", bbox_to_anchor=(0.5, -0.3), ncol=2, frameon=False, fontsize=7.5, labelcolor=INK2)
    # (b) the part of the change common to all forget people, added at a retain person's name
    ax = axes[1]
    clean(ax, axis="x")
    for y, (_, _, _, _, d) in zip(ys, rows):
        z = d["layers"][L]
        ax.plot([z["u+mean_top1"], d["top1"]["retain_u"]], [y, y], color=AXIS, lw=1.6, zorder=2)
        ax.plot(z["u+rand@d_top1"], y, "o", ms=12.5, mfc="none", mec=INK2, mew=1.2, zorder=3)
        ax.plot(z["0+mean_top1"], y, "o", ms=8.5, mfc=INK2, mec=SURFACE, mew=1.5, zorder=4)
        ax.plot(z["u+mean_top1"], y, "o", ms=8.5, mfc=NAME, mec=SURFACE, mew=1.5, zorder=5)
    print("delta L10", [(r[3], round(r[4]["layers"][L]["norm_delta"] / r[4]["layers"][L]["norm_h0_name"], 2),
                         round(r[4]["layers"][L]["norm_delta_retain"] / r[4]["layers"][L]["norm_h0_name"], 2),
                         round(r[4]["layers"][L]["shared_share"], 2), round(r[4]["layers"][L]["0+mean_top1"], 2),
                         round(r[4]["layers"][L]["u+mean_top1"], 2), round(r[4]["layers"][L]["u+rand@d_top1"], 2),
                         round(r[4]["top1"]["retain_u"], 2)) for r in rows])
    ax.set_yticks(ys)
    ax.set_yticklabels([])
    ax.set_ylim(len(rows) - 0.4, -0.6)
    ax.set_xlim(-0.04, 1.06)
    ax.set_xticks([0, 0.25, 0.5, 0.75, 1.0])
    ax.set_xlabel("share of retain facts still answered\n(first answer token)")
    ax.set_title("b  Shared part, on a retain name", loc="left", color=INK)
    ax.legend([Line2D([0], [0], marker="o", ls="", ms=8, mfc=NAME, mec=SURFACE), Line2D([0], [0], marker="o", ls="", ms=8, mfc=INK2, mec=SURFACE),
               Line2D([0], [0], marker="o", ls="", ms=11, mfc="none", mec=INK2, mew=1.2)],
              ["added in the unlearned model", "added in the original model", "random direction, unlearned model"],
              loc="upper center", bbox_to_anchor=(0.5, -0.3), ncol=1, frameon=False, fontsize=7.5, labelcolor=INK2)
    # (c) the same person's change in two runs
    ax = axes[2]
    clean(ax)
    groups = collections.OrderedDict()
    for key, tag in (("qwen2vl7b", "qwen"), ("llava", "llava")):
        for pair in json.load(open(ROOT / key / "mech_umu" / "_compare" / "delta_what" / f"{tag}.json")):
            kind = "two NPO\nseeds" if "ga_diff" not in pair["a"] + pair["b"] else "NPO and\nGD"
            groups.setdefault(f"{MODELS[key]['short']}\n{kind}", []).append(pair["layers"][L])
    for x, (lab, pairs) in enumerate(groups.items()):
        jit = np.linspace(-0.16, 0.16, len(pairs)) if len(pairs) > 1 else [0.0]
        for j, p in zip(jit, pairs):
            ax.plot(x + j, p["cos_person_centred_shuffled"], "o", ms=7.5, mfc=SURFACE, mec=INK2, mew=1.4, zorder=3)
            ax.plot(x + j, p["cos_person_centred"], "o", ms=8.5, mfc=NAME, mec=SURFACE, mew=1.5, zorder=4)
    print("cross-run", {k.replace("\n", " "): [(round(p["cos_person_centred"], 2), round(p["cos_person_centred_shuffled"], 2), round(p["cos_mean"], 2)) for p in v]
                        for k, v in groups.items()})
    ax.axhline(0, color=AXIS, lw=0.8)
    ax.set_xticks(range(len(groups)))
    ax.set_xticklabels(list(groups), color=INK2, fontsize=7.5)
    ax.set_xlim(-0.5, len(groups) - 0.5)
    ax.set_ylim(-0.1, 0.8)
    ax.set_ylabel("cosine between two runs")
    ax.set_title("c  Person-specific part", loc="left", color=INK)
    ax.legend([Line2D([0], [0], marker="o", ls="", ms=8, mfc=NAME, mec=SURFACE), Line2D([0], [0], marker="o", ls="", ms=7.5, mfc=SURFACE, mec=INK2, mew=1.4)],
              ["the same person in both runs", "different people"], loc="upper center", bbox_to_anchor=(0.5, -0.3), ncol=1,
              frameon=False, fontsize=7.5, labelcolor=INK2)
    fig.tight_layout(w_pad=1.6)
    save(fig, "mech_delta")


# ------------------------------------------------------------------------------------- person code
def fig_person_code():
    fig, axes = plt.subplots(2, 2, figsize=(9.6, 5.8))
    fig.patch.set_facecolor(SURFACE)
    taus = np.linspace(0, 0.6, 61)
    for c, (key, m) in enumerate(MODELS.items()):
        base = ROOT / key / "mech_umu" / m["vanilla"]
        # top: can the person be read across the two forms?
        ax = axes[0, c]
        clean(ax)
        s = json.load(open(base / "identity_code" / "summary.json"))
        layers = sorted(int(k) for k in s["layers"])
        get = lambda k: [s["layers"][str(L)][k] for L in layers]  # noqa: E731
        ax.plot(layers, get("N->N"), color=MUTED, lw=1.2, ls=(0, (4, 2)))
        ax.plot(layers, get("N->F"), color=NAME, lw=2, solid_capstyle="round")
        ax.plot(layers, get("F->N"), color=FACE, lw=2, solid_capstyle="round")
        merge_line(ax, m["L_star"], y=1.04, side="left")
        ax.set_ylim(-0.04, 1.2)
        ax.set_yticks([0, 0.25, 0.5, 0.75, 1.0])
        ax.set_xlim(layers[0], layers[-1])
        ax.set_xticks([x for x in layers if x % 4 == 0])
        ax.set_xlabel("decoder layer (state at the last prompt token)")
        ax.set_title(f"{m['label']}: the same person found among {s['n_people']}", loc="left", color=INK)
        if c == 0:
            ax.set_ylabel("share of questions")
        # bottom: a recogniser built from the forget people's name questions only
        ax = axes[1, c]
        clean(ax)
        g = json.load(open(base / "merge_gate" / "summary.json"))
        off = [x for x in jl(base / "merge_gate" / "items.jsonl") if x["cond"] == "off"]

        def fires(st, mod):
            rs = [x for x in off if x["set"] == st and x["modality"] == mod]
            cs, nf = np.array([x["cos_nearest"] for x in rs]), np.array([x["nearest"] == "forget" for x in rs])
            return [float(np.mean(nf & (cs > t))) for t in taus]
        ax.plot(taus, fires("forget", "txt"), color=NAME, lw=2, solid_capstyle="round")
        ax.plot(taus, fires("forget", "img"), color=FACE, lw=2, solid_capstyle="round")
        ax.plot(taus, fires("retain", "img"), color=INK2, lw=1.6, ls=(0, (4, 2)))
        tau = g["tau"]["protocol"]
        ax.axvline(tau, color=MUTED, lw=0.8)
        ax.annotate("5% false alarms on\npeople it does not know", (tau, 1.04), xytext=(3, 0), textcoords="offset points",
                    ha="left", va="bottom", fontsize=7, color=INK2)
        print(key, "tau", tau, "fires at tau: forget txt/img, strangers img",
              [round(fires(a, b)[int(round(tau * 100))], 2) for a, b in (("forget", "txt"), ("forget", "img"), ("retain", "img"))])
        ax.set_ylim(-0.04, 1.26)
        ax.set_yticks([0, 0.25, 0.5, 0.75, 1.0])
        ax.set_xlim(0, 0.6)
        ax.set_xlabel("threshold on the cosine to the nearest person code")
        ax.set_title(f"{m['short']}, layer {g['layer']}: questions the recogniser fires on", loc="left", color=INK)
        if c == 0:
            ax.set_ylabel("share of questions")
    fig.legend([Line2D([0], [0], color=FACE, lw=2), Line2D([0], [0], color=NAME, lw=2), Line2D([0], [0], color=MUTED, lw=1.2, ls=(0, (4, 2)))],
               ["portrait question matched to name-question codes", "name question matched to portrait-question codes",
                "within the name form"], loc="center", ncol=3, frameon=False, fontsize=7.5, labelcolor=INK2,
               bbox_to_anchor=(0.5, 0.505))
    fig.legend([Line2D([0], [0], color=NAME, lw=2), Line2D([0], [0], color=FACE, lw=2), Line2D([0], [0], color=INK2, lw=1.6, ls=(0, (4, 2)))],
               ["forget people, name questions", "forget people, portrait questions (never seen by the recogniser)",
                "people unknown to the recogniser, portrait questions"], loc="lower center", ncol=3, frameon=False,
               fontsize=7.5, labelcolor=INK2, bbox_to_anchor=(0.5, -0.03))
    fig.tight_layout(w_pad=2.0, h_pad=3.4)
    save(fig, "mech_person_code")


# --------------------------------------------------------------------------------------- schematic
def lane(ax, x, w, y0, y1, color, alpha):
    ax.add_patch(FancyBboxPatch((x, y0), w, y1 - y0, boxstyle="round,pad=0,rounding_size=0.08", fc=color, ec="none",
                                alpha=alpha, zorder=1))
    ax.add_patch(FancyBboxPatch((x, y0), w, y1 - y0, boxstyle="round,pad=0,rounding_size=0.08", fc="none", ec=AXIS,
                                lw=0.8, zorder=3))


def hatch(ax, x, w, y0, y1):
    ax.add_patch(Rectangle((x, y0), w, y1 - y0, fc="none", ec=INK2, hatch="////", lw=0, zorder=2))


def arrow(ax, p0, p1, color, rad=0.0):
    ax.add_patch(FancyArrowPatch(p0, p1, arrowstyle="-|>", mutation_scale=11, lw=2, color=color, zorder=4,
                                 connectionstyle=f"arc3,rad={rad}", shrinkA=0, shrinkB=0))


def fig_schematic():
    fig, axes = plt.subplots(1, 2, figsize=(9.6, 4.5))
    fig.patch.set_facecolor(SURFACE)
    Y0, Y1, YM = 0.0, 6.0, 3.75            # first layer, last layer, merge layer (drawn at 20/32 of the height)
    W, xs = 1.05, [0.0, 2.15, 4.3, 6.45]   # lanes: referent tokens, rest of the question, last prompt token, answer tokens
    panels = [
        ("a  Asked by name (the form used for unlearning)", NAME, True,
         ["Amelia\nBrockhurst", "'s occupation?", "ASSISTANT:", "Graphic\nDesigner"], "name tokens",
         "the edit is written\nat the name tokens,\nbelow the merge layer", "wrong value, or\none fixed string"),
        ("b  Asked by portrait (not used for unlearning)", FACE, False,
         ["[portrait]", "this person's\noccupation?", "ASSISTANT:", "Graphic\nDesigner"], "image tokens",
         "no name tokens:\nnothing on this\npath is edited", "correct answer"),
    ]
    for ax, (title, color, edited, labels, ref, note, out) in zip(axes, panels):
        ax.set_facecolor(SURFACE)
        ax.set_xlim(-1.0, 7.7)
        ax.set_ylim(-1.35, 7.75)
        ax.axis("off")
        ax.set_title(title, loc="left", color=INK)
        ax.plot([-0.2, 7.6], [YM, YM], color=MUTED, lw=0.8, zorder=0)
        ax.text(-0.3, Y0, "first\nlayer", ha="right", va="bottom", fontsize=7, color=MUTED)
        ax.text(-0.3, Y1, "last\nlayer", ha="right", va="top", fontsize=7, color=MUTED)
        ax.text(-0.3, YM, "merge\nlayer", ha="right", va="center", fontsize=7, color=INK2)
        for x, lab, sub in zip(xs, labels, [ref, "rest of the\nquestion", "last prompt\ntoken", "answer tokens\n(generated)"]):
            ax.text(x + W / 2, -0.18, lab, ha="center", va="top", fontsize=7.5, color=INK)
            ax.text(x + W / 2, -0.95, sub, ha="center", va="top", fontsize=7, color=INK2, style="italic")
        lane(ax, xs[0], W, Y0, Y1, color, 0.22)
        lane(ax, xs[1], W, Y0, Y1, GRID, 0.55)
        lane(ax, xs[2], W, Y0, Y1, color, 0.22)
        lane(ax, xs[3], W, Y0, Y1, color, 0.22)
        if edited:
            hatch(ax, xs[0], W, Y0 + 0.08, YM - 0.35)      # written and carried on the name tokens
            hatch(ax, xs[2], W, YM - 0.35, Y1 - 0.08)      # after the hand-over: at the positions that produce the answer
            hatch(ax, xs[3], W, YM - 0.35, Y1 - 0.08)
        arrow(ax, (xs[0] + W, YM - 1.05), (xs[2], YM - 0.4), color, rad=-0.12)
        arrow(ax, (xs[0] + W, YM - 1.45), (xs[3], YM - 0.4), color, rad=-0.22)
        ax.text(xs[1] + W / 2, 0.95, note, ha="center", va="center", fontsize=7.2, color=INK)
        ax.text((xs[2] + xs[3] + W) / 2, 0.95, "every position that\nproduces an answer\ntoken reads the person\nfrom the " + ref,
                ha="center", va="center", fontsize=7.2, color=INK)
        for x in (xs[2], xs[3]):
            arrow(ax, (x + W / 2, Y1), (x + W / 2, Y1 + 0.5), INK2)
        ax.text((xs[2] + xs[3] + W) / 2, Y1 + 0.62, out, ha="center", va="bottom", fontsize=8, color=INK, fontweight="semibold")
    handles = [Rectangle((0, 0), 1, 1, fc=NAME, alpha=0.3, ec=AXIS), Rectangle((0, 0), 1, 1, fc=FACE, alpha=0.3, ec=AXIS),
               Rectangle((0, 0), 1, 1, fc=SURFACE, ec=INK2, hatch="////", lw=0)]
    fig.legend(handles, ["carries the person, identified by name", "carries the person, identified by portrait",
                         "carries the change made by unlearning with names"],
               loc="lower center", ncol=3, frameon=False, fontsize=7.5, labelcolor=INK2, bbox_to_anchor=(0.5, -0.06))
    fig.tight_layout(w_pad=1.0)
    save(fig, "mech_schematic")


FIGS = collections.OrderedDict([("schematic", fig_schematic), ("matrix", fig_matrix), ("answer_level", fig_answer_level),
                                ("convergence", fig_convergence), ("path", fig_path), ("first_token", fig_first_token),
                                ("delta", fig_delta), ("person_code", fig_person_code)])

if __name__ == "__main__":
    setup()
    for name in (sys.argv[1:] or list(FIGS)):
        FIGS[name]()
