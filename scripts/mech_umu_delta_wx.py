"""UMU track: where on the input does the unlearning LoRA act? ||Delta W x|| by layer and token group.

theta_u = UMU's vanilla + the run's LoRA at --step. For every LoRA module (q/k/v/o, gate/up/down in each of the 32
decoder layers) a forward hook computes the LoRA's own contribution d(x) = scaling * B A x on the module input
(vlmu/mech/tools.py:delta_wx_norm; passive, Milestone J check J1) and averages ||d(x)|| over token groups:
  name      tokens of the person's name (name-based questions, with or without the portrait)
  image     the 576 image tokens (portrait questions)
  template  every other prompt token (chat frame, question words, options, instruction)
  last      the last prompt token, where the first answer token is predicted
  answer    the teacher-forced answer tokens (reported, not used for the conclusions)
For o_proj and down_proj (the two writers into the residual stream) it also reports the projection of d(x) on the
unit direction of the correct first answer token (final RMSNorm scale x unembedding row): negative = the LoRA writes
against the answer at that layer and position. Items: the run's doubly-correct facts, UMU-protocol prompt + the
correct option text (teacher forcing), routes forget txt / img / txt+img and retain txt / img.

Name tokens are found by character offsets in the prompt text and mapped to the batch after the processor expands
<image> to 576 tokens (verified against the LLaVA processor: identical token ids). Also writes the LoRA weight norm
||B A||_F * scaling per layer, to set "how big is the change" against "where does it matter" (layer masking, 4c).

Reading (layer masking found the effective edit in layers 0-19, DEV-012):
  name >> template, last in layers 0-19 on name-based questions        the edit is bound to the name tokens (H1')
  name tokens get the same ||dW x|| with the portrait (txt+img)        the edit fires on the name regardless of the
    and for retain people (txt)                                          image: the portrait route bypasses it, and every
                                                                         name pays (collateral)
  template / last as large as name                                      the edit keys on the text-only prompt form
  large late-layer ||dW x|| or ||BA|| with no masking effect            change size is not causal effect

  python scripts/mech_umu_delta_wx.py --config configs/llava/umu_npo_rt.yaml --run-id <run_id> --step <N>
Writes results/llava/mech_umu/<run_id>/step<N>/delta_wx.json and delta_wx.png.
"""
import collections
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np  # noqa: E402
import torch  # noqa: E402

from vlmu.config import base_parser, resolve  # noqa: E402
from vlmu.mech import tools  # noqa: E402
from vlmu.mech.hooks import decoder_layers, layer_index, lm_parts, lora_modules  # noqa: E402
from vlmu.models import get_adapter  # noqa: E402
from vlmu.runs import free, model_dir, results_dir  # noqa: E402
from vlmu.umu import encode_teacher_forced, load_umu, protocol_items, token_groups_umu as umu_token_groups, \
    umu_run_id  # noqa: E402
from vlmu.utils import Logger, atomic_write_json, read_json, tracked_run  # noqa: E402

FORMS = {"forget": ("txt", "img", "txt+img"), "retain": ("txt", "img")}
GROUPS = ("name", "image", "template", "last", "answer")
ATTN, MLP, WRITERS = ("q_proj", "k_proj", "v_proj", "o_proj"), ("gate_proj", "up_proj", "down_proj"), ("o_proj", "down_proj")
BANDS = {"L0-9": (0, 10), "L10-19": (10, 20), "L20-31": (20, 32)}


@torch.no_grad()
def lora_weight_norms(model):
    """{kind: [L]} of ||B A||_F * scaling via trace((B^T B)(A A^T)), no dense product."""
    out = collections.defaultdict(dict)
    for n, m in lora_modules(model).items():
        a = m.active_adapters[0]
        A, B = m.lora_A[a].weight.float(), m.lora_B[a].weight.float()
        fro = torch.sqrt(torch.clamp(torch.trace((B.T @ B) @ (A @ A.T)), min=0)) * m.scaling[a]
        out[n.rsplit(".", 1)[-1]][layer_index(n)] = float(fro)
    n_layers = 1 + max(max(v) for v in out.values())
    return {k: [v.get(i, float("nan")) for i in range(n_layers)] for k, v in out.items()}


def nanmean(xs):
    """Column mean ignoring NaN; NaN where a column has no values."""
    import warnings
    xs = np.array(xs, dtype=float)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        return np.nanmean(xs, axis=0)


def row_groups(res, j, n_layers):
    """One row's curves from tools.delta_wx_norm output: {group: {norm_attn, norm_mlp, proj_writers}} ([L], NaN where
    the row has no token of that group)."""
    out = {}
    col = lambda v: v[:, j].numpy() if v is not None else np.full(n_layers, np.nan)  # noqa: E731
    for g in GROUPS:
        nk = {k: res["norm"].get(k, {}).get(g) for k in ATTN + MLP}
        if all(v is None for v in nk.values()):
            continue
        empty = np.isnan(col(nk["down_proj"]))
        attn = np.nansum([col(nk[k]) for k in ATTN], axis=0)
        mlp = np.nansum([col(nk[k]) for k in MLP], axis=0)
        pw = [res["proj"].get(k, {}).get(g) for k in WRITERS]
        proj = np.nansum([col(v) for v in pw], axis=0) if all(v is not None for v in pw) else np.full(n_layers, np.nan)
        rk = [res.get("rel", {}).get(k, {}).get(g) for k in ATTN + MLP]
        rel = nanmean([col(v) for v in rk if v is not None]) if any(v is not None for v in rk) else np.full(n_layers, np.nan)
        attn[empty], mlp[empty], proj[empty], rel[empty] = np.nan, np.nan, np.nan, np.nan
        out[g] = {"norm_attn": attn.round(4).tolist(), "norm_mlp": mlp.round(4).tolist(), "proj_writers": proj.round(4).tolist(),
                  "rel_mean": rel.round(5).tolist()}
    return out


def summarize(per_row):
    """Mean curves per route and group, and band means (layers 0-9 / 10-19 / 20-31)."""
    summ = {}
    for key in sorted({f"{r['set']}/{r['modality']}" for r in per_row}):
        rows = [r for r in per_row if f"{r['set']}/{r['modality']}" == key]
        summ[key] = {}
        for g in GROUPS:
            have = [r[g] for r in rows if g in r and not np.all(np.isnan(np.array(r[g]["norm_mlp"], dtype=float)))]
            if not have:
                continue
            d = {k: nanmean([h[k] for h in have]) for k in ("norm_attn", "norm_mlp", "proj_writers", "rel_mean")}
            d["norm_all"] = d["norm_attn"] + d["norm_mlp"]
            summ[key][g] = {k: [None if np.isnan(v) else round(float(v), 4) for v in arr] for k, arr in d.items()}
            summ[key][g]["n_rows"] = len(have)
    bands = {}
    for key, gs in summ.items():
        for g, d in gs.items():
            arr = np.array([np.nan if v is None else v for v in d["norm_all"]], dtype=float)
            pr = np.array([np.nan if v is None else v for v in d["proj_writers"]], dtype=float)
            rl = np.array([np.nan if v is None else v for v in d["rel_mean"]], dtype=float)
            bands[f"{key}/{g}"] = {b: {"norm": None if np.isnan(arr[a:z]).all() else float(np.nanmean(arr[a:z])),
                                       "proj": None if np.isnan(pr[a:z]).all() else float(np.nanmean(pr[a:z])),
                                       "rel": None if np.isnan(rl[a:z]).all() else float(np.nanmean(rl[a:z]))}
                                   for b, (a, z) in BANDS.items()}
    return summ, bands


def plot(summ, wnorm, n_layers, title, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    INK2, MUTED, GRID, AXIS = "#52514e", "#898781", "#e1e0d9", "#c3c2b7"
    GC = {"name": "#6250d6", "image": "#eda100", "template": "#898781", "last": "#e34948"}
    RC = {"forget/txt": ("#2a78d6", "-"), "forget/img": ("#eb6834", "-"), "forget/txt+img": ("#1baf7a", "-"),
          "retain/txt": ("#2a78d6", "--")}
    x = np.arange(n_layers)
    fig, axes = plt.subplots(2, 2, figsize=(11, 6.6), dpi=200)
    for ax in axes.flat:
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
        ax.grid(axis="y", color=GRID, lw=0.6)
        ax.set_axisbelow(True)
        ax.axvspan(-0.5, 19.5, color="#2a78d6", alpha=0.05, lw=0)
        ax.tick_params(colors=MUTED)
        ax.set_xlim(-0.5, n_layers - 0.5)
        ax.set_xticks([0, 8, 16, 24, 31])
    def line(ax, y, **kw):
        if y is not None:
            ax.plot(x, np.array([np.nan if v is None else v for v in y], dtype=float), lw=1.7, **kw)
    for ax, route, groups, t in ((axes[0, 0], "forget/txt", ("name", "template", "last"), "a  Name, no image (forget)"),
                                 (axes[0, 1], "forget/img", ("image", "template", "last"), "b  Portrait (forget)")):
        for g in groups:
            line(ax, summ.get(route, {}).get(g, {}).get("norm_all"), color=GC[g], label=g)
        ax.set_title(t, loc="left", fontsize=9)
        ax.set_ylabel("mean ||ΔW x|| per token\n(sum over the 7 LoRA modules)", color=INK2, fontsize=8)
        ax.legend(frameon=False, fontsize=7.5, loc="lower left")
    ax = axes[1, 0]
    for route, label in (("forget/txt", "name, no image (forget)"), ("forget/txt+img", "name + portrait (forget)"),
                         ("retain/txt", "name, no image (retain)")):
        c, ls = RC[route]
        line(ax, summ.get(route, {}).get("name", {}).get("norm_all"), color=c, ls=ls, label=label)
    ax.set_title("c  Name tokens: with and without the portrait, forget vs retain", loc="left", fontsize=9)
    ax.set_ylabel("mean ||ΔW x|| on name tokens", color=INK2, fontsize=8)
    ax.legend(frameon=False, fontsize=7.5)
    ax.set_xlabel("decoder layer (shaded: layers 0-19, where masking found the effect)", color=INK2, fontsize=8)
    ax = axes[1, 1]
    ax.axhline(0, color=AXIS, lw=0.9)
    for route, label in (("forget/txt", "name, no image (forget)"), ("forget/img", "portrait (forget)"),
                         ("forget/txt+img", "name + portrait (forget)"), ("retain/txt", "name, no image (retain)")):
        c, ls = RC[route]
        line(ax, summ.get(route, {}).get("last", {}).get("proj_writers"), color=c, ls=ls, label=label)
    ax.set_title("d  Last prompt token: LoRA write along the answer direction (o + down)", loc="left", fontsize=9)
    ax.set_ylabel("projection (negative = against the answer)", color=INK2, fontsize=8)
    ax.legend(frameon=False, fontsize=7.5)
    ax.set_xlabel("decoder layer", color=INK2, fontsize=8)
    if wnorm:
        ax2 = axes[0, 0].inset_axes([0.62, 0.62, 0.36, 0.32])
        tot = np.nansum([wnorm[k] for k in wnorm], axis=0)
        ax2.bar(x, tot, color=MUTED, width=0.8)
        ax2.set_title("||BA|| per layer", fontsize=6.5, color=INK2)
        ax2.tick_params(labelsize=5, colors=MUTED)
    fig.suptitle(title, x=0.01, ha="left", fontsize=9, color=INK2)
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def main():
    p = base_parser(__doc__)
    p.add_argument("--step", type=int, required=True)
    p.add_argument("--run-id", default=None, help="default: derived from the config (as scripts/unlearn_umu.py)")
    p.add_argument("--batch-img", type=int, default=4)
    p.add_argument("--batch-txt", type=int, default=16)
    args = p.parse_args()
    cfg = resolve(args)
    run_id = args.run_id or umu_run_id(cfg)
    run_dir = model_dir(cfg) / "umu_unlearn" / run_id
    out = results_dir(cfg) / "mech_umu" / run_id / f"step{args.step}"
    log = Logger(out / "delta_wx_log.txt")
    adapter_dir = run_dir / "ckpt" / f"step{args.step}"
    if not (adapter_dir / "adapter_config.json").exists():
        raise SystemExit(f"no adapter at {adapter_dir}")
    with tracked_run(results_dir(cfg), out, f"mech_umu_delta_wx_{run_id}_step{args.step}", "mech_umu_delta_wx"):
        from peft import PeftModel
        dc = set(read_json(run_dir / "baseline_dc.json")["pair_ids"])
        dfs = load_umu(cfg["paths"]["root"], cfg["umu"]["n_retain_eval"], 0, log, cfg["umu"]["forget_split"])
        items = [r for r in protocol_items(dfs, forms=FORMS) if r["pair_id"] in dc]
        log(f"{run_id} step {args.step}: {len(items)} rows over {len(dc)} doubly-correct facts")
        adapter = get_adapter({**cfg["model"], "id": cfg["umu"]["ckpt"], "processor_id": cfg["model"]["id"]})
        model = PeftModel.from_pretrained(adapter.load_model(), str(adapter_dir), is_trainable=False)
        model.eval()
        n_layers = len(decoder_layers(model))
        wnorm = lora_weight_norms(model)
        per_row, counts, mism = [], collections.Counter(), 0
        for m in sorted({r["modality"] for r in items}):
            group = sorted([r for r in items if r["modality"] == m], key=lambda r: len(r["prompt"]))
            bs = args.batch_txt if group[0]["image_bytes"] is None else args.batch_img
            for i in range(0, len(group), bs):
                chunk = group[i:i + bs]
                batch = encode_teacher_forced(adapter, chunk)
                targets = tools.answer_targets(batch)
                groups, bad = umu_token_groups(adapter, model, chunk, batch, targets)
                mism += bad
                for g in GROUPS:
                    counts[(m, g)] += int(groups[g].sum())
                res = tools.delta_wx_norm(adapter, model, batch, targets, groups)
                for j, r in enumerate(chunk):
                    rec = {k: r[k] for k in ("uid", "set", "modality", "profile", "field", "pair_id")}
                    rec.update(row_groups(res, j, n_layers))
                    per_row.append(rec)
            log(f"  {m}: {len(group)} rows")
        model = None
        free()

        summ, bands = summarize(per_row)
        atomic_write_json(out / "delta_wx.json", {"run_id": run_id, "step": args.step, "n_layers": n_layers,
                                                  "token_counts": {f"{m}/{g}": c for (m, g), c in counts.items()},
                                                  "name_mask_mismatches": mism, "lora_weight_norm": wnorm,
                                                  "summary": summ, "bands": bands, "per_row": per_row})
        log(f"tokens per group: {dict((f'{m}/{g}', c) for (m, g), c in sorted(counts.items()) if c)}; "
            f"rows without a name mask (tokenisation mismatch): {mism}")
        log("mean ||ΔW x|| per token (sum over the 7 LoRA modules), by layer band; (rel) = mean ||ΔW x|| / ||W x|| over the "
            "7 modules, in %; [proj] = LoRA write along the answer direction at that position (o + down)")
        log(f"{'route/group':26s} " + " ".join(f"{b:>25s}" for b in BANDS))
        for k, v in bands.items():
            if k.endswith("/answer"):
                continue
            cells = []
            for b in BANDS:
                n, pj, rl = v[b]["norm"], v[b]["proj"], v[b].get("rel")
                cells.append(f"{'-' if n is None else format(n, '.3f'):>6s} ({'-' if rl is None else format(100 * rl, '.2f'):>5s}%) "
                             f"[{'-' if pj is None else format(pj, '+.3f'):>6s}]")
            log(f"{k:26s} " + " ".join(cells))
        tot = np.nansum([wnorm[k] for k in wnorm], axis=0)
        log("||B A||_F per layer band (sum over modules): " + ", ".join(
            f"{b} {np.nansum(tot[a:z]):.2f}" for b, (a, z) in BANDS.items()))
        plot(summ, wnorm, n_layers, f"{run_id} step {args.step}: where the unlearning LoRA acts (||ΔW x|| by token group)",
             out / "delta_wx.png")
        log(f"wrote {out / 'delta_wx.json'} and delta_wx.png")


if __name__ == "__main__":
    main()
