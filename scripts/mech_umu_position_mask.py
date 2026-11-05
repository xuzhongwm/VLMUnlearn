"""UMU track: which token positions carry the effective unlearning edit? Position-specific LoRA masking.

Layer masking (4c) found the effective edit in layers 0-19; ||dW x|| (4d) found it largest on the name tokens and on the
last prompt token. This experiment switches the unlearning LoRA off (or keeps it on) at chosen token positions, in every
LoRA module of every layer, and measures how much of the unlearning on the name-only questions comes back:
  rm_<group>      LoRA off at that group's positions, on everywhere else        -> is the group NECESSARY?
  only_<group>    LoRA on only at that group's positions (off on the rest of    -> is the group SUFFICIENT?
                  the prompt)
groups: name, last (the last prompt token, where the first answer token is predicted), template (all other prompt tokens),
image (portrait questions), and name+last. Exact switch (vlmu/mech/hooks.py:mask_contributions): the base output is
recomputed, so "nothing off" reproduces the unlearned model and "everything off" the vanilla bit for bit (checked and
logged at the start). Teacher-forced on the run's doubly-correct facts as the logit lens; readouts per condition:
log p of the first answer token at the last position, its recovery (0 = unlearned, 1 = vanilla), and top-1 accuracy of
the first answer token (does the correct answer start win?).

Reading:
  rm_name restores and only_name reproduces the forgetting      the edit is in how the name tokens are processed
  rm_last restores and only_last reproduces it                  the edit is in how the last position reads the identity
  both needed / neither enough alone                            the edit is the name -> last-token pathway as a whole
  template matters as much                                      the edit keys on the text-only prompt form

  python scripts/mech_umu_position_mask.py --config configs/llava/umu_npo_rt.yaml --run-id <run_id> --step <N>
Writes results/llava/mech_umu/<run_id>/step<N>/position_mask.json and position_mask.png.
"""
import collections
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np  # noqa: E402
import torch  # noqa: E402

from vlmu.config import base_parser, resolve  # noqa: E402
from vlmu.mech import tools  # noqa: E402
from vlmu.mech.hooks import lora_modules, mask_contributions  # noqa: E402
from vlmu.models import get_adapter  # noqa: E402
from vlmu.runs import free, model_dir, results_dir  # noqa: E402
from vlmu.umu import encode_teacher_forced, first_target, load_umu, protocol_items, token_groups_umu, \
    umu_run_id  # noqa: E402
from vlmu.utils import Logger, atomic_write_json, read_json, read_jsonl, tracked_run  # noqa: E402

FORMS = {"forget": ("txt", "img", "txt+img"), "retain": ("txt", "img")}
CONDITIONS = ("unlearned", "vanilla", "rm_name", "rm_last", "rm_template", "rm_image", "rm_name+last",
              "only_name", "only_last", "only_template", "only_name+last")


def remove_mask(cond, g):
    """[B, T] bool: positions where the LoRA is switched OFF for this condition (answer positions keep it: they come
    after the scored last position and cannot affect the first answer token)."""
    prompt = g["name"] | g["image"] | g["template"] | g["last"]
    if cond == "unlearned":
        return torch.zeros_like(prompt)
    if cond == "vanilla":
        return prompt | g["answer"]
    kind, what = cond.split("_", 1)
    sel = torch.zeros_like(prompt)
    for part in what.split("+"):
        sel = sel | g[part]
    return sel if kind == "rm" else prompt & ~sel


@torch.no_grad()
def score(adapter, model, names, batch, t_first, off):
    keep = (~off).to(torch.float32)
    with mask_contributions(model, names, keep):
        logits = tools.forward_logits(adapter, model, batch)
    at = logits[t_first["b"], t_first["pos"]].float()
    lp = at.log_softmax(-1).gather(-1, t_first["tok"][:, None]).squeeze(-1)
    top1 = at.argmax(-1) == t_first["tok"]
    return lp.cpu().tolist(), top1.cpu().tolist()


def summarize(items, res, lost):
    """{cond: {group: {lp, top1, n, recovery}}}; groups: forget/txt/lost, forget/txt, retain/txt/lost, ..."""
    groups = collections.defaultdict(list)
    for r in items:
        g = f"{r['set']}/{r['modality']}"
        groups[g].append(r["uid"])
        if r["modality"] == "txt":
            groups[f"{g}/{'lost' if lost.get((r['set'], r['pair_id'])) else 'kept'}"].append(r["uid"])
    out = {}
    for cond, per in res.items():
        out[cond] = {}
        for g, uids in groups.items():
            out[cond][g] = {"n": len(uids), "lp": float(np.mean([per[u]["lp"] for u in uids])),
                            "top1": float(np.mean([per[u]["top1"] for u in uids]))}
    for cond in out:
        for g, v in out[cond].items():
            u, z = out["unlearned"][g]["lp"], out["vanilla"][g]["lp"]
            v["recovery"] = (v["lp"] - u) / (z - u) if abs(z - u) > 1e-6 else None
    return out


def plot(summ, title, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    INK2, MUTED, GRID, AXIS = "#52514e", "#898781", "#e1e0d9", "#c3c2b7"
    rm = ["rm_name", "rm_last", "rm_template", "rm_name+last"]
    only = ["only_name", "only_last", "only_template", "only_name+last"]
    labels = ["name", "last", "template", "name+last"]
    series = [("forget/txt/lost", "Forget, name only, lost facts", 1.0), ("retain/txt/lost", "Retain, name only, lost facts", 0.45)]
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 3.4), dpi=200, sharey=True)
    for ax, conds, ttl, flip in ((axes[0], rm, "LoRA OFF at these tokens: how much comes back (necessary?)", False),
                                 (axes[1], only, "LoRA ON only at these tokens: how much forgetting remains (sufficient?)", True)):
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
        ax.grid(axis="y", color=GRID, lw=0.6)
        ax.set_axisbelow(True)
        ax.axhline(0, color=AXIS, lw=0.9)
        ax.axhline(1, color=AXIS, lw=0.9)
        x = np.arange(len(conds))
        for k, (g, lab, alpha) in enumerate(series):
            if g not in summ["unlearned"]:
                continue
            ys = []
            for c in conds:
                r = summ[c][g]["recovery"]
                ys.append(0 if r is None else (1 - r if flip else r))
            ax.bar(x + (k - 0.5) * 0.36, ys, width=0.32, color="#2a78d6", alpha=alpha, label=f"{lab} (n={summ['unlearned'][g]['n']})")
        ax.set_xticks(x, labels, fontsize=8)
        ax.set_title(ttl, loc="left", fontsize=8.5, color=INK2)
        ax.tick_params(colors=MUTED)
    axes[0].set_ylabel("fraction of the unlearning effect\n(log p of the first answer token)", color=INK2, fontsize=8)
    axes[0].legend(frameon=False, fontsize=7)
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
    log = Logger(out / "position_mask_log.txt")
    adapter_dir = run_dir / "ckpt" / f"step{args.step}"
    if not (adapter_dir / "adapter_config.json").exists():
        raise SystemExit(f"no adapter at {adapter_dir}")
    with tracked_run(results_dir(cfg), out, f"mech_umu_position_mask_{run_id}_step{args.step}", "mech_umu_position_mask"):
        from peft import PeftModel
        dc = set(read_json(run_dir / "baseline_dc.json")["pair_ids"])
        dfs = load_umu(cfg["paths"]["root"], cfg["umu"]["n_retain_eval"], 0, log, cfg["umu"]["forget_split"])
        items = [r for r in protocol_items(dfs, forms=FORMS) if r["pair_id"] in dc]
        at_step = {r["uid"]: r["correct_exact"] for r in read_jsonl(run_dir / "evals" / f"step{args.step}.jsonl")}
        lost = {(r["set"], r["pair_id"]): at_step.get(r["uid"]) is False for r in items if r["modality"] == "txt"}
        adapter = get_adapter({**cfg["model"], "id": cfg["umu"]["ckpt"], "processor_id": cfg["model"]["id"]})
        model = PeftModel.from_pretrained(adapter.load_model(), str(adapter_dir), is_trainable=False)
        model.eval()
        names = list(lora_modules(model))
        batches = []
        for m in sorted({r["modality"] for r in items}):
            group = sorted([r for r in items if r["modality"] == m], key=lambda r: len(r["prompt"]))
            bs = args.batch_txt if group[0]["image_bytes"] is None else args.batch_img
            for i in range(0, len(group), bs):
                chunk = group[i:i + bs]
                batch = encode_teacher_forced(adapter, chunk)
                t_all = tools.answer_targets(batch)
                g, bad = token_groups_umu(adapter, model, chunk, batch, t_all)
                if bad:
                    log(f"  warning: {bad} rows without a name mask in a {m} batch")
                batches.append((chunk, batch, first_target(t_all), g))
        log(f"{run_id} step {args.step}: {len(items)} rows, {len(batches)} batches; name route lost at this step: "
            f"forget {sum(v for (s, _), v in lost.items() if s == 'forget')}, retain {sum(v for (s, _), v in lost.items() if s == 'retain')}")

        # exactness of the switch: nothing off = unlearned model, everything off = vanilla (adapter disabled)
        for chunk, batch, t_first, g in (batches[0], batches[-1]):
            ref_u = tools.forward_logits(adapter, model, batch).float()
            with model.disable_adapter():
                ref_0 = tools.forward_logits(adapter, model, batch).float()
            with mask_contributions(model, names, torch.ones_like(g["last"], dtype=torch.float32)):
                hu = tools.forward_logits(adapter, model, batch).float()
            with mask_contributions(model, names, (~remove_mask("vanilla", g)).to(torch.float32)):
                h0 = tools.forward_logits(adapter, model, batch).float()
            v = batch["attention_mask"].bool()
            log(f"  switch check ({chunk[0]['modality']}): max |all-on - unlearned| = {(hu - ref_u)[v].abs().max():.2e}, "
                f"max |all-off - vanilla| = {(h0 - ref_0)[v].abs().max():.2e}")

        res = {}
        for cond in CONDITIONS:
            per = {}
            for chunk, batch, t_first, g in batches:
                lp, top1 = score(adapter, model, names, batch, t_first, remove_mask(cond, g))
                for r, a, b in zip(chunk, lp, top1):
                    per[r["uid"]] = {"lp": round(a, 4), "top1": bool(b)}
            res[cond] = per
            log(f"  scored {cond}")
        model = None
        free()
        summ = summarize(items, res, lost)
        atomic_write_json(out / "position_mask.json", {"run_id": run_id, "step": args.step, "conditions": list(CONDITIONS),
                                                       "summary": summ, "per_item": res})
        groups = ["forget/txt/lost", "forget/txt", "retain/txt/lost", "forget/txt+img", "forget/img"]
        log("recovery of log p(first answer token) (0 = unlearned, 1 = vanilla) | top-1 accuracy of the first answer token")
        log(f"{'condition':16s} " + " ".join(f"{g:>22s}" for g in groups))
        for c in CONDITIONS:
            cells = []
            for g in groups:
                v = summ[c].get(g)
                if v is None:
                    cells.append(f"{'-':>22s}")
                    continue
                rec = v["recovery"]
                cells.append(f"{'-' if rec is None else format(rec, '+.2f'):>8s} | top1 {v['top1']:.2f}  ")
            log(f"{c:16s} " + " ".join(cells))
        plot(summ, f"{run_id} step {args.step}: switching the unlearning LoRA off by token position", out / "position_mask.png")
        log(f"wrote {out / 'position_mask.json'} and position_mask.png")


if __name__ == "__main__":
    main()
