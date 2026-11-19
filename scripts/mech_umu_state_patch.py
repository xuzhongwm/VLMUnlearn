"""UMU track: where is the suppression carried? Cross-model state patching on the training form (DEVIATIONS DEV-015).

Layer and position masking located where the unlearning LoRA writes (layers 0-19, the whole text prompt). This
experiment locates what those writes change: the hidden state that carries the suppression to the output. Prompt: the
training form "USER: {name question}\nASSISTANT:" with the teacher-forced answer. At one decoder layer and one group of
prompt tokens, the residual state of one model is replaced by the other model's state from the same prompt:
  restore   theta_u runs with theta_0's state      recovery = (lp_patched - lp_u) / (lp_0 - lp_u)   1 = answer is back
  induce    theta_0 runs with theta_u's state      induced  = (lp_0 - lp_patched) / (lp_0 - lp_u)   1 = fully suppressed
lp = log p of the first answer token (only prompt positions can affect it; the answer-text log p is saved too).
Ratio of means over facts, 95% bootstrap over profiles. Second readout, top-1: is the correct first answer token the
most likely one? The input-cue sweep showed that a large drop in log p and a changed generated answer are different
things (3 nats cost as many answers as 14), so the map is reported on both.
Token groups: opening (BOS and "USER:"), name, question (the other question tokens up to the question mark), frame (the
newline and "ASSISTANT"), last (the final ":", where the first answer token is predicted), prompt (all of them). The
first run (2026-11-19) had no opening group: its question group included those tokens; its records are not reused.

Reading (exploratory, no thresholds): a group with high recovery at a layer carries the suppression there. Recovery for
`prompt` at layer l is the share of the suppression produced by layers up to l. The usual picture for a fact recalled
from a name is: carried on the name / question tokens in early layers, then on the last token.

Also, descriptive: how far theta_u's state is from theta_0's, ||h_u - h_0|| / ||h_0|| per layer and group, on the
training form and on the same question behind (a) a blank image and (b) 576 tokens of unrelated text. The input-cue
sweep found that both remove most of the log-p suppression but only the image restores the generated answers.

Default: the frozen doubly-correct forget facts, every other layer, residual stream; resumable per fact. Opt-in:
--with-retain, --layer-step 1 (every layer, about twice as slow), --no-contexts. First batch: a model's own state patched
back in must reproduce the unpatched run (logged; aborts otherwise).

  python scripts/mech_umu_state_patch.py --config configs/llava/umu_npo_rt.yaml \
    --run-id umu_npo_rt_txt_lr5e-05_s0 --step 150
Writes results/llava/mech_umu/<run>/step<N>/state_patch/{summary.json,items.jsonl,state_patch.png,log.txt}.
"""
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np  # noqa: E402
import torch  # noqa: E402

from scripts.mech_umu_form_sweep import blank_image_bytes, facts_for, filler_string  # noqa: E402
from vlmu.config import base_parser, resolve  # noqa: E402
from vlmu.mech import tools  # noqa: E402
from vlmu.mech.hooks import decoder_layers, patch_outputs, record_outputs  # noqa: E402
from vlmu.models import get_adapter  # noqa: E402
from vlmu.runs import free, model_dir, results_dir  # noqa: E402
from vlmu.umu import encode_teacher_forced, first_target, load_umu, token_groups_umu, umu_run_id  # noqa: E402
from vlmu.utils import Logger, append_jsonl, atomic_write_json, read_json, read_jsonl, tracked_run  # noqa: E402

GROUPS = ("opening", "name", "question", "frame", "last", "prompt")
CONTEXTS = ("blank", "filler")     # the same question behind a blank image / behind 576 filler text tokens
CTX_GROUPS = ("opening", "image", "context", "name", "question", "frame", "last")
SELF_TOL = 0.05      # nats; a model's own state patched back in must reproduce the unpatched run


def make_rows(fact, blank, filler):
    """{'text' | 'blank' | 'filler': row}; the prompts are those of the input-cue sweep (base, blank_before,
    text_prefix_576)."""
    q = fact["q_uni"]
    shared = {"pair_id": fact["pair_id"], "set": fact["set"], "profile": fact["profile"], "name": fact["name"],
              "correct_text": fact["answer_text"]}
    prompts = {"text": (f"USER: {q}\nASSISTANT:", None), "blank": (f"USER: <image>\n{q}\nASSISTANT:", blank),
               "filler": (f"USER: {filler}\n{q}\nASSISTANT:", None)}
    return {k: {**shared, "uid": f"{fact['pair_id']}/{k}", "prompt": prompt, "image_bytes": img}
            for k, (prompt, img) in prompts.items()}


def split_groups(adapter, model, rows, batch, t_all):
    """token_groups_umu, with its `template` group split in four: opening = BOS and "USER:"; frame = the last newline
    before the end of the prompt and "ASSISTANT"; context = everything else up to an earlier newline, if there is one
    (the text before the question in the blank-image and filler prompts; empty on the training form); question = the
    rest. Returns ({group: bool [B, T]}, number of rows without a name mask)."""
    g, _ = token_groups_umu(adapter, model, rows, batch, t_all)
    nl = adapter.tok.convert_tokens_to_ids("<0x0A>")
    ids = batch["input_ids"]
    frame, context, opening = (torch.zeros_like(g["last"]) for _ in range(3))
    head = adapter.tok("USER:", add_special_tokens=True)["input_ids"]
    for i in range(ids.shape[0]):
        if ids[i, :len(head)].tolist() != head:
            raise ValueError(f"the prompt does not start with the tokens of 'USER:': {rows[i]['uid']}")
        opening[i, :len(head)] = True
        last = int(g["last"][i].nonzero()[0])
        start = last - 1
        while start > 0 and int(ids[i, start]) != nl:
            start -= 1
        if int(ids[i, start]) != nl:
            raise ValueError(f"no newline before the end of the prompt: {rows[i]['uid']}")
        frame[i, start:last] = True
        earlier = (ids[i, :start] == nl).nonzero()
        if len(earlier):
            context[i, :int(earlier[0]) + 1] = True
    context = context & g["template"] & ~opening
    out = {"opening": opening, "name": g["name"], "question": g["template"] & ~frame & ~context & ~opening,
           "frame": frame, "last": g["last"], "image": g["image"], "context": context}
    out["prompt"] = out["opening"] | out["name"] | out["question"] | out["frame"] | out["last"]
    return out, int((~g["name"].any(1)).sum())


def readout(adapter, model, batch, t_all, t_first):
    logits = tools.forward_logits(adapter, model, batch)
    at = logits[t_first["b"], t_first["pos"]]
    first = tools.target_logprob(at, t_first["tok"])
    answer = tools._per_row(tools.target_logprob(logits[t_all["b"], t_all["pos"]], t_all["tok"]), t_all)
    top1 = at.float().argmax(-1) == t_first["tok"]
    return first.float().cpu().tolist(), answer.float().cpu().tolist(), top1.cpu().tolist()


def as_dict(lp, i):
    return {"first": round(lp[0][i], 4), "answer": round(lp[1][i], 4), "top1": bool(lp[2][i])}


def record_both(adapter, model, batch, t_all, t_first, n_layers):
    """Residual states of every layer and the readouts (first, answer, top1), for theta_u ('u') and theta_0 ('0')."""
    store, lp = {"u": {}, "0": {}}, {}
    with record_outputs(model, "resid", range(n_layers), store["u"]):
        lp["u"] = readout(adapter, model, batch, t_all, t_first)
    with model.disable_adapter(), record_outputs(model, "resid", range(n_layers), store["0"]):
        lp["0"] = readout(adapter, model, batch, t_all, t_first)
    return store, lp


def relative_change(store, groups, names, n_layers):
    """{group: [B, n_layers]} mean over the group's tokens of ||h_u - h_0|| / ||h_0||; NaN where a row has no such token."""
    out = {}
    for name in names:
        m = groups[name].float()
        count = m.sum(1)
        cols = []
        for layer in range(n_layers):
            u, z = store["u"][layer].float(), store["0"][layer].float()
            rel = (u - z).norm(dim=-1) / z.norm(dim=-1).clamp_min(1e-8)
            cols.append((rel * m).sum(1) / count.clamp_min(1))
        vals = torch.stack(cols, 1)
        vals[count == 0] = float("nan")
        out[name] = vals.cpu().tolist()
    return out


@torch.no_grad()
def score_text(adapter, model, rows, layers, n_layers, check):
    """One batch of training-form prompts: every (direction, group, layer) patch. Returns (records, self-check error)."""
    batch = encode_teacher_forced(adapter, rows)
    t_all = tools.answer_targets(batch)
    t_first = first_target(t_all)
    groups, no_name = split_groups(adapter, model, rows, batch, t_all)
    store, lp = record_both(adapter, model, batch, t_all, t_first, n_layers)
    err = None
    if check:
        b, p = groups["prompt"].nonzero(as_tuple=True)
        mid = layers[len(layers) // 2]
        with patch_outputs(model, "resid", {mid: store["u"][mid]}, (b, p, p)):
            same = readout(adapter, model, batch, t_all, t_first)
        with model.disable_adapter(), patch_outputs(model, "resid", {mid: store["0"][mid]}, (b, p, p)):
            same0 = readout(adapter, model, batch, t_all, t_first)
        err = max(abs(x - y) for got, ref in ((same, lp["u"]), (same0, lp["0"])) for a, c in zip(got, ref)
                  for x, y in zip(a, c))
    patched = {d: {g: {} for g in GROUPS} for d in ("restore", "induce")}
    for g in GROUPS:
        b, p = groups[g].nonzero(as_tuple=True)
        if len(b) == 0:
            continue
        for layer in layers:
            with patch_outputs(model, "resid", {layer: store["0"][layer]}, (b, p, p)):
                patched["restore"][g][layer] = readout(adapter, model, batch, t_all, t_first)
            with model.disable_adapter(), patch_outputs(model, "resid", {layer: store["u"][layer]}, (b, p, p)):
                patched["induce"][g][layer] = readout(adapter, model, batch, t_all, t_first)
    rel = relative_change(store, groups, GROUPS[:-1], n_layers)
    recs = []
    for i, r in enumerate(rows):
        rec = {"pair_id": r["pair_id"], "set": r["set"], "profile": r["profile"], "layers": list(layers),
               "groups": list(GROUPS),
               "n_tokens": {g: int(groups[g][i].sum()) for g in GROUPS},
               "lp": {tag: as_dict(lp[tag], i) for tag in ("u", "0")},
               "rel_text": {g: [round(v, 5) for v in rel[g][i]] for g in rel}}
        for d in ("restore", "induce"):
            # a row has no entry for a group it has no token of (e.g. the name was not found in the question)
            rec[d] = {g: {"first": [round(patched[d][g][layer][0][i], 4) for layer in layers],
                          "answer": [round(patched[d][g][layer][1][i], 4) for layer in layers],
                          "top1": [bool(patched[d][g][layer][2][i]) for layer in layers]}
                      for g in GROUPS if patched[d][g] and bool(groups[g][i].any())}
        recs.append(rec)
    return recs, err, no_name


@torch.no_grad()
def score_context(adapter, model, rows, n_layers, key):
    """{pair_id: {'rel_<key>': {group: [n_layers]}, 'lp_<key>': {...}}} for one batch of prompts of one context."""
    batch = encode_teacher_forced(adapter, rows)
    t_all = tools.answer_targets(batch)
    t_first = first_target(t_all)
    groups, _ = split_groups(adapter, model, rows, batch, t_all)
    store, lp = record_both(adapter, model, batch, t_all, t_first, n_layers)
    rel = relative_change(store, groups, CTX_GROUPS, n_layers)
    return {r["pair_id"]: {f"rel_{key}": {g: [round(v, 5) for v in rel[g][i]] for g in rel},
                           f"lp_{key}": {tag: as_dict(lp[tag], i) for tag in ("u", "0")}}
            for i, r in enumerate(rows)}


def effect(recs, direction, group, k, key="first"):
    """Recovery (restore) or induced suppression (induce) at layer index k: ratio of means over the facts that have
    the group. None without such facts or without forgetting."""
    rows = [r for r in recs if group in r[direction]]
    if not rows:
        return None
    gap = float(np.mean([r["lp"]["0"][key] - r["lp"]["u"][key] for r in rows]))
    if abs(gap) < 1e-8:
        return None
    if direction == "restore":
        num = np.mean([r["restore"][group][key][k] - r["lp"]["u"][key] for r in rows])
    else:
        num = np.mean([r["lp"]["0"][key] - r["induce"][group][key][k] for r in rows])
    return float(num / gap)


def bootstrap(recs, direction, group, n_layers_scored, n_boot=300):
    """[[lo, hi] per layer] 95% profile bootstrap of effect() on the first answer token."""
    profiles = sorted({r["profile"] for r in recs})
    by = {p: [r for r in recs if r["profile"] == p] for p in profiles}
    rng = np.random.default_rng(0)
    draws = [[] for _ in range(n_layers_scored)]
    for _ in range(n_boot):
        sample = [r for p in rng.choice(profiles, len(profiles), replace=True) for r in by[p]]
        for k in range(n_layers_scored):
            v = effect(sample, direction, group, k)
            if v is not None:
                draws[k].append(v)
    return [[float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))] if d else [None, None] for d in draws]


def nanmean_cols(rows):
    """Per-layer mean over facts of lists that may hold NaN / None (rows without that token group)."""
    out = []
    for col in zip(*rows):
        vals = [v for v in col if v is not None and v == v]
        out.append(float(np.mean(vals)) if vals else None)
    return out


def summarize(recs_all, layers):
    out = {}
    for set_name in ("forget", "retain"):
        recs = [r for r in recs_all if r["set"] == set_name]
        if not recs:
            continue
        s = {"n_facts": len(recs), "n_profiles": len({r["profile"] for r in recs}), "layers": list(layers),
             "n_tokens_mean": {g: float(np.mean([r["n_tokens"][g] for r in recs])) for g in GROUPS},
             "forgetting": {k: float(np.mean([r["lp"]["0"][k] - r["lp"]["u"][k] for r in recs])) for k in ("first", "answer")},
             "top1": {tag: float(np.mean([float(r["lp"][tag]["top1"]) for r in recs])) for tag in ("u", "0")}}
        for d in ("restore", "induce"):
            s[d] = {}
            for g in GROUPS:
                s[d][g] = {"n_facts": sum(g in r[d] for r in recs),
                           "first": [effect(recs, d, g, k) for k in range(len(layers))],
                           "answer": [effect(recs, d, g, k, "answer") for k in range(len(layers))],
                           "top1": [float(np.mean([float(r[d][g]["top1"][k]) for r in recs if g in r[d]]))
                                    if any(g in r[d] for r in recs) else None for k in range(len(layers))],
                           "first_ci95": bootstrap(recs, d, g, len(layers))}
        s["rel_text"] = {g: nanmean_cols([r["rel_text"][g] for r in recs]) for g in GROUPS[:-1]}
        for c in CONTEXTS:
            have = [r for r in recs if f"rel_{c}" in r]
            if have:
                s[f"rel_{c}"] = {g: nanmean_cols([r[f"rel_{c}"][g] for r in have]) for g in CTX_GROUPS}
                s[f"forgetting_{c}"] = {k: float(np.mean([r[f"lp_{c}"]["0"][k] - r[f"lp_{c}"]["u"][k] for r in have]))
                                        for k in ("first", "answer")}
                s[f"top1_{c}"] = {tag: float(np.mean([float(r[f"lp_{c}"][tag]["top1"]) for r in have])) for tag in ("u", "0")}
                s[f"n_facts_{c}"] = len(have)
        out[set_name] = s
    return out


def fmt(v):
    return "    -" if v is None else f"{v:+5.2f}"


def plot(s, title, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    INK2, MUTED, GRID, AXIS = "#52514e", "#898781", "#e1e0d9", "#c3c2b7"
    colors = {"opening": "#1baf7a", "name": "#6a4fd8", "question": "#8a8880", "frame": "#eda100", "last": "#e34948",
              "prompt": "#111111"}
    layers = s["layers"]
    num = lambda xs: [np.nan if v is None else v for v in xs]  # noqa: E731
    fig, axes = plt.subplots(1, 4, figsize=(17.5, 3.7), dpi=200)
    panels = (("restore", "Unlearned model, one state taken from the vanilla:\nhow much of the answer comes back"),
              ("induce", "Vanilla, one state taken from the unlearned model:\nhow much suppression it carries over"))
    for ax, (d, ttl) in zip(axes, panels):
        ax.axvspan(-0.5, 19.5, color="#eef3fa", lw=0)
        for g in GROUPS:
            ys = num(s[d][g]["first"])
            ax.plot(layers, ys, color=colors[g], lw=1.2 if g == "prompt" else 1.7, ls="--" if g == "prompt" else "-",
                    label=f"{g} ({s['n_tokens_mean'][g]:.1f} tok)")
            ci = s[d][g]["first_ci95"]
            if all(c[0] is not None for c in ci):
                ax.fill_between(layers, [c[0] for c in ci], [c[1] for c in ci], color=colors[g], alpha=0.12, lw=0)
        for y in (0, 1):
            ax.axhline(y, color=AXIS, lw=0.9)
        ax.set_title(ttl, loc="left", fontsize=8, color=INK2)
        ax.set_xlabel("decoder layer whose output state is replaced (shaded: 0-19)", color=INK2, fontsize=8)
    axes[0].set_ylabel("share of the forgetting\n(log p of the first answer token)", color=INK2, fontsize=8)
    axes[0].legend(frameon=False, fontsize=6.5)
    full = list(range(len(s["rel_text"]["last"])))
    styles = (("rel_text", "no image", "-", 1.8), ("rel_filler", "576 text tokens before", "--", 1.4),
              ("rel_blank", "blank image before", ":", 1.6))
    for ax, g, where in ((axes[2], "last", "last prompt token"), (axes[3], "name", "name tokens")):
        ax.axvspan(-0.5, 19.5, color="#eef3fa", lw=0)
        for key, lab, ls, lw in styles:
            if key in s:
                ax.plot(full, num(s[key][g]), color=colors[g], lw=lw, ls=ls, label=lab)
        ax.set_title(f"How far the unlearned state is from the vanilla state,\n{where}: ||h_u - h_0|| / ||h_0||", loc="left",
                     fontsize=8, color=INK2)
        ax.set_xlabel("decoder layer", color=INK2, fontsize=8)
        ax.set_ylim(bottom=0)
        ax.legend(frameon=False, fontsize=6.5)
    for ax in axes:
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        ax.grid(axis="y", color=GRID, lw=0.6)
        ax.set_axisbelow(True)
        ax.tick_params(colors=MUTED, labelsize=8)
    fig.suptitle(title, x=0.01, ha="left", fontsize=9, color=INK2)
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def main():
    p = base_parser(__doc__)
    p.add_argument("--run-id", default=None)
    p.add_argument("--step", type=int, required=True)
    p.add_argument("--with-retain", action="store_true", help="also score the doubly-correct retain facts")
    p.add_argument("--layer-step", type=int, default=2, help="patch every n-th layer (the last layer is always included)")
    p.add_argument("--no-contexts", action="store_true", help="skip the blank-image / filler-text state comparison")
    p.add_argument("--max-facts", type=int, default=0, help="0 = all selected facts")
    p.add_argument("--batch-txt", type=int, default=16)
    p.add_argument("--batch-img", type=int, default=2, help="batch size for the ~600-token blank-image and filler prompts")
    p.add_argument("--restart", action="store_true", help="set existing per-fact results aside instead of resuming")
    args = p.parse_args()
    cfg = resolve(args)
    run_id = args.run_id or umu_run_id(cfg)
    run_dir = model_dir(cfg) / "umu_unlearn" / run_id
    ckpt = run_dir / "ckpt" / f"step{args.step}"
    if not (ckpt / "adapter_config.json").exists():
        raise SystemExit(f"no adapter at {ckpt}")
    out = results_dir(cfg) / "mech_umu" / run_id / f"step{args.step}" / "state_patch"
    log = Logger(out / "log.txt")
    with tracked_run(results_dir(cfg), out, f"mech_umu_state_patch_{run_id}_step{args.step}", "mech_umu_state_patch"):
        from peft import PeftModel
        dc = set(read_json(run_dir / "baseline_dc.json")["pair_ids"])
        dfs = load_umu(cfg["paths"]["root"], cfg["umu"]["n_retain_eval"], 0, log, cfg["umu"]["forget_split"])
        sets = ("forget", "retain") if args.with_retain else ("forget",)
        facts = sorted((f for f in facts_for(dfs, dc) if f["set"] in sets), key=lambda f: (f["set"], f["pair_id"]))
        if args.max_facts:
            facts = facts[:args.max_facts]
        if not facts:
            raise SystemExit("no doubly-correct facts selected")
        path = out / "items.jsonl"
        if args.restart and path.exists():
            path.rename(out / f"items.{int(time.time())}.jsonl.bak")

        adapter = get_adapter({**cfg["model"], "id": cfg["umu"]["ckpt"], "processor_id": cfg["model"]["id"]})
        model = PeftModel.from_pretrained(adapter.load_model(), str(ckpt), is_trainable=False)
        model.eval()
        n_layers = len(decoder_layers(model))
        layers = sorted(set(range(0, n_layers, max(1, args.layer_step))) | {n_layers - 1})
        contexts = () if args.no_contexts else CONTEXTS
        done = {r["pair_id"]: r for r in read_jsonl(path)
                if r.get("layers") == layers and r.get("groups") == list(GROUPS)
                and all(f"rel_{c}" in r for c in contexts)}
        blank, filler = blank_image_bytes(), filler_string(adapter.tok, 576)
        pairs = {f["pair_id"]: make_rows(f, blank, filler) for f in facts}
        todo = sorted((f for f in facts if f["pair_id"] not in done), key=lambda f: len(pairs[f["pair_id"]]["text"]["prompt"]))
        n_fwd = 2 * len(layers) * len(GROUPS) + 2
        log(f"{run_id} step {args.step}: {len(facts)} facts ({', '.join(sets)}), {len(todo)} to score; {len(layers)} layers x "
            f"{len(GROUPS)} token groups x 2 directions = {n_fwd} forward passes per batch of {args.batch_txt}; "
            f"state comparison on: {', '.join(contexts) or 'none'}")

        t0, worst, n_done, no_name = time.time(), None, 0, 0
        for i in range(0, len(todo), args.batch_txt):
            chunk = todo[i:i + args.batch_txt]
            recs, err, missing = score_text(adapter, model, [pairs[f["pair_id"]]["text"] for f in chunk], layers, n_layers,
                                            check=i == 0)
            no_name += missing
            if err is not None:
                worst = err
                log(f"  self-patch check: max |change in log p| = {err:.2e} nats (tolerance {SELF_TOL})")
                if err > SELF_TOL:
                    raise RuntimeError(f"patching a model's own state back in changed log p by {err:.3f} nats: "
                                       "the patch is not exact, results would not be interpretable")
            for c in contexts:
                extra = {}
                for j in range(0, len(chunk), args.batch_img):
                    extra.update(score_context(adapter, model, [pairs[f["pair_id"]][c] for f in chunk[j:j + args.batch_img]],
                                               n_layers, c))
                recs = [{**r, **extra[r["pair_id"]]} for r in recs]
            append_jsonl(path, recs)
            done.update({r["pair_id"]: r for r in recs})
            n_done += len(chunk)
            per = (time.time() - t0) / n_done
            log(f"  {n_done}/{len(todo)} facts, {per:.1f} s per fact, about {per * (len(todo) - n_done) / 60:.1f} min left")
        if no_name:
            log(f"  {no_name} facts without a name mask (the name was not found in the question): no 'name' entry for them")
        model = None
        free()

        summary = summarize([done[f["pair_id"]] for f in facts], layers)
        atomic_write_json(out / "summary.json", {
            "run_id": run_id, "step": args.step, "sets": list(sets), "layers": layers, "groups": list(GROUPS),
            "readout": "log p of the first answer token ('first'); answer-text log p ('answer'); 'top1' = the correct first answer token is the argmax",
            "definitions": {"restore": "theta_u with theta_0's residual state at one layer and token group; (lp_patched - lp_u) / (lp_0 - lp_u)",
                            "induce": "theta_0 with theta_u's residual state at one layer and token group; (lp_0 - lp_patched) / (lp_0 - lp_u)",
                            "rel": "mean over a group's tokens of ||h_u - h_0|| / ||h_0|| at each layer output"},
            "self_patch_error_nats": worst, "summary": summary})
        show = sorted(set(range(0, len(layers), 2)) | {len(layers) - 1})
        for set_name, s in summary.items():
            log(f"{set_name}: n={s['n_facts']} facts / {s['n_profiles']} profiles; forgetting {s['forgetting']['first']:.2f} nats "
                f"(first token), {s['forgetting']['answer']:.2f} nats (answer); tokens per group "
                + ", ".join(f"{g} {v:.1f}" for g, v in s["n_tokens_mean"].items()))
            log(f"  top-1 of the first answer token: unlearned {s['top1']['u']:.2f}, vanilla {s['top1']['0']:.2f}")
            for d, what in (("restore", "recovery with theta_0's state (1 = answer back)"),
                            ("induce", "suppression carried by theta_u's state (1 = all of it)")):
                log(f"  {what}; layer: " + " ".join(f"{layers[k]:5d}" for k in show))
                for g in GROUPS:
                    log(f"    {g:9s}        " + " ".join(fmt(s[d][g]["first"][k]) for k in show))
                log(f"  the same as top-1 accuracy of the first answer token ({'theta_u' if d == 'restore' else 'theta_0'} patched)")
                for g in GROUPS:
                    log(f"    {g:9s}        " + " ".join(fmt(s[d][g]["top1"][k]) for k in show))
            log("  ||h_u - h_0|| / ||h_0||; layer:        " + " ".join(f"{layer:5d}" for layer in range(n_layers) if layer % 4 == 3 or layer == 0))
            for key, tag in (("rel_text", "no image"), ("rel_filler", "filler text"), ("rel_blank", "blank image")):
                for g, vals in s.get(key, {}).items():
                    if any(v is not None for v in vals):
                        log(f"    {g:9s} {tag:12s}       " + " ".join(fmt(v) for k, v in enumerate(vals) if k % 4 == 3 or k == 0))
            for c in CONTEXTS:
                if f"forgetting_{c}" in s:
                    log(f"  {c} prompt: forgetting {s[f'forgetting_{c}']['first']:.2f} nats (first token), "
                        f"{s[f'forgetting_{c}']['answer']:.2f} nats (answer); top-1 unlearned {s[f'top1_{c}']['u']:.2f}, "
                        f"vanilla {s[f'top1_{c}']['0']:.2f}; n={s[f'n_facts_{c}']}")
        if "forget" in summary:
            plot(summary["forget"], f"{run_id} step {args.step}: where the suppression is carried (forget facts, "
                 f"n={summary['forget']['n_facts']})", out / "state_patch.png")
        log(f"wrote {out / 'summary.json'}" + (" and state_patch.png" if "forget" in summary else ""))


if __name__ == "__main__":
    main()
