"""UMU track: is the suppression a signal that transfers to other examples? Mean-difference vector at the last prompt
token (DEVIATIONS DEV-016).

State patching (DEV-015) found that from about layer 16 the state of the last prompt token alone decides whether the
answer is suppressed. That test moved a whole state between the two models on the SAME fact, so it could not tell a
shared "do not answer" signal from fact-specific damage. Here the change that unlearning makes to that state,
delta = h_u - h_0 at the last prompt token of the training form "USER: {name question}\nASSISTANT:", is averaged over
facts about OTHER people and moved to examples it was not computed from:
  add_mean          vanilla, held-out fact: h_0 + mean delta            is the answer suppressed?
  add_mean_normed   the same with mean delta rescaled to the average norm of a single delta
  add_own           h_0 + the fact's own delta (= theta_u's state; reference, the `induce` of the state patching)
  add_random        h_0 + a random vector with the norm of mean delta   (control)
  sub_mean          unlearned model, held-out fact: h_u - mean delta    does the answer come back?
  sub_own           h_u - own delta (= theta_0's state; reference, the `restore` of the state patching)
  sub_random        control
  blank_add_mean / blank_add_random   unlearned model, the same question behind a blank image, where the answer is
                    not suppressed: state + mean delta of the training form. Does the suppression appear?
"Other people": the forget profiles are split in two halves; a forget fact gets the mean over the half its person is
not in. Retain facts get the mean over all forget facts (different people by construction).

Readout: log p of the first answer token and top-1 (is the correct first token the most likely one?), plus the most
frequent predicted first tokens. add_*: share = (lp_0 - lp_patched) / (lp_0 - lp_u), 1 = as suppressed as the unlearned
model. sub_*: share = (lp_patched - lp_u) / (lp_0 - lp_u), 1 = the answer is fully back. blank_add_*: suppression added
to the blank-image prompt, as a share of the same text-form forgetting. Ratios of means, 95% bootstrap over profiles.

Reading (thresholds fixed in DEV-016 before the run; forget facts, layer 16):
  add_mean >= 0.5, sub_mean >= 0.5 and add_random <= 0.1   a signal shared across people carries the suppression
  add_mean <= 0.2 and sub_mean <= 0.2                      the change is specific to each fact
  otherwise                                                partly shared, no verdict
Also: cosine between each fact's delta and the mean of the other people, and the norm of that mean relative to a single
delta (descriptive).

All doubly-correct forget and retain facts; about 5 minutes on an L4, not resumable.

  python scripts/mech_umu_signal_transfer.py --config configs/llava/umu_npo_rt.yaml \
    --run-id umu_npo_rt_txt_lr5e-05_s0 --step 150
Writes results/llava/mech_umu/<run>/step<N>/signal_transfer/{summary.json,items.jsonl,signal_transfer.png,log.txt}.
"""
import collections
import contextlib
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np  # noqa: E402
import torch  # noqa: E402

from scripts.mech_umu_form_sweep import blank_image_bytes, facts_for  # noqa: E402
from vlmu.config import base_parser, resolve  # noqa: E402
from vlmu.mech import tools  # noqa: E402
from vlmu.mech.hooks import decoder_layers, patch_outputs, record_outputs  # noqa: E402
from vlmu.models import get_adapter  # noqa: E402
from vlmu.runs import free, model_dir, results_dir  # noqa: E402
from vlmu.umu import encode_teacher_forced, first_target, load_umu, umu_run_id  # noqa: E402
from vlmu.utils import Logger, atomic_write_json, read_json, tracked_run, write_jsonl  # noqa: E402

LAYERS = (8, 12, 14, 16, 18, 20, 24)
BLANK_LAYERS = (12, 16, 20)
PRIMARY = 16
ADD = ("add_mean", "add_mean_normed", "add_own", "add_random")
SUB = ("sub_mean", "sub_own", "sub_random")
BLANK = ("blank_add_mean", "blank_add_random")
SELF_TOL = 0.05
SHARED, SPECIFIC, RANDOM_MAX = 0.5, 0.2, 0.1      # DEV-016


def make_rows(fact, blank):
    q = fact["q_uni"]
    shared = {"pair_id": fact["pair_id"], "set": fact["set"], "profile": fact["profile"],
              "correct_text": fact["answer_text"]}
    return {"text": {**shared, "uid": f"{fact['pair_id']}/text", "prompt": f"USER: {q}\nASSISTANT:", "image_bytes": None},
            "blank": {**shared, "uid": f"{fact['pair_id']}/blank", "prompt": f"USER: <image>\n{q}\nASSISTANT:",
                      "image_bytes": blank}}


def encode(adapter, rows):
    batch = encode_teacher_forced(adapter, rows)
    return batch, first_target(tools.answer_targets(batch))


def readout(adapter, model, batch, t):
    """Per row: log p of the first answer token, whether it is the argmax, and the argmax token id."""
    at = tools.forward_logits(adapter, model, batch)[t["b"], t["pos"]].float()
    lp = at.log_softmax(-1).gather(-1, t["tok"][:, None]).squeeze(-1)
    arg = at.argmax(-1)
    return lp.cpu().tolist(), (arg == t["tok"]).cpu().tolist(), arg.cpu().tolist()


@torch.no_grad()
def record(adapter, model, batch, t, layers, vanilla):
    """The last prompt token's residual state at `layers` ([B, n_layers, D], float32 on the CPU) and the readout."""
    store = {}
    ctx = model.disable_adapter() if vanilla else contextlib.nullcontext()
    with ctx, record_outputs(model, "resid", layers, store, select=(t["b"], t["pos"])):
        out = readout(adapter, model, batch, t)
    return torch.stack([store[layer].float().cpu() for layer in layers], 1), out


@torch.no_grad()
def patched(adapter, model, batch, t, layer, state, vanilla):
    """Readout with the last prompt token's state at `layer` replaced by `state` ([B, D])."""
    b, p = t["b"], t["pos"]
    n_rows, n_tok = batch["input_ids"].shape
    cached = torch.zeros(n_rows, n_tok, state.shape[-1], device=b.device, dtype=adapter.dtype)
    cached[b, p] = state.to(device=b.device, dtype=adapter.dtype)
    ctx = model.disable_adapter() if vanilla else contextlib.nullcontext()
    with ctx, patch_outputs(model, "resid", {layer: cached}, (b, p, p)):
        return readout(adapter, model, batch, t)


def held_out_means(facts, delta):
    """{pair_id: mean delta over forget facts about other people} and each forget fact's fold. Forget profiles are
    split in two halves (alternating in sorted order); retain facts use all forget facts."""
    profiles = sorted({f["profile"] for f in facts if f["set"] == "forget"}, key=str)
    fold_of = {p: i % 2 for i, p in enumerate(profiles)}
    by_fold = {k: [delta[f["pair_id"]] for f in facts if f["set"] == "forget" and fold_of[f["profile"]] == k] for k in (0, 1)}
    if not by_fold[0] or not by_fold[1]:
        raise SystemExit("need forget facts from at least two people")
    mean = {k: torch.stack(v).mean(0) for k, v in by_fold.items()}
    mean_all = torch.stack(by_fold[0] + by_fold[1]).mean(0)
    vec, fold = {}, {}
    for f in facts:
        if f["set"] == "forget":
            fold[f["pair_id"]] = fold_of[f["profile"]]
            vec[f["pair_id"]] = mean[1 - fold_of[f["profile"]]]
        else:
            fold[f["pair_id"]] = None
            vec[f["pair_id"]] = mean_all
    return vec, fold


def share(recs, cond, k):
    """Effect of one condition at layer index k as a share of the text-form forgetting (ratio of means); None if
    there is no forgetting or the condition was not scored."""
    rows = [r for r in recs if cond in r["res"]]
    if not rows:
        return None
    gap = float(np.mean([r["lp_0"] - r["lp_u"] for r in rows]))
    if abs(gap) < 1e-8:
        return None
    if cond.startswith("add"):
        num = np.mean([r["lp_0"] - r["res"][cond]["lp"][k] for r in rows])
    elif cond.startswith("sub"):
        num = np.mean([r["res"][cond]["lp"][k] - r["lp_u"] for r in rows])
    else:
        num = np.mean([r["blank_lp_u"] - r["res"][cond]["lp"][k] for r in rows])
    return float(num / gap)


def bootstrap(recs, cond, k, n_boot=300):
    profiles = sorted({r["profile"] for r in recs}, key=str)
    by = {p: [r for r in recs if r["profile"] == p] for p in profiles}
    rng = np.random.default_rng(0)
    vals = []
    for _ in range(n_boot):
        v = share([r for p in rng.choice(profiles, len(profiles), replace=True) for r in by[p]], cond, k)
        if v is not None:
            vals.append(v)
    return [float(np.percentile(vals, q)) for q in (2.5, 97.5)] if vals else [None, None]


def common(tokens, n=4):
    return [[tok, cnt] for tok, cnt in collections.Counter(tokens).most_common(n)]


def summarize(recs_all):
    out = {}
    for set_name in ("forget", "retain"):
        recs = [r for r in recs_all if r["set"] == set_name]
        if not recs:
            continue
        s = {"n_facts": len(recs), "n_profiles": len({r["profile"] for r in recs}),
             "forgetting": float(np.mean([r["lp_0"] - r["lp_u"] for r in recs])),
             "top1": {"vanilla": float(np.mean([float(r["top1_0"]) for r in recs])),
                      "unlearned": float(np.mean([float(r["top1_u"]) for r in recs]))},
             "cos_own_vs_mean": [float(np.mean([r["cos"][k] for r in recs])) for k in range(len(LAYERS))],
             "norm_mean_over_single": [float(np.mean([r["norm_v"][k] for r in recs]) / np.mean([r["norm_delta"][k] for r in recs]))
                                       for k in range(len(LAYERS))],
             "first_tokens": {"vanilla": common([r["arg_0"] for r in recs]), "unlearned": common([r["arg_u"] for r in recs])}}
        for cond, layers in [(c, LAYERS) for c in ADD + SUB] + [(c, BLANK_LAYERS) for c in BLANK]:
            have = [r for r in recs if cond in r["res"]]
            if not have:
                continue
            s[cond] = {"layers": list(layers), "n_facts": len(have),
                       "share": [share(recs, cond, k) for k in range(len(layers))],
                       "ci95": [bootstrap(recs, cond, k) for k in range(len(layers))],
                       "top1": [float(np.mean([float(r["res"][cond]["top1"][k]) for r in have])) for k in range(len(layers))]}
            if PRIMARY in layers:
                s["first_tokens"][cond] = common([r["res"][cond]["arg"][layers.index(PRIMARY)] for r in have])
        have = [r for r in recs if "blank_lp_u" in r]
        if have:
            s["blank"] = {"n_facts": len(have),
                          "forgetting": float(np.mean([r["blank_lp_0"] - r["blank_lp_u"] for r in have])),
                          "top1": {"vanilla": float(np.mean([float(r["blank_top1_0"]) for r in have])),
                                   "unlearned": float(np.mean([float(r["blank_top1_u"]) for r in have]))}}
        out[set_name] = s
    return out


def fmt(v):
    return "    -" if v is None else f"{v:+5.2f}"


def verdict(summary):
    f = summary.get("forget")
    if not f or "add_mean" not in f:
        return ["no forget facts scored: no verdict"]
    k = LAYERS.index(PRIMARY)
    a, s_, r = f["add_mean"]["share"][k], f["sub_mean"]["share"][k], f["add_random"]["share"][k]
    head = f"layer {PRIMARY}, held-out people: add_mean {fmt(a)}, sub_mean {fmt(s_)}, add_random {fmt(r)}"
    if None in (a, s_, r):
        lines = [head + " -> no verdict"]
    elif a >= SHARED and s_ >= SHARED and r <= RANDOM_MAX:
        lines = [head + " -> a signal shared across people carries the suppression"]
    elif a <= SPECIFIC and s_ <= SPECIFIC:
        lines = [head + " -> the change is specific to each fact"]
    else:
        lines = [head + " -> partly shared, no verdict under DEV-016"]
    if "blank_add_mean" in f and PRIMARY in BLANK_LAYERS:
        kb = BLANK_LAYERS.index(PRIMARY)
        lines.append(f"blank-image prompt, layer {PRIMARY}: adding the signal moves {fmt(f['blank_add_mean']['share'][kb])} of the "
                     f"text-form forgetting (random vector {fmt(f['blank_add_random']['share'][kb])}); top-1 "
                     f"{f['blank']['top1']['unlearned']:.2f} -> {f['blank_add_mean']['top1'][kb]:.2f}")
    return lines


def plot(summary, title, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    INK2, MUTED, GRID, AXIS = "#52514e", "#898781", "#e1e0d9", "#c3c2b7"
    f = summary["forget"]
    num = lambda xs: [np.nan if v is None else v for v in xs]  # noqa: E731
    fig, axes = plt.subplots(1, 3, figsize=(13.5, 3.7), dpi=200)
    spec = {"mean": ("signal from other people", "#e34948", "-"), "mean_normed": ("the same, rescaled", "#e34948", ":"),
            "own": ("the fact's own change", "#111111", "--"), "random": ("random vector", "#898781", "-")}
    for ax, conds, ttl in ((axes[0], ADD, "Vanilla + signal: how much suppression appears"),
                           (axes[1], SUB, "Unlearned model - signal: how much of the answer comes back")):
        for cond in conds:
            if cond not in f:
                continue
            lab, color, ls = spec[cond.split("_", 1)[1]]
            ax.plot(LAYERS, num(f[cond]["share"]), color=color, ls=ls, lw=1.7, marker="o", ms=3, label=lab)
            ci = f[cond]["ci95"]
            if cond.endswith("_mean") and all(c[0] is not None for c in ci):
                ax.fill_between(LAYERS, [c[0] for c in ci], [c[1] for c in ci], color=color, alpha=0.13, lw=0)
            if cond.endswith("_mean") and "retain" in summary and cond in summary["retain"]:
                ax.plot(LAYERS, num(summary["retain"][cond]["share"]), color="#2a78d6", lw=1.3, marker="s", ms=3,
                        label="signal from forget people, on retain facts")
        for y in (0, 1):
            ax.axhline(y, color=AXIS, lw=0.9)
        ax.axvline(PRIMARY, color=AXIS, lw=0.6, ls=":")
        ax.set_title(ttl, loc="left", fontsize=8, color=INK2)
        ax.set_xlabel("decoder layer (state of the last prompt token)", color=INK2, fontsize=8)
        ax.legend(frameon=False, fontsize=6.5)
    axes[0].set_ylabel("share of the forgetting\n(log p of the first answer token)", color=INK2, fontsize=8)
    k = LAYERS.index(PRIMARY)
    bars = [("vanilla", f["top1"]["vanilla"], "#c3c2b7"), ("vanilla\n+ signal", f["add_mean"]["top1"][k], "#e34948"),
            ("vanilla\n+ random", f["add_random"]["top1"][k], "#898781"), ("unlearned", f["top1"]["unlearned"], "#c3c2b7"),
            ("unlearned\n- signal", f["sub_mean"]["top1"][k], "#2a78d6")]
    if "blank" in f and "blank_add_mean" in f and PRIMARY in BLANK_LAYERS:
        kb = BLANK_LAYERS.index(PRIMARY)
        bars += [("unlearned,\nblank image", f["blank"]["top1"]["unlearned"], "#c3c2b7"),
                 ("blank image\n+ signal", f["blank_add_mean"]["top1"][kb], "#e34948")]
    ax = axes[2]
    ax.bar(range(len(bars)), [b[1] for b in bars], color=[b[2] for b in bars], width=0.7)
    ax.set_xticks(range(len(bars)), [b[0] for b in bars], fontsize=6.5)
    ax.set_ylim(0, 1)
    ax.set_title(f"Correct first answer token is the most likely one\n(forget facts, signal applied at layer {PRIMARY})",
                 loc="left", fontsize=8, color=INK2)
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
    p.add_argument("--no-blank", action="store_true", help="skip the blank-image prompts")
    p.add_argument("--max-facts", type=int, default=0, help="0 = all doubly-correct facts")
    p.add_argument("--batch-txt", type=int, default=16)
    p.add_argument("--batch-img", type=int, default=4)
    args = p.parse_args()
    cfg = resolve(args)
    run_id = args.run_id or umu_run_id(cfg)
    run_dir = model_dir(cfg) / "umu_unlearn" / run_id
    ckpt = run_dir / "ckpt" / f"step{args.step}"
    if not (ckpt / "adapter_config.json").exists():
        raise SystemExit(f"no adapter at {ckpt}")
    out = results_dir(cfg) / "mech_umu" / run_id / f"step{args.step}" / "signal_transfer"
    log = Logger(out / "log.txt")
    with tracked_run(results_dir(cfg), out, f"mech_umu_signal_transfer_{run_id}_step{args.step}", "mech_umu_signal_transfer"):
        from peft import PeftModel
        dc = set(read_json(run_dir / "baseline_dc.json")["pair_ids"])
        dfs = load_umu(cfg["paths"]["root"], cfg["umu"]["n_retain_eval"], 0, log, cfg["umu"]["forget_split"])
        facts = sorted(facts_for(dfs, dc), key=lambda f: (f["set"], f["pair_id"]))
        if args.max_facts:
            facts = facts[:args.max_facts]
        adapter = get_adapter({**cfg["model"], "id": cfg["umu"]["ckpt"], "processor_id": cfg["model"]["id"]})
        model = PeftModel.from_pretrained(adapter.load_model(), str(ckpt), is_trainable=False)
        model.eval()
        n_layers = len(decoder_layers(model))
        if max(LAYERS) >= n_layers:
            raise SystemExit(f"the model has {n_layers} decoder layers; LAYERS = {LAYERS}")
        blank = blank_image_bytes()
        rows = {f["pair_id"]: make_rows(f, blank) for f in facts}
        tok = adapter.tok
        log(f"{run_id} step {args.step}: {sum(f['set'] == 'forget' for f in facts)} forget + "
            f"{sum(f['set'] == 'retain' for f in facts)} retain facts; layers {LAYERS}; blank-image layers "
            f"{() if args.no_blank else BLANK_LAYERS}")

        # pass 1: both models' state at the last prompt token of the training form
        order = sorted(facts, key=lambda f: len(rows[f["pair_id"]]["text"]["prompt"]))
        batches, state, rec = [], {}, {}
        for i in range(0, len(order), args.batch_txt):
            chunk = order[i:i + args.batch_txt]
            batch, t = encode(adapter, [rows[f["pair_id"]]["text"] for f in chunk])
            h_u, (lp_u, top_u, arg_u) = record(adapter, model, batch, t, LAYERS, vanilla=False)
            h_0, (lp_0, top_0, arg_0) = record(adapter, model, batch, t, LAYERS, vanilla=True)
            if i == 0:
                k = LAYERS.index(PRIMARY)
                same = patched(adapter, model, batch, t, PRIMARY, h_u[:, k], vanilla=False)[0]
                err = max(abs(a - b) for a, b in zip(same, lp_u))
                log(f"  self-patch check: max |change in log p| = {err:.2e} nats (tolerance {SELF_TOL})")
                if err > SELF_TOL:
                    raise RuntimeError(f"a state patched back in changed log p by {err:.3f} nats: the patch is not exact")
            batches.append((chunk, batch, t))
            for j, f in enumerate(chunk):
                pid = f["pair_id"]
                state[pid] = {"u": h_u[j], "0": h_0[j]}
                rec[pid] = {"pair_id": pid, "set": f["set"], "profile": f["profile"],
                            "lp_u": round(lp_u[j], 4), "lp_0": round(lp_0[j], 4), "top1_u": bool(top_u[j]), "top1_0": bool(top_0[j]),
                            "arg_u": tok.convert_ids_to_tokens(arg_u[j]), "arg_0": tok.convert_ids_to_tokens(arg_0[j]), "res": {}}

        # the signal: mean change over other people's facts, its rescaled version and a random control
        delta = {pid: s["u"] - s["0"] for pid, s in state.items()}
        vec, fold = held_out_means(facts, delta)
        mean_norm = torch.stack([delta[f["pair_id"]] for f in facts if f["set"] == "forget"]).norm(dim=-1).mean(0)   # [n_layers]
        gen = torch.Generator().manual_seed(0)
        normed, rand = {}, {}
        for f in facts:
            pid = f["pair_id"]
            v = vec[pid]
            normed[pid] = v / v.norm(dim=-1, keepdim=True).clamp_min(1e-8) * mean_norm[:, None]
            r = torch.randn(v.shape, generator=gen)
            rand[pid] = r / r.norm(dim=-1, keepdim=True) * v.norm(dim=-1, keepdim=True)
            rec[pid].update(fold=fold[pid],
                            cos=[round(float(c), 4) for c in torch.nn.functional.cosine_similarity(delta[pid], v, dim=-1)],
                            norm_delta=[round(float(x), 3) for x in delta[pid].norm(dim=-1)],
                            norm_v=[round(float(x), 3) for x in v.norm(dim=-1)])

        # pass 2: move the signal
        def run(conds, chunk, batch, t, layers, base_of, vanilla_of):
            for cond in conds:
                for pid in (f["pair_id"] for f in chunk):
                    rec[pid]["res"][cond] = {"lp": [], "top1": [], "arg": []}
            for layer in layers:
                k = LAYERS.index(layer)
                for cond in conds:
                    kind = cond.split("_", 1)[1] if not cond.startswith("blank") else cond.split("_", 2)[2]
                    sign = -1.0 if cond.startswith("sub") else 1.0
                    src = {"mean": vec, "mean_normed": normed, "own": delta, "random": rand}[kind]
                    st = torch.stack([base_of(f["pair_id"], cond)[k] + sign * src[f["pair_id"]][k] for f in chunk])
                    lp, top, arg = patched(adapter, model, batch, t, layer, st, vanilla=vanilla_of(cond))
                    for j, f in enumerate(chunk):
                        r = rec[f["pair_id"]]["res"][cond]
                        r["lp"].append(round(lp[j], 4))
                        r["top1"].append(bool(top[j]))
                        r["arg"].append(tok.convert_ids_to_tokens(arg[j]))

        for n, (chunk, batch, t) in enumerate(batches):
            run(ADD + SUB, chunk, batch, t, LAYERS,
                base_of=lambda pid, cond: state[pid]["0" if cond.startswith("add") else "u"],
                vanilla_of=lambda cond: cond.startswith("add"))
            log(f"  training form: batch {n + 1}/{len(batches)} done")
        batches = None

        if not args.no_blank:
            forget = [f for f in facts if f["set"] == "forget"]
            for i in range(0, len(forget), args.batch_img):
                chunk = forget[i:i + args.batch_img]
                batch, t = encode(adapter, [rows[f["pair_id"]]["blank"] for f in chunk])
                h_b, (lp_u, top_u, _) = record(adapter, model, batch, t, LAYERS, vanilla=False)
                lp_0, top_0, _ = readout_vanilla(adapter, model, batch, t)
                blank_state = {f["pair_id"]: h_b[j] for j, f in enumerate(chunk)}
                for j, f in enumerate(chunk):
                    rec[f["pair_id"]].update(blank_lp_u=round(lp_u[j], 4), blank_lp_0=round(lp_0[j], 4),
                                             blank_top1_u=bool(top_u[j]), blank_top1_0=bool(top_0[j]))
                run(BLANK, chunk, batch, t, BLANK_LAYERS, base_of=lambda pid, cond: blank_state[pid],
                    vanilla_of=lambda cond: False)
                if (i // args.batch_img + 1) % 5 == 0 or i + args.batch_img >= len(forget):
                    log(f"  blank-image prompts: {min(i + args.batch_img, len(forget))}/{len(forget)} facts done")
        model = None
        free()

        recs = [rec[f["pair_id"]] for f in facts]
        write_jsonl(out / "items.jsonl", recs)
        summary = summarize(recs)
        lines = verdict(summary)
        atomic_write_json(out / "summary.json", {
            "run_id": run_id, "step": args.step, "layers": list(LAYERS), "blank_layers": list(BLANK_LAYERS), "primary_layer": PRIMARY,
            "readout": "log p of the first answer token; top1 = the correct first answer token is the argmax",
            "thresholds": {"shared_min": SHARED, "specific_max": SPECIFIC, "random_max": RANDOM_MAX, "self_patch_tol_nats": SELF_TOL},
            "verdict": lines, "summary": summary})
        for set_name, s in summary.items():
            log(f"{set_name}: n={s['n_facts']} facts / {s['n_profiles']} profiles; forgetting {s['forgetting']:.2f} nats; top-1 vanilla "
                f"{s['top1']['vanilla']:.2f}, unlearned {s['top1']['unlearned']:.2f}")
            log("  layer                        " + " ".join(f"{layer:5d}" for layer in LAYERS))
            log("  cos(own change, signal)      " + " ".join(fmt(v) for v in s["cos_own_vs_mean"]))
            log("  ||signal|| / ||own change||  " + " ".join(fmt(v) for v in s["norm_mean_over_single"]))
            for cond in ADD + SUB:
                if cond in s:
                    log(f"  {cond:16s} share       " + " ".join(fmt(v) for v in s[cond]["share"]))
                    log(f"  {'':16s} top-1       " + " ".join(fmt(v) for v in s[cond]["top1"]))
            for cond in BLANK:
                if cond in s:
                    log(f"  {cond:16s} layers {s[cond]['layers']}: share " + " ".join(fmt(v) for v in s[cond]["share"])
                        + " | top-1 " + " ".join(fmt(v) for v in s[cond]["top1"])
                        + f" (unpatched {s['blank']['top1']['unlearned']:.2f})")
            log(f"  most frequent first tokens: {s['first_tokens']}")
        for line in lines:
            log(line)
        if "forget" in summary and "add_mean" in summary["forget"]:
            plot(summary, f"{run_id} step {args.step}: moving the unlearning signal to other examples", out / "signal_transfer.png")
        log(f"wrote {out / 'summary.json'}")


@torch.no_grad()
def readout_vanilla(adapter, model, batch, t):
    with model.disable_adapter():
        return readout(adapter, model, batch, t)


if __name__ == "__main__":
    main()
