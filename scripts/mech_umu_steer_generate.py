"""UMU track: does the unlearning signal switch the response style? Steering during generation (DEVIATIONS DEV-017).

DEV-016 moved the mean change of the last prompt token's state (unlearned - vanilla, averaged over facts about other
people) between the two models and read only the first answer token. Reading the generations of the unlearned model
suggested what the change does: the vanilla replies with the bare answer, the unlearned model writes a long sentence
about the person. This script applies the same vector while the model GENERATES, so the claim can be checked directly.

Prompt: the training form "USER: {name question}\nASSISTANT:" (left-padded, no answer). At one decoder layer (16) the
vector is added to the output state of the last prompt token during the first forward pass of generation; the tokens
generated afterwards read that state through attention. Conditions (greedy, 32 new tokens):
  vanilla                                   unlearned
  vanilla + mean vector                     unlearned - mean vector
  vanilla + mean vector, rescaled           unlearned - random vector
  vanilla + random vector
  unlearned, blank image before the question          (answers normally survive there)
  the same + mean vector / + random vector
The mean vector for a fact comes from the other half of the forget people (retain facts: all forget people), as in
DEV-016. Random vectors have the norm of the vector they are compared with.

Readout per response: number of words; style = short (fewer than 6 words) / bio (longer and opening with the person's
first name) / refusal (longer and opening with "I'm", "I's", "I don't", ...) / other; keyword hit of the correct answer.

Reading (thresholds fixed in DEV-017 before the run; forget facts):
  switch on    vanilla + rescaled mean vector: >= 50% long responses, and vanilla + random <= 10%
  switch off   unlearned - mean vector: >= 50% short responses, and unlearned - random <= 25%
A direction passes, fails (< 20% long / < 25% short) or is partial.

  python scripts/mech_umu_steer_generate.py --config configs/llava/umu_npo_rt.yaml \
    --run-id umu_npo_rt_txt_lr5e-05_s0 --step 150
Writes results/llava/mech_umu/<run>/step<N>/steer_generate/{summary.json,items.jsonl,steer_generate.png,log.txt}.
About 7 minutes on an L4; not resumable.
"""
import contextlib
import io
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np  # noqa: E402
import torch  # noqa: E402
from PIL import Image  # noqa: E402

from scripts.mech_umu_form_sweep import blank_image_bytes, facts_for  # noqa: E402
from scripts.mech_umu_signal_transfer import held_out_means  # noqa: E402
from vlmu.config import base_parser, resolve  # noqa: E402
from vlmu.evaluate import keyword_hit  # noqa: E402
from vlmu.mech import tools  # noqa: E402
from vlmu.mech.hooks import decoder_layers, record_outputs  # noqa: E402
from vlmu.models import get_adapter  # noqa: E402
from vlmu.runs import free, model_dir, results_dir  # noqa: E402
from vlmu.umu import load_umu, umu_run_id  # noqa: E402
from vlmu.utils import Logger, atomic_write_json, read_json, tracked_run, write_jsonl  # noqa: E402

LONG = 6                      # words; a response with fewer is "short"
STYLES = ("short", "bio", "refusal", "other")
REFUSAL = re.compile(r"^(i'm|i am|i's|i don't|i do not|i cannot|i can't|it's|it is|as an ai|sorry)\b", re.I)
TEXT = ("vanilla", "vanilla+mean", "vanilla+mean_normed", "vanilla+random", "unlearned", "unlearned-mean", "unlearned-random")
BLANK = ("blank_unlearned", "blank_unlearned+mean", "blank_unlearned+random")
ON_MIN, ON_FAIL, ON_RANDOM_MAX, OFF_MIN, OFF_FAIL, OFF_RANDOM_MAX = 0.5, 0.2, 0.1, 0.5, 0.25, 0.25   # DEV-017


def style(response, name):
    text = (response or "").strip()
    if len(text.split()) < LONG:
        return "short"
    if REFUSAL.match(text):
        return "refusal"
    first = (name or "").split()[0].lower() if name else ""
    return "bio" if first and text.lower().startswith(first) else "other"


def encode(adapter, prompts, image):
    """Left-padded prompt-only batch: the last prompt token is at position -1 in every row."""
    imgs = [Image.open(io.BytesIO(image)).convert("RGB") for _ in prompts] if image is not None else None
    return adapter.to_device(adapter._process(prompts, imgs, "left"))


@contextlib.contextmanager
def steer(module, vec, n_tok):
    """Add vec ([B, D]) to the output state of the last prompt token while the whole prompt is processed (the first
    forward pass of generation). Later steps process one new token and are left alone; they read the changed state
    through the cached keys and values of the layers above."""
    def hook(_module, _inputs, out):
        h = out[0] if isinstance(out, tuple) else out
        if h.shape[1] != n_tok:
            return out
        h = h.clone()
        h[:, -1] = h[:, -1] + vec.to(device=h.device, dtype=h.dtype)
        return (h,) + tuple(out[1:]) if isinstance(out, tuple) else h
    handle = module.register_forward_hook(hook)
    try:
        yield
    finally:
        handle.remove()


@torch.no_grad()
def last_state(adapter, model, batch, layer, vanilla):
    """[B, D] float32 on the CPU: output state of decoder layer `layer` at the last prompt token."""
    n_rows, n_tok = batch["input_ids"].shape
    dev = batch["input_ids"].device
    store = {}
    ctx = model.disable_adapter() if vanilla else contextlib.nullcontext()
    with ctx, record_outputs(model, "resid", [layer], store,
                             select=(torch.arange(n_rows, device=dev), torch.full((n_rows,), n_tok - 1, device=dev))):
        tools.forward_logits(adapter, model, batch)
    return store[layer].float().cpu()


@torch.no_grad()
def generate(adapter, model, module, batch, vec, vanilla, max_new):
    ctx = model.disable_adapter() if vanilla else contextlib.nullcontext()
    hook = steer(module, vec, batch["input_ids"].shape[1]) if vec is not None else contextlib.nullcontext()
    with ctx, hook:
        return [s.strip() for s in adapter.generate(model, batch, max_new)]


def summarize(recs, conds):
    out = {}
    for set_name in ("forget", "retain"):
        rows = [r for r in recs if r["set"] == set_name]
        if not rows:
            continue
        out[set_name] = {"n_facts": len(rows)}
        for cond in conds:
            have = [r["gen"][cond] for r in rows if cond in r["gen"]]
            if not have:
                continue
            counts = {s: sum(g["style"] == s for g in have) for s in STYLES}
            out[set_name][cond] = {"n": len(have), "styles": counts,
                                   "long_share": 1 - counts["short"] / len(have), "short_share": counts["short"] / len(have),
                                   "words_mean": float(np.mean([g["words"] for g in have])),
                                   "keyword_hit": float(np.mean([float(g["hit"]) for g in have])),
                                   "same_as_vanilla": float(np.mean([float(g["same_as_vanilla"]) for g in have]))}
    return out


def verdict(summary):
    f = summary.get("forget", {})
    if "vanilla+mean_normed" not in f:
        return ["no forget facts scored: no verdict"]
    on, ctrl = f["vanilla+mean_normed"]["long_share"], f["vanilla+random"]["long_share"]
    off, ctrl2 = f["unlearned-mean"]["short_share"], f["unlearned-random"]["short_share"]
    word = lambda v, lo, hi, ok: "PASS" if v >= hi and ok else ("FAIL" if v < lo else "partial")  # noqa: E731
    return [f"switch on: vanilla + rescaled mean vector gives {on:.2f} long responses (vanilla {f['vanilla']['long_share']:.2f}, "
            f"+ random {ctrl:.2f}) -> {word(on, ON_FAIL, ON_MIN, ctrl <= ON_RANDOM_MAX)}",
            f"switch off: unlearned - mean vector gives {off:.2f} short responses (unlearned {f['unlearned']['short_share']:.2f}, "
            f"- random {ctrl2:.2f}) -> {word(off, OFF_FAIL, OFF_MIN, ctrl2 <= OFF_RANDOM_MAX)}"]


def plot(summary, conds, title, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    INK2, MUTED, GRID = "#52514e", "#898781", "#e1e0d9"
    colors = {"short": "#2a78d6", "bio": "#e34948", "refusal": "#eda100", "other": "#898781"}
    labels = {"short": "short answer", "bio": "long, opens with the person's name", "refusal": "long, refusal", "other": "long, other"}
    f = summary["forget"]
    shown = [c for c in conds if c in f]
    fig, ax = plt.subplots(figsize=(8.8, 0.42 * len(shown) + 1.3), dpi=200)
    left = np.zeros(len(shown))
    for s in STYLES:
        vals = np.array([f[c]["styles"][s] / f[c]["n"] for c in shown])
        ax.barh(range(len(shown)), vals, left=left, color=colors[s], height=0.68, label=labels[s])
        left += vals
    for i, c in enumerate(shown):
        ax.text(1.01, i, f"answer present {f[c]['keyword_hit']:.2f}", va="center", fontsize=7, color=INK2)
    ax.set_yticks(range(len(shown)), [c.replace("_", " ") for c in shown], fontsize=8)
    ax.invert_yaxis()
    ax.set_xlim(0, 1)
    ax.set_xlabel("share of responses (forget facts)", color=INK2, fontsize=8)
    ax.legend(frameon=False, fontsize=7, ncol=4, loc="lower left", bbox_to_anchor=(0, 1.0))
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.grid(axis="x", color=GRID, lw=0.6)
    ax.set_axisbelow(True)
    ax.tick_params(colors=MUTED, labelsize=8)
    fig.suptitle(title, x=0.01, y=1.04, ha="left", fontsize=9, color=INK2)
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def main():
    p = base_parser(__doc__)
    p.add_argument("--run-id", default=None)
    p.add_argument("--step", type=int, required=True)
    p.add_argument("--layer", type=int, default=16)
    p.add_argument("--max-new-tokens", type=int, default=32)
    p.add_argument("--no-blank", action="store_true", help="skip the blank-image prompts")
    p.add_argument("--max-facts", type=int, default=0, help="0 = all doubly-correct facts")
    p.add_argument("--batch-txt", type=int, default=16)
    p.add_argument("--batch-img", type=int, default=8)
    args = p.parse_args()
    cfg = resolve(args)
    run_id = args.run_id or umu_run_id(cfg)
    run_dir = model_dir(cfg) / "umu_unlearn" / run_id
    ckpt = run_dir / "ckpt" / f"step{args.step}"
    if not (ckpt / "adapter_config.json").exists():
        raise SystemExit(f"no adapter at {ckpt}")
    out = results_dir(cfg) / "mech_umu" / run_id / f"step{args.step}" / "steer_generate"
    log = Logger(out / "log.txt")
    with tracked_run(results_dir(cfg), out, f"mech_umu_steer_generate_{run_id}_step{args.step}", "mech_umu_steer_generate"):
        from peft import PeftModel
        dc = set(read_json(run_dir / "baseline_dc.json")["pair_ids"])
        dfs = load_umu(cfg["paths"]["root"], cfg["umu"]["n_retain_eval"], 0, log, cfg["umu"]["forget_split"])
        facts = sorted(facts_for(dfs, dc), key=lambda f: (f["set"], f["pair_id"]))
        if args.max_facts:
            facts = facts[:args.max_facts]
        adapter = get_adapter({**cfg["model"], "id": cfg["umu"]["ckpt"], "processor_id": cfg["model"]["id"]})
        model = PeftModel.from_pretrained(adapter.load_model(), str(ckpt), is_trainable=False)
        model.eval()
        layers = decoder_layers(model)
        if not 0 <= args.layer < len(layers):
            raise SystemExit(f"--layer must be in 0..{len(layers) - 1}")
        module = layers[args.layer][1]
        blank = blank_image_bytes()
        text_prompt = {f["pair_id"]: f"USER: {f['q_uni']}\nASSISTANT:" for f in facts}
        blank_prompt = {f["pair_id"]: f"USER: <image>\n{f['q_uni']}\nASSISTANT:" for f in facts}
        log(f"{run_id} step {args.step}: {sum(f['set'] == 'forget' for f in facts)} forget + {sum(f['set'] == 'retain' for f in facts)} "
            f"retain facts; vector at layer {args.layer}, last prompt token; {args.max_new_tokens} new tokens")

        # the vector: mean change of the last prompt token's state over other people's facts
        order = sorted(facts, key=lambda f: len(text_prompt[f["pair_id"]]))
        batches, delta = [], {}
        for i in range(0, len(order), args.batch_txt):
            chunk = order[i:i + args.batch_txt]
            batch = encode(adapter, [text_prompt[f["pair_id"]] for f in chunk], None)
            d = last_state(adapter, model, batch, args.layer, vanilla=False) - last_state(adapter, model, batch, args.layer, vanilla=True)
            batches.append((chunk, batch))
            delta.update({f["pair_id"]: d[j][None] for j, f in enumerate(chunk)})          # [1, D], as held_out_means expects
        mean, _ = held_out_means(facts, delta)
        mean = {pid: v[0] for pid, v in mean.items()}
        single_norm = torch.stack([delta[f["pair_id"]][0] for f in facts if f["set"] == "forget"]).norm(dim=-1).mean()
        gen = torch.Generator().manual_seed(0)
        unit = {}
        for f in facts:
            r = torch.randn(mean[f["pair_id"]].shape, generator=gen)
            unit[f["pair_id"]] = r / r.norm()
        normed = {pid: v / v.norm().clamp_min(1e-8) * single_norm for pid, v in mean.items()}
        log(f"  norm of the mean vector {float(torch.stack(list(mean.values())).norm(dim=-1).mean()):.2f}, of a single change "
            f"{float(single_norm):.2f}")

        def vectors(cond, chunk):
            """[B, D] to add for this condition, or None for an unsteered run."""
            kind = cond.split("+")[-1] if "+" in cond else (cond.split("-")[-1] if "-" in cond else None)
            if kind is None:
                return None
            sign = -1.0 if "-" in cond else 1.0
            if kind == "random":      # the norm of the vector it is compared with
                ref = normed if cond == "vanilla+random" else mean
                return torch.stack([sign * unit[f["pair_id"]] * ref[f["pair_id"]].norm() for f in chunk])
            src = normed if kind == "mean_normed" else mean
            return torch.stack([sign * src[f["pair_id"]] for f in chunk])

        rec = {f["pair_id"]: {"pair_id": f["pair_id"], "set": f["set"], "profile": f["profile"], "name": f["name"],
                              "answer": f["answer_text"], "gen": {}} for f in facts}

        def run(conds, chunk, batch):
            for cond in conds:
                texts = generate(adapter, model, module, batch, vectors(cond, chunk), cond.startswith("vanilla"),
                                 args.max_new_tokens)
                for f, text in zip(chunk, texts):
                    r = rec[f["pair_id"]]
                    base = r["gen"].get("vanilla", {}).get("response")
                    r["gen"][cond] = {"response": text, "words": len(text.split()), "style": style(text, f["name"]),
                                      "hit": bool(keyword_hit(f["answer_text"], text)),
                                      "same_as_vanilla": base is not None and text.strip().rstrip(".") == base.strip().rstrip(".")}

        for n, (chunk, batch) in enumerate(batches):
            if n == 0:   # a zero vector must not change the generation
                plain = generate(adapter, model, module, batch, None, False, args.max_new_tokens)
                zero = generate(adapter, model, module, batch, torch.zeros(len(chunk), mean[chunk[0]["pair_id"]].shape[-1]), False,
                                args.max_new_tokens)
                same = sum(a == b for a, b in zip(plain, zero))
                log(f"  zero-vector check: {same}/{len(chunk)} generations identical to the unsteered run")
                if same != len(chunk):
                    raise RuntimeError("steering with a zero vector changed the generations: the hook is not neutral")
            run(TEXT, chunk, batch)
            log(f"  training form: batch {n + 1}/{len(batches)} done")
        batches = None

        conds = list(TEXT)
        if not args.no_blank:
            forget = [f for f in order if f["set"] == "forget"]
            for i in range(0, len(forget), args.batch_img):
                chunk = forget[i:i + args.batch_img]
                run(BLANK, chunk, encode(adapter, [blank_prompt[f["pair_id"]] for f in chunk], blank))
                log(f"  blank-image prompts: {min(i + args.batch_img, len(forget))}/{len(forget)} facts done")
            conds += list(BLANK)
        model = None
        free()

        recs = [rec[f["pair_id"]] for f in facts]
        write_jsonl(out / "items.jsonl", recs)
        summary = summarize(recs, conds)
        lines = verdict(summary)
        atomic_write_json(out / "summary.json", {
            "run_id": run_id, "step": args.step, "layer": args.layer, "max_new_tokens": args.max_new_tokens, "conditions": conds,
            "styles": {"short": f"fewer than {LONG} words", "bio": "longer, opens with the person's first name",
                       "refusal": "longer, opens with I'm / I's / I don't / It's ...", "other": "longer, anything else"},
            "thresholds": {"on_min_long": ON_MIN, "on_fail_long": ON_FAIL, "on_random_max_long": ON_RANDOM_MAX,
                           "off_min_short": OFF_MIN, "off_fail_short": OFF_FAIL, "off_random_max_short": OFF_RANDOM_MAX},
            "verdict": lines, "summary": summary})
        for set_name, s in summary.items():
            log(f"{set_name}: n={s['n_facts']}")
            log(f"  {'condition':28s} {'short':>6s} {'bio':>6s} {'refus':>6s} {'other':>6s} {'words':>6s} {'answer present':>15s} {'= vanilla':>10s}")
            for cond in conds:
                if cond in s:
                    v = s[cond]
                    log(f"  {cond:28s} " + " ".join(f"{v['styles'][k] / v['n']:6.2f}" for k in STYLES)
                        + f" {v['words_mean']:6.1f} {v['keyword_hit']:15.2f} {v['same_as_vanilla']:10.2f}")
        log("examples (first 6 forget facts):")
        for r in [r for r in recs if r["set"] == "forget"][:6]:
            log(f"  [{r['name']}] answer: {r['answer']!r}")
            for cond in conds:
                if cond in r["gen"]:
                    log(f"    {cond:28s} {r['gen'][cond]['response'][:110]!r}")
        for line in lines:
            log(line)
        if "forget" in summary:
            plot(summary, conds, f"{run_id} step {args.step}: response style when the unlearning vector is added or removed "
                 f"(layer {args.layer})", out / "steer_generate.png")
        log(f"wrote {out / 'summary.json'}")


if __name__ == "__main__":
    main()
