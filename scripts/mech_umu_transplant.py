"""UMU track: does an image stop the unlearning LoRA from writing its suppression, or is the suppression written and
then not used? LoRA-contribution transplant between two prompts of the same fact (DEVIATIONS DEV-014).

  text    "USER: {name question}\nASSISTANT:"            the training form; the answer is suppressed
  blank   "USER: <image>\n{name question}\nASSISTANT:"   a grey image added; the form sweep found R = 0.88

Both prompts end in the same tokens (question, "ASSISTANT:", teacher-forced answer). Under theta_u each LoRA module's
contribution d(x) = scaling * B(A(x)) is recorded at those shared tokens in both prompts. The LoRA weights are the same
in the two runs, so the two recordings differ only through the hidden states the LoRA reads. Conditions, per layer band
(0-19: where layer masking found the effect; 20-31: control):
  text<-blank            text prompt, but at the shared tokens the LoRA writes what it wrote in the blank run
  blank<-text            blank prompt, but at the shared tokens the LoRA writes what it wrote in the text run
  text_off / blank_off   LoRA switched off at the shared tokens (reference)

Readout: teacher-forced log p of the answer text (and of its first token) under theta_u and adapter-off theta_0. With
S = log p_0 - log p (suppression of a prompt relative to its own vanilla) and gap = S_text - S_blank under theta_u, each
condition reports the moved fraction M = change of S in the target prompt towards the other prompt's level / gap:
  text<-blank   M = 1: the text prompt is as little suppressed as the blank prompt;  M = 0: unchanged
  blank<-text   M = 1: the blank prompt is as suppressed as the text prompt;         M = 0: unchanged
Ratio of means over facts, 95% bootstrap over profiles.

Reading (thresholds fixed in DEV-014 before the run; forget facts, answer log p, layers 0-19):
  M >= 0.5 in both directions   the image changes what the LoRA writes: the suppression is not written
  M <= 0.2 in both directions   the LoRA writes the same thing in effect; with an image the model does not use it
  otherwise                     mixed, no verdict
Also per layer (descriptive): cosine and norm ratio between the two recordings at the shared prompt tokens.

Default: the frozen doubly-correct forget facts (65 on run 2), bands 0-19 and 20-31, one fact per forward pass, 12
forward passes per fact. Opt-in: --with-retain (retain facts too), --bands 0-19,20-31,all. Resumable per fact. For the
first facts, transplanting a prompt's own contributions must reproduce the unhooked run (logged; aborts otherwise).

  python scripts/mech_umu_transplant.py --config configs/llava/umu_npo_rt.yaml \
    --run-id umu_npo_rt_txt_lr5e-05_s0 --step 150
Writes results/llava/mech_umu/<run>/step<N>/transplant/{summary.json,items.jsonl,transplant.png,log.txt}.
"""
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np  # noqa: E402
import torch  # noqa: E402

from scripts.mech_umu_form_sweep import blank_image_bytes, facts_for  # noqa: E402
from vlmu.config import base_parser, resolve  # noqa: E402
from vlmu.mech import tools  # noqa: E402
from vlmu.mech.hooks import (decoder_layers, layer_index, lora_modules, mask_contributions, record_contributions,
                             transplant_contributions)  # noqa: E402
from vlmu.models import get_adapter  # noqa: E402
from vlmu.runs import free, model_dir, results_dir  # noqa: E402
from vlmu.umu import encode_teacher_forced, load_umu, umu_run_id  # noqa: E402
from vlmu.utils import Logger, append_jsonl, atomic_write_json, read_json, read_jsonl, tracked_run  # noqa: E402

BANDS = {"0-19": (0, 20), "20-31": (20, 32), "all": (0, 32)}
ROUTES = ("text", "blank")
OTHER = {"text": "blank", "blank": "text"}
READOUTS = ("answer", "first")
SELF_TOL = 0.05          # nats; a prompt's own contributions transplanted back must reproduce the unhooked run
TRIGGER, OVERRIDE, CONTROL, BYPASS = 0.5, 0.2, 0.1, 0.5   # DEV-014


def routes(fact, blank):
    q = fact["q_uni"]
    shared = {"pair_id": fact["pair_id"], "set": fact["set"], "profile": fact["profile"],
              "correct_text": fact["answer_text"]}
    return {"text": {**shared, "uid": f"{fact['pair_id']}/text", "prompt": f"USER: {q}\nASSISTANT:",
                     "image_bytes": None},
            "blank": {**shared, "uid": f"{fact['pair_id']}/blank", "prompt": f"USER: <image>\n{q}\nASSISTANT:",
                      "image_bytes": blank}}


def align(batches, targets):
    """End-aligned shared tokens of the two prompts: ({route: positions}, index of the last prompt token within
    them, ids of the text-prompt tokens that are in neither the shared suffix nor the identical prefix). Prefix tokens
    have the same hidden states in both runs (causal attention), so only the returned leftover tokens keep
    contributions that are specific to the text run."""
    ids = {k: batches[k]["input_ids"][0].tolist() for k in ROUTES}
    n = 0
    while n < min(len(v) for v in ids.values()) and ids["text"][-1 - n] == ids["blank"][-1 - n]:
        n += 1
    pre = 0
    while pre < len(ids["text"]) - n and ids["text"][pre] == ids["blank"][pre]:
        pre += 1
    dev = batches["text"]["input_ids"].device
    pos = {k: torch.arange(len(ids[k]) - n, len(ids[k]), device=dev) for k in ROUTES}
    last = {k: int(targets[k]["pos"][0]) - (len(ids[k]) - n) for k in ROUTES}
    if last["text"] != last["blank"] or last["text"] < 1:
        raise ValueError(f"shared suffix ({n} tokens) does not cover the end of the prompt: {last}")
    return pos, last["text"], ids["text"][pre:len(ids["text"]) - n]


def readout(adapter, model, batch, t):
    logits = tools.forward_logits(adapter, model, batch)
    lp = tools.target_logprob(logits[t["b"], t["pos"]], t["tok"])
    return {"first": round(float(lp[0]), 4), "answer": round(float(lp.sum()), 4)}


def similarity(store, names_by_layer, k_last):
    """Per layer, averaged over its LoRA modules: cos(d_text, d_blank) at the shared prompt tokens and at the last
    prompt token, and ||d_blank|| / ||d_text|| at the shared prompt tokens."""
    out = {"cos_prompt": [], "cos_last": [], "norm_ratio_prompt": []}
    for names in names_by_layer:
        if not names:
            for v in out.values():
                v.append(None)
            continue
        cos, ratio = [], []
        for n in names:
            a = store["text"][n][0, :k_last + 1].float()
            b = store["blank"][n][0, :k_last + 1].float()
            cos.append(torch.nn.functional.cosine_similarity(a, b, dim=-1))
            ratio.append(b.norm(dim=-1) / a.norm(dim=-1).clamp_min(1e-8))
        cos, ratio = torch.stack(cos), torch.stack(ratio)          # [modules, shared prompt tokens]
        out["cos_prompt"].append(round(float(cos.mean()), 4))
        out["cos_last"].append(round(float(cos[:, -1].mean()), 4))
        out["norm_ratio_prompt"].append(round(float(ratio.mean()), 4))
    return out


@torch.no_grad()
def evaluate(adapter, model, fact, blank, names, band_names, names_by_layer, check):
    """One fact: all conditions. Returns (record, max self-transplant error or None, tokens for the log)."""
    rows = routes(fact, blank)
    batches = {k: encode_teacher_forced(adapter, [rows[k]]) for k in ROUTES}
    targets = {k: tools.answer_targets(batches[k]) for k in ROUTES}
    pos, k_last, leftover = align(batches, targets)
    lp, store = {}, {k: {} for k in ROUTES}
    for k in ROUTES:
        with record_contributions(model, names, store[k], pos[k]):
            lp[f"{k}/u"] = readout(adapter, model, batches[k], targets[k])
        with model.disable_adapter():
            lp[f"{k}/0"] = readout(adapter, model, batches[k], targets[k])
    worst = None
    if check:
        worst = 0.0
        for k in ROUTES:
            with transplant_contributions(model, names, store[k], pos[k]):
                same = readout(adapter, model, batches[k], targets[k])
            worst = max([worst] + [abs(same[r] - lp[f"{k}/u"][r]) for r in READOUTS])
    for band, bn in band_names.items():
        for k in ROUTES:
            with transplant_contributions(model, bn, store[OTHER[k]], pos[k]):
                lp[f"{k}<-{OTHER[k]}/{band}"] = readout(adapter, model, batches[k], targets[k])
            keep = torch.ones_like(batches[k]["input_ids"], dtype=torch.float32)
            keep[:, pos[k]] = 0
            with mask_contributions(model, bn, keep):
                lp[f"{k}_off/{band}"] = readout(adapter, model, batches[k], targets[k])
    shared = batches["text"]["input_ids"][0, pos["text"]].tolist()
    tokens = {"shared_prompt": adapter.tok.convert_ids_to_tokens(shared[:k_last + 1]),
              "text_leftover": adapter.tok.convert_ids_to_tokens(leftover)}
    rec = {"pair_id": fact["pair_id"], "set": fact["set"], "profile": fact["profile"],
           "n_shared_prompt": k_last + 1, "n_text_leftover": len(leftover),
           "n_tokens": {k: int(batches[k]["input_ids"].shape[1]) for k in ROUTES},
           "lp": lp, **similarity(store, names_by_layer, k_last)}
    return rec, worst, tokens


def gap(rec, key):
    lp = rec["lp"]
    return (lp["text/0"][key] - lp["text/u"][key]) - (lp["blank/0"][key] - lp["blank/u"][key])


def moved(recs, cond, key):
    """Moved fraction M of one condition (ratio of means over facts); None if the prompts do not differ."""
    target = "text" if cond.startswith("text") else "blank"
    sign = 1.0 if target == "text" else -1.0
    g = float(np.mean([gap(r, key) for r in recs]))
    if abs(g) < 1e-8:
        return None
    return float(np.mean([sign * (r["lp"][cond][key] - r["lp"][f"{target}/u"][key]) for r in recs]) / g)


def bootstrap(recs, cond, key, n_boot=1000):
    profiles = sorted({r["profile"] for r in recs})
    by = {p: [r for r in recs if r["profile"] == p] for p in profiles}
    rng = np.random.default_rng(0)
    vals = []
    for _ in range(n_boot):
        v = moved([r for p in rng.choice(profiles, len(profiles), replace=True) for r in by[p]], cond, key)
        if v is not None:
            vals.append(v)
    return [float(np.percentile(vals, q)) for q in (2.5, 97.5)] if vals else [None, None]


def summarize(recs_all, conds):
    out = {}
    for set_name in ("forget", "retain"):
        recs = [r for r in recs_all if r["set"] == set_name]
        if not recs:
            continue
        s = {"n_facts": len(recs), "n_profiles": len({r["profile"] for r in recs}),
             "n_shared_prompt_mean": float(np.mean([r["n_shared_prompt"] for r in recs])),
             "n_text_leftover_max": int(max(r["n_text_leftover"] for r in recs))}
        for key in READOUTS:
            mean = {c: float(np.mean([r["lp"][c][key] for r in recs])) for c in ("text/u", "text/0", "blank/u", "blank/0")}
            f_text, f_blank = mean["text/0"] - mean["text/u"], mean["blank/0"] - mean["blank/u"]
            s[key] = {"lp": mean, "forgetting_text": f_text, "forgetting_blank": f_blank,
                      "R_blank": 1 - f_blank / f_text if abs(f_text) > 1e-8 else None,
                      "moved": {c: {"M": moved(recs, c, key), "ci95": bootstrap(recs, c, key)} for c in conds}}
        for k in ("cos_prompt", "cos_last", "norm_ratio_prompt"):
            cols = list(zip(*[r[k] for r in recs]))
            s[k] = [None if any(v is None for v in col) else float(np.mean(col)) for col in cols]
        out[set_name] = s
    return out


def fmt(v):
    return "-" if v is None else f"{v:+.2f}"


def verdict(summary, bands):
    """Lines for the log: the DEV-014 reading on forget facts, answer log p."""
    f = summary.get("forget")
    if not f:
        return ["no forget facts scored: no verdict"]
    a, m = f["answer"], f["answer"]["moved"]
    lines = [f"bypass on these facts: R_blank = {fmt(a['R_blank'])} "
             f"({'reproduced' if a['R_blank'] is not None and a['R_blank'] >= BYPASS else 'NOT reproduced: do not interpret M'}"
             f"; threshold {BYPASS})"]
    if "0-19" in bands:
        x, y = m["text<-blank/0-19"]["M"], m["blank<-text/0-19"]["M"]
        if x is None or y is None:
            lines.append("layers 0-19: the two prompts do not differ: no verdict")
        elif x >= TRIGGER and y >= TRIGGER:
            lines.append(f"layers 0-19: M = {x:+.2f} / {y:+.2f} (both >= {TRIGGER}) -> the image changes what the LoRA writes")
        elif x <= OVERRIDE and y <= OVERRIDE:
            lines.append(f"layers 0-19: M = {x:+.2f} / {y:+.2f} (both <= {OVERRIDE}) -> the LoRA writes the same in effect; "
                         "with an image the model does not use it")
        else:
            lines.append(f"layers 0-19: M = {x:+.2f} / {y:+.2f} -> mixed, no verdict under DEV-014")
    if "20-31" in bands:
        x, y = m["text<-blank/20-31"]["M"], m["blank<-text/20-31"]["M"]
        ok = x is not None and y is not None and abs(x) <= CONTROL and abs(y) <= CONTROL
        lines.append(f"control, layers 20-31: M = {fmt(x)} / {fmt(y)} -> "
                     f"{'no effect, as expected' if ok else 'NOT negligible: report'} (threshold |M| <= {CONTROL})")
    return lines


def plot(summary, bands, title, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    INK2, MUTED, GRID, AXIS = "#52514e", "#898781", "#e1e0d9", "#c3c2b7"
    num = lambda v: 0.0 if v is None else v  # noqa: E731
    f = summary["forget"]
    rt = summary.get("retain")
    fig, axes = plt.subplots(1, 3, figsize=(12.5, 3.5), dpi=200)
    x = np.arange(len(bands))
    panels = (("text", "#2a78d6", "Name question: LoRA writes what it\nwrote with the blank image (1 = escapes too)"),
              ("blank", "#ec6a35", "Name question + blank image: LoRA writes what\nit wrote without it (1 = suppressed too)"))
    for ax, (tgt, color, ttl) in zip(axes, panels):
        main = [f["answer"]["moved"][f"{tgt}<-{OTHER[tgt]}/{b}"] for b in bands]
        off = [f["answer"]["moved"][f"{tgt}_off/{b}"] for b in bands]
        ys = [num(v["M"]) for v in main]
        err = [[max(0.0, y - num(v["ci95"][0])) if v["ci95"][0] is not None else 0.0 for y, v in zip(ys, main)],
               [max(0.0, num(v["ci95"][1]) - y) if v["ci95"][1] is not None else 0.0 for y, v in zip(ys, main)]]
        ax.bar(x - 0.18, ys, width=0.32, color=color, yerr=err, error_kw={"lw": 0.9, "ecolor": INK2},
               label=f"transplant (forget, n={f['n_facts']}, 95% CI)")
        ax.bar(x + 0.18, [num(v["M"]) for v in off], width=0.32, color=MUTED, alpha=0.45, label="LoRA off at these tokens")
        if rt:
            ax.scatter(x - 0.18, [num(rt["answer"]["moved"][f"{tgt}<-{OTHER[tgt]}/{b}"]["M"]) for b in bands], s=22,
                       facecolor="white", edgecolor="black", zorder=3, label=f"transplant (retain, n={rt['n_facts']})")
        for y in (0, 1):
            ax.axhline(y, color=AXIS, lw=0.9)
        for y in (OVERRIDE, TRIGGER):
            ax.axhline(y, color=AXIS, lw=0.6, ls=":")
        ax.set_xticks(x, [f"layers {b}" for b in bands], fontsize=8)
        ax.set_title(ttl, loc="left", fontsize=8, color=INK2)
        ax.legend(frameon=False, fontsize=6.5)
    axes[0].set_ylabel("moved fraction M of the gap between\nthe two prompts (answer log p)", color=INK2, fontsize=8)
    ax = axes[2]
    layers = np.arange(len(f["cos_prompt"]))
    ax.axvspan(-0.5, 19.5, color="#eef3fa", lw=0)
    for key, lab, color in (("cos_prompt", "shared prompt tokens", "#2a78d6"), ("cos_last", "last prompt token", "#e34948")):
        ys = [np.nan if v is None else v for v in f[key]]
        ax.plot(layers, ys, color=color, lw=1.6, label=lab)
    ax.axhline(1, color=AXIS, lw=0.9)
    ax.axhline(0, color=AXIS, lw=0.9)
    ax.set_xlabel("decoder layer (shaded: 0-19)", color=INK2, fontsize=8)
    ax.set_ylabel("cosine", color=INK2, fontsize=8)
    ax.set_title("LoRA contribution with vs without the blank image\n(same tokens, forget facts)", loc="left", fontsize=8, color=INK2)
    ax.legend(frameon=False, fontsize=6.5)
    for ax in axes:
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
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
    p.add_argument("--bands", default="0-19,20-31", help=f"comma-separated, from {list(BANDS)}")
    p.add_argument("--with-retain", action="store_true", help="also score the doubly-correct retain facts")
    p.add_argument("--max-facts", type=int, default=0, help="0 = all selected facts")
    p.add_argument("--check-facts", type=int, default=3, help="facts on which the self-transplant check runs")
    p.add_argument("--restart", action="store_true", help="set existing per-fact results aside instead of resuming")
    args = p.parse_args()
    bands = [b.strip() for b in args.bands.split(",") if b.strip()]
    if not bands or any(b not in BANDS for b in bands):
        raise SystemExit(f"--bands must be chosen from {list(BANDS)}")
    cfg = resolve(args)
    run_id = args.run_id or umu_run_id(cfg)
    run_dir = model_dir(cfg) / "umu_unlearn" / run_id
    ckpt = run_dir / "ckpt" / f"step{args.step}"
    if not (ckpt / "adapter_config.json").exists():
        raise SystemExit(f"no adapter at {ckpt}")
    out = results_dir(cfg) / "mech_umu" / run_id / f"step{args.step}" / "transplant"
    log = Logger(out / "log.txt")
    with tracked_run(results_dir(cfg), out, f"mech_umu_transplant_{run_id}_step{args.step}", "mech_umu_transplant"):
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
        conds = [f"{k}<-{OTHER[k]}/{b}" for b in bands for k in ROUTES] + [f"{k}_off/{b}" for b in bands for k in ROUTES]
        done = {r["pair_id"]: r for r in read_jsonl(path) if all(c in r["lp"] for c in conds)}

        adapter = get_adapter({**cfg["model"], "id": cfg["umu"]["ckpt"], "processor_id": cfg["model"]["id"]})
        model = PeftModel.from_pretrained(adapter.load_model(), str(ckpt), is_trainable=False)
        model.eval()
        n_layers = len(decoder_layers(model))
        names = [n for n in lora_modules(model) if layer_index(n) is not None]
        skipped = len(lora_modules(model)) - len(names)
        band_names = {b: [n for n in names if BANDS[b][0] <= layer_index(n) < min(BANDS[b][1], n_layers)] for b in bands}
        names_by_layer = [[n for n in names if layer_index(n) == i] for i in range(n_layers)]
        blank = blank_image_bytes()
        todo = [f for f in facts if f["pair_id"] not in done]
        log(f"{run_id} step {args.step}: {len(facts)} facts ({', '.join(sets)}), {len(todo)} to score; bands {bands}; "
            f"{len(names)} LoRA modules in {n_layers} layers ({skipped} outside decoder layers, left untouched); "
            f"{4 + 4 * len(bands)} forward passes per fact")

        worst, checked, t0 = 0.0, 0, time.time()
        for i, fact in enumerate(todo):
            rec, err, tokens = evaluate(adapter, model, fact, blank, names, band_names, names_by_layer,
                                        check=checked < args.check_facts)
            if err is not None:
                checked += 1
                worst = max(worst, err)
                if err > SELF_TOL:
                    raise RuntimeError(f"self-transplant changed log p by {err:.3f} nats on {fact['pair_id']} "
                                       f"(tolerance {SELF_TOL}): the hook is not exact, results would not be interpretable")
            append_jsonl(path, [rec])
            done[fact["pair_id"]] = rec
            if i == 0:
                log(f"  shared prompt tokens ({rec['n_shared_prompt']}): {' '.join(tokens['shared_prompt'])}")
                log(f"  text-prompt tokens outside the shared suffix and the identical prefix: {tokens['text_leftover']}")
                log(f"  prompt lengths with answer: {rec['n_tokens']}")
            if i + 1 == checked == args.check_facts or (i + 1 == len(todo) and 0 < checked < args.check_facts):
                log(f"  self-transplant check on {checked} facts: max |change in log p| = {worst:.2e} nats (tolerance {SELF_TOL})")
            if i + 1 in (5, len(todo)) or (i + 1) % 20 == 0:
                per = (time.time() - t0) / (i + 1)
                log(f"  {i + 1}/{len(todo)} facts, {per:.1f} s per fact, about {per * (len(todo) - i - 1) / 60:.1f} min left")
        model = None
        free()

        recs = [done[f["pair_id"]] for f in facts]
        summary = summarize(recs, conds)
        lines = verdict(summary, bands)
        atomic_write_json(out / "summary.json", {
            "run_id": run_id, "step": args.step, "bands": bands, "sets": list(sets), "conditions": conds,
            "readout": "teacher-forced log p of the answer text ('answer') and of its first token ('first')",
            "thresholds": {"trigger_M_min": TRIGGER, "override_M_max": OVERRIDE, "control_abs_M_max": CONTROL,
                           "bypass_R_blank_min": BYPASS, "self_transplant_tol_nats": SELF_TOL},
            "verdict": lines, "summary": summary})
        for set_name, s in summary.items():
            a = s["answer"]
            log(f"{set_name}: n={s['n_facts']} facts / {s['n_profiles']} profiles; shared prompt tokens {s['n_shared_prompt_mean']:.1f}; "
                f"forgetting text {a['forgetting_text']:.2f} nats, blank {a['forgetting_blank']:.2f} nats, R_blank {fmt(a['R_blank'])}")
            log(f"  {'condition':24s} {'M answer [95% CI]':>28s} {'M first token':>14s}")
            for c in conds:
                m = a["moved"][c]
                ci = "[" + ", ".join(fmt(v) for v in m["ci95"]) + "]"
                log(f"  {c:24s} {fmt(m['M']) + ' ' + ci:>28s} {fmt(s['first']['moved'][c]['M']):>14s}")
        for line in lines:
            log(line)
        if "forget" in summary:
            plot(summary, bands, f"{run_id} step {args.step}: transplanting the unlearning LoRA's contributions between prompts",
                 out / "transplant.png")
        log(f"wrote {out / 'summary.json'}" + (" and transplant.png" if "forget" in summary else ""))


if __name__ == "__main__":
    main()
