"""UMU input-cue experiment: what part of adding a blank image bypasses text unlearning?

Same name question, fact and answer throughout. Conditions: no image, blank image before/after
the question, the same blank image with projected visual features set to zero while retaining
image token slots and positions, and neutral text prefixes of 16/64/256/576 tokens or a
576-token suffix. The zero-projection intervention is out of distribution; a vanilla-model
readout is mandatory and included. This experiment does not block image-token attention.

Both theta_u and adapter-off theta_0 are scored by greedy generation (keyword hit) and by
teacher-forced log p of the correct answer. All frozen doubly-correct facts are included,
without selecting on their post-unlearning outcome. Results are resumable per condition.

  python scripts/mech_umu_input_cues.py --config configs/llava/umu_npo_rt.yaml \
    --run-id umu_npo_rt_txt_lr5e-05_s0 --step 150

Writes results/llava/mech_umu/<run>/step<N>/input_cues/{summary.json,items_*.jsonl}.
"""
import contextlib
import io
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np  # noqa: E402
import torch  # noqa: E402
from PIL import Image  # noqa: E402

from scripts.mech_umu_form_sweep import blank_image_bytes, facts_for, filler_string  # noqa: E402
from vlmu.config import base_parser, resolve  # noqa: E402
from vlmu.evaluate import keyword_hit  # noqa: E402
from vlmu.mech import tools  # noqa: E402
from vlmu.models import get_adapter  # noqa: E402
from vlmu.runs import free, model_dir, results_dir  # noqa: E402
from vlmu.umu import encode_teacher_forced, load_umu, umu_run_id  # noqa: E402
from vlmu.utils import Logger, append_jsonl, atomic_write_json, read_json, read_jsonl, tracked_run  # noqa: E402

FORMS = ("base", "blank_before", "blank_after", "blank_zero_projector",
         "text_prefix_16", "text_prefix_64", "text_prefix_256", "text_prefix_576",
         "text_suffix_576")


def make_rows(facts, tok):
    blank = blank_image_bytes()
    filler = {n: filler_string(tok, n) for n in (16, 64, 256, 576)}
    rows = {form: [] for form in FORMS}
    for f in facts:
        q = f["q_uni"]
        prompts = {
            "base": (f"USER: {q}\nASSISTANT:", None),
            "blank_before": (f"USER: <image>\n{q}\nASSISTANT:", blank),
            "blank_after": (f"USER: {q}\n<image>\nASSISTANT:", blank),
            "blank_zero_projector": (f"USER: <image>\n{q}\nASSISTANT:", blank),
            **{f"text_prefix_{n}": (f"USER: {filler[n]}\n{q}\nASSISTANT:", None)
               for n in (16, 64, 256, 576)},
            "text_suffix_576": (f"USER: {q}\n{filler[576]}\nASSISTANT:", None),
        }
        for form, (prompt, image) in prompts.items():
            rows[form].append({"uid": f"{f['set']}/{f['pair_id']}/{form}", "pair_id": f["pair_id"],
                               "set": f["set"], "profile": f["profile"], "field": f["field"],
                               "prompt": prompt, "image_bytes": image, "correct_text": f["answer_text"]})
    return rows


def projector_module(model):
    found = [(name, module) for name, module in model.named_modules()
             if name.split(".")[-1] == "multi_modal_projector"]
    if len(found) != 1:
        raise RuntimeError(f"Expected one LLaVA multi_modal_projector, found {[name for name, _ in found]}")
    return found[0][1]


@contextlib.contextmanager
def projected_features(module, mode):
    if mode is None:
        yield
        return
    if mode == "identity":
        hook = lambda _m, _i, out: out
    elif mode == "zero":
        hook = lambda _m, _i, out: torch.zeros_like(out)
    else:
        raise ValueError(mode)
    handle = module.register_forward_hook(hook)
    try:
        yield
    finally:
        handle.remove()


def logprobs(adapter, model, batch):
    targets = tools.answer_targets(batch)
    logits = tools.forward_logits(adapter, model, batch)
    lp = tools.target_logprob(logits[targets["b"], targets["pos"]], targets["tok"])
    total = tools._per_row(lp, targets).float().cpu().tolist()
    first = torch.full((targets["n_rows"],), float("nan"), device=lp.device)
    idx = tools._first(targets)
    first[targets["b"][idx]] = lp[idx]
    prompt_tokens = (targets["pos"][idx] + 1).cpu().tolist()
    return total, first.float().cpu().tolist(), prompt_tokens


def generate(adapter, model, rows):
    imgs = [Image.open(io.BytesIO(r["image_bytes"])).convert("RGB") for r in rows if r["image_bytes"] is not None]
    assert len(imgs) in (0, len(rows))
    batch = adapter.to_device(adapter._process([r["prompt"] for r in rows], imgs or None, "left"))
    return adapter.generate(model, batch, 32)


def score_chunk(adapter, model, projector, rows, tag, form, no_gen):
    mode = "zero" if form == "blank_zero_projector" else None
    ctx = model.disable_adapter() if tag == "0" else contextlib.nullcontext()
    with ctx, projected_features(projector, mode):
        batch = encode_teacher_forced(adapter, rows)
        answer_lp, first_lp, prompt_tokens = logprobs(adapter, model, batch)
        responses = [None] * len(rows) if no_gen else generate(adapter, model, rows)
    return [{"uid": r["uid"], "pair_id": r["pair_id"], "set": r["set"],
             "profile": r["profile"], "form": form, "tag": tag,
             "answer_lp": float(a), "first_lp": float(b), "prompt_tokens": int(n),
             "response": s, "gen_correct": None if s is None else bool(keyword_hit(r["correct_text"], s))}
            for r, a, b, n, s in zip(rows, answer_lp, first_lp, prompt_tokens, responses)]


def profile_bootstrap(data, base, n_boot=1000):
    """Paired profile bootstrap of 1 - F_form/F_base; one profile may have multiple facts."""
    profiles = sorted({r["profile"] for r in data})
    by_profile = {p: [r for r in data if r["profile"] == p] for p in profiles}
    rng = np.random.default_rng(0)
    values = []
    for _ in range(n_boot):
        sample = [r for p in rng.choice(profiles, len(profiles), replace=True) for r in by_profile[p]]
        fb = np.mean([base[r["pair_id"]] for r in sample])
        fv = np.mean([r["drop"] for r in sample])
        if abs(fb) > 1e-8:
            values.append(1 - fv / fb)
    return [float(np.percentile(values, x)) for x in (2.5, 97.5)] if values else [None, None]


def summarize(all_rows):
    by_form = {form: {tag: {r["uid"]: r for r in all_rows[(form, tag)]} for tag in ("u", "0")}
               for form in FORMS}
    out = {}
    for set_name in ("forget", "retain"):
        base = {}
        for uid, u in by_form["base"]["u"].items():
            if u["set"] == set_name and uid in by_form["base"]["0"]:
                base[u["pair_id"]] = by_form["base"]["0"][uid]["answer_lp"] - u["answer_lp"]
        out[set_name] = {}
        for form in FORMS:
            pairs = []
            for uid, u in by_form[form]["u"].items():
                z = by_form[form]["0"].get(uid)
                if u["set"] == set_name and z and u["pair_id"] in base:
                    pairs.append({"pair_id": u["pair_id"], "profile": u["profile"],
                                  "drop": z["answer_lp"] - u["answer_lp"], "u": u, "z": z})
            if not pairs:
                continue
            fb = np.mean([base[r["pair_id"]] for r in pairs])
            fv = np.mean([r["drop"] for r in pairs])
            ci = profile_bootstrap(pairs, base)
            gen_u = [r["u"]["gen_correct"] for r in pairs if r["u"]["gen_correct"] is not None]
            gen_0 = [r["z"]["gen_correct"] for r in pairs if r["z"]["gen_correct"] is not None]
            out[set_name][form] = {
                "n_facts": len(pairs), "n_profiles": len({r["profile"] for r in pairs}),
                "prompt_tokens_mean": float(np.mean([r["u"]["prompt_tokens"] for r in pairs])),
                "answer_lp_u": float(np.mean([r["u"]["answer_lp"] for r in pairs])),
                "answer_lp_0": float(np.mean([r["z"]["answer_lp"] for r in pairs])),
                "forgetting_drop": float(fv), "recovery_R": float(1 - fv / fb) if abs(fb) > 1e-8 else None,
                "recovery_R_profile_bootstrap_95": ci,
                "generation_keyword_u": float(np.mean(gen_u)) if gen_u else None,
                "generation_keyword_0": float(np.mean(gen_0)) if gen_0 else None,
            }
    return out


def main():
    p = base_parser(__doc__)
    p.add_argument("--run-id", default=None)
    p.add_argument("--step", type=int, required=True)
    p.add_argument("--max-facts", type=int, default=0, help="0 = all frozen doubly-correct facts")
    p.add_argument("--batch-img", type=int, default=4)
    p.add_argument("--batch-txt", type=int, default=16)
    p.add_argument("--no-gen", action="store_true", help="teacher-forced readout only")
    args = p.parse_args()
    cfg = resolve(args)
    run_id = args.run_id or umu_run_id(cfg)
    run_dir = model_dir(cfg) / "umu_unlearn" / run_id
    ckpt = run_dir / "ckpt" / f"step{args.step}"
    if not (ckpt / "adapter_config.json").exists():
        raise SystemExit(f"no adapter at {ckpt}")
    out = results_dir(cfg) / "mech_umu" / run_id / f"step{args.step}" / "input_cues"
    log = Logger(out / "log.txt")
    with tracked_run(results_dir(cfg), out, f"mech_umu_input_cues_{run_id}_step{args.step}", "mech_umu_input_cues"):
        from peft import PeftModel
        dc = set(read_json(run_dir / "baseline_dc.json")["pair_ids"])
        dfs = load_umu(cfg["paths"]["root"], cfg["umu"]["n_retain_eval"], 0, log, cfg["umu"]["forget_split"])
        facts = sorted(facts_for(dfs, dc), key=lambda f: (f["set"], f["pair_id"]))
        if args.max_facts:
            facts = facts[:args.max_facts]
        adapter = get_adapter({**cfg["model"], "id": cfg["umu"]["ckpt"], "processor_id": cfg["model"]["id"]})
        rows = make_rows(facts, adapter.tok)
        model = PeftModel.from_pretrained(adapter.load_model(), str(ckpt), is_trainable=False)
        model.eval()
        projector = projector_module(model)

        # A passive projector hook must reproduce the unhooked readout exactly.
        probe = encode_teacher_forced(adapter, [rows["blank_before"][0]])
        ref = logprobs(adapter, model, probe)[1][0]
        with projected_features(projector, "identity"):
            same = logprobs(adapter, model, probe)[1][0]
        if abs(ref - same) > 1e-6:
            raise RuntimeError(f"passive projector hook changed first-token log p: {ref} -> {same}")
        log(f"passive projector hook: first-token log p difference {same - ref:+.2e}")
        log(f"{len(facts)} doubly-correct facts; forms={FORMS}; generation={'off' if args.no_gen else 'on'}")

        all_rows = {}
        for form in FORMS:
            group = rows[form]
            bs = args.batch_img if group[0]["image_bytes"] is not None else args.batch_txt
            for tag in ("u", "0"):
                path = out / f"items_{form}_{tag}.jsonl"
                done = {r["uid"]: r for r in read_jsonl(path)}
                todo = [r for r in group if r["uid"] not in done or
                        (not args.no_gen and done[r["uid"]].get("gen_correct") is None)]
                for i in range(0, len(todo), bs):
                    chunk = todo[i:i + bs]
                    recs = score_chunk(adapter, model, projector, chunk, tag, form, args.no_gen)
                    append_jsonl(path, recs)
                    done.update({r["uid"]: r for r in recs})
                all_rows[(form, tag)] = [done[r["uid"]] for r in group]
                log(f"{form:24s} theta_{tag}: {len(group)} rows, {len(todo)} newly scored")
        summary = summarize(all_rows)
        atomic_write_json(out / "summary.json", {
            "run_id": run_id, "step": args.step, "forms": list(FORMS),
            "readout": "answer-text log p (teacher forced, no EOS) and generation keyword hit",
            "zero_projector_caveat": "Out-of-distribution intervention; inspect vanilla scores before interpretation.",
            "summary": summary,
        })
        for set_name in ("forget", "retain"):
            log(f"{set_name}: form | n | R [95% profile bootstrap] | generation vanilla -> unlearned | prompt tokens")
            for form, s in summary[set_name].items():
                log(f"  {form:24s} {s['n_facts']:2d} R={s['recovery_R']:+.2f} {s['recovery_R_profile_bootstrap_95']}"
                    f" gen={s['generation_keyword_0']} -> {s['generation_keyword_u']}"
                    f" tokens={s['prompt_tokens_mean']:.1f}")
        log(f"wrote {out / 'summary.json'}")
        model = None
        free()


if __name__ == "__main__":
    main()
