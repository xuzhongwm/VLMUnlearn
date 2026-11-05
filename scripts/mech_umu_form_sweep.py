"""UMU track, DEV-013: query-form component sweep. Which part of the question does text-only unlearning key on?

Base = the unlearning training form: "USER: {UM_QA question}\\nASSISTANT: {answer}", no image, no options. Each form
changes ONE component of it (same facts, same answer text):
  A image context   own portrait | blank image | another person's portrait | 576 neutral filler text tokens
                    (+ reference: the portrait question "this person" with the own portrait)
  B referent        first name only | last name only | "Mr./Ms. <last name>" | name order reversed
  C wording         two paraphrases per attribute
  D answer format   UMU option list ("Select answer in [...]") | A-D lettered options | fill-in-the-blank sentence
  E chat frame      vicuna system prompt | no chat frame ("Question: ... Answer:")
Model pair: theta_u = UMU's vanilla + the run's LoRA at --step, theta_0 = the same model with the LoRA off.
Readouts (the run's doubly-correct facts; forget and retain separately):
  F_v = mean over facts of [log p_theta0 - log p_thetau] of the correct answer text (teacher forced, EOS excluded)
  R_v = 1 - F_v / F_base   (0 = forgotten as much as in the training form, 1 = unaffected); 95% bootstrap CI over facts
  gen_v = greedy generation, correct if the normalised answer text occurs in the response (theta_0 and theta_u)
  ||A x||_v = mean norm of the LoRA input projection A x over the text prompt tokens, summed over the LoRA modules of
              layers 0-19 (where layer masking found the effect), under theta_u
Pre-registered predictions (DEVIATIONS.md DEV-013), evaluated and printed PASS / FAIL:
  P1 form not identity   R(blank) >= 0.5 and R(filler) >= 0.5   [counter: R(blank) <= 0.2 and R(own portrait) >= 0.8]
  P2 spelling irrelevant R(first), R(last), R(title) each <= 0.2
  P3 beyond modality     some text-only change (C, D, E) R >= 0.3 [counter: all <= 0.2]
  P4 collateral by form  Spearman(R_retain, R_forget) over forms >= 0.7
  P5 mechanism-behaviour Spearman(||A x||, 1 - R_forget) over forms >= 0.6

  python scripts/mech_umu_form_sweep.py --config configs/llava/umu_npo_rt.yaml --run-id <run_id> --step <N> [--no-gen]
Writes results/llava/mech_umu/<run_id>/step<N>/form_sweep.json, form_sweep.png (+ form_sweep_gen_*.jsonl).
"""
import collections
import io
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np  # noqa: E402
import torch  # noqa: E402

from vlmu.config import base_parser, resolve  # noqa: E402
from vlmu.evaluate import keyword_hit  # noqa: E402
from vlmu.mech import tools  # noqa: E402
from vlmu.mech.hooks import layer_index, lora_modules  # noqa: E402
from vlmu.models import get_adapter  # noqa: E402
from vlmu.runs import free, model_dir, results_dir  # noqa: E402
from vlmu.umu import encode_teacher_forced, image_bytes, load_umu, parse, umu_run_id  # noqa: E402
from vlmu.utils import Logger, append_jsonl, atomic_write_json, read_json, read_jsonl, tracked_run  # noqa: E402

VICUNA_SYSTEM = ("A chat between a curious user and an artificial intelligence assistant. "
                 "The assistant gives helpful, detailed, and polite answers to the user's questions.")
FILLER_TEXT = ("The museum opens at nine in the morning and closes at five in the afternoon. Visitors can walk through "
               "the garden, see the old maps in the east hall, and buy postcards at the small shop near the entrance. ")
PARAPHRASES = {   # two per attribute; {n} = the name
    "Occupation": ["What does {n} do for a living?", "Which profession does {n} work in?"],
    "Born": ["In which place was {n} born?", "What is the birthplace of {n}?"],
    "Birth": ["What is the date of birth of {n}?", "On what date was {n} born?"],
    "Education": ["Which institution did {n} graduate from?", "Where did {n} study?"],
    "Height": ["How tall is {n}?", "What height is {n}?"],
    "Annual_Salary": ["How much does {n} earn per year?", "What annual income does {n} have?"],
    "Residence": ["In which city does {n} live?", "Where is the home of {n}?"],
    "Medical_Conditions": ["What health conditions does {n} have?", "Is {n} diagnosed with any medical condition?"],
}
CLOZE = {
    "Occupation": "{n}'s occupation is ___.", "Born": "{n} was born in ___.", "Birth": "{n} was born on ___.",
    "Education": "{n} completed their education at ___.", "Height": "{n}'s height is ___.",
    "Annual_Salary": "{n}'s annual salary is ___.", "Residence": "{n} currently lives in ___.",
    "Medical_Conditions": "{n}'s medical conditions: ___.",
}
# (form, component, label); "base" first
FORMS = [("base", "base", "training form"),
         ("own_portrait", "A image", "+ own portrait"), ("blank_image", "A image", "+ blank image"),
         ("other_portrait", "A image", "+ other person's portrait"), ("filler_text", "A image", "+ 576 filler text tokens"),
         ("portrait_only", "A image", "portrait question (this person)"),
         ("first_name", "B referent", "first name only"), ("last_name", "B referent", "last name only"),
         ("title_last", "B referent", "Mr./Ms. + last name"), ("reversed_name", "B referent", "name order reversed"),
         ("paraphrase1", "C wording", "paraphrase 1"), ("paraphrase2", "C wording", "paraphrase 2"),
         ("option_list", "D format", "UMU option list"), ("lettered", "D format", "A-D lettered options"),
         ("cloze", "D format", "fill in the blank"),
         ("vicuna_frame", "E frame", "vicuna system prompt"), ("no_frame", "E frame", "no chat frame")]
TEXT_ONLY_CHANGES = ("paraphrase1", "paraphrase2", "option_list", "lettered", "cloze", "vicuna_frame", "no_frame")
MASK_LAYERS = 20


def blank_image_bytes():
    from PIL import Image
    buf = io.BytesIO()
    Image.new("RGB", (336, 336), (128, 128, 128)).save(buf, "PNG")
    return buf.getvalue()


def filler_string(tok, n_tokens=576):
    """A neutral passage cut to n_tokens tokens (the number of image tokens LLaVA-1.5 inserts)."""
    ids = tok(FILLER_TEXT * 20, add_special_tokens=False)["input_ids"][:n_tokens]
    return tok.decode(ids)


def facts_for(dfs, dc):
    """[{set, profile, field, pair_id, name, gender, q_uni, q_mul, options, answer_text, img}] for doubly-correct facts."""
    out = []
    for set_name in ("forget", "retain"):
        for _, row in dfs[set_name].iterrows():
            bio = parse(row["Biography"])
            cls = parse(row["Classify"])
            uni, mul = cls.get("unimodal", {}), cls.get("muitimodal") or cls.get("multimodal") or {}
            for key, u in uni.items():
                pid = f"umu/{row['ID']}/{key}"
                if pid not in dc or key not in mul:
                    continue
                opts = mul[key]["options"]
                out.append({"set": set_name, "profile": row["ID"], "field": key, "pair_id": pid, "name": bio.get("Name"),
                            "gender": bio.get("Gender", ""), "q_uni": u["question"], "q_mul": mul[key]["question"],
                            "options": [opts[k] for k in "ABCD"], "answer_text": opts[u["answer"].split(".")[0]],
                            "img": image_bytes(row["image"])})
    return out


def referent_variants(name, gender):
    parts = name.split()
    first, last = parts[0], parts[-1]
    title = {"female": "Ms.", "male": "Mr."}.get(str(gender).strip().lower(), "Mx.")
    return {"first_name": first, "last_name": last, "title_last": f"{title} {last}",
            "reversed_name": " ".join(parts[1:] + parts[:1]) if len(parts) > 1 else name}


def build_rows(facts, tok):
    """One row per (fact, form): {uid, set, form, profile, field, pair_id, prompt, image_bytes, correct_text}."""
    blank, filler = blank_image_bytes(), filler_string(tok)
    by_set = collections.defaultdict(list)
    for f in facts:
        by_set[f["set"]].append(f)
    rows = []
    for set_name, fs in by_set.items():
        people = sorted({f["profile"] for f in fs})
        img_of = {f["profile"]: f["img"] for f in fs}
        other = {p: img_of[people[(i + 1) % len(people)]] for i, p in enumerate(people)}   # deterministic other person
        for f in fs:
            n, q, ans = f["name"], f["q_uni"], f["answer_text"]
            if n not in q:
                continue
            ref = referent_variants(n, f["gender"])
            opts = f["options"]
            lettered = "\n".join(f"{L}. {o}" for L, o in zip("ABCD", opts))
            p = {
                "base": (f"USER: {q}\nASSISTANT:", None),
                "own_portrait": (f"USER: <image>\n{q}\nASSISTANT:", f["img"]),
                "blank_image": (f"USER: <image>\n{q}\nASSISTANT:", blank),
                "other_portrait": (f"USER: <image>\n{q}\nASSISTANT:", other[f["profile"]]),
                "filler_text": (f"USER: {filler}\n{q}\nASSISTANT:", None),
                "portrait_only": (f"USER: <image>\n{f['q_mul']}\nASSISTANT:", f["img"]),
                "paraphrase1": (f"USER: {PARAPHRASES[f['field']][0].format(n=n)}\nASSISTANT:", None),
                "paraphrase2": (f"USER: {PARAPHRASES[f['field']][1].format(n=n)}\nASSISTANT:", None),
                "option_list": (f"USER:\n{q}\nSelect answer in {str(opts)}\nASSISTANT:", None),
                "lettered": (f"USER: {q}\n{lettered}\nAnswer with the text of the correct option.\nASSISTANT:", None),
                "cloze": (f"USER: Fill in the blank: {CLOZE[f['field']].format(n=n)}\nASSISTANT:", None),
                "vicuna_frame": (f"{VICUNA_SYSTEM} USER: {q} ASSISTANT:", None),
                "no_frame": (f"Question: {q}\nAnswer:", None),
            }
            for k, v in ref.items():
                p[k] = (f"USER: {q.replace(n, v)}\nASSISTANT:", None)
            for form, _, _ in FORMS:
                prompt, img = p[form]
                rows.append({"uid": f"{set_name}/{f['pair_id']}/{form}", "set": set_name, "form": form,
                             "profile": f["profile"], "field": f["field"], "pair_id": f["pair_id"],
                             "prompt": prompt, "image_bytes": img, "correct_text": ans, "name": n})
    return rows


def answer_logprobs(adapter, model, batch):
    """Per row: (sum of log p over the answer tokens, log p of the first answer token)."""
    t = tools.answer_targets(batch)
    logits = tools.forward_logits(adapter, model, batch)
    lp = tools.target_logprob(logits[t["b"], t["pos"]], t["tok"])
    total = tools._per_row(lp, t)
    first = torch.full((t["n_rows"],), float("nan"), device=lp.device)
    f = tools._first(t)
    first[t["b"][f]] = lp[f]
    return total.float().cpu().tolist(), first.float().cpu().tolist()


def prompt_text_mask(batch, image_token_id):
    """[B, T] bool: prompt positions (before the first answer label) that are not image tokens."""
    labels, valid = batch["labels"], batch["attention_mask"].bool()
    T = labels.shape[1]
    pos = torch.arange(T, device=labels.device)[None, :]
    has = labels != -100
    first_ans = torch.where(has.any(1), has.float().argmax(1), torch.full_like(has[:, 0], T, dtype=torch.long))
    return valid & (pos < first_ans[:, None]) & (batch["input_ids"] != image_token_id)


def spearman(x, y):
    x, y = np.asarray(x, float), np.asarray(y, float)
    ok = ~(np.isnan(x) | np.isnan(y))
    if ok.sum() < 3:
        return float("nan")
    rx, ry = np.argsort(np.argsort(x[ok])), np.argsort(np.argsort(y[ok]))
    return float(np.corrcoef(rx, ry)[0, 1])


def recovery(per, forms, set_name, n_boot=1000, seed=0):
    """{form: {R, lo, hi, F, n}} from per-fact drops d = lp0 - lpu (answer sum), R = 1 - mean(d_v) / mean(d_base)."""
    facts = sorted({k[1] for k in per if k[0] == set_name})
    D = np.array([[per[(set_name, pid, f)]["lp0"] - per[(set_name, pid, f)]["lpu"] if (set_name, pid, f) in per else np.nan
                   for f in forms] for pid in facts])
    if not len(D):
        return {}
    rng = np.random.default_rng(seed)
    base = 0
    out = {}
    for j, f in enumerate(forms):
        col, b = D[:, j], D[:, base]
        ok = ~(np.isnan(col) | np.isnan(b))
        if ok.sum() == 0:
            continue
        R = 1 - col[ok].mean() / b[ok].mean()
        boots = []
        idx = np.flatnonzero(ok)
        for _ in range(n_boot):
            s = rng.choice(idx, size=len(idx), replace=True)
            boots.append(1 - D[s, j].mean() / D[s, base].mean())
        out[f] = {"R": float(R), "lo": float(np.percentile(boots, 2.5)), "hi": float(np.percentile(boots, 97.5)),
                  "F": float(col[ok].mean()), "n": int(ok.sum())}
    return out


def predictions(Rf, Rr, ax_norm):
    g = lambda f: Rf.get(f, {}).get("R", float("nan"))  # noqa: E731
    forms = [f for f, _, _ in FORMS if f != "base" and f in Rf]
    p = {}
    p["P1"] = {"pass": g("blank_image") >= 0.5 and g("filler_text") >= 0.5,
               "counter": g("blank_image") <= 0.2 and g("own_portrait") >= 0.8,
               "values": {k: g(k) for k in ("blank_image", "filler_text", "own_portrait", "other_portrait")}}
    p["P2"] = {"pass": all(g(k) <= 0.2 for k in ("first_name", "last_name", "title_last")),
               "values": {k: g(k) for k in ("first_name", "last_name", "title_last", "reversed_name")}}
    txt = {k: g(k) for k in TEXT_ONLY_CHANGES}
    p["P3"] = {"pass": any(v >= 0.3 for v in txt.values()), "counter": all(v <= 0.2 for v in txt.values()), "values": txt}
    rho4 = spearman([Rr.get(f, {}).get("R", np.nan) for f in forms], [g(f) for f in forms])
    p["P4"] = {"pass": rho4 >= 0.7, "spearman": rho4}
    rho5 = spearman([ax_norm.get(f, np.nan) for f in forms], [1 - g(f) for f in forms])
    p["P5"] = {"pass": rho5 >= 0.6, "spearman": rho5}
    return p


def plot(Rf, Rr, ax_norm, title, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    INK2, MUTED, GRID, AXIS = "#52514e", "#898781", "#e1e0d9", "#c3c2b7"
    COMP = {"base": "#898781", "A image": "#eb6834", "B referent": "#6250d6", "C wording": "#2a78d6",
            "D format": "#1baf7a", "E frame": "#eda100"}
    forms = [(f, c, lab) for f, c, lab in FORMS if f in Rf]
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(11.5, 5.6), dpi=200, gridspec_kw={"width_ratios": [1.6, 1]})
    y = np.arange(len(forms))[::-1]
    for yi, (f, c, lab) in zip(y, forms):
        r = Rf[f]
        a1.barh(yi, r["R"], color=COMP[c], height=0.62)
        a1.plot([r["lo"], r["hi"]], [yi, yi], color="#0b0b0b", lw=1)
        if f in Rr:
            a1.plot([Rr[f]["R"]], [yi], "o", ms=4.5, mfc="white", mec="#0b0b0b", mew=1)
    a1.set_yticks(y, [f"{lab}" for _, _, lab in forms], fontsize=7.5)
    for v in (0.2, 0.5):
        a1.axvline(v, color=AXIS, lw=0.9)
    a1.axvline(0, color=INK2, lw=0.9)
    a1.set_xlim(min(-0.2, min(Rf[f]["lo"] for f, _, _ in forms) - 0.05), max(1.1, max(Rf[f]["hi"] for f, _, _ in forms) + 0.05))
    a1.set_xlabel("recovery R (0 = forgotten as in the training form, 1 = unaffected)\nbars: forget facts ±95% CI; circles: retain facts",
                  color=INK2, fontsize=8)
    for s in ("top", "right"):
        a1.spines[s].set_visible(False)
        a2.spines[s].set_visible(False)
    a1.grid(axis="x", color=GRID, lw=0.6)
    a1.set_axisbelow(True)
    handles = [plt.Rectangle((0, 0), 1, 1, color=v) for v in COMP.values()]
    a1.legend(handles, list(COMP), frameon=False, fontsize=7, loc="lower right")
    xs = [ax_norm.get(f, np.nan) for f, _, _ in forms]
    ys = [1 - Rf[f]["R"] for f, _, _ in forms]
    for (f, c, lab), x, yv in zip(forms, xs, ys):
        a2.plot([x], [yv], "o", ms=6, color=COMP[c], mec="white", mew=1.2)
    a2.set_xlabel("mean ||A x|| over text prompt tokens, LoRA in layers 0-19", color=INK2, fontsize=8)
    a2.set_ylabel("forgetting left (1 - R)", color=INK2, fontsize=8)
    a2.grid(color=GRID, lw=0.6)
    a2.set_axisbelow(True)
    a2.set_title(f"Spearman = {spearman(xs, ys):+.2f}", fontsize=8.5, color=INK2, loc="left")
    for ax in (a1, a2):
        ax.tick_params(colors=MUTED)
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
    p.add_argument("--gen-batch-img", type=int, default=8)
    p.add_argument("--no-gen", action="store_true", help="skip the generation readout")
    args = p.parse_args()
    cfg = resolve(args)
    run_id = args.run_id or umu_run_id(cfg)
    run_dir = model_dir(cfg) / "umu_unlearn" / run_id
    out = results_dir(cfg) / "mech_umu" / run_id / f"step{args.step}"
    log = Logger(out / "form_sweep_log.txt")
    adapter_dir = run_dir / "ckpt" / f"step{args.step}"
    if not (adapter_dir / "adapter_config.json").exists():
        raise SystemExit(f"no adapter at {adapter_dir}")
    with tracked_run(results_dir(cfg), out, f"mech_umu_form_sweep_{run_id}_step{args.step}", "mech_umu_form_sweep"):
        from peft import PeftModel
        dc = set(read_json(run_dir / "baseline_dc.json")["pair_ids"])
        dfs = load_umu(cfg["paths"]["root"], cfg["umu"]["n_retain_eval"], 0, log, cfg["umu"]["forget_split"])
        adapter = get_adapter({**cfg["model"], "id": cfg["umu"]["ckpt"], "processor_id": cfg["model"]["id"]})
        facts = facts_for(dfs, dc)
        rows = build_rows(facts, adapter.tok)
        n_fill = len(adapter.tok(filler_string(adapter.tok), add_special_tokens=False)["input_ids"])
        log(f"{run_id} step {args.step}: {len(facts)} doubly-correct facts, {len(FORMS)} forms, {len(rows)} rows; "
            f"filler re-tokenises to {n_fill} tokens")
        log("example prompts (first forget fact):")
        first_pid = next(f["pair_id"] for f in facts if f["set"] == "forget")
        for r in rows:
            if r["pair_id"] == first_pid and r["set"] == "forget" and r["form"] != "filler_text":
                log(f"  {r['form']:15s} {r['prompt']!r} -> {r['correct_text']!r}{' [image]' if r['image_bytes'] else ''}")

        model = PeftModel.from_pretrained(adapter.load_model(), str(adapter_dir), is_trainable=False)
        model.eval()
        img_id = model.config.image_token_index if hasattr(model.config, "image_token_index") else 32000
        layers = set(range(MASK_LAYERS))
        ax_mods = {n: m for n, m in lora_modules(model).items() if layer_index(n) in layers}
        per, ax_row = {}, {}
        for form, _, _ in FORMS:
            group = sorted([r for r in rows if r["form"] == form], key=lambda r: len(r["prompt"]))
            bs = args.batch_txt if group[0]["image_bytes"] is None else args.batch_img
            for i in range(0, len(group), bs):
                chunk = group[i:i + bs]
                batch = encode_teacher_forced(adapter, chunk)
                tmask = prompt_text_mask(batch, img_id)
                acc = {"v": None}

                def hook(mod, inp, o, acc=acc):
                    a = mod.active_adapters[0]
                    nrm = mod.lora_A[a](inp[0].to(mod.lora_A[a].weight.dtype)).float().norm(dim=-1)
                    acc["v"] = nrm if acc["v"] is None else acc["v"] + nrm
                handles = [m.register_forward_hook(hook) for m in ax_mods.values()]
                try:
                    lpu, fu = answer_logprobs(adapter, model, batch)
                finally:
                    for h in handles:
                        h.remove()
                axn = ((acc["v"] * tmask).sum(1) / tmask.sum(1).clamp_min(1)).cpu().tolist()
                with model.disable_adapter():
                    lp0, f0 = answer_logprobs(adapter, model, batch)
                for r, a, b, c, d, e in zip(chunk, lpu, lp0, fu, f0, axn):
                    per[(r["set"], r["pair_id"], form)] = {"lpu": a, "lp0": b, "first_u": c, "first_0": d}
                    ax_row[(r["set"], r["pair_id"], form)] = e
            log(f"  scored {form} ({len(group)} rows)")

        gen = {}
        if not args.no_gen:
            for form, _, _ in FORMS:
                group = sorted([r for r in rows if r["form"] == form], key=lambda r: len(r["prompt"]))
                for tag, ctx in (("u", None), ("0", "off")):
                    path = out / f"form_sweep_gen_{form}_{tag}.jsonl"
                    done = {x["uid"]: x for x in read_jsonl(path)}
                    todo = [r for r in group if r["uid"] not in done]
                    bs = args.batch_txt if group[0]["image_bytes"] is None else args.gen_batch_img
                    for i in range(0, len(todo), bs):
                        chunk = todo[i:i + bs]
                        from PIL import Image
                        imgs = [Image.open(io.BytesIO(r["image_bytes"])).convert("RGB") for r in chunk if r["image_bytes"]]
                        b = adapter.to_device(adapter._process([r["prompt"] for r in chunk], imgs or None, "left"))
                        if ctx == "off":
                            with model.disable_adapter():
                                texts = adapter.generate(model, b, 32)
                        else:
                            texts = adapter.generate(model, b, 32)
                        recs = [{"uid": r["uid"], "set": r["set"], "form": form, "pair_id": r["pair_id"], "response": t,
                                 "correct": keyword_hit(r["correct_text"], t)} for r, t in zip(chunk, texts)]
                        append_jsonl(path, recs)
                        done.update({x["uid"]: x for x in recs})
                    res = [done[r["uid"]] for r in group]
                    for s in ("forget", "retain"):
                        v = [x["correct"] for x in res if x["set"] == s]
                        if v:
                            gen.setdefault(form, {}).setdefault(s, {})[tag] = float(np.mean(v))
                log(f"  generated {form}")
        model = None
        free()

        forms = [f for f, _, _ in FORMS]
        Rf, Rr = recovery(per, forms, "forget"), recovery(per, forms, "retain")
        ax_norm = {f: float(np.mean([v for (s, _, ff), v in ax_row.items() if s == "forget" and ff == f])) for f in forms}
        pred = predictions(Rf, Rr, ax_norm)
        atomic_write_json(out / "form_sweep.json", {
            "run_id": run_id, "step": args.step, "forms": [{"form": f, "component": c, "label": l} for f, c, l in FORMS],
            "recovery_forget": Rf, "recovery_retain": Rr, "lora_input_norm_L0_19": ax_norm, "generation": gen,
            "predictions": pred, "per_fact": {"|".join(k): v for k, v in per.items()},
            "lora_input_norm_per_fact": {"|".join(k): v for k, v in ax_row.items()}})
        log("recovery R (0 = forgotten as in the training form, 1 = unaffected), 95% CI over facts; generation correct "
            "(vanilla -> unlearned); ||A x|| in layers 0-19")
        log(f"{'form':34s} {'R forget':>20s} {'R retain':>9s} {'gen forget':>13s} {'gen retain':>13s} {'||Ax||':>8s}")
        for f, c, lab in FORMS:
            rf, rr = Rf.get(f), Rr.get(f)
            gf, gr = gen.get(f, {}).get("forget", {}), gen.get(f, {}).get("retain", {})
            gtxt = lambda g: f"{g.get('0', float('nan')):.2f}->{g.get('u', float('nan')):.2f}" if g else "-"  # noqa: E731
            log(f"{c[:10]:10s} {lab:23s} "
                f"{(format(rf['R'], '+.2f') + ' [' + format(rf['lo'], '+.2f') + ',' + format(rf['hi'], '+.2f') + ']') if rf else '-':>20s} "
                f"{format(rr['R'], '+.2f') if rr else '-':>9s} {gtxt(gf):>13s} {gtxt(gr):>13s} {ax_norm.get(f, float('nan')):8.2f}")
        log(f"base form: mean drop in log p (vanilla - unlearned) forget {Rf.get('base', {}).get('F', float('nan')):.2f}, "
            f"retain {Rr.get('base', {}).get('F', float('nan')):.2f}")
        for k, v in pred.items():
            extra = ", ".join(f"{a}={b:+.2f}" for a, b in v.get("values", {}).items()) if "values" in v else f"spearman={v['spearman']:+.2f}"
            counter = " (counter-evidence pattern)" if v.get("counter") else ""
            log(f"{k}: {'PASS' if v['pass'] else 'FAIL'}{counter} — {extra}")
        plot(Rf, Rr, ax_norm, f"{run_id} step {args.step}: which part of the question does the unlearning key on?",
             out / "form_sweep.png")
        log(f"wrote {out / 'form_sweep.json'} and form_sweep.png")


if __name__ == "__main__":
    main()
