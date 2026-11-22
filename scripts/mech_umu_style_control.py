"""UMU track, DEV-019 gate 2: is the knowledge still there once the response style is controlled? (claim C1)

On the training form the unlearned model mostly answers a name question with a long sentence about the person instead
of the short fine-tuned answer (DEV-016, DEV-017), so exact match cannot tell a changed response style from lost
knowledge. Three readouts that do not depend on the response style, on the run's doubly-correct facts:
  FC-train  forced choice among the item's four UMU options under the training form "USER: {q}\\nASSISTANT:" (no
            options in the prompt): the option whose answer tokens get the highest summed log p (teacher forced, end
            of turn not scored) is the model's choice. Secondary: mean log p per answer token.
  FC-opt    the same under the UMU readout prompt "USER:\\n{q}\\nSelect answer in [...]\\nASSISTANT:" (options listed)
  FS        three-shot generation: three solved readout items "<readout prompt> {answer}</s>" before the target's
            readout prompt. Each shot asks about a different attribute from the target's and from the other shots,
            about people outside every readout and training set (the last 50 retain_90 profiles in ID order, rows
            401-450), and no shot's answer equals the target's. Greedy, 50 new tokens, UMU parsing (the text before
            a repeated "ASSISTANT:"; here also before a new "USER:"). Scored by exact match after removing one
            trailing period from response and answer (DEV-019 amendment: shot answers that end with a period are
            copied, "NA." for "NA"); raw exact match, UMU's difflib judge and keyword hit as secondary.
Std_u, the standard readout, is recomputed under the same period rule from the run's evals/step<N>.jsonl (txt form,
doubly-correct facts); the checkpoint's recorded value (raw exact match) is reported next to it.
Models: theta_0 (UMU's vanilla, LoRA off), theta_u (+ the run's LoRA at --step) and, as a prior control, the public
base llava-hf/llava-1.5-7b-hf, which never saw the profiles.

Reading (thresholds fixed in DEV-019 before the run; forget dc facts; Std_u = the checkpoint's forget dc txt):
  C1 pass   FC-train(theta_u) >= 0.85 and FS recovery (FS_u - Std_u) / (FS_0 - Std_u) >= 0.5
  C1 fail   FC-train(theta_u) <= 0.5 and FS recovery <= 0.2
  otherwise partial
  not interpretable   theta_0 FC-train < 0.90, theta_0 FS < 0.90, FS_0 - Std_u < 0.15, or base FC-train > 0.60

  python scripts/mech_umu_style_control.py --config configs/llava/umu_npo_rt.yaml --run-id <run> --step <N>
The retain readout size (umu.n_retain_eval) is taken from the run's own config.json. Writes
results/llava/mech_umu/<run>/step<N>/style_control/{summary.json,items.jsonl,style_control.png,log.txt}.
Not resumable; a few minutes on an A6000.
"""
import contextlib
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np  # noqa: E402
import torch  # noqa: E402

from scripts.mech_umu_form_sweep import answer_logprobs, facts_for  # noqa: E402
from vlmu.config import base_parser, resolve  # noqa: E402
from vlmu.evaluate import keyword_hit  # noqa: E402
from vlmu.models import get_adapter  # noqa: E402
from vlmu.runs import free, model_dir, results_dir  # noqa: E402
from vlmu.umu import _response, encode_teacher_forced, load_umu, parse, select_answer, umu_run_id  # noqa: E402
from vlmu.utils import Logger, atomic_write_json, read_json, read_jsonl, tracked_run, write_jsonl  # noqa: E402

N_SHOTS = 3
SHOT_ROWS = (400, 450)                       # retain_90 rows (sorted by ID) used for the shots: profiles 401-450
FC_PASS, FS_PASS, FC_FAIL, FS_FAIL = 0.85, 0.5, 0.5, 0.2               # DEV-019
V0_FC_MIN, V0_FS_MIN, GAP_MIN, BASE_FC_MAX = 0.90, 0.90, 0.15, 0.60    # DEV-019 validity conditions
FORMS = ("train", "opt")
MODELS = ("vanilla", "unlearned", "base")


def train_prompt(q):
    return f"USER: {q}\nASSISTANT:"


def readout_prompt(q, options):
    return f"USER:\n{q}\nSelect answer in {str(options)}\nASSISTANT:"


def shot_pool(df):
    """Readout items (unimodal question, the multimodal item's options, as vlmu/umu.py:protocol_items) of the shot people."""
    pool = []
    for _, row in df.iterrows():
        cls = parse(row["Classify"])
        uni, mul = cls.get("unimodal", {}), cls.get("muitimodal") or cls.get("multimodal") or {}
        for key, u in uni.items():
            if key not in mul:
                continue
            opts = mul[key]["options"]
            pool.append({"profile": row["ID"], "field": key, "q": u["question"], "options": [opts[k] for k in "ABCD"],
                         "answer": opts[u["answer"].split(".")[0]]})
    return pool


def pick_shots(fact, pool):
    """N_SHOTS items, deterministic per fact: distinct people, distinct attributes, none the target's attribute, and no
    shot answer equal to the target's correct option text."""
    rng = random.Random(fact["pair_id"])
    cand = [s for s in pool if s["field"] != fact["field"] and s["answer"] != fact["answer_text"]]
    rng.shuffle(cand)
    shots, fields, people = [], set(), set()
    for s in cand:
        if s["field"] in fields or s["profile"] in people:
            continue
        shots.append(s)
        fields.add(s["field"])
        people.add(s["profile"])
        if len(shots) == N_SHOTS:
            break
    assert len(shots) == N_SHOTS, f"not enough shot candidates for {fact['pair_id']}"
    assert all(s["answer"] != fact["answer_text"] for s in shots), f"a shot answer equals the target's: {fact['pair_id']}"
    return shots


def fs_prompt(adapter, fact, shots):
    ctx = "".join(readout_prompt(s["q"], s["options"]) + adapter.answer_suffix(s["answer"]) for s in shots)
    return ctx + readout_prompt(fact["q_uni"], fact["options"])


def parse_fs(text):
    return _response(text).split("USER:")[0].strip()


def norm(text):
    """Answer comparison of the DEV-019 amendment: surrounding spaces and one trailing period removed."""
    t = (text or "").strip()
    return t[:-1].strip() if t.endswith(".") else t


def match(response, answer):
    return norm(response) == norm(answer)


def standard_readout(path, dc, sets=("forget", "retain")):
    """{set: accuracy} of the run's standard readout (txt form, doubly-correct facts) under the period rule."""
    rows = [r for r in read_jsonl(path) if r["modality"] == "txt" and r["pair_id"] in dc and r["set"] in sets]
    return {s: float(np.mean([match(r["response"], r["correct_text"]) for r in rows if r["set"] == s]))
            for s in sets if any(r["set"] == s for r in rows)}


def fc_rows(facts):
    rows = []
    for f in facts:
        for form in FORMS:
            prompt = train_prompt(f["q_uni"]) if form == "train" else readout_prompt(f["q_uni"], f["options"])
            for k, o in enumerate(f["options"]):
                rows.append({"uid": f"{f['pair_id']}/{form}/{k}", "prompt": prompt, "correct_text": o, "image_bytes": None})
    return rows


def no_adapter(model, off):
    return model.disable_adapter() if off else contextlib.nullcontext()


@torch.no_grad()
def score_fc(adapter, model, rows, bs, off):
    """{uid: (summed log p of the option's answer tokens, number of answer tokens)}."""
    out = {}
    order = sorted(rows, key=lambda r: len(r["prompt"]) + len(r["correct_text"]))
    for i in range(0, len(order), bs):
        chunk = order[i:i + bs]
        batch = encode_teacher_forced(adapter, chunk)
        n_tok = (batch["labels"][:, 1:] != -100).sum(1).tolist()
        with no_adapter(model, off):
            total, _ = answer_logprobs(adapter, model, batch)
        out.update({r["uid"]: (t, n) for r, t, n in zip(chunk, total, n_tok)})
    return out


@torch.no_grad()
def generate_fs(adapter, model, prompts, bs, off, max_new=50):
    order = sorted(range(len(prompts)), key=lambda i: len(prompts[i]))
    out = [None] * len(prompts)
    for i in range(0, len(order), bs):
        idx = order[i:i + bs]
        batch = adapter.to_device(adapter._process([prompts[j] for j in idx], None, "left"))
        with no_adapter(model, off):
            texts = adapter.generate(model, batch, max_new)
        for j, t in zip(idx, texts):
            out[j] = t
    return out


def record(facts, fc, gen):
    recs = []
    for i, f in enumerate(facts):
        r = {k: f[k] for k in ("pair_id", "set", "profile", "field", "name", "options", "answer_text")}
        r["shots"], r["models"] = f["shots"], {}
        k_true = f["options"].index(f["answer_text"])
        for m in fc:
            res = {}
            for form in FORMS:
                vals = [fc[m][f"{f['pair_id']}/{form}/{k}"] for k in range(4)]
                for tag, score in (("sum", [s for s, _ in vals]), ("mean", [s / max(n, 1) for s, n in vals])):
                    k = int(np.argmax(score))
                    res[f"fc_{form}_{tag}"] = {"choice": k, "correct": f["options"][k] == f["answer_text"],
                                               "margin": float(score[k_true] - max(s for j, s in enumerate(score) if j != k_true))}
            raw = gen[m][i]
            resp = parse_fs(raw)
            res["fs"] = {"raw": raw, "response": resp, "exact": match(resp, f["answer_text"]),
                         "exact_raw": resp == f["answer_text"],
                         "difflib": select_answer(resp, f["options"]) == f["answer_text"],
                         "hit": bool(keyword_hit(f["answer_text"], resp)), "words": len(resp.split())}
            r["models"][m] = res
        recs.append(r)
    return recs


def summarize(recs, std):
    out = {}
    for set_name in ("forget", "retain"):
        rows = [r for r in recs if r["set"] == set_name]
        if not rows:
            continue
        s = {"n_facts": len(rows), "std_u": std.get(set_name)}
        for m in rows[0]["models"]:
            g = lambda key: float(np.mean([r["models"][m][key]["correct"] for r in rows]))  # noqa: E731
            s[m] = {"fc_train": g("fc_train_sum"), "fc_train_mean": g("fc_train_mean"),
                    "fc_opt": g("fc_opt_sum"), "fc_opt_mean": g("fc_opt_mean"),
                    "fs_exact": float(np.mean([r["models"][m]["fs"]["exact"] for r in rows])),
                    "fs_exact_raw": float(np.mean([r["models"][m]["fs"]["exact_raw"] for r in rows])),
                    "fs_difflib": float(np.mean([r["models"][m]["fs"]["difflib"] for r in rows])),
                    "fs_hit": float(np.mean([r["models"][m]["fs"]["hit"] for r in rows])),
                    "fs_words": float(np.mean([r["models"][m]["fs"]["words"] for r in rows]))}
        out[set_name] = s
    return out


def verdict(summary):
    f = summary.get("forget")
    if not f:
        return {"verdict": "no forget facts scored"}
    u, v, std = f["unlearned"], f["vanilla"], f["std_u"]
    gap = v["fs_exact"] - std
    rec = (u["fs_exact"] - std) / gap if gap > 0 else float("nan")
    checks = {"vanilla FC-train >= 0.90": v["fc_train"] >= V0_FC_MIN, "vanilla FS >= 0.90": v["fs_exact"] >= V0_FS_MIN,
              "FS_0 - Std_u >= 0.15": gap >= GAP_MIN,
              "base FC-train <= 0.60": (f["base"]["fc_train"] <= BASE_FC_MAX) if "base" in f else None}
    if any(ok is False for ok in checks.values()):
        word = "NOT INTERPRETABLE"
    elif checks["base FC-train <= 0.60"] is None:
        word = "INCOMPLETE (no base control)"
    elif u["fc_train"] >= FC_PASS and rec >= FS_PASS:
        word = "PASS"
    elif u["fc_train"] <= FC_FAIL and rec <= FS_FAIL:
        word = "FAIL"
    else:
        word = "partial"
    return {"verdict": word, "fc_train_u": u["fc_train"], "fs_recovery": rec, "std_u": std, "fs_0": v["fs_exact"],
            "fs_u": u["fs_exact"], "validity": checks}


def plot(summary, title, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    INK, INK2, MUTED, GRID = "#1f1e1c", "#52514e", "#898781", "#e1e0d9"
    colors = {"vanilla": "#2a78d6", "unlearned": "#eb6834", "base": "#1baf7a"}       # categorical slots 1-3, fixed order
    labels = {"vanilla": "θ0 (UMU vanilla)", "unlearned": "θu (unlearned)", "base": "public base (never saw the people)"}
    groups = [("standard\nreadout", None), ("forced choice\ntraining form", "fc_train"),
              ("forced choice\noptions listed", "fc_opt"), ("3-shot\ngeneration", "fs_exact")]
    sets = [s for s in ("forget", "retain") if s in summary]
    fig, axes = plt.subplots(1, len(sets), figsize=(4.6 * len(sets), 3.2), dpi=200, sharey=True, squeeze=False)
    models = [m for m in MODELS if m in summary[sets[0]]]
    w = 0.8 / len(models)
    for ax, set_name in zip(axes[0], sets):
        s = summary[set_name]
        for j, m in enumerate(models):
            vals = []
            for _, key in groups:
                if key is None:     # the standard readout: vanilla 1.00 by construction, theta_u from the checkpoint
                    vals.append({"vanilla": 1.0, "unlearned": s["std_u"]}.get(m, np.nan))
                else:
                    vals.append(s[m][key])
            x = np.arange(len(groups)) + (j - (len(models) - 1) / 2) * w
            ax.bar(x, vals, width=w - 0.03, color=colors[m], label=labels[m], zorder=2)
            for xi, val in zip(x, vals):
                if not np.isnan(val):
                    ax.text(xi, val + 0.015, f"{val:.2f}", ha="center", va="bottom", fontsize=5.5, color=INK2)
        ax.axhline(0.25, xmin=0.27, xmax=0.73, color=MUTED, lw=0.8, ls=(0, (3, 2)), zorder=1)
        ax.text(1.5, 0.265, "chance (4 options)", ha="center", fontsize=5.5, color=MUTED)
        ax.set_xticks(range(len(groups)), [g for g, _ in groups], fontsize=6.5, color=INK2)
        ax.set_ylim(0, 1.08)
        ax.set_title(f"{set_name} facts (n={s['n_facts']})", fontsize=8, color=INK, loc="left")
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        ax.spines["left"].set_color(GRID)
        ax.spines["bottom"].set_color(GRID)
        ax.grid(axis="y", color=GRID, lw=0.5, zorder=0)
        ax.tick_params(colors=MUTED, labelsize=6.5)
    axes[0][0].set_ylabel("accuracy", fontsize=7, color=INK2)
    axes[0][0].legend(frameon=False, fontsize=6.5, ncol=len(models), loc="lower left", bbox_to_anchor=(0, 1.08))
    fig.suptitle(title, x=0.01, y=1.1, ha="left", fontsize=8.5, color=INK2)
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def main():
    p = base_parser(__doc__)
    p.add_argument("--run-id", default=None)
    p.add_argument("--step", type=int, required=True)
    p.add_argument("--no-base", action="store_true", help="skip the public-base control (the verdict is then incomplete)")
    p.add_argument("--max-facts", type=int, default=0, help="0 = all doubly-correct facts")
    p.add_argument("--batch-txt", type=int, default=32)
    p.add_argument("--gen-batch", type=int, default=16)
    args = p.parse_args()
    cfg = resolve(args)
    run_id = args.run_id or umu_run_id(cfg)
    run_dir = model_dir(cfg) / "umu_unlearn" / run_id
    ckpt = run_dir / "ckpt" / f"step{args.step}"
    if not (ckpt / "adapter_config.json").exists():
        raise SystemExit(f"no adapter at {ckpt}")
    run_cfg = read_json(run_dir / "config.json")
    n_eval, n_ru = run_cfg["umu"]["n_retain_eval"], run_cfg["umu"].get("n_retain_unlearn", 0)
    if n_eval + n_ru > SHOT_ROWS[0]:
        raise SystemExit(f"shot people (rows {SHOT_ROWS}) overlap the run's readout/training people ({n_eval} + {n_ru})")
    std_rec = {r["step"]: r for r in read_jsonl(run_dir / "checkpoints.jsonl")}.get(args.step)
    evals = run_dir / "evals" / f"step{args.step}.jsonl"
    if std_rec is None or not evals.exists():
        raise SystemExit(f"no checkpoint record or generations for step {args.step} in {run_dir}")
    std_recorded = {"forget": std_rec["forget_dc_txt"], "retain": std_rec["retain_dc_txt"]}
    out =results_dir(cfg) / "mech_umu" / run_id / f"step{args.step}" / "style_control"
    log = Logger(out / "log.txt")
    with tracked_run(results_dir(cfg), out, f"mech_umu_style_control_{run_id}_step{args.step}", "mech_umu_style_control"):
        from peft import PeftModel
        dc = set(read_json(run_dir / "baseline_dc.json")["pair_ids"])
        std = standard_readout(evals, dc)
        dfs = load_umu(cfg["paths"]["root"], n_eval, 0, log, cfg["umu"]["forget_split"])
        facts = sorted(facts_for(dfs, dc), key=lambda f: (f["set"], f["pair_id"]))
        if args.max_facts:
            facts = facts[:args.max_facts]
        pool_df = load_umu(cfg["paths"]["root"], SHOT_ROWS[1], 0, log, cfg["umu"]["forget_split"])["retain"].iloc[SHOT_ROWS[0]:]
        pool = shot_pool(pool_df)
        used = {f["profile"] for f in facts}
        assert not used & {s["profile"] for s in pool}, "shot people overlap the scored people"
        for f in facts:
            f["_shots"] = pick_shots(f, pool)
            f["shots"] = [{k: s[k] for k in ("profile", "field", "answer")} for s in f["_shots"]]
        n_f, n_r = sum(f["set"] == "forget" for f in facts), sum(f["set"] == "retain" for f in facts)
        log(f"{run_id} step {args.step}: {n_f} forget + {n_r} retain dc facts (retain readout {n_eval} people); "
            f"standard readout theta_u (period rule) forget {std['forget']:.3f} retain {std['retain']:.3f} (recorded, raw exact: "
            f"{std_recorded['forget']:.3f} / {std_recorded['retain']:.3f}); shot pool {len(pool)} items "
            f"from {pool_df['ID'].iloc[0]}-{pool_df['ID'].iloc[-1]}")

        rows = fc_rows(facts)
        fc, gen = {}, {}
        adapter = get_adapter({**cfg["model"], "id": cfg["umu"]["ckpt"], "processor_id": cfg["model"]["id"]})
        prompts = [fs_prompt(adapter, f, f["_shots"]) for f in facts]
        log("example 3-shot prompt:\n" + prompts[0])
        model = PeftModel.from_pretrained(adapter.load_model(), str(ckpt), is_trainable=False)
        model.eval()
        for m, off in (("unlearned", False), ("vanilla", True)):
            fc[m] = score_fc(adapter, model, rows, args.batch_txt, off)
            gen[m] = generate_fs(adapter, model, prompts, args.gen_batch, off)
            log(f"  {m}: {len(rows)} option scores, {len(prompts)} 3-shot generations")
        model = None
        free()
        if not args.no_base:
            badapter = get_adapter(cfg["model"])
            bmodel = badapter.load_model()
            bmodel.eval()
            fc["base"] = score_fc(badapter, bmodel, rows, args.batch_txt, False)
            gen["base"] = generate_fs(badapter, bmodel, prompts, args.gen_batch, False)
            log(f"  base ({cfg['model']['id']}): done")
            bmodel = None
            free()

        for f in facts:
            f.pop("_shots")
        recs = record(facts, fc, gen)
        write_jsonl(out / "items.jsonl", recs)
        summary = summarize(recs, std)
        v = verdict(summary)
        atomic_write_json(out / "summary.json", {
            "run_id": run_id, "step": args.step, "n_shots": N_SHOTS, "std_recorded_raw_exact": std_recorded, "shot_profiles": [str(pool_df["ID"].iloc[0]), str(pool_df["ID"].iloc[-1])],
            "thresholds": {"fc_pass": FC_PASS, "fs_recovery_pass": FS_PASS, "fc_fail": FC_FAIL, "fs_recovery_fail": FS_FAIL,
                           "vanilla_fc_min": V0_FC_MIN, "vanilla_fs_min": V0_FS_MIN, "gap_min": GAP_MIN, "base_fc_max": BASE_FC_MAX},
            "verdict": v, "summary": summary})
        for set_name, s in summary.items():
            log(f"{set_name}: n={s['n_facts']}, standard readout theta_u {s['std_u']:.3f} (theta_0 1.000 by construction)")
            log(f"  {'model':10s} {'FC-train':>9s} {'(mean)':>7s} {'FC-opt':>7s} {'(mean)':>7s} {'FS':>6s} {'raw':>6s} "
                f"{'difflib':>8s} {'hit':>6s} {'words':>6s}")
            for m in MODELS:
                if m in s:
                    x = s[m]
                    log(f"  {m:10s} {x['fc_train']:9.3f} {x['fc_train_mean']:7.3f} {x['fc_opt']:7.3f} {x['fc_opt_mean']:7.3f} "
                        f"{x['fs_exact']:6.3f} {x['fs_exact_raw']:6.3f} {x['fs_difflib']:8.3f} {x['fs_hit']:6.3f} {x['fs_words']:6.1f}")
        log("examples (first 6 forget facts, 3-shot responses):")
        for r in [r for r in recs if r["set"] == "forget"][:6]:
            log(f"  [{r['name']} / {r['field']}] answer {r['answer_text']!r}: " +
                " | ".join(f"{m} {r['models'][m]['fs']['response'][:50]!r}" for m in MODELS if m in r["models"]))
        log(f"C1 (DEV-019 gate 2): {v['verdict']}; FC-train theta_u {v.get('fc_train_u', float('nan')):.3f}, "
            f"FS recovery {v.get('fs_recovery', float('nan')):.3f}; validity {v.get('validity')}")
        plot(summary, f"{run_id} step {args.step}: accuracy when the response style is controlled", out / "style_control.png")
        log(f"wrote {out / 'summary.json'}")


if __name__ == "__main__":
    main()
