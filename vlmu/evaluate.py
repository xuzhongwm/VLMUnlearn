"""MC evaluation by A-D letter logits. Resumable: per-item results are appended to a jsonl
file and already-scored items are skipped on restart.

Per item: prediction (argmax over the option letters), margin
m = z_correct - logsumexp(z_wrong) over the letter logits, probability mass on A-D, and the
chosen option text (for the option-permutation check).

Open-ended scoring (score_open) uses the same paired facts without options: the question stem
is the prompt and the correct option text is the reference answer. Per item:
  - answer log-prob under teacher forcing: per-token mean and sum over the answer tokens plus
    the end-of-turn token (the same tokens the training loss covers);
  - greedy generation (max_new_tokens): keyword hit = the normalised reference occurs in the
    normalised generation; ROUGE-L recall = LCS(reference, generation) / len(reference) over
    normalised whitespace tokens.
"""
import collections
import re
import time
from pathlib import Path

import numpy as np
import torch

from .utils import append_jsonl, read_jsonl

LETTERS = "ABCD"

# Suites: (set, split or None for all, source, modalities)
CHECKPOINT_SUITE = [  # calibration entities only: all selection happens here
    ("forget", "calibration", "paired", ("img", "txt")),
    ("retain", "calibration", "paired", ("img", "txt")),
]
FULL_SUITE = [
    ("forget", None, "paired", ("img", "txt", "pose")), ("forget", None, "official", ("img", "txt")),
    ("retain", None, "paired", ("img", "txt", "pose")), ("retain", None, "official", ("img", "txt")),
    ("celeb", None, "official", ("img", "txt")), ("utility", None, "mmbench", ("img",)),
]
PRIOR_SUITE = [  # pretrained (no FT) priors: only what scripts/sanity.py:per_field reads, original option order
    ("forget", None, "paired", ("img", "txt", "pose")),
]
SHIFTED = ("img", "txt")  # paired primary forms re-scored under the 3 non-trivial cyclic shifts
OPEN_SUITE = [  # open-ended scoring: paired primary forms, forget + retain eval entities
    ("forget", None, "paired", ("img", "txt")), ("retain", None, "paired", ("img", "txt")),
]


def cyclic_shift(row, k):
    """Options rotated by k: new option j = old option (j+k) mod n; answer letter remapped."""
    letters = list(row["options"].keys())
    vals = [row["options"][L] for L in letters]
    n = len(vals)
    new = {letters[j]: vals[(j + k) % n] for j in range(n)}
    c = letters.index(row["answer"])
    return {**row, "uid": f"{row['uid']}/shift{k}", "options": new, "answer": letters[(c - k) % n], "shift": k}


def load_items(processed, suite, max_per_group=None, shifts=False, pair_ids=None):
    rows = []
    for set_name in sorted({s[0] for s in suite}):
        all_rows = read_jsonl(Path(processed) / f"items_{set_name}.jsonl")
        for s, split, src, mods in suite:
            if s != set_name:
                continue
            for m in mods:
                g = [r for r in all_rows if r["source"] == src and r["modality"] == m
                     and (split is None or r.get("split") == split)
                     and (pair_ids is None or src != "paired" or r["pair_id"] in pair_ids)]
                if max_per_group:  # debug cap, applied per split so every split stays represented
                    by_split = {}
                    for r in g:
                        by_split.setdefault(r.get("split"), []).append(r)
                    g = [r for v in by_split.values() for r in v[:max_per_group]]
                rows += g
    for r in rows:
        r.setdefault("shift", 0)
    if shifts:
        rows += [cyclic_shift(r, k) for r in list(rows)
                 if r["source"] == "paired" and r["modality"] in SHIFTED for k in (1, 2, 3)]
    return rows


def group_key(r):
    k = f"{r['set']}/{r.get('split', 'all')}/{r['source']}/{r['modality']}"
    return k + (f"/shift{r['shift']}" if r.get("shift") else "")


@torch.no_grad()
def score(adapter, model, items, image_root, ecfg, out_path, log=print):
    out_path = Path(out_path)
    done = {r["uid"]: r for r in read_jsonl(out_path)}
    todo = [r for r in items if r["uid"] not in done]
    if not todo:
        return [done[r["uid"]] for r in items]
    model.eval()
    t0 = time.time()
    by_mod = collections.defaultdict(list)
    for r in todo:
        by_mod["txt" if r["image"] is None else "img"].append(r)
    letter_ids = adapter.letter_ids
    all_letter = None
    for mod, rows in by_mod.items():
        rows.sort(key=lambda r: len(r["question"]) + sum(len(v) for v in r["options"].values()))
        bs = ecfg["batch_size_txt"] if mod == "txt" else ecfg["batch_size_img"]
        for i in range(0, len(rows), bs):
            chunk = rows[i:i + bs]
            batch = adapter.encode_mc(chunk, image_root)
            with adapter.autocast():
                lp = adapter.letter_scores(model, batch)
            if all_letter is None:
                all_letter = torch.tensor([letter_ids[L] for L in LETTERS], device=lp.device)
            p_letters = lp[:, all_letter].exp().sum(-1)
            top1 = lp.argmax(-1)
            res = []
            for j, r in enumerate(chunk):
                opts = list(r["options"].keys())
                scores = {L: lp[j, letter_ids[L]].item() for L in opts}
                pred = max(scores, key=scores.get)
                wrong = torch.tensor([scores[L] for L in opts if L != r["answer"]], dtype=torch.float64)
                margin = scores[r["answer"]] - torch.logsumexp(wrong, 0).item()
                res.append({"uid": r["uid"], "set": r["set"], "split": r.get("split", "all"), "source": r["source"],
                            "modality": r["modality"], "shift": r.get("shift", 0), "profile": r["profile"],
                            "field": r["field"], "pair_id": r["pair_id"], "answer": r["answer"], "pred": pred,
                            "pred_text": r["options"][pred], "correct": pred == r["answer"],
                            "margin": round(margin, 4), "n_options": len(opts),
                            "p_letters": round(p_letters[j].item(), 4),
                            "top1_is_option_letter": top1[j].item() in {letter_ids[L] for L in opts},
                            "logp": {k: round(v, 4) for k, v in scores.items()}})
            append_jsonl(out_path, res)
            done.update({x["uid"]: x for x in res})
    log(f"scored {len(todo)} items in {time.time() - t0:.0f}s -> {out_path.name}")
    return [done[r["uid"]] for r in items]


def summarize(results, pair_ids=None) -> dict:
    """Accuracy etc. per group. With pair_ids, paired groups are restricted to those pairs
    (e.g. the frozen doubly-correct set)."""
    groups = collections.defaultdict(list)
    for r in results:
        if pair_ids is not None and r["source"] == "paired" and r["pair_id"] not in pair_ids:
            continue
        groups[group_key(r)].append(r)
    out = {}
    for k, rows in sorted(groups.items()):
        out[k] = {"acc": float(np.mean([r["correct"] for r in rows])), "n": len(rows),
                  "chance": float(np.mean([1.0 / r["n_options"] for r in rows])),
                  "margin": float(np.mean([r["margin"] for r in rows])),
                  "p_letters": float(np.mean([r["p_letters"] for r in rows])),
                  "top1_is_option_letter": float(np.mean([r["top1_is_option_letter"] for r in rows]))}
    return out


def permutation_check(results, pair_ids=None) -> dict:
    """Accuracy per cyclic shift and agreement (same underlying option chosen under all 4
    shifts), per (set, split, modality) of the paired primary forms."""
    by_item = collections.defaultdict(dict)
    for r in results:
        if r["source"] != "paired" or r["modality"] not in SHIFTED:
            continue
        if pair_ids is not None and r["pair_id"] not in pair_ids:
            continue
        base = r["uid"].split("/shift")[0]
        by_item[(r["set"], r["split"], r["modality"], base)][r["shift"]] = r
    out = collections.defaultdict(lambda: {"n": 0, "acc_by_shift": collections.defaultdict(list), "agree": []})
    for (s, split, m, _), shifts in by_item.items():
        if len(shifts) != 4:
            continue
        g = out[f"{s}/{split}/{m}"]
        g["n"] += 1
        for k, r in shifts.items():
            g["acc_by_shift"][k].append(r["correct"])
        g["agree"].append(len({r["pred_text"] for r in shifts.values()}) == 1)
    return {k: {"n": v["n"], "acc_by_shift": {str(s): float(np.mean(x)) for s, x in sorted(v["acc_by_shift"].items())},
                "agreement_all_4_shifts": float(np.mean(v["agree"])) if v["agree"] else None}
            for k, v in sorted(out.items())}


# ---------------------------------------------------------------------------------------- open-ended
def _norm(text):
    text = re.sub(r"[^\w\s]", "", text.lower().replace("-", " "))
    return " ".join(text.split())


def rouge_l_recall(ref, hyp):
    r, h = _norm(ref).split(), _norm(hyp).split()
    if not r:
        return float("nan")
    prev = [0] * (len(h) + 1)
    for a in r:
        cur = [0]
        for j, b in enumerate(h):
            cur.append(prev[j] + 1 if a == b else max(prev[j + 1], cur[j]))
        prev = cur
    return prev[-1] / len(r)


def keyword_hit(ref, hyp):
    return bool(_norm(ref)) and f" {_norm(ref)} " in f" {_norm(hyp)} "


def open_items(items):
    """Shift-0 paired items -> open-ended rows (reference answer = correct option text)."""
    return [{**r, "uid": f"{r['uid']}/open", "reference": r["options"][r["answer"]]}
            for r in items if r["source"] == "paired" and r.get("shift", 0) == 0]


@torch.no_grad()
def score_open(adapter, model, items, image_root, ecfg, out_path, log=print):
    """Resumable like score(); items come from open_items()."""
    out_path = Path(out_path)
    done = {r["uid"]: r for r in read_jsonl(out_path)}
    todo = [r for r in items if r["uid"] not in done]
    if not todo:
        return [done[r["uid"]] for r in items]
    model.eval()
    t0 = time.time()
    by_mod = collections.defaultdict(list)
    for r in todo:
        by_mod["txt" if r["image"] is None else "img"].append(r)
    max_new = ecfg.get("max_new_tokens", 32)
    for mod, rows in by_mod.items():
        rows.sort(key=lambda r: len(r["question"]))
        bs = ecfg["batch_size_txt"] if mod == "txt" else ecfg["batch_size_img"]
        for i in range(0, len(rows), bs):
            chunk = rows[i:i + bs]
            with adapter.autocast():
                tf = adapter.encode_train([{**r, "answer": r["reference"]} for r in chunk], image_root)
                _, lp_sum, n_tok = adapter.answer_logprobs(model, tf)
                gens = adapter.generate(model, adapter.encode_open(chunk, image_root), max_new)
            res = []
            for j, r in enumerate(chunk):
                res.append({"uid": r["uid"], "set": r["set"], "split": r.get("split", "all"), "source": r["source"],
                            "modality": r["modality"], "shift": 0, "profile": r["profile"], "field": r["field"],
                            "pair_id": r["pair_id"], "reference": r["reference"], "generation": gens[j],
                            "answer_logprob_sum": round(lp_sum[j].item(), 4),
                            "answer_logprob_mean": round(lp_sum[j].item() / max(1, n_tok[j].item()), 4),
                            "n_answer_tokens": int(n_tok[j].item()),
                            "keyword_hit": keyword_hit(r["reference"], gens[j]),
                            "rouge_l_recall": round(rouge_l_recall(r["reference"], gens[j]), 4)})
            append_jsonl(out_path, res)
            done.update({x["uid"]: x for x in res})
    log(f"open-scored {len(todo)} items in {time.time() - t0:.0f}s -> {out_path.name}")
    return [done[r["uid"]] for r in items]


def summarize_open(results, pair_ids=None) -> dict:
    groups = collections.defaultdict(list)
    for r in results:
        if pair_ids is not None and r["pair_id"] not in pair_ids:
            continue
        groups[group_key(r)].append(r)
    return {k: {"n": len(rows), **{f: float(np.mean([r[f] for r in rows]))
                                   for f in ("keyword_hit", "rouge_l_recall", "answer_logprob_mean", "answer_logprob_sum")}}
            for k, rows in sorted(groups.items())}
