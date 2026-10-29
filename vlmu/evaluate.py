"""MC evaluation by restricted letter logits (D3). Resumable: per-item results are appended
to a jsonl file and already-scored items are skipped on restart."""
import collections
import time
from pathlib import Path

import numpy as np
import torch

from .utils import append_jsonl, read_jsonl

# Eval suites: (set, source, modalities)
CHECKPOINT_SUITE = [("forget", "paired", ("img", "txt")), ("retain", "paired", ("img", "txt"))]
FULL_SUITE = [
    ("forget", "paired", ("img", "txt", "pose")), ("forget", "official", ("img", "txt")),
    ("retain", "paired", ("img", "txt", "pose")), ("retain", "official", ("img", "txt")),
    ("celeb", "official", ("img", "txt")), ("utility", "mmbench", ("img",)),
]


def load_items(processed, suite, max_per_group=None):
    rows = []
    for set_name in sorted({s for s, _, _ in suite}):
        all_rows = read_jsonl(Path(processed) / f"items_{set_name}.jsonl")
        for s, src, mods in suite:
            if s != set_name:
                continue
            for m in mods:
                g = [r for r in all_rows if r["source"] == src and r["modality"] == m]
                rows += g[:max_per_group] if max_per_group else g
    return rows


def group_key(r):
    return f"{r['set']}/{r['source']}/{r['modality']}"


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
    for mod, rows in by_mod.items():
        rows.sort(key=lambda r: len(r["question"]) + sum(len(v) for v in r["options"].values()))
        bs = ecfg["batch_size_txt"] if mod == "txt" else ecfg["batch_size_img"]
        for i in range(0, len(rows), bs):
            chunk = rows[i:i + bs]
            batch = adapter.encode_mc(chunk, image_root)
            with torch.autocast(device_type=adapter.device.type, dtype=adapter.dtype,
                                enabled=adapter.device.type == "cuda"):
                lp = adapter.letter_scores(model, batch)
            all_letter = torch.tensor(list(letter_ids.values()), device=lp.device)
            p_letters = lp[:, all_letter].exp().sum(-1)
            top1 = lp.argmax(-1)
            res = []
            for j, r in enumerate(chunk):
                opts = list(r["options"].keys())
                scores = {L: lp[j, letter_ids[L]].item() for L in opts}
                pred = max(scores, key=scores.get)
                res.append({"uid": r["uid"], "set": r["set"], "source": r["source"], "modality": r["modality"],
                            "profile": r["profile"], "field": r["field"], "pair_id": r["pair_id"],
                            "answer": r["answer"], "pred": pred, "correct": pred == r["answer"],
                            "n_options": len(opts), "p_letters": round(p_letters[j].item(), 4),
                            "top1_is_option_letter": top1[j].item() in {letter_ids[L] for L in opts},
                            "logp": {k: round(v, 4) for k, v in scores.items()}})
            append_jsonl(out_path, res)
            done.update({x["uid"]: x for x in res})
    log(f"scored {len(todo)} items in {time.time() - t0:.0f}s -> {out_path.name}")
    return [done[r["uid"]] for r in items]


def summarize(results) -> dict:
    groups = collections.defaultdict(list)
    for r in results:
        groups[group_key(r)].append(r)
    out = {}
    for k, rows in sorted(groups.items()):
        out[k] = {"acc": 100.0 * np.mean([r["correct"] for r in rows]), "n": len(rows),
                  "chance": 100.0 * np.mean([1.0 / r["n_options"] for r in rows]),
                  "p_letters": float(np.mean([r["p_letters"] for r in rows])),
                  "top1_is_option_letter": 100.0 * np.mean([r["top1_is_option_letter"] for r in rows])}
    return out
