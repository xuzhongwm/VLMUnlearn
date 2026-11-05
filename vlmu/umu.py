"""UMU-Bench (Wang et al., NeurIPS 2025 D&B; github.com/QDRhhhh/UMU-bench): data loading, the
unimodal unlearning set, and the benchmark's own classification readout.

UMU profiles = MLLMU-Bench's 500 fictitious profiles (re-written) + 153 real persons. Columns are
python-repr strings: MM_QA / UM_QA (fine-tuning QA, 12 per profile; UM_QA names the person, MM_QA
says "this person"), Classify / Cloze / Generation (eval; keys 'unimodal' and the misspelled
'muitimodal').

Readout ("UMU protocol"), re-implemented from eval.py:evaluate_classification (LLaVA branch):
  prompt   USER: <image>\\n{q}\\nSelect answer in ['opt A', 'opt B', 'opt C', 'opt D']\\nASSISTANT:
           (text-only: "USER:\\n{q}..." and no image)
  decode   greedy, max_new_tokens=50, response = text after "ASSISTANT:" (up to a repeated one), stripped
  correct  response == correct option text (exact; forget/retain in their code), plus difflib
           closest-option for reference (their real-person mode)
Scored at batch size 1 this reproduces the paper's Table 1 Origin forget-10% Class AccF (0.823 vs 0.8233,
2026-11-05). Batched (left padding) is used inside training for speed; the step-0 eval of every run is
its own baseline, so all checkpoints of a run share one readout setup.
"""
import ast
import collections
import difflib
import io
import json
import re
import time
from pathlib import Path

import numpy as np

from .utils import append_jsonl, read_jsonl

UMU_REPO = "chengyewang/UMU-bench"
UMU_FILES = {"retain": "retain_90", "celeb": "real_person"}   # + the forget split, e.g. forget_10


# ---------------------------------------------------------------------------------------- parsing
def parse(v):
    """str (json / python repr) or nested dict -> nested dict without None leaves. Parquet struct
    columns carry the union of keys over all rows, so absent fields come back as None."""
    if isinstance(v, str):
        try:
            v = json.loads(v)
        except json.JSONDecodeError:
            v = ast.literal_eval(v)
    return _drop_none(v) if isinstance(v, dict) else {}


def _drop_none(d):
    return {k: _drop_none(x) if isinstance(x, dict) else x for k, x in d.items() if x is not None}


def letter(ans):
    m = re.match(r"\s*\(?([A-D])\b", str(ans))
    return m.group(1) if m else None


def clean_ref(ans):
    a = str(ans).strip()
    return a[:-1].strip() if a.endswith(".") else a


def image_bytes(img):
    return img.get("bytes") if isinstance(img, dict) else img


def save_image(img, path):
    """Original bytes, no re-encoding (PIL detects the format from the content)."""
    path = Path(path)
    if path.exists():
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(image_bytes(img))


# ---------------------------------------------------------------------------------------- data
def load_umu(root, n_retain, n_celeb, log, forget_split="forget_10"):
    """{set: dataframe sorted by ID}; forget = the whole split, retain / celeb = the first N profiles."""
    import pandas as pd
    from huggingface_hub import hf_hub_download
    out = {}
    for set_name, sub, n in (("forget", forget_split, None), ("retain", UMU_FILES["retain"], n_retain),
                             ("celeb", UMU_FILES["celeb"], n_celeb)):
        if n == 0:
            continue
        p = hf_hub_download(UMU_REPO, f"{sub}/train-00000-of-00001.parquet", repo_type="dataset",
                            local_dir=str(Path(root) / "data" / "raw" / "umu"))
        df = pd.read_parquet(p)
        df["ID"] = df["ID"].astype(str).str.zfill(3)
        df = df.sort_values("ID")
        out[set_name] = df if n is None else df.head(n)
        log(f"UMU {sub}: {len(df)} profiles, using {len(out[set_name])}")
    return out


def unlearn_rows(df, form="txt"):
    """UMU's unlearning set for one modality, in this repo's training-row format. txt = UM_QA, no
    image (UMU unlearn_dataset.py:Unimodal_Dataset; its prompt "USER: {q}\\nASSISTANT: {a}" is
    LlavaAdapter's text-only format). img = MM_QA + the portrait (not used yet)."""
    assert form == "txt", "only text-only (T->M) unlearning is wired up"
    rows = []
    for _, row in df.iterrows():
        qa = parse(row["UM_QA"])
        for field, q in qa.get("question", {}).items():
            a = qa.get("answer", {}).get(field)
            if a is None:
                continue
            rows.append({"uid": f"umu_unlearn/{row['ID']}/{field}/{form}", "profile": row["ID"], "field": field,
                         "form": form, "question": q, "answer": str(a), "image": None})
    return rows


# ---------------------------------------------------------------------------------------- UMU protocol
def protocol_items(dfs, sets=("forget", "retain")):
    """Classify items as UMU's eval.py builds them: one img (multimodal) + one txt (unimodal) row per fact."""
    rows = []
    for set_name in sets:
        if set_name not in dfs:
            continue
        for _, row in dfs[set_name].iterrows():
            cls = parse(row["Classify"])
            uni = cls.get("unimodal", {})
            mul = cls.get("muitimodal") or cls.get("multimodal") or {}
            for key in uni:
                if key not in mul:
                    continue
                opts = mul[key]["options"]
                temp = [opts["A"], opts["B"], opts["C"], opts["D"]]
                base = {"set": set_name, "profile": row["ID"], "field": key, "pair_id": f"umu/{row['ID']}/{key}",
                        "temp": temp, "correct_text": opts[uni[key]["answer"].split(".")[0]]}
                rows.append({**base, "uid": f"umuproto/{set_name}/{row['ID']}/{key}/img", "modality": "img",
                             "image_bytes": image_bytes(row["image"]),
                             "prompt": f"USER: <image>\n{mul[key]['question']}\nSelect answer in {str(temp)}\nASSISTANT:"})
                rows.append({**base, "uid": f"umuproto/{set_name}/{row['ID']}/{key}/txt", "modality": "txt",
                             "image_bytes": None,
                             "prompt": f"USER:\n{uni[key]['question']}\nSelect answer in {str(temp)}\nASSISTANT:"})
    return rows


def select_answer(response, temp):
    sims = [difflib.SequenceMatcher(None, response, t).ratio() for t in temp]
    return temp[sims.index(max(sims))]


def _response(new_text):
    """UMU: decode prompt+generation, take split('ASSISTANT:')[1]; on the generated part alone this is
    the text before any repeated 'ASSISTANT:'."""
    return new_text.split("ASSISTANT:")[0].strip()


def generate_protocol(model, processor, rows, out_path, log, batch_size=1, dtype=None):
    """Greedy UMU-protocol generation; resumable (per-batch jsonl append). batch_size (int or
    {'img': n, 'txt': m}) = 1 is UMU exactly; larger batches are left-padded and homogeneous in modality."""
    import torch
    from PIL import Image
    out_path = Path(out_path)
    done = {r["uid"]: r for r in read_jsonl(out_path)}
    todo = [r for r in rows if r["uid"] not in done]
    if not todo:
        return [done[r["uid"]] for r in rows]
    tok = processor.tokenizer
    tok.padding_side = "left"
    device = next(model.parameters()).device
    was_training = model.training
    model.eval()
    t0 = time.time()
    for mod in ("img", "txt"):
        group = sorted([r for r in todo if r["modality"] == mod], key=lambda r: len(r["prompt"]))
        bs = batch_size[mod] if isinstance(batch_size, dict) else batch_size
        for i in range(0, len(group), bs):
            chunk = group[i:i + bs]
            prompts = [r["prompt"] for r in chunk]
            if mod == "img":
                imgs = [Image.open(io.BytesIO(r["image_bytes"])).convert("RGB") for r in chunk]
                inputs = processor(images=imgs, text=prompts, padding=True, return_tensors="pt")
            else:
                inputs = tok(prompts, padding=True, return_tensors="pt")
            inputs = {k: v.to(device) for k, v in inputs.items()}
            if "pixel_values" in inputs:
                inputs["pixel_values"] = inputs["pixel_values"].to(dtype or next(model.parameters()).dtype)
            with torch.no_grad():
                out = model.generate(**inputs, max_new_tokens=50, do_sample=False, num_beams=1, use_cache=True,
                                     pad_token_id=tok.pad_token_id)
            texts = tok.batch_decode(out[:, inputs["input_ids"].shape[1]:], skip_special_tokens=True)
            recs = []
            for r, t in zip(chunk, texts):
                resp = _response(t)
                rec = {k: r[k] for k in ("uid", "set", "modality", "profile", "field", "pair_id", "correct_text")}
                rec.update(response=resp, correct_exact=resp == r["correct_text"],
                           correct_difflib=select_answer(resp, r["temp"]) == r["correct_text"])
                recs.append(rec)
            append_jsonl(out_path, recs)
            done.update({x["uid"]: x for x in recs})
    log(f"UMU protocol: generated {len(todo)} items in {time.time() - t0:.0f}s (batch {batch_size}) -> {out_path.name}")
    if was_training:
        model.train()
    return [done[r["uid"]] for r in rows]


def fact_table(res, judge="correct_exact"):
    """{(set, pair_id): {'txt': bool, 'img': bool}} for facts scored in both forms."""
    facts = collections.defaultdict(dict)
    for r in res:
        facts[(r["set"], r["pair_id"])][r["modality"]] = r[judge]
    return {k: v for k, v in facts.items() if len(v) == 2}


def protocol_summary(res, pair_ids=None):
    """Per judge and set: Acc_uni (txt), Acc_mul (img), Acc_any, Acc_all and the paper's AccF / AccR (Eq. 1).
    With pair_ids, restricted to those facts (e.g. the run's doubly-correct baseline set)."""
    out = {}
    for judge in ("correct_exact", "correct_difflib"):
        facts = fact_table(res, judge)
        for s in sorted({k[0] for k in facts}):
            fs = [v for (ss, pid), v in facts.items() if ss == s and (pair_ids is None or pid in pair_ids)]
            if not fs:
                continue
            uni, mul = float(np.mean([v["txt"] for v in fs])), float(np.mean([v["img"] for v in fs]))
            any_ = float(np.mean([v["txt"] or v["img"] for v in fs]))
            all_ = float(np.mean([v["txt"] and v["img"] for v in fs]))
            out[f"{judge}/{s}"] = {"n_facts": len(fs), "uni": uni, "mul": mul, "any": any_, "all": all_,
                                   "AccF": (uni + mul + any_) / 3, "AccR": (uni + mul + all_) / 3}
    return out
