"""Evaluate an external, already fine-tuned checkpoint as a candidate vanilla (no training).

Default: UMU-Bench's released LLaVA-1.5-7B fine-tune `chengyewang/llava_smu_ft` (full merged
weights, fp16; the README calls it the fine-tuned model, i.e. before unlearning). Compared with
the public base `llava-hf/llava-1.5-7b-hf` to separate learned facts from priors.

Subsets (all small, all resumable):
  umu_mc    UMU-Bench `Classify` items (A-D), per profile ~2 fields, three forms:
              txt      unimodal question (name), no image          -> our text route
              img      multimodal question ("this person") + image -> our image route
              txt+img  unimodal question (name) + image            -> UMU's training format for UM_QA
            sets: forget = UMU forget_10 (50 profiles); retain = first N of retain_90; celeb = first N
            of real_person (pretraining knowledge, utility only).
  umu_open  UMU MM_QA / UM_QA short fields (born, birth date, occupation, height, education, salary,
            residence; img also the name), greedy generation, keyword hit; same three forms;
            forget + retain only.
  paired    this repo's paired MC items (MLLMU-based; forget all 50 entities + retain calibration),
            forms img / txt as in the gate, plus img_this (image referent "this person").
            Needs the processed data from scripts/prepare_data.py; skipped if missing.

Prompt formats (the checkpoint's tokenizer carries LLaMA-Factory's vicuna template, so it was most
likely trained in that format; our LlavaAdapter uses MLLMU's format):
  vicuna   "<system> USER: <image>{q} ASSISTANT:"   (LLaMA-Factory llava/vicuna template)
  repo     "USER: <image>\\n{q}\\nASSISTANT:"       (vlmu/models.py LlavaAdapter)

Reading the result:
  ckpt forget ~ ckpt retain, both high       it is a vanilla (not unlearned), knowledge injected
  ckpt umu_mc txt  >= 0.80 and img >= 0.80   usable as vanilla for the T->M gate (checkpoint A)
  txt+img >> txt                              text route works only with the photo present
  paired << umu_mc                            our MLLMU-based items do not match UMU's facts/wording

  python scripts/eval_external.py --config configs/llava/vanilla.yaml
  python scripts/eval_external.py --config configs/llava/vanilla.yaml --runs ckpt:vicuna --n-retain 20
"""
import ast
import collections
import io
import json
import re
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from vlmu.config import base_parser, resolve  # noqa: E402
from vlmu.evaluate import load_items, score, score_open  # noqa: E402
from vlmu.models import LlavaAdapter  # noqa: E402
from vlmu.runs import free, model_dir, processed_dir, results_dir  # noqa: E402
from vlmu.utils import Logger, atomic_write_json, read_jsonl, tracked_run  # noqa: E402

UMU_REPO = "chengyewang/UMU-bench"
UMU_FILES = {"forget": "forget_10", "retain": "retain_90", "celeb": "real_person"}
OPEN_FIELDS = ("Born", "Birth", "Occupation", "Height", "Education", "Annual_Salary", "Residence")
PAIRED_SUITE = [("forget", None, "paired", ("img", "txt")), ("retain", "calibration", "paired", ("img", "txt"))]
REFERENT_SWAP = [("the pictured person's", "this person's"), ("the pictured person", "this person")]
VICUNA_SYSTEM = ("A chat between a curious user and an artificial intelligence assistant. "
                 "The assistant gives helpful, detailed, and polite answers to the user's questions.")


class VicunaLlavaAdapter(LlavaAdapter):
    """LLaMA-Factory llava/vicuna template; <image> directly before the question (as UMU's info_pre.py)."""

    def prompt(self, question, has_image):
        return f"{VICUNA_SYSTEM} USER: {'<image>' if has_image else ''}{question} ASSISTANT:"


FORMATS = {"vicuna": VicunaLlavaAdapter, "repo": LlavaAdapter}


# ---------------------------------------------------------------------------------------- UMU data
def _parse(v):
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


def _letter(ans):
    m = re.match(r"\s*\(?([A-D])\b", str(ans))
    return m.group(1) if m else None


def _clean_ref(ans):
    a = str(ans).strip()
    return a[:-1].strip() if a.endswith(".") else a


def load_umu(root, n_retain, n_celeb, log):
    """{set: dataframe} with at most the requested number of profiles, sorted by ID."""
    import pandas as pd
    from huggingface_hub import hf_hub_download
    out = {}
    for set_name, sub in UMU_FILES.items():
        n = {"forget": None, "retain": n_retain, "celeb": n_celeb}[set_name]
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


def save_image(img, path):
    if path.exists():
        return
    from PIL import Image
    data = img.get("bytes") if isinstance(img, dict) else img
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.open(io.BytesIO(data)).convert("RGB").save(path, quality=95)


def umu_items(dfs, image_root, log):
    """UMU rows -> MC rows (score) and open rows (score_open) in this repo's item format."""
    mc, op = [], []
    bad = collections.Counter()
    for set_name, df in dfs.items():
        for _, row in df.iterrows():
            pid = row["ID"]
            rel = f"images/{set_name}/{pid}.jpg"
            save_image(row["image"], Path(image_root) / rel)
            base = {"set": set_name, "split": "umu", "profile": pid}
            cls = _parse(row["Classify"])
            uni = cls.get("unimodal", {})
            multi = cls.get("multimodal") or cls.get("muitimodal") or {}  # dataset key is misspelled
            for field, u in uni.items():
                ans = _letter(u.get("answer"))
                opts = {k: str(v) for k, v in (u.get("options") or {}).items()}
                if ans not in opts:
                    bad["mc_answer_not_option"] += 1
                    continue
                forms = [("txt", u["question"], None), ("txt+img", u["question"], rel)]
                m = multi.get(field)
                if m and _letter(m.get("answer")) == ans and m.get("options") == u.get("options"):
                    forms.append(("img", m["question"], rel))
                else:
                    bad["mc_no_matching_multimodal"] += 1
                for mod, q, img in forms:
                    mc.append({**base, "uid": f"umu/{set_name}/{pid}/{field}/{mod}", "source": "umu_mc",
                               "modality": mod, "field": field, "pair_id": f"umu/{pid}/{field}",
                               "question": q, "image": img, "options": opts, "answer": ans})
            if set_name == "celeb":
                continue
            mm, um = _parse(row["MM_QA"]), _parse(row["UM_QA"])
            fields = [f for f in OPEN_FIELDS if f in mm.get("question", {}) and f in um.get("question", {})]
            for field in fields + ["Name"]:
                ref = _clean_ref(mm["answer"].get(field, ""))
                if not ref or ref.upper() == "NA":
                    bad["open_no_reference"] += 1
                    continue
                forms = [("img", mm["question"][field], rel)]
                if field != "Name":
                    forms += [("txt", um["question"][field], None), ("txt+img", um["question"][field], rel)]
                for mod, q, img in forms:
                    op.append({**base, "uid": f"umu/{set_name}/{pid}/{field}/{mod}/open", "source": "umu_open",
                               "modality": mod, "field": field, "pair_id": f"umu/{pid}/{field}",
                               "question": q, "image": img, "reference": ref})
    if bad:
        log(f"UMU items skipped: {dict(bad)}")
    log(f"UMU items: {len(mc)} MC, {len(op)} open")
    return mc, op


# ---------------------------------------------------------------------------------------- paired
def paired_items(cfg, log):
    try:
        processed = processed_dir(cfg, log)
    except RuntimeError as e:
        log(f"paired subset skipped: {e}")
        return None, [], []
    items = load_items(processed, PAIRED_SUITE, cfg["eval"].get("max_per_group"))
    extra = []
    for r in items:
        if r["modality"] != "img":
            continue
        q = r["question"]
        for old, new in REFERENT_SWAP:
            if old in q:
                extra.append({**r, "uid": f"{r['uid']}/this", "modality": "img_this", "question": q.replace(old, new)})
                break
    mc = items + extra
    op = [{**r, "uid": f"{r['uid']}/open", "reference": r["options"][r["answer"]]} for r in items]
    log(f"paired items: {len(mc)} MC (incl. {len(extra)} img_this), {len(op)} open")
    return processed, mc, op


def overlap_report(processed, dfs, log):
    """Do UMU forget_10 and this repo's forget split name the same profiles?"""
    if processed is None or "forget" not in dfs:
        return None
    ours = {int(r["profile"]) for r in read_jsonl(Path(processed) / "items_forget.jsonl") if r["source"] == "paired"}
    umu = {int(x) for x in dfs["forget"]["ID"]}
    rep = {"umu_forget_10": len(umu), "repo_forget": len(ours), "intersection": len(umu & ours)}
    log(f"forget profiles: UMU forget_10 {rep['umu_forget_10']}, repo {rep['repo_forget']}, "
        f"shared {rep['intersection']}")
    return rep


# ---------------------------------------------------------------------------------------- scoring
def summarize(rows, metric):
    g = collections.defaultdict(list)
    for r in rows:
        g[f"{r['source']}/{r['set']}/{r['modality']}"].append(float(r[metric]))
    return {k: {"acc": float(np.mean(v)), "n": len(v)} for k, v in sorted(g.items())}


def run_one(cfg, tag, model_id, fmt, subsets, image_roots, out_root, log):
    out = out_root / f"{tag}__{fmt}"
    mcfg = {**cfg["model"], "id": model_id, "processor_id": model_id}
    adapter = FORMATS[fmt](mcfg)
    log(f"=== {tag} ({model_id}), prompt format {fmt}")
    log("example prompt: " + repr(adapter.prompt("<question>", True)))
    model = adapter.load_model()
    summ = {}
    for name, (mc, op) in subsets.items():
        root = image_roots[name]
        if mc:
            summ.update(summarize(score(adapter, model, mc, root, cfg["eval"], out / f"{name}_mc.jsonl", log), "correct"))
        if op:
            summ.update(summarize(score_open(adapter, model, op, root, cfg["eval"], out / f"{name}_open.jsonl", log),
                                  "keyword_hit"))
    model = None
    free()
    atomic_write_json(out / "summary.json", summ)
    return summ


def print_table(all_summ, log):
    cols = list(all_summ)
    keys = sorted({k for s in all_summ.values() for k in s})
    order = {"umu_mc": 0, "umu_open": 1, "paired": 2}
    keys.sort(key=lambda k: (order.get(k.split("/")[0], 9), k))
    log("metric: MC accuracy (umu_mc, paired) / greedy keyword hit (*_open); chance for MC = 0.25")
    log(f"{'subset/set/form':32s} {'n':>5s} " + " ".join(f"{c:>16s}" for c in cols))
    for k in keys:
        n = next(s[k]["n"] for s in all_summ.values() if k in s)
        vals = [f"{all_summ[c][k]['acc']:16.3f}" if k in all_summ[c] else f"{'-':>16s}" for c in cols]
        log(f"{k:32s} {n:5d} " + " ".join(vals))


def main():
    p = base_parser(__doc__)
    p.add_argument("--ckpt", default="chengyewang/llava_smu_ft", help="external fine-tuned checkpoint (HF id or path)")
    p.add_argument("--tag", default="umu_llava_smu_ft", help="output folder name for this checkpoint")
    p.add_argument("--runs", nargs="+", default=["ckpt:vicuna", "ckpt:repo", "base:vicuna"],
                   help="model:format pairs; model in {ckpt, base}, format in {vicuna, repo}")
    p.add_argument("--n-retain", type=int, default=50, help="UMU retain profiles (first N of retain_90)")
    p.add_argument("--n-celeb", type=int, default=50, help="UMU real-person profiles (0 = skip)")
    p.add_argument("--no-paired", action="store_true", help="skip this repo's paired items")
    args = p.parse_args()
    cfg = resolve(args)
    assert cfg["model"]["family"] == "llava", "this script evaluates LLaVA checkpoints: use configs/llava/*.yaml"
    if cfg["debug"]:
        args.n_retain, args.n_celeb = min(args.n_retain, 3), min(args.n_celeb, 3)
    out_root = model_dir(cfg) / "external" / args.tag
    log = Logger(out_root / "log.txt")
    with tracked_run(results_dir(cfg), out_root, f"{cfg['model']['name']}_external_{args.tag}", "eval_external"):
        umu_root = Path(cfg["paths"]["out"]) / "data" / "umu"
        dfs = load_umu(cfg["paths"]["root"], args.n_retain, args.n_celeb, log)
        if cfg["debug"]:
            dfs = {k: v.head(3) for k, v in dfs.items()}
        subsets = {"umu": umu_items(dfs, umu_root, log)}
        image_roots = {"umu": umu_root}
        overlap = None
        if not args.no_paired:
            processed, mc, op = paired_items(cfg, log)
            if processed is not None:
                subsets["paired"], image_roots["paired"] = (mc, op), processed
                overlap = overlap_report(processed, dfs, log)
        models = {"ckpt": args.ckpt, "base": cfg["model"]["id"]}
        all_summ = {}
        for spec in args.runs:
            tag, fmt = spec.split(":")
            all_summ[f"{tag}:{fmt}"] = run_one(cfg, tag, models[tag], fmt, subsets, image_roots, out_root, log)
        atomic_write_json(out_root / "summary.json", {"ckpt": args.ckpt, "base": cfg["model"]["id"],
                                                     "forget_overlap": overlap, "results": all_summ})
        print_table(all_summ, log)
        log(f"wrote {out_root / 'summary.json'}")


if __name__ == "__main__":
    main()
