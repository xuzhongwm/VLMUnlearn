"""Build all processed data from the raw MLLMU-Bench / MMBench files.

Outputs (in `processed/`):
  splits.json              forget / retain_eval / retain_train / celeb IDs
  images/                  portraits, held-out poses, celebrity and MMBench images (raw bytes)
  images.json              path -> sha1, role, profile (for overlap checks)
  train_vanilla.jsonl      FT QA for all 500 profiles, image form (all QA) + text form (name-substituted)
  unlearn_forget.jsonl     forget-profile QA minus the name QA, both forms, same facts
  unlearn_retain.jsonl     retain_train QA minus the name QA, both forms (GA_Diff retain data)
  items_{forget,retain,celeb,utility}.jsonl   MC eval items (one row per item x modality)
  build_report.json        counts, dropped items with reasons, paraphrase-distance check
  STAMP                    build id (used for local caching)
"""
import collections
import hashlib
import io
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

from ..utils import atomic_write_json, now, write_jsonl
from .fields import FIELDS, REFERENTS, acceptable_distractor, classify_question, fact_value, format_class, \
    is_occupation_question
from .rewrite import rewrite_qa, to_text_form

LETTERS = "ABCD"


def stable_int(s: str) -> int:
    return int(hashlib.md5(s.encode()).hexdigest()[:8], 16)


def sha1(b: bytes) -> str:
    return hashlib.sha1(b).hexdigest()


class ImageStore:
    def __init__(self, root: Path):
        self.root = root
        self.manifest = {}

    def put(self, data: bytes, subdir: str, key: str, role: str, profile=None) -> str:
        fmt = (Image.open(io.BytesIO(data)).format or "jpeg").lower()
        ext = {"jpeg": "jpg"}.get(fmt, fmt)
        rel = f"images/{subdir}/{key}.{ext}"
        path = self.root / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        if not path.exists():
            path.write_bytes(data)
        self.manifest[rel] = {"sha1": sha1(data), "role": role, "profile": profile}
        return rel


def ft_qas(metadata_json: str):
    """FT QA list for one profile (the 'Additional' description QA is skipped, as in the
    official loader). Adds the field label and marks the first QA per field."""
    out, seen = [], set()
    for i, qa in enumerate(json.loads(metadata_json)):
        if not qa.get("Question") or not qa.get("Answer"):
            continue
        f = classify_question(qa["Question"])
        primary = f not in seen and (f not in ("father", "mother") or is_occupation_question(qa["Question"]))
        if primary:
            seen.add(f)
        out.append({"idx": i, "field": f, "primary": primary, "question": qa["Question"], "answer": qa["Answer"]})
    return out


def render_stem(field: str, modality: str, name: str) -> str:
    ref = REFERENTS["img"] if modality in ("img", "pose") else {"R": name, "Rp": f"{name}'s"}
    return FIELDS[field]["stem"].format(**ref)


def word_set(q: str) -> set:
    return set(re.findall(r"[a-z<>']+", q.lower()))


def build(cfg, raw_files: dict, out_dir, log=print) -> dict:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    dcfg = cfg["data"]
    seed = dcfg["seed"]
    debug = cfg.get("debug", False)
    report = {"built": now(), "forget_split": dcfg["forget_split"], "debug": debug, "dropped": [],
              "counts": {}}
    drop = lambda **kw: report["dropped"].append(kw)  # noqa: E731

    log("reading parquet files")
    ft = pd.read_parquet(raw_files["ft"])
    full = pd.read_parquet(raw_files["full"])
    celeb = pd.read_parquet(raw_files["celeb"])
    forget_df = pd.read_parquet(raw_files["forget"], columns=["ID"])
    test = pd.concat([pd.read_parquet(p, columns=["ID", "images"]) for p in raw_files["test"]])
    mmb = pd.read_parquet(raw_files["mmbench"])

    # ---------------- splits ----------------
    all_ids = sorted(full["ID"])
    forget = sorted(forget_df["ID"])
    official_forget = sorted(json.load(open(raw_files["forget_ids"])))
    official_retain = sorted(json.load(open(raw_files["retain_ids"])))
    assert forget == official_forget, "forget parquet IDs differ from baseline_train_split forget_ids.json"
    retain = sorted(set(all_ids) - set(forget))
    assert retain == official_retain, "retain IDs differ from baseline_train_split retain_ids.json"
    no_mc = {r.ID for r in full.itertuples() if len(r.Classification_Task["Image_Textual_Questions"]) == 0}
    rng = np.random.default_rng(seed)
    retain_eval = sorted(rng.choice([i for i in retain if i not in no_mc], dcfg["n_retain_eval"], replace=False))
    retain_train = sorted(set(retain) - set(retain_eval))
    celeb_ids = sorted(celeb["ID"])
    if debug:
        forget, retain_eval, retain_train, celeb_ids = forget[:4], retain_eval[:4], retain_train[:10], celeb_ids[:4]
    splits = {"forget": forget, "retain_eval": retain_eval, "retain_train": retain_train, "celeb": celeb_ids,
              "no_official_mc": sorted(no_mc)}
    atomic_write_json(out / "splits.json", splits)

    bios = {r.ID: json.loads(r.biography) for r in full.itertuples()}
    names = {i: b["Name"] for i, b in bios.items()}
    full_by_id = full.set_index("ID")
    ft_by_id = {r.ID: ft_qas(r.metadata) for r in ft.itertuples()}
    ft_img = {r.ID: r.image["bytes"] for r in ft.itertuples()}
    store = ImageStore(out)

    # ---------------- images ----------------
    portraits = {}
    for pid in all_ids:
        portraits[pid] = store.put(full_by_id.loc[pid, "image"]["bytes"], "portraits", pid, "portrait", pid)
        if sha1(ft_img[pid]) != store.manifest[portraits[pid]]["sha1"]:
            drop(kind="warning", profile=pid, reason="ft_image_differs_from_eval_portrait")
    poses = {}
    test_by_id = test.set_index("ID")
    for pid in forget + retain_eval:
        imgs = list(test_by_id.loc[pid, "images"])
        k = stable_int(f"pose/{pid}/{seed}") % len(imgs)
        poses[pid] = store.put(imgs[k]["bytes"], "poses", pid, "pose", pid)

    # ---------------- FT / unlearning data ----------------
    vanilla_profiles = all_ids if not debug else sorted(set(forget + retain_eval + retain_train))
    train_rows, rewrite_fail = [], 0
    unlearn = {"forget": [], "retain": []}
    for pid in vanilla_profiles:
        name = names[pid]
        for qa in ft_by_id[pid]:
            fid = f"{pid}/{qa['idx']}"
            base = {"profile": pid, "field": qa["field"], "fact_id": fid}
            img_row = {**base, "uid": f"ft/{fid}/img", "form": "img", "question": qa["question"],
                       "answer": qa["answer"], "image": portraits[pid]}
            train_rows.append(img_row)
            if qa["field"] == "name":
                continue
            q, a, ok = rewrite_qa(qa["question"], qa["answer"], name)
            if not ok:
                rewrite_fail += 1
                drop(kind="text_rewrite", profile=pid, fact_id=fid, question=qa["question"], rewritten=q)
                continue
            txt_row = {**base, "uid": f"ft/{fid}/txt", "form": "txt", "question": q, "answer": a, "image": None}
            train_rows.append(txt_row)
            # unlearning sets: only facts available in BOTH forms, name QA excluded
            target = "forget" if pid in forget else "retain" if pid in retain_train else None
            if target:
                unlearn[target] += [img_row, txt_row]
    write_jsonl(out / "train_vanilla.jsonl", train_rows)
    write_jsonl(out / "unlearn_forget.jsonl", unlearn["forget"])
    write_jsonl(out / "unlearn_retain.jsonl", unlearn["retain"])

    # ---------------- paired MC items ----------------
    facts = {}  # (pid, field) -> (value, ft_answer)
    for pid in all_ids:
        primary = {qa["field"]: qa["answer"] for qa in ft_by_id[pid] if qa["primary"]}
        for field in FIELDS:
            v, why = fact_value(field, bios[pid], primary.get(field))
            if v is None:
                if pid in forget or pid in retain_eval:
                    drop(kind="paired_item", profile=pid, field=field, reason=why)
                continue
            facts[(pid, field)] = (v, primary[field])
    pools = collections.defaultdict(list)
    for (pid, field), (v, _) in facts.items():
        pools[(field, format_class(field, v))].append((pid, v))

    def paired_items(set_name, ids):
        rows = []
        for pid in ids:
            for field in FIELDS:
                if (pid, field) not in facts:
                    continue
                ans, ft_ans = facts[(pid, field)]
                pair_id = f"paired/{set_name}/{pid}/{field}"
                r = np.random.default_rng(stable_int(pair_id) + seed)
                pool = [v for p, v in pools[(field, format_class(field, ans))] if p != pid]
                if len(pool) < 10:  # rare format class: fall back to all values of the field
                    pool = [v for (f, _), vals in pools.items() if f == field for p, v in vals if p != pid]
                distractors, seen = [], {ans.lower()}
                for j in r.permutation(len(pool)):
                    d = pool[j]
                    if d.lower() not in seen and acceptable_distractor(field, d, ans, ft_ans):
                        distractors.append(d)
                        seen.add(d.lower())
                    if len(distractors) == 3:
                        break
                if len(distractors) < 3:
                    drop(kind="paired_item", profile=pid, field=field, reason="not_enough_distractors")
                    continue
                opts = distractors + [ans]
                order = r.permutation(4)
                options = {LETTERS[k]: opts[o] for k, o in enumerate(order)}
                answer = LETTERS[int(np.where(order == 3)[0][0])]
                for modality, image in (("img", portraits[pid]), ("txt", None), ("pose", poses[pid])):
                    rows.append({"uid": f"{pair_id}/{modality}", "pair_id": pair_id, "source": "paired",
                                 "set": set_name, "profile": pid, "field": field, "modality": modality,
                                 "question": render_stem(field, modality, names[pid]), "options": options,
                                 "answer": answer, "image": image, "fact_answer": ans, "ft_answer": ft_ans})
        return rows

    # ---------------- official MLLMU MC (secondary) ----------------
    def official_items(set_name, df, ids, img_subdir):
        rows = []
        by_id = df.set_index("ID")
        for pid in ids:
            row = by_id.loc[pid]
            image = portraits.get(pid) if img_subdir == "portraits" else \
                store.put(row["image"]["bytes"], img_subdir, pid, img_subdir, pid)
            ct = row["Classification_Task"]
            for modality, key in (("img", "Image_Textual_Questions"), ("txt", "Pure_Text_Questions")):
                for k, q in enumerate(ct[key]):
                    options = {L: str(v) for L, v in dict(q["Options"]).items() if v is not None}
                    rows.append({"uid": f"official/{set_name}/{pid}/{modality}/{k}", "pair_id": None,
                                 "source": "official", "set": set_name, "profile": pid, "field": None,
                                 "modality": modality, "question": q["Question"], "options": options,
                                 "answer": q["Correct_Answer"], "image": image if modality == "img" else None})
        return rows

    items = {
        "forget": paired_items("forget", forget) + official_items("forget", full, forget, "portraits"),
        "retain": paired_items("retain", retain_eval) + official_items("retain", full, retain_eval, "portraits"),
        "celeb": official_items("celeb", celeb, celeb_ids, "celeb"),
    }

    # ---------------- MMBench utility subset ----------------
    n_util = dcfg["n_utility"] if not debug else 20
    sel = mmb.sample(n=n_util, random_state=seed)
    util = []
    for r in sel.itertuples():
        options = {L: str(getattr(r, L)) for L in LETTERS
                   if getattr(r, L) is not None and str(getattr(r, L)).lower() != "nan"}
        hint = "" if r.hint is None or str(r.hint).lower() == "nan" else f"{r.hint}\n"
        util.append({"uid": f"mmbench/{r.index}", "pair_id": None, "source": "mmbench", "set": "utility",
                     "profile": None, "field": r.category, "modality": "img", "question": hint + r.question,
                     "options": options, "answer": r.answer,
                     "image": store.put(r.image["bytes"], "mmbench", str(r.index), "mmbench")})
    items["utility"] = util
    for k, rows in items.items():
        write_jsonl(out / f"items_{k}.jsonl", rows)
    atomic_write_json(out / "images.json", store.manifest)

    # ---------------- paraphrase check (D8) ----------------
    ft_q_by_field = collections.defaultdict(set)
    for pid, qas in ft_by_id.items():
        for qa in qas:
            ft_q_by_field[qa["field"]].add(to_text_form(qa["question"], "<R>").lower().rstrip("?").strip())
    para = {}
    for field, spec in FIELDS.items():
        stem = spec["stem"].format(R="<R>", Rp="<R>'s").lower().rstrip("?").strip()
        sims = [len(word_set(stem) & word_set(q)) / len(word_set(stem) | word_set(q)) for q in ft_q_by_field[field]]
        para[field] = {"stem": stem, "exact_match_with_ft_question": stem in ft_q_by_field[field],
                       "max_jaccard_vs_ft_questions": round(max(sims), 3) if sims else None,
                       "n_ft_question_variants": len(ft_q_by_field[field])}
    report["paraphrase_check"] = para
    if any(v["exact_match_with_ft_question"] for v in para.values()):
        raise AssertionError(f"eval stem identical to an FT question: {para}")

    c = report["counts"]
    c["splits"] = {k: len(v) for k, v in splits.items()}
    c["train_vanilla"] = collections.Counter(r["form"] for r in train_rows)
    c["unlearn_forget"] = collections.Counter(r["form"] for r in unlearn["forget"])
    c["unlearn_retain"] = collections.Counter(r["form"] for r in unlearn["retain"])
    c["text_rewrite_failures"] = rewrite_fail
    for k, rows in items.items():
        c[f"items_{k}"] = collections.Counter(f"{r['source']}/{r['modality']}" for r in rows)
    c["dropped_by_reason"] = collections.Counter(f"{d['kind']}:{d.get('reason', '')}" for d in report["dropped"])
    atomic_write_json(out / "build_report.json", report)
    (out / "STAMP").write_text(report["built"])
    log("build done:", json.dumps(c, default=dict))
    return report
