"""E0 data audit: neutral facts about MLLMU-Bench as distributed, plus a per-attribute overlap
audit of the official image+text vs text-only MC questions. CPU only; reads ROOT/data/raw.

Writes analysis/data_audit.md and analysis/official_mc_attributes.csv (one row per official
MC question with its attribute label, for review).

  python scripts/data_audit.py [--root ROOT]
"""
import argparse
import collections
import csv
import hashlib
import json
import os
import re
import sys
from pathlib import Path

import pandas as pd

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))
from vlmu.config import DEFAULT_ROOT  # noqa: E402
from vlmu.data.fields import FIELDS, classify_question, fact_value, is_occupation_question  # noqa: E402
from vlmu.data.rewrite import residue, to_text_form  # noqa: E402

# Attribute taxonomy for official MC questions (rules over the question; the correct answer is
# used only to split father/mother questions into name vs occupation and pet into type vs name).
RULES = [
    ("parents", r"\bfather\b|\bdad\b|\bmother\b|\bmom\b|\bparents?\b"),
    ("pet", r"\bpet\b|\bpets\b|\bcat\b|\bdog\b|companion|\banimal\b.*\b(own|have|has|keep)\b"),
    ("dob", r"date of birth|birth ?date|birthday|when was .* born|\bage\b|how old|year .* born"),
    ("birthplace", r"\bborn\b|birthplace|place of birth|hometown|originally from"),
    ("residence", r"resid|\blive\b|\blives\b|living|located|based in"),
    ("salary", r"salary|income|\bearn"), ("height", r"height|\btall\b"),
    ("gender", r"gender|\bsex\b"), ("medical", r"medical|health|condition|allerg"),
    ("works_achievements", r"film|movie|series|tv show|album|song|appeared in|award|famous|known for|notable work|competition"),
    ("education", r"educat|universit|college|school|graduat|degree|stud(y|ied)|\battend|alma mater|institution"),
    ("hobby_leisure", r"skill|hobb|free time|weekend|leisure|pastime|passion|instrument|sport|recreation|outdoor|wildlife|collect"),
    ("profession", r"profession|occupation|\bjob\b|employ|career|\bwork|for a living|role"),
    ("food", r"food|cuisine|dish|\beat\b|meal|cook"),
    ("hobby_leisure", r"enjoy|activit|interest|play\b|travel"),
    ("name", r"\bname\b|who is (this|the) (person|individual)"),
    ("appearance", r"wear|hair|physical|look|appear|feature|cloth|glasses|tattoo|eye"),
    ("personality", r"characteristic|personality|trait|describe|lifestyle|nature of"),
]
_JOB = re.compile(r"teacher|engineer|manager|doctor|nurse|professor|artist|designer|librarian|owner|scientist|chef|"
                  r"lawyer|accountant|architect|pharmacist|director|developer|consultant|educator|writer|musician|"
                  r"farmer|officer|analyst|technician|curator|surgeon|dentist|biologist|journalist|banker", re.I)


def attribute(question: str, answer: str):
    s = question.lower()
    for label, pat in RULES:
        if re.search(pat, s):
            if label == "parents":
                who = "father" if re.search(r"father|dad", s) else "mother" if re.search(r"mother|mom", s) else "parents"
                is_name = not _JOB.search(answer) and (re.search(r"\bname\b|who (is|are)", s) is not None)
                return label, f"{who}_{'name' if is_name else 'occupation'}"
            if label == "pet":
                return label, "pet_name" if re.search(r"\bname\b", s) else "pet_type"
            return label, label
    return "other", "other"


# Manual review of attribute labels (reviewer: Claude, the coding assistant — not a human annotator).
SPOT_CHECK = """Labels were reviewed against the question text (and correct answer) on random samples:

- Round 1 (`random.seed(42)`, 60 fictitious + 20 celebrity questions, rules before revision): 54/60 and 14/20 judged
  correct. Error types: no 'parents' category ("professions of X's parents", "parents of X"); "companion during
  activities" (a pet) labelled hobby; "skill in addition to their profession" labelled profession; capitalised job
  titles taken as parent names; celebrity film/role questions had no category. The rules were revised once in response.
- Round 2 (`random.seed(2026)`, a fresh sample, final rules): fictitious 58/60 correct at the coarse attribute level
  used for the overlap figures (57/60 fine-grained); errors: "physical stature" -> appearance (should be height),
  "where likely based, considering their profession" -> profession (should be residence), "Who is X's mother? ->
  Realtor" -> mother_name (should be occupation; coarse label correct). Celebrities 29/30 (error: "type of vehicle
  known for riding" -> works_achievements).

The overlap percentages in §5 therefore carry label noise of roughly a few percent of questions.
"""


def norm_tokens(s):
    return {t for t in re.findall(r"[a-z0-9]+", s.lower()) if len(t) > 1}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--root", default=None)
    a = p.parse_args()
    raw = Path(a.root or os.environ.get("VLMU_ROOT") or DEFAULT_ROOT) / "data" / "raw"
    M = raw / "MLLMU-Bench"
    ft = pd.read_parquet(M / "ft_Data/train-00000-of-00001.parquet")
    full = pd.read_parquet(M / "Full_Set/train-00000-of-00001.parquet")
    celeb = pd.read_parquet(M / "Retain_Set/train-00000-of-00001.parquet")
    f10 = set(pd.read_parquet(M / "forget_10/train-00000-of-00001.parquet", columns=["ID"])["ID"])
    test = pd.concat([pd.read_parquet(x, columns=["ID", "images"]) for x in sorted((M / "Test_Set").glob("*.parquet"))])
    L = []  # markdown lines
    w = L.append

    # ---------------- vanilla training data
    n_qa, n_add, ref = collections.Counter(), 0, collections.Counter()
    field_pos = collections.defaultdict(collections.Counter)
    for r in ft.itertuples():
        meta = json.loads(r.metadata)
        qas = [q for q in meta if q.get("Question")]
        n_add += sum(1 for q in meta if "Additional_question" in q)
        n_qa[len(qas)] += 1
        for i, q in enumerate(qas):
            field_pos[i][classify_question(q["Question"])] += 1
            for m in re.finditer(r"\b(this person|the person in the image|the person|this individual|the individual)\b",
                                 q["Question"], re.I):
                ref[m.group(1).lower()] += 1
            if not re.search(r"\b(person|individual|image)\b", q["Question"], re.I):
                ref["(no person/image referent)"] += 1
    sha = lambda b: hashlib.sha1(b).hexdigest()  # noqa: E731
    full_sha = {r.ID: sha(r.image["bytes"]) for r in full.itertuples()}
    same_img = sum(full_sha[r.ID] == sha(r.image["bytes"]) for r in ft.itertuples())
    w("# E0 data audit — MLLMU-Bench (facts only)\n")
    w("Generated by `scripts/data_audit.py` from the HF files in `ROOT/data/raw` (dataset `MLLMMU/MLLMU-Bench`, "
      "ID lists from `MLLMMU/baseline_train_split`). Code facts refer to github.com/franciscoliu/MLLMU-Bench at commit "
      "`0e3098e5ea7e4878ce4f19cdf873de1410b4d427` (2025-03-12).\n")
    w("## 1. Vanilla training data (`ft_Data`)\n")
    w(f"- Rows: {len(ft)} profiles; columns: {list(ft.columns)}.")
    w(f"- QA pairs with `Question`/`Answer` per profile: {dict(sorted(n_qa.items()))} (count of profiles).")
    w(f"- Each profile additionally has one `Additional_question`/`Additional_answer` entry ({n_add} in total). "
      "The official loader skips entries without `Question` (`data_process/data_preprocess.py:63`), so these are not used in official fine-tuning.")
    w(f"- Every training example has an image: the profile portrait. Portrait bytes equal the `Full_Set` eval portrait for "
      f"{same_img}/{len(ft)} profiles.")
    w("- Referring expressions in training questions (counts of occurrences): "
      + ", ".join(f"'{k}': {v}" for k, v in ref.most_common()) + ".")
    w("- No training question contains a profile name instead of a referent; there is no text-only training form.")
    w("- Field order by position (most frequent field label at each position, rule-based classifier `vlmu/data/fields.py`):")
    for i in sorted(field_pos)[:12]:
        top, c = field_pos[i].most_common(1)[0]
        w(f"  - position {i}: {top} ({c}/{sum(field_pos[i].values())})")
    w("- Positions >= 12 hold 2–7 free-form 'fun fact' questions per profile (pets, food, hobbies, ...).\n")

    # ---------------- splits
    no_mc = sorted(r.ID for r in full.itertuples() if len(r.Classification_Task["Image_Textual_Questions"]) == 0)
    f5 = set(pd.read_parquet(M / "forget_5/train-00000-of-00001.parquet", columns=["ID"])["ID"]) \
        if (M / "forget_5/train-00000-of-00001.parquet").exists() else None
    names = collections.Counter(json.loads(r.biography)["Name"] for r in full.itertuples())
    w("## 2. Splits and profiles\n")
    w(f"- `Full_Set`: {len(full)} fictitious profiles; `Retain_Set`: {len(celeb)} real celebrities; `forget_10`: {len(f10)} profiles.")
    if f5 is not None:
        w(f"- `forget_5` ({len(f5)}) is {'a subset' if f5 <= f10 else 'not a subset'} of `forget_10` (overlap {len(f5 & f10)}).")
    else:
        w("- `forget_5` is not downloaded by this pipeline; checked on 2026-10-29 via the HF datasets-server API: "
          "its 25 IDs are not a subset of `forget_10`.")
    w(f"- Profiles without official MC questions: {no_mc}.")
    w(f"- Duplicate profile names among the 500: {sum(1 for v in names.values() if v > 1)}.")
    tsha = {r.ID: [sha(x['bytes']) for x in r.images] for r in test.itertuples()}
    n_pose = collections.Counter(len(v) for v in tsha.values())
    clash = sum(1 for pid, hs in tsha.items() if full_sha.get(pid) in hs)
    w(f"- `Test_Set`: {len(tsha)} profiles; images per profile {dict(n_pose)}; profiles whose Test_Set images include "
      f"the training portrait (byte-identical): {clash}.\n")

    # ---------------- biography vs FT answers
    w("## 3. Biography fields vs fine-tuning answers\n")
    w("For each of the 9 fields used by the paired items, whether the biography value is stated by the profile's FT answer "
      "(rule-based check `vlmu/data/fields.py:supported`; parent fields are compared by token overlap of the FT-answer "
      "occupation with the serialised `Parents` field).\n")
    w("| field | profiles with FT QA | biography value missing/NA | stated by FT answer | not stated |")
    w("|---|---|---|---|---|")
    bios = {r.ID: json.loads(r.biography) for r in full.itertuples()}
    ftq = {}
    for r in ft.itertuples():
        seen = {}
        for q in json.loads(r.metadata):
            if not q.get("Question"):
                continue
            f = classify_question(q["Question"])
            if f in ("father", "mother") and not is_occupation_question(q["Question"]):
                continue
            seen.setdefault(f, q["Answer"])
        ftq[r.ID] = seen
    for field in FIELDS:
        have = [pid for pid in bios if field in ftq[pid]]
        miss = stated = 0
        for pid in have:
            if field in ("father", "mother"):
                occ, _ = fact_value(field, bios[pid], ftq[pid][field])
                par = json.dumps(bios[pid].get("Parents", "")).lower()
                if occ is None:
                    miss += 1
                elif norm_tokens(occ) and len(norm_tokens(occ) & norm_tokens(par)) / len(norm_tokens(occ)) >= 0.6:
                    stated += 1
            else:
                v, why = fact_value(field, bios[pid], ftq[pid][field])
                if why == "missing_or_na":
                    miss += 1
                elif v is not None:
                    stated += 1
        w(f"| {field} | {len(have)} | {miss} | {stated} | {len(have) - miss - stated} |")
    w("")

    # ---------------- text rewrite
    fails = 0
    for r in ft.itertuples():
        name = bios[r.ID]["Name"]
        for q in json.loads(r.metadata):
            if q.get("Question") and classify_question(q["Question"]) != "name":
                t = to_text_form(q["Question"], name)
                if residue(t) or name not in t:
                    fails += 1
    w(f"Name substitution ('this person' -> name) leaves an image/person reference or no name in {fails} non-name training "
      "questions (see `build_report.json`).\n")

    # ---------------- official MC
    rows = []
    for df, set_name in ((full, "fictitious"), (celeb, "celebrity")):
        for r in df.itertuples():
            ct = r.Classification_Task
            for mod, key in (("img", "Image_Textual_Questions"), ("txt", "Pure_Text_Questions")):
                for k, q in enumerate(ct[key]):
                    opts = {kk: str(v) for kk, v in dict(q["Options"]).items() if v is not None}
                    ans = opts.get(q["Correct_Answer"], str(q["Correct_Answer"]))
                    coarse, fine = attribute(q["Question"], ans)
                    rows.append({"set": set_name, "profile": r.ID, "forget_10": r.ID in f10, "modality": mod, "k": k,
                                 "question": q["Question"], "correct_answer": ans, "answer_is_option": q["Correct_Answer"] in opts,
                                 "attribute": coarse, "attribute_fine": fine})
    out_csv = REPO / "analysis" / "official_mc_attributes.csv"
    out_csv.parent.mkdir(exist_ok=True)
    with open(out_csv, "w", newline="") as f:
        cw = csv.DictWriter(f, fieldnames=list(rows[0]))
        cw.writeheader()
        cw.writerows(rows)
    w("## 4. Official evaluation protocol (code facts)\n")
    w("- Released fine-tuned checkpoints on the HF Hub (org `MLLMMU`): `LLaVA_Vanilla` (llava-1.5-7b-hf architecture), "
      "`Idefics2_Vanilla`. None for Qwen2-VL / Qwen2.5-VL. Not used in this project.")
    w("- Reported (not measured here) in the MLLMU-Bench paper, Table 2: vanilla LLaVA-1.5-7B classification accuracy, "
      "modalities pooled, 5% forget: forget 51.70%, test 47.86%, retain 46.11%, real celebrity 51.80%.")
    w("- MC scoring is generative: greedy decoding, the first character of the reply must be one of the option letters, "
      "otherwise the item is counted wrong (`eval.py:376`, `eval.py:432`).")
    w("- Forget/retain/test MC prompts include few-shot demonstrations from 1 profile sampled with `random.sample(id_list, 1)` "
      "from the IDs of the set being evaluated (`eval.py:263-265`); the demo profile's own questions are skipped. "
      "`eval.py` contains no `random.seed` call. Celebrity MC is zero-shot (`eval.py:344`, `eval.py:398`).")
    w("- Fine-tuning/unlearning labels are the full sequence with only padding masked (`data_process/data_preprocess.py:373`).")
    w("- NPO default `--beta 0.4` (`baselines/NPO.py:299`) with a separately fine-tuned retain-only reference model "
      "(`--oracle_model_id`). GA_Diff iterates a full forget pass, then a full retain pass, per epoch "
      "(`baselines/GA_Difference.py:360-381`).")
    n_bad = sum(not r["answer_is_option"] for r in rows)
    w(f"- Official MC questions whose `Correct_Answer` is not one of the option letters: {n_bad} "
      + "(" + "; ".join(f"profile {r['profile']} {r['modality']}: '{r['question']}' -> '{r['correct_answer']}'" for r in rows if not r["answer_is_option"]) + ").\n")

    w("## 5. Official MC: per-attribute overlap of image+text vs text-only questions\n")
    w("**Method.** Each official MC question is assigned one attribute by ordered rules over the question wording "
      "(`scripts/data_audit.py:RULES`; father/mother questions are split into name vs occupation using the correct answer, "
      "pet questions into type vs name). The labels for every question are in `analysis/official_mc_attributes.csv`. "
      "For each profile, the attribute sets of its image+text and text-only questions are compared. "
      "A label-accuracy spot check is reported in §5.4.\n")
    for scope, pred in (("all 500 fictitious profiles", lambda r: r["set"] == "fictitious"),
                        ("forget_10 (50 profiles)", lambda r: r["set"] == "fictitious" and r["forget_10"]),
                        ("153 celebrities", lambda r: r["set"] == "celebrity")):
        R = [r for r in rows if pred(r)]
        by_p = collections.defaultdict(lambda: {"img": [], "txt": []})
        for r in R:
            by_p[r["profile"]][r["modality"]].append(r)
        txt_q_with_img_attr = txt_q = 0
        fine_hit = same_ans = pairs = 0
        n_shared = []
        for pid, d in by_p.items():
            ia = {r["attribute"] for r in d["img"]}
            fa = {r["attribute_fine"] for r in d["img"]}
            n_shared.append(len(ia & {r["attribute"] for r in d["txt"]}))
            for r in d["txt"]:
                txt_q += 1
                if r["attribute"] in ia:
                    txt_q_with_img_attr += 1
                if r["attribute_fine"] in fa:
                    fine_hit += 1
                    for s in d["img"]:
                        if s["attribute_fine"] == r["attribute_fine"]:
                            pairs += 1
                            same_ans += norm_tokens(s["correct_answer"]) == norm_tokens(r["correct_answer"])
        w(f"### {scope}\n")
        w(f"- Text-only questions whose attribute (coarse) is also asked by an image+text question of the same profile: "
          f"{txt_q_with_img_attr}/{txt_q} ({100 * txt_q_with_img_attr / max(1, txt_q):.1f}%); fine-grained: "
          f"{fine_hit}/{txt_q} ({100 * fine_hit / max(1, txt_q):.1f}%).")
        w(f"- Shared coarse attributes per profile: mean {sum(n_shared) / max(1, len(n_shared)):.2f}; distribution "
          f"{dict(sorted(collections.Counter(n_shared).items()))}.")
        w(f"- Among fine-grained attribute-matched (image question, text question) pairs: {pairs} pairs; correct answers "
          f"with identical normalised token sets: {same_ans}/{pairs}.")
        cnt = {m: collections.Counter(r["attribute"] for r in R if r["modality"] == m) for m in ("img", "txt")}
        attrs = sorted(set(cnt["img"]) | set(cnt["txt"]), key=lambda x: -(cnt["img"][x] + cnt["txt"][x]))
        w("\n| attribute | image+text questions | text-only questions |")
        w("|---|---|---|")
        for x in attrs:
            w(f"| {x} | {cnt['img'][x]} | {cnt['txt'][x]} |")
        w("")
    w("### 5.4 Label spot check\n")
    w(SPOT_CHECK)
    w("## 6. Paired items built for this project (coverage of the official attributes)\n")
    w("The paired MC items (`vlmu/data/build.py`) cover 9 attributes: " + ", ".join(FIELDS) + ". "
      "In the official taxonomy above these correspond to birthplace, dob, profession, height, education, salary, residence "
      "and parents (occupation). Official attributes not covered by paired items: name, gender, medical, pet, food, "
      "hobby/leisure, works/achievements, appearance, personality, other.\n")
    (REPO / "analysis" / "data_audit.md").write_text("\n".join(L) + "\n")
    print(f"wrote analysis/data_audit.md and {out_csv.relative_to(REPO)} ({len(rows)} questions)")


if __name__ == "__main__":
    main()
