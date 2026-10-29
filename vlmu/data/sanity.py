"""Data-level sanity checks (run by prepare_data; re-run by scripts/sanity.py).

- paired items: the image+text, text-only and pose forms refer to the same fact
  (identical options/answer; answer stated by the FT answer)
- text-only inputs carry no image; image inputs carry exactly one
- eval wording differs from training wording (paraphrase requirement)
- no forget-image overlap with any other eval set; held-out poses differ from portraits
- forget / retain-eval / retain-train ID sets are disjoint
"""
import collections
from pathlib import Path

from ..utils import atomic_write_json, read_json, read_jsonl
from .fields import supported
from .rewrite import residue


def check_data(processed, log=print) -> dict:
    P = Path(processed)
    splits = read_json(P / "splits.json")
    manifest = read_json(P / "images.json")
    items = {k: read_jsonl(P / f"items_{k}.jsonl") for k in ("forget", "retain", "celeb", "utility")}
    train = read_jsonl(P / "train_vanilla.jsonl")
    uf = read_jsonl(P / "unlearn_forget.jsonl")
    ur = read_jsonl(P / "unlearn_retain.jsonl")
    checks = []

    def check(name, ok, detail=None):
        checks.append({"check": name, "passed": bool(ok), "detail": detail})
        log(("PASS " if ok else "FAIL ") + name + ("" if ok or detail is None else f"  {detail}"))

    # --- ID sets
    f, re_, rt = set(splits["forget"]), set(splits["retain_eval"]), set(splits["retain_train"])
    check("forget/retain_eval/retain_train disjoint", not (f & re_ or f & rt or re_ & rt))
    check("unlearn_forget profiles == forget", {r["profile"] for r in uf} <= f and len(uf) > 0)
    check("unlearn_retain profiles within retain_train", {r["profile"] for r in ur} <= rt and len(ur) > 0)

    # --- paired items: same fact in every form
    paired = [r for k in ("forget", "retain") for r in items[k] if r["source"] == "paired"]
    by_pair = collections.defaultdict(list)
    for r in paired:
        by_pair[r["pair_id"]].append(r)
    bad = [pid for pid, rows in by_pair.items()
           if sorted(r["modality"] for r in rows) != ["img", "pose", "txt"]
           or len({(tuple(sorted(r["options"].items())), r["answer"]) for r in rows}) != 1]
    check("paired items: img/txt/pose share options+answer", not bad, bad[:5])
    bad = [r["uid"] for r in paired if r["options"][r["answer"]] != r["fact_answer"]
           or not supported(r["field"], r["fact_answer"], r["ft_answer"])]
    check("paired items: answer is the fact stated in the FT answer", not bad, bad[:5])
    bad = [r["uid"] for r in paired if len(set(o.lower() for o in r["options"].values())) != 4]
    check("paired items: 4 distinct options", not bad, bad[:5])
    letters = collections.Counter(r["answer"] for r in paired if r["modality"] == "txt")
    n = sum(letters.values())
    lo, hi = 0.25 - 3 * (0.1875 / n) ** 0.5, 0.25 + 3 * (0.1875 / n) ** 0.5  # +-3 sd of a fair draw
    check("paired items: answer letters balanced (within 3 sd of uniform)",
          all(lo <= letters.get(L, 0) / n <= hi for L in "ABCD"), dict(letters))

    # --- modality integrity of inputs
    all_items = [r for v in items.values() for r in v]
    bad = [r["uid"] for r in all_items if (r["modality"] == "txt") != (r["image"] is None)]
    check("eval: text-only items have no image, image items have one", not bad, bad[:5])
    bad = [r["uid"] for r in train + uf + ur if (r["form"] == "txt") != (r["image"] is None)]
    check("train/unlearn: text form has no image, image form has one", not bad, bad[:5])
    bad = [r["uid"] for r in train + uf + ur if r["form"] == "txt" and residue(r["question"] + " " + r["answer"])]
    check("train/unlearn: text form has no residual image reference", not bad, bad[:5])
    bad = [r["uid"] for r in paired if r["modality"] == "img" and "pictured person" not in r["question"]]
    check("paired img stems use the visual referent", not bad, bad[:5])

    # --- unlearning sets: same facts in both forms, no name QA
    for label, rows in (("unlearn_forget", uf), ("unlearn_retain", ur)):
        forms = collections.defaultdict(set)
        for r in rows:
            forms[r["fact_id"]].add(r["form"])
        bad = [k for k, v in forms.items() if v != {"img", "txt"}]
        check(f"{label}: every fact present in both forms", not bad, bad[:5])
        check(f"{label}: no name QA", not any(r["field"] == "name" for r in rows))

    # --- paraphrase: eval wording never equals training wording
    train_q = {r["question"].strip().lower() for r in train}
    bad = [r["uid"] for r in paired if r["question"].strip().lower() in train_q]
    check("paired eval questions differ from every training question", not bad, bad[:5])
    report = read_json(P / "build_report.json")
    check("paired stems: no exact match with FT question templates",
          not any(v["exact_match_with_ft_question"] for v in report["paraphrase_check"].values()),
          {k: v["max_jaccard_vs_ft_questions"] for k, v in report["paraphrase_check"].items()})

    # --- image overlap
    sha_of = lambda pred: {v["sha1"] for k, v in manifest.items() if pred(k, v)}  # noqa: E731
    forget_portraits = sha_of(lambda k, v: v["role"] == "portrait" and v["profile"] in f)
    other_eval = sha_of(lambda k, v: (v["role"] == "portrait" and v["profile"] in re_)
                        or v["role"] in ("celeb", "mmbench"))
    check("forget portraits do not appear in retain/celeb/utility eval images",
          not (forget_portraits & other_eval), len(forget_portraits & other_eval))
    portraits = sha_of(lambda k, v: v["role"] == "portrait")
    pose_sha = sha_of(lambda k, v: v["role"] == "pose")
    check("held-out poses differ from every portrait (incl. training images)", not (pose_sha & portraits))
    n_port = sum(1 for v in manifest.values() if v["role"] == "portrait")
    check("portraits are unique per profile", len(portraits) == n_port, f"{len(portraits)} unique / {n_port}")
    warn = [d for d in report["dropped"] if d.get("reason") == "ft_image_differs_from_eval_portrait"]
    check("FT images identical to eval portraits", not warn, len(warn))

    out = {"passed": all(c["passed"] for c in checks), "checks": checks}
    atomic_write_json(P / "sanity_data.json", out)
    return out
