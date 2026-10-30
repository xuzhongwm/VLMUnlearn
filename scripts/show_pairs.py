"""Print a random sample of paired items (same fact, image vs text question) and of the
rewritten FT QA (image form vs name-substituted text form) for a manual check.

  python scripts/show_pairs.py [--n 8] [--seed 0] [--debug]
"""
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from vlmu.config import base_parser, resolve  # noqa: E402
from vlmu.utils import read_jsonl  # noqa: E402


def main():
    p = base_parser(__doc__, config_required=False)
    p.add_argument("--n", type=int, default=8)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()
    args.config = args.config or "configs/data.yaml"
    proc = Path(resolve(args)["paths"]["processed"])
    rng = random.Random(args.seed)

    items = [r for r in read_jsonl(proc / "items_forget.jsonl") if r["source"] == "paired"]
    by_pair = {}
    for r in items:
        by_pair.setdefault(r["pair_id"], {})[r["modality"]] = r
    print(f"=== paired MC items (forget): {len(by_pair)} facts, showing {args.n}")
    for pid in rng.sample(sorted(by_pair), min(args.n, len(by_pair))):
        f = by_pair[pid]
        print(f"\n{pid}  answer={f['txt']['answer']} ({f['txt']['options'][f['txt']['answer']]})")
        print(f"  IMG [{f['img']['image']}]: {f['img']['question']}")
        print(f"  TXT [no image]: {f['txt']['question']}")
        print("  options:", " | ".join(f"{k}. {v}" for k, v in f["txt"]["options"].items()))
        assert f["img"]["options"] == f["txt"]["options"] and f["img"]["answer"] == f["txt"]["answer"]
        assert f["txt"]["image"] is None and f["img"]["image"]

    qa = read_jsonl(proc / "unlearn_forget.jsonl")
    by_fact = {}
    for r in qa:
        by_fact.setdefault(r["fact_id"], {})[r["form"]] = r
    print(f"\n=== FT QA rewrite (forget unlearning set): {len(by_fact)} facts, showing {args.n}")
    for fid in rng.sample(sorted(by_fact), min(args.n, len(by_fact))):
        f = by_fact[fid]
        print(f"\n{fid}")
        print(f"  IMG: Q {f['img']['question']!r}  A {f['img']['answer']!r}")
        print(f"  TXT: Q {f['txt']['question']!r}  A {f['txt']['answer']!r}")


if __name__ == "__main__":
    main()
