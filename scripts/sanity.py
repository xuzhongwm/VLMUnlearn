"""Sanity checks before reporting (D2/D8 + task rules). Writes results/<model>/sanity.json.

- all data checks (pairing, modality integrity, paraphrase, image overlap, disjoint IDs)
- vanilla gate: paired forget MC >= threshold in BOTH modalities (per-field breakdown)
- vanilla answers in letter format (probability mass on A-D)
- pretrained (no-FT) priors: fields answerable without the fine-tuned knowledge are flagged

  python scripts/sanity.py --config configs/qwen2vl/grid.yaml [--debug]
"""
import collections
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from vlmu.config import base_parser, resolve  # noqa: E402
from vlmu.data.sanity import check_data  # noqa: E402
from vlmu.runs import model_dir, results_dir, vanilla_eval_dir  # noqa: E402
from vlmu.utils import Logger, atomic_write_json, read_json, read_jsonl  # noqa: E402


def per_field(items_file, set_name="forget"):
    acc = collections.defaultdict(list)
    for r in read_jsonl(items_file):
        if r["set"] == set_name and r["source"] == "paired":
            acc[(r["field"], r["modality"])].append(r["correct"])
    return {f"{f}/{m}": round(100 * sum(v) / len(v), 1) for (f, m), v in sorted(acc.items())}


def main():
    cfg = resolve(base_parser(__doc__).parse_args())
    log = Logger()
    out = {"data": check_data(cfg["paths"]["processed"], log=log)}
    checks = list(out["data"]["checks"])
    gate_f = model_dir(cfg) / "vanilla" / "gate.json"
    if gate_f.exists():
        gate = read_json(gate_f)
        checks.append({"check": f"vanilla gate (>= {gate['threshold']}% paired forget MC, both modalities)",
                       "passed": gate["passed"], "detail": gate})
        summ = read_json(vanilla_eval_dir(cfg) / "summary.json")
        pl = {m: summ[f"forget/paired/{m}"]["p_letters"] for m in ("img", "txt")}
        checks.append({"check": "vanilla puts >= 50% probability on option letters", "passed": min(pl.values()) >= 0.5,
                       "detail": pl})
        out["vanilla_per_field"] = per_field(vanilla_eval_dir(cfg) / "items.jsonl")
    else:
        checks.append({"check": "vanilla gate", "passed": False, "detail": "no vanilla model yet"})
    pre = model_dir(cfg) / "pretrained" / "full_eval" / "items.jsonl"
    if pre.exists():
        pf = per_field(pre)
        flagged = {k: v for k, v in pf.items() if v > 40.0}
        out["pretrained_per_field"] = pf
        checks.append({"check": "pretrained (no FT) per-field accuracy <= 40% (chance 25%) — priors",
                       "passed": not flagged, "detail": flagged or "none above 40%", "severity": "warning"})
    out["checks"] = checks
    out["passed"] = all(c["passed"] for c in checks if c.get("severity") != "warning")
    for c in checks[len(out["data"]["checks"]):]:
        log(("PASS " if c["passed"] else ("WARN " if c.get("severity") else "FAIL ")) + c["check"], c["detail"])
    atomic_write_json(results_dir(cfg) / "sanity.json", out)
    log("SANITY", "PASSED" if out["passed"] else "FAILED")
    sys.exit(0 if out["passed"] else 1)


if __name__ == "__main__":
    main()
