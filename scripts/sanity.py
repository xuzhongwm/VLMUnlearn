"""Sanity checks before reporting. Writes results/<model>/sanity.json.

- all data checks (pairing, balance, modality integrity, paraphrase, image overlap, splits)
- vanilla gate: paired forget MC >= threshold in BOTH modalities (primary forms), per-field breakdown
- doubly-correct set frozen; coverage per split (entity and attribute coverage in the file)
- vanilla answers in letter format (probability mass on A-D)
- Milestone J hook checks passed (trained adapter)
- pretrained (no-FT) priors: fields answerable without the fine-tuned knowledge are flagged (warning)

  python scripts/sanity.py --config configs/llava/grid.yaml [--debug]
"""
import collections
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from vlmu.config import base_parser, resolve  # noqa: E402
from vlmu.data.sanity import check_data  # noqa: E402
from vlmu.runs import dc_file, model_dir, results_dir, vanilla_eval_dir  # noqa: E402
from vlmu.utils import Logger, atomic_write_json, read_json, read_jsonl  # noqa: E402


def per_field(items_file, set_name="forget"):
    acc = collections.defaultdict(list)
    for r in read_jsonl(items_file):
        if r["set"] == set_name and r["source"] == "paired" and r.get("shift", 0) == 0:
            acc[(r["field"], r["modality"])].append(r["correct"])
    return {f"{f}/{m}": round(sum(v) / len(v), 3) for (f, m), v in sorted(acc.items())}


def main():
    cfg = resolve(base_parser(__doc__).parse_args())
    log = Logger()
    out = {"model": cfg["model"]["name"], "role": cfg["model"].get("role"),
           "data": check_data(cfg["paths"]["processed"], log=log)}
    checks = list(out["data"]["checks"])
    add = lambda name, ok, detail, severity=None: checks.append(  # noqa: E731
        {"check": name, "passed": bool(ok), "detail": detail, **({"severity": severity} if severity else {})})
    gate_f = model_dir(cfg) / "vanilla" / "gate.json"
    if gate_f.exists():
        gate = read_json(gate_f)
        add(f"vanilla gate (>= {gate['threshold']} paired forget MC, both modalities, primary forms)", gate["passed"], gate)
        summ = read_json(vanilla_eval_dir(cfg) / "summary.json")["all_items"]
        pl = {m: round(sum(summ[k]["p_letters"] * summ[k]["n"] for k in summ if k.startswith("forget/") and k.endswith(f"paired/{m}"))
                       / max(1, sum(summ[k]["n"] for k in summ if k.startswith("forget/") and k.endswith(f"paired/{m}"))), 3)
              for m in ("img", "txt")}
        add("vanilla puts >= 50% probability on option letters", min(pl.values()) >= 0.5, pl)
        out["vanilla_per_field"] = per_field(vanilla_eval_dir(cfg) / "items.jsonl")
    else:
        add("vanilla gate", False, "no vanilla model yet")
    if dc_file(cfg).exists():
        cov = read_json(dc_file(cfg))["coverage"]["by_split"]
        add("doubly-correct set frozen (coverage per split)", all(v["n_doubly_correct"] > 0 for v in cov.values()),
            {k: v["coverage"] for k, v in cov.items()})
    else:
        add("doubly-correct set frozen", False, "missing")
    mj = results_dir(cfg) / "milestone_j.json"
    add("Milestone J hook checks (trained adapter)", mj.exists() and read_json(mj)["passed"],
        read_json(mj)["checks"] if mj.exists() else "not run")
    pre = model_dir(cfg) / "pretrained" / "full_eval" / "items.jsonl"
    if pre.exists():
        pf = per_field(pre)
        flagged = {k: v for k, v in pf.items() if v > 0.40}
        out["pretrained_per_field"] = pf
        add("pretrained (no FT) per-field accuracy <= 0.40 (chance 0.25) — priors", not flagged,
            flagged or "none above 0.40", "warning")
    if cfg["model"].get("role") == "pipeline_check":
        add("model role: pipeline check only — excluded from confirmatory evidence", True, cfg["model"]["role"], "info")
    out["checks"] = checks
    out["passed"] = all(c["passed"] for c in checks if c.get("severity") not in ("warning", "info"))
    for c in checks[len(out["data"]["checks"]):]:
        tag = "PASS" if c["passed"] else ("WARN" if c.get("severity") else "FAIL")
        log(f"{tag} {c['check']}", str(c["detail"])[:300])
    atomic_write_json(results_dir(cfg) / "sanity.json", out)
    log("SANITY", "PASSED" if out["passed"] else "FAILED")
    sys.exit(0 if out["passed"] else 1)


if __name__ == "__main__":
    main()
