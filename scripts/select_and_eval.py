"""Select, per (method, unlearning modality), the lowest lr whose runs reach an eligible
checkpoint in every seed (calibration entities only), take each seed's earliest eligible
checkpoint, and full-eval it (confirmation entities are first touched here). All runs and their
statuses are reported. Writes results/<model>/{selection.json, grid.jsonl, selected_runs.json}.

  python scripts/select_and_eval.py --config configs/llava/grid.yaml [--debug] [--allow-incomplete]
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from vlmu.config import base_parser, resolve  # noqa: E402
from vlmu.runs import eval_adapter, results_dir, select  # noqa: E402
from vlmu.utils import Logger, atomic_write_json, tracked_run, write_jsonl  # noqa: E402


def main():
    p = base_parser(__doc__)
    p.add_argument("--allow-incomplete", action="store_true", help="select even if some grid runs are not DONE")
    args = p.parse_args()
    cfg = resolve(args)
    rdir = results_dir(cfg)
    log = Logger(rdir / "select_log.txt")
    sel, recs = select(cfg, log)
    write_jsonl(rdir / "grid.jsonl", recs)
    atomic_write_json(rdir / "selection.json", sel)
    for rid, r in sel["runs"].items():
        log(f"run {rid:40s} {r['status']}" + (f" step {r['step']} acc_in {r['acc_in']:.3f}" if "step" in r else ""))
    for k, v in sel["cells"].items():
        log(f"cell {k:14s} {v['status']}" + (f" lr={v['lr']}" if "lr" in v else ""))
    incomplete = [k for k, r in sel["runs"].items() if r["status"] == "incomplete"]
    if incomplete and not (args.allow_incomplete or cfg["debug"]):
        raise SystemExit(f"grid runs not DONE: {incomplete} (run run_grid.py, or --allow-incomplete)")
    selected, failed = {}, []
    for cell, info in sel["cells"].items():
        if info["status"] != "selected":
            continue
        entry = {"lr": info["lr"], "seeds": {}}
        for seed, run in info["seeds"].items():
            rd = Path(run["dir"])
            try:
                with tracked_run(rdir, rd / "full_eval", run["run_id"], "full_eval_selected"):
                    eval_adapter(cfg, rd / "selected_adapter", rd / "full_eval", log)
                entry["seeds"][seed] = {**run, "full_eval": str(rd / "full_eval")}
            except Exception as e:
                log(f"FAILED full eval {cell} seed {seed}: {type(e).__name__}: {e}")
                failed.append(f"{cell}/s{seed}")
        selected[cell] = entry
        atomic_write_json(rdir / "selected_runs.json", selected)
    atomic_write_json(rdir / "selected_runs.json", selected)
    if failed:
        raise SystemExit(f"failed full evals: {failed}")


if __name__ == "__main__":
    main()
