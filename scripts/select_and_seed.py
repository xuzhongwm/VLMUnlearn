"""Select, per (method, unlearning modality), the grid checkpoint closest to the target
in-modality forgetting among those within the retain margin (same rule for every cell);
full-eval it (seed 0); rerun it with the extra seed(s) to the same step and full-eval.
Writes results/<model>/{selection.json, grid.jsonl, selected_runs.json}. Resumable.

  python scripts/select_and_seed.py --config configs/qwen2vl/grid.yaml [--debug] [--allow-incomplete]
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from vlmu.config import base_parser, resolve  # noqa: E402
from vlmu.runs import eval_adapter, model_dir, rerun_cfg, results_dir, run_unlearn, save_generated_cfg, select  # noqa: E402
from vlmu.utils import Logger, atomic_write_json, read_json, tracked_run, write_jsonl  # noqa: E402


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
    for k, v in sel["cells"].items():
        log(f"{k:14s} {v}")
    if sel["incomplete_runs"] and not (args.allow_incomplete or cfg["debug"]):
        raise SystemExit(f"grid runs not DONE: {sel['incomplete_runs']} (run run_grid.py, or --allow-incomplete)")
    selected, failed = {}, []
    for cell, info in sel["cells"].items():
        if info["status"] == "no_feasible_checkpoint":
            log(f"{cell}: no feasible checkpoint — reported as such, no rerun")
            continue
        method, modality = cell.split("/")
        run0 = model_dir(cfg) / "unlearn" / info["run_id"]
        best = read_json(run0 / "best.json")
        assert best["step"] == info["step"], f"{cell}: best adapter step {best['step']} != selected {info['step']}"
        entry = {**info, "seeds": {}}
        try:
            with tracked_run(rdir, run0 / "full_eval_selected", info["run_id"], "full_eval_selected"):
                eval_adapter(cfg, run0 / "best_adapter", run0 / "full_eval_selected", log)
            entry["seeds"][str(cfg["grid"]["seed"])] = {"run_id": info["run_id"], "full_eval": str(run0 / "full_eval_selected")}
            for seed in cfg["selection"]["extra_seeds"]:
                rc = rerun_cfg(cfg, info, method, modality, seed)
                log(f"{cell}: seed {seed} rerun -> {rc['run_id']} (config {save_generated_cfg(cfg, rc)})")
                run_unlearn(rc)
                entry["seeds"][str(seed)] = {"run_id": rc["run_id"],
                                             "full_eval": str(model_dir(cfg) / "unlearn" / rc["run_id"] / "full_eval")}
        except Exception as e:
            log(f"FAILED {cell}: {type(e).__name__}: {e}")
            failed.append(cell)
        selected[cell] = entry
        atomic_write_json(rdir / "selected_runs.json", selected)
    if failed:
        raise SystemExit(f"failed cells: {failed}")


if __name__ == "__main__":
    main()
