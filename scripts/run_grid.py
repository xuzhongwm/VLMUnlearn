"""Run every grid run in configs/<model>/runs/ (all methods, modalities, lrs, seeds), skipping DONE runs. A failed run is
logged (FAILED marker + results/run_log.csv) and the grid continues; exit code is non-zero if
any run failed. Safe to re-invoke after a disconnect.

  python scripts/run_grid.py --config configs/qwen2vl/grid.yaml [--debug] [--only RUN_ID ...]
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from vlmu.config import base_parser, resolve  # noqa: E402
from vlmu.runs import load_run_cfg, run_configs, run_unlearn  # noqa: E402
from vlmu.utils import Logger  # noqa: E402


def main():
    p = base_parser(__doc__)
    p.add_argument("--only", nargs="*", default=None)
    args = p.parse_args()
    cfg = resolve(args)
    log = Logger()
    paths = run_configs(cfg["model"]["name"])
    cfgs = [load_run_cfg(x, cfg) for x in paths]
    if cfg["debug"]:
        want = {tuple(x) for x in cfg["debug_runs"]}
        lr0 = cfg["grid"]["lrs"][0]
        cfgs = [c for c in cfgs if (c["unlearn"]["method"], c["unlearn"]["modality"]) in want and c["train"]["lr"] == lr0]
    if args.only:
        cfgs = [c for c in cfgs if c["run_id"] in args.only]
    failed = []
    for i, c in enumerate(cfgs, 1):
        log(f"=== [{i}/{len(cfgs)}] {c['run_id']}")
        try:
            run_unlearn(c)
        except Exception as e:  # logged by tracked_run; keep going
            log(f"FAILED {c['run_id']}: {type(e).__name__}: {e}")
            failed.append(c["run_id"])
    log(f"grid finished: {len(cfgs) - len(failed)} ok, {len(failed)} failed {failed}")
    if failed:
        sys.exit(1)


if __name__ == "__main__":
    main()
