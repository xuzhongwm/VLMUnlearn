"""Job 0: download raw data to Drive and build processed items. CPU only, ~10 min.

  python scripts/prepare_data.py --config configs/data.yaml [--debug]
"""
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from vlmu.config import base_parser, resolve  # noqa: E402
from vlmu.data.build import build  # noqa: E402
from vlmu.data.raw import download  # noqa: E402
from vlmu.data.sanity import check_data  # noqa: E402
from vlmu.utils import Logger, is_done, mark_done, tracked_run  # noqa: E402


def main():
    p = base_parser(__doc__, config_required=False)
    p.add_argument("--force", action="store_true", help="rebuild even if DONE")
    p.set_defaults(config="configs/data.yaml")
    args = p.parse_args()
    cfg = resolve(args)
    P = cfg["paths"]
    out = Path(P["processed"])
    log = Logger(out / "prepare_log.txt")
    if is_done(out) and not args.force:
        log(f"{out} already built (DONE). Use --force to rebuild.")
        return
    with tracked_run(P["results"], out, "prepare_data", "prepare_data"):
        t0 = time.time()
        raw = download(P["raw"], cfg["data"]["forget_split"], log=log)
        report = build(cfg, raw, out, log=log)
        d = cfg["data"]
        sizes = None if cfg["debug"] else {
            "forget_calibration": d["n_forget_calibration"], "forget_confirmation": 50 - d["n_forget_calibration"],
            "retain_unlearn": 450 - d["n_retain_calibration"] - d["n_retain_confirmation"],
            "retain_calibration": d["n_retain_calibration"], "retain_confirmation": d["n_retain_confirmation"]}
        checks = check_data(out, log=log, expected_sizes=sizes)
        if not checks["passed"]:
            raise RuntimeError(f"data sanity checks FAILED — see {out / 'sanity_data.json'}")
        mark_done(out, {"seconds": time.time() - t0, "counts": report["counts"]["splits"]})


if __name__ == "__main__":
    main()
