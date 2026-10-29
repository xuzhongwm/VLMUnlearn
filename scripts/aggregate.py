"""Build results/<model>/{transfer_table.csv, asymmetry.csv, grid.csv, verdict.json,
transfer_<model>.png}; then combine all models into results/transfer_table.csv and mirror
results to the repo's results/ directory (non-debug).

  python scripts/aggregate.py --config configs/qwen2vl/grid.yaml [--debug]
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import shutil  # noqa: E402

from vlmu.config import REPO, base_parser, resolve  # noqa: E402
from vlmu.report import aggregate, write_csv  # noqa: E402
from vlmu.utils import Logger  # noqa: E402


def main():
    cfg = resolve(base_parser(__doc__).parse_args())
    log = Logger()
    aggregate(cfg, log)
    res = Path(cfg["paths"]["results"])
    rows = []
    for f in sorted(res.glob("*/transfer_table.csv")):
        import csv
        rows += list(csv.DictReader(open(f)))
    write_csv(res / "transfer_table.csv", rows)
    log(f"wrote {res / 'transfer_table.csv'} ({len(rows)} rows)")
    if not cfg["debug"]:
        dst = REPO / "results"
        for f in list(res.glob("*.csv")) + list(res.glob("*/*.csv")) + list(res.glob("*/*.png")) + list(res.glob("*/*.json")):
            out = dst / f.relative_to(res)
            out.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(f, out)
        log(f"mirrored results to {dst}")


if __name__ == "__main__":
    main()
