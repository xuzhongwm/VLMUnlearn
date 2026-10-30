"""Single-cell pilot (research.md Section 6 step 1) with checkpoint A (vanilla gate) and
checkpoint B (leakage exists). Resumable. Exits 1 if checkpoint B fails (not in --debug).

  python scripts/pilot.py --config configs/llava/pilot.yaml [--debug]
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from vlmu.config import base_parser, resolve  # noqa: E402
from vlmu.runs import run_pilot  # noqa: E402

if __name__ == "__main__":
    cfg = resolve(base_parser(__doc__).parse_args())
    summary = run_pilot(cfg)
    if not summary["checkpoint_b"]["passed"]:
        print("CHECKPOINT B FAILED: no checkpoint shows the leak. Stop and report; do not run the grid.")
        sys.exit(0 if cfg["debug"] else 1)
