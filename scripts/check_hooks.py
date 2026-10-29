"""Milestone J: hook-machinery checks on the vanilla model + a briefly TRAINED adapter.
Writes runs/<model>/milestone_j/milestone_j.json and results/<model>/milestone_j.json.

  python scripts/check_hooks.py --config configs/qwen2vl/grid.yaml [--debug]
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from vlmu.config import base_parser, resolve  # noqa: E402
from vlmu.mech.checks import run_milestone_j  # noqa: E402

if __name__ == "__main__":
    run_milestone_j(resolve(base_parser(__doc__).parse_args()))
