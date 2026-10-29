"""Run ONE unlearning run defined by its YAML (configs/<model>/runs/<run_id>.yaml). Resumable.

  python scripts/unlearn.py --config configs/qwen2vl/runs/qwen2vl_ga_txt_lr2e-05_s0.yaml [--debug]
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from vlmu.config import base_parser, resolve  # noqa: E402
from vlmu.runs import run_unlearn  # noqa: E402

if __name__ == "__main__":
    run_unlearn(resolve(base_parser(__doc__).parse_args()))
