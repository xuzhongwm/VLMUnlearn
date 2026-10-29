"""Evaluate the public base model (no fine-tuning) on all eval sets: measures how much of
each MC set is answerable from priors / visual cues alone.

  python scripts/eval_pretrained.py --config configs/qwen2vl/vanilla.yaml [--debug]
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from vlmu.config import base_parser, resolve  # noqa: E402
from vlmu.runs import run_pretrained_eval  # noqa: E402

if __name__ == "__main__":
    run_pretrained_eval(resolve(base_parser(__doc__).parse_args()))
