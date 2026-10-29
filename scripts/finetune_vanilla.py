"""Fine-tune our own vanilla model on ft_Data in both forms, then full-eval it and apply the
gate (paired forget MC >= 80% in both modalities). Resumable.

  python scripts/finetune_vanilla.py --config configs/qwen2vl/vanilla.yaml [--debug]
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from vlmu.config import base_parser, resolve  # noqa: E402
from vlmu.runs import run_vanilla  # noqa: E402

if __name__ == "__main__":
    run_vanilla(resolve(base_parser(__doc__).parse_args()))
