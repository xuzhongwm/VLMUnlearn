"""Milestone J on the UMU track (DEV-011): hook checks (i)-(iii) and tool tests J1-J6 with theta_0 = UMU's
released vanilla and theta_u = theta_0 + the unlearning LoRA the analyses will use.

dtype: float32 needs ~30 GB (A100 40 GB). On an L4 run --set mech.dtype=bfloat16: the exact checks ((i),
(iii-a), J2, J4) still pass bit-for-bit; the subtraction-based ones ((ii), (iii-b), J5) are limited by bf16
rounding (DEV-007) -- read their top-1 agreement, not only pass/fail.

  python scripts/check_hooks_umu.py --config configs/llava/umu_npo_rt.yaml --step 75 [--run-id ...]
Writes runs/llava/umu_unlearn/<run_id>/milestone_j_step<N>/ and results/llava/milestone_j_umu.json.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from vlmu.config import base_parser, resolve  # noqa: E402
from vlmu.mech.checks import run_milestone_j  # noqa: E402
from vlmu.runs import model_dir  # noqa: E402


def umu_run_id(cfg):
    u, t = cfg["unlearn"], cfg["train"]
    return f"umu_{u['method']}_{u['modality']}_lr{t['lr']:.0e}_s{u['seed']}"


if __name__ == "__main__":
    p = base_parser(__doc__)
    p.add_argument("--step", type=int, required=True, help="checkpoint step whose LoRA is checked")
    p.add_argument("--run-id", default=None, help="default: derived from the config (as scripts/unlearn_umu.py)")
    args = p.parse_args()
    cfg = resolve(args)
    run_dir = model_dir(cfg) / "umu_unlearn" / (args.run_id or umu_run_id(cfg))
    adapter_dir = run_dir / "ckpt" / f"step{args.step}"
    if not (adapter_dir / "adapter_config.json").exists():
        raise SystemExit(f"no adapter at {adapter_dir}")
    run_milestone_j(cfg, umu_adapter_dir=adapter_dir, out_dir=run_dir / f"milestone_j_step{args.step}")
