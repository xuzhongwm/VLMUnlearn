"""Expand configs/<model>/grid.yaml into one YAML per run in configs/<model>/runs/ (committed).

  python scripts/make_run_configs.py
"""
import sys
from pathlib import Path

import yaml

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))
from vlmu.config import load_yaml  # noqa: E402


def main():
    for model in ("qwen2vl", "llava"):
        grid = load_yaml(REPO / "configs" / model / "grid.yaml")["grid"]
        out = REPO / "configs" / model / "runs"
        out.mkdir(exist_ok=True)
        for old in out.glob("*.yaml"):
            old.unlink()
        for method in grid["methods"]:
            for modality in grid["modalities"]:
                for lr in grid["lrs"]:
                    run_id = f"{model}_{method}_{modality}_lr{lr:.0e}_s{grid['seed']}"
                    cfg = {"base": "../grid.yaml", "run_id": run_id,
                           "unlearn": {"method": method, "modality": modality, "seed": grid["seed"], "mode": "grid"},
                           "train": {"lr": lr}}
                    (out / f"{run_id}.yaml").write_text(yaml.safe_dump(cfg, sort_keys=False))
        print(model, len(list(out.glob('*.yaml'))), "run configs")


if __name__ == "__main__":
    main()
