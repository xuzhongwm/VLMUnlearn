"""Run every job with --debug, in order, and report pass/fail + wall time per job.
Clears previous debug runs/models/results first (debug data is kept).

  python scripts/smoke_test.py --model qwen2vl [--set model.id=...]
"""
import argparse
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))
from vlmu.config import DEFAULT_ROOT  # noqa: E402


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True, choices=["qwen2vl", "llava"])
    p.add_argument("--root", default=None)
    p.add_argument("--set", nargs="*", default=[])
    p.add_argument("--keep", action="store_true", help="do not clear previous debug outputs")
    a = p.parse_args()
    root = Path(a.root or os.environ.get("VLMU_ROOT") or DEFAULT_ROOT)
    if not a.keep:
        for d in ("runs", "models", "results"):
            shutil.rmtree(root / "debug" / d, ignore_errors=True)
    extra = ["--debug", "--root", str(root)] + (["--set", *a.set] if a.set else [])
    m = a.model
    jobs = [
        ("prepare_data", ["scripts/prepare_data.py", "--config", "configs/data.yaml"]),
        ("eval_pretrained", ["scripts/eval_pretrained.py", "--config", f"configs/{m}/vanilla.yaml"]),
        ("finetune_vanilla", ["scripts/finetune_vanilla.py", "--config", f"configs/{m}/vanilla.yaml"]),
        ("run_grid", ["scripts/run_grid.py", "--config", f"configs/{m}/grid.yaml"]),
        ("select_and_seed", ["scripts/select_and_seed.py", "--config", f"configs/{m}/grid.yaml"]),
        ("aggregate", ["scripts/aggregate.py", "--config", f"configs/{m}/grid.yaml"]),
        ("sanity", ["scripts/sanity.py", "--config", f"configs/{m}/grid.yaml"]),
    ]
    report = []
    for name, cmd in jobs:
        args = cmd + (["--debug", "--root", str(root)] if name == "prepare_data" else extra)
        t0 = time.time()
        r = subprocess.run([sys.executable, *args], cwd=REPO)
        dt = time.time() - t0
        # sanity is expected to FAIL the gate in debug (vanilla trained for a few steps)
        ok = r.returncode == 0 or name == "sanity"
        report.append((name, ok, dt, r.returncode))
        print(f"\n>>> {name}: {'OK' if ok else 'FAILED'} ({dt:.0f}s, exit {r.returncode})\n", flush=True)
        if not ok:
            break
    print("\nSMOKE TEST SUMMARY")
    for name, ok, dt, rc in report:
        print(f"  {name:18s} {'OK ' if ok else 'FAIL'} {dt:6.0f}s  (exit {rc})")
    sys.exit(0 if all(ok for _, ok, _, _ in report) and len(report) == len(jobs) else 1)


if __name__ == "__main__":
    main()
