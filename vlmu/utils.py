"""I/O, seeding, run bookkeeping (DONE/FAILED markers, run log)."""
import contextlib
import csv
import datetime as dt
import json
import os
import random
import time
import traceback
from pathlib import Path

import numpy as np


def seed_all(seed: int):
    import torch
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _json_default(o):
    """numpy scalars -> native (np.bool_ would otherwise be written as the truthy string 'False')."""
    if hasattr(o, "item") and callable(o.item):
        return o.item()
    return str(o)


def now() -> str:
    return dt.datetime.now().isoformat(timespec="seconds")


def atomic_write_json(path, obj):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    with open(tmp, "w") as f:
        json.dump(obj, f, indent=2, sort_keys=False, default=_json_default)
    os.replace(tmp, path)


def read_json(path):
    with open(path) as f:
        return json.load(f)


def write_jsonl(path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    with open(tmp, "w") as f:
        for r in rows:
            f.write(json.dumps(r, default=_json_default) + "\n")
    os.replace(tmp, path)


def append_jsonl(path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a") as f:
        for r in rows:
            f.write(json.dumps(r, default=_json_default) + "\n")
        f.flush()
        os.fsync(f.fileno())


def read_jsonl(path):
    """Read a jsonl file, tolerating a truncated last line (e.g. after a disconnect)."""
    path = Path(path)
    if not path.exists():
        return []
    rows = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                break
    return rows


class Logger:
    """Prints with timestamps and mirrors to a log file in the run dir."""

    def __init__(self, path=None):
        self.path = Path(path) if path else None
        if self.path:
            self.path.parent.mkdir(parents=True, exist_ok=True)

    def __call__(self, *msg):
        line = f"[{now()}] " + " ".join(str(m) for m in msg)
        print(line, flush=True)
        if self.path:
            with open(self.path, "a") as f:
                f.write(line + "\n")


def is_done(run_dir) -> bool:
    return (Path(run_dir) / "DONE").exists()


def mark_done(run_dir, info=None):
    atomic_write_json(Path(run_dir) / "DONE", {"time": now(), **(info or {})})
    failed = Path(run_dir) / "FAILED"
    if failed.exists():
        failed.rename(Path(run_dir) / f"FAILED.resolved.{int(time.time())}")


RUN_LOG_FIELDS = ["time", "run_id", "job", "status", "seconds", "message"]


def log_run(results_dir, run_id, job, status, seconds, message=""):
    path = Path(results_dir) / "run_log.csv"
    path.parent.mkdir(parents=True, exist_ok=True)
    new = not path.exists()
    with open(path, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=RUN_LOG_FIELDS)
        if new:
            w.writeheader()
        w.writerow({"time": now(), "run_id": run_id, "job": job, "status": status,
                    "seconds": round(seconds, 1), "message": str(message)[:500]})


@contextlib.contextmanager
def tracked_run(results_dir, run_dir, run_id, job):
    """Logs start/finish/failure of a run. Failures are never dropped: they write
    FAILED (with traceback) in the run dir and a row in results/run_log.csv, then re-raise."""
    run_dir = Path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    log_run(results_dir, run_id, job, "started", 0)
    try:
        yield
    except BaseException as e:  # includes KeyboardInterrupt (Colab "stop")
        tb = traceback.format_exc()
        (run_dir / "FAILED").write_text(f"{now()}\n{tb}")
        log_run(results_dir, run_id, job, "failed" if isinstance(e, Exception) else "interrupted",
                time.time() - t0, f"{type(e).__name__}: {e}")
        raise
    log_run(results_dir, run_id, job, "done", time.time() - t0)
