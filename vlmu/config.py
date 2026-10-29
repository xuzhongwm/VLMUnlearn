"""YAML configs with `base:` inheritance, CLI overrides, and path resolution.

Every run is fully defined by its (resolved) config, which is saved to the run dir.
Paths:
  ROOT            --root > $VLMU_ROOT > /content/drive/MyDrive/vlm_unlearn
  ROOT/data/raw   HF downloads (shared by debug and full runs)
  OUT             ROOT (full) or ROOT/debug (--debug): data/processed, models, runs, results
"""
import argparse
import copy
import os
import shutil
from pathlib import Path

import yaml

REPO = Path(__file__).resolve().parent.parent
DEFAULT_ROOT = "/content/drive/MyDrive/vlm_unlearn"


def deep_merge(a: dict, b: dict) -> dict:
    out = copy.deepcopy(a)
    for k, v in b.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = deep_merge(out[k], v)
        else:
            out[k] = copy.deepcopy(v)
    return out


def load_yaml(path) -> dict:
    path = Path(path)
    if not path.is_absolute() and not path.exists():
        path = REPO / path
    with open(path) as f:
        cfg = yaml.safe_load(f) or {}
    bases = cfg.pop("base", [])
    bases = [bases] if isinstance(bases, str) else bases
    merged = {}
    for b in bases:
        merged = deep_merge(merged, load_yaml((path.parent / b).resolve()))
    return deep_merge(merged, cfg)


def set_dotted(cfg: dict, key: str, value):
    parts = key.split(".")
    d = cfg
    for p in parts[:-1]:
        d = d.setdefault(p, {})
    d[parts[-1]] = yaml.safe_load(value) if isinstance(value, str) else value


def base_parser(desc: str, config_required=True) -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=desc)
    p.add_argument("--config", required=config_required, help="YAML config (see configs/)")
    p.add_argument("--debug", action="store_true", help="~20 samples, <5 min, writes under ROOT/debug")
    p.add_argument("--root", default=None, help="output root (default $VLMU_ROOT or Drive)")
    p.add_argument("--set", nargs="*", default=[], metavar="KEY=VALUE", help="dotted config overrides")
    return p


def apply_overrides(cfg: dict, overrides) -> dict:
    for kv in overrides:
        k, v = kv.split("=", 1)
        set_dotted(cfg, k, v)
    return cfg


def resolve(args) -> dict:
    cfg = load_yaml(args.config) if args.config else {}
    if args.debug:
        cfg = deep_merge(cfg, cfg.get("debug_overrides", {}))
    apply_overrides(cfg, args.set)  # CLI wins over config and debug overrides
    cfg["cli_overrides"] = list(args.set)
    cfg["debug"] = bool(args.debug)
    root = Path(args.root or os.environ.get("VLMU_ROOT") or DEFAULT_ROOT)
    cfg["paths"] = paths_for(root, args.debug)
    cfg["config_file"] = str(args.config)
    return cfg


def paths_for(root: Path, debug: bool) -> dict:
    out = root / "debug" if debug else root
    return {
        "root": str(root),
        "raw": str(root / "data" / "raw"),
        "out": str(out),
        "processed": str(out / "data" / "processed"),
        "models": str(out / "models"),
        "runs": str(out / "runs"),
        "results": str(out / "results"),
    }


def localize(src, log=print) -> str:
    """Copy a Drive directory to fast local disk ($VLMU_LOCAL_CACHE) once per session.
    A directory is (re)copied when its STAMP file differs from the cached copy's."""
    cache = os.environ.get("VLMU_LOCAL_CACHE")
    src = Path(src)
    if not cache or not src.exists():
        return str(src)
    dst = Path(cache) / src.relative_to(src.anchor)
    stamp_src, stamp_dst = src / "STAMP", dst / "STAMP"
    if stamp_dst.exists() and stamp_src.exists() and stamp_dst.read_text() == stamp_src.read_text():
        return str(dst)
    log(f"copying {src} -> {dst} (local cache)")
    if dst.exists():
        shutil.rmtree(dst)
    shutil.copytree(src, dst)
    return str(dst)
