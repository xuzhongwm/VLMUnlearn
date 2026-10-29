"""Download the raw benchmark files from the HF Hub into ROOT/data/raw (idempotent)."""
from pathlib import Path

MLLMU = "MLLMMU/MLLMU-Bench"            # dataset repo (eval data + ft_Data)
SPLITS = "MLLMMU/baseline_train_split"  # model repo holding the official ID lists
MMBENCH = "lmms-lab/MMBench_EN"         # dataset repo


def mllmu_files(forget_split: str):
    ratio = int(forget_split.split("_")[1])
    return {
        "ft": "ft_Data/train-00000-of-00001.parquet",
        "full": "Full_Set/train-00000-of-00001.parquet",
        "celeb": "Retain_Set/train-00000-of-00001.parquet",
        "forget": f"{forget_split}/train-00000-of-00001.parquet",
        "test": ["Test_Set/train-00000-of-00002.parquet", "Test_Set/train-00001-of-00002.parquet"],
        "forget_ids": f"forget_{ratio}/forget_ids.json",
        "retain_ids": f"retain_{100 - ratio}/retain_ids.json",
    }


def download(raw_dir, forget_split: str, log=print) -> dict:
    """Returns {key: local path(s)}. Files already present are not re-downloaded."""
    from huggingface_hub import hf_hub_download, list_repo_files

    raw = Path(raw_dir)
    files = mllmu_files(forget_split)

    def get(repo, repo_type, fname, sub):
        dst = raw / sub / fname
        if not dst.exists() or dst.stat().st_size == 0:
            log(f"downloading {repo}/{fname}")
            hf_hub_download(repo, fname, repo_type=repo_type, local_dir=str(raw / sub))
        return str(dst)

    out = {}
    for k in ("ft", "full", "celeb", "forget"):
        out[k] = get(MLLMU, "dataset", files[k], "MLLMU-Bench")
    out["test"] = [get(MLLMU, "dataset", f, "MLLMU-Bench") for f in files["test"]]
    for k in ("forget_ids", "retain_ids"):
        out[k] = get(SPLITS, "model", files[k], "baseline_train_split")
    local_mmb = sorted((raw / "MMBench_EN" / "data").glob("dev-*.parquet"))
    if local_mmb:
        out["mmbench"] = str(local_mmb[0])
    else:
        fname = next(f for f in list_repo_files(MMBENCH, repo_type="dataset") if f.startswith("data/dev-"))
        out["mmbench"] = get(MMBENCH, "dataset", fname, "MMBench_EN")
    return out
