"""Exploratory: classify every checkpoint's standard-readout responses (txt form, dc facts) by style, for every run."""
import json, re, sys, glob, os, collections
sys.path.insert(0, "/scratch/xuzhong/VLMUnlearn")
import pandas as pd
from vlmu.umu import parse
V = "/scratch/xuzhong/vlm_unlearn"
names = {}
for f in glob.glob(f"{V}/data/raw/umu/*/train-*.parquet"):
    df = pd.read_parquet(f, columns=["ID", "Biography"])
    for _, r in df.iterrows():
        names[str(r["ID"]).zfill(3)] = parse(r["Biography"]).get("Name") or ""
REF = re.compile(r"^(i'm|i am|i's|i don't|i do not|i cannot|i can't|it's|it is|as an ai|sorry)\b", re.I)
n_ = lambda t: t.strip().rstrip(".")
def style(t, nm):
    if len(t.split()) < 6: return "short"
    if REF.match(t): return "refusal"
    return "bio" if nm and t.lower().startswith(nm.split()[0].lower()) else "other"
RUNS = {"run2 lr5e-5 w1 n50": "umu_npo_rt_txt_lr5e-05_s0", "run4 lr1e-4 w3 n200": "umu_npo_rt_txt_lr1e-04_s0_rw3n200",
        "G1 lr1e-4 w3": "umu_npo_rt_txt_lr1e-04_s0_gate_rw3n200", "G2 lr2e-4 w3": "umu_npo_rt_txt_lr2e-04_s0_gate_rw3n200",
        "G3s0 lr2e-4 w5": "umu_npo_rt_txt_lr2e-04_s0_gate_rw5n200", "G3s1 lr2e-4 w5": "umu_npo_rt_txt_lr2e-04_s1_gate_rw5n200",
        "G3s2 lr2e-4 w5": "umu_npo_rt_txt_lr2e-04_s2_gate_rw5n200", "G4 lr1e-4 w3 n300": "umu_npo_rt_txt_lr1e-04_s0_gate_rw3n300"}
def logratio(run_dir):
    out = {}
    for line in open(f"{run_dir}/log.txt"):
        m = re.search(r"step (\d+)/\d+ .*'log_ratio': (-?[\d.]+)", line)
        if m: out[int(m.group(1))] = float(m.group(2))
    return out
print(f"{'run':20s} {'step':>4s} | forget dc txt: {'acc':>5s} {'bio':>4s} {'refus':>5s} {'shortW':>6s} | retain: {'acc':>5s} {'bio':>4s} | log_ratio(nearest)")
for tag, rid in RUNS.items():
    rd = f"{V}/runs/llava/umu_unlearn/{rid}"
    if not os.path.exists(f"{rd}/baseline_dc.json"): print(tag, "missing"); continue
    dc = set(json.load(open(f"{rd}/baseline_dc.json"))["pair_ids"]); lr = logratio(rd)
    for p in sorted(glob.glob(f"{rd}/evals/step*.jsonl"), key=lambda x: int(re.search(r"step(\d+)", x).group(1))):
        step = int(re.search(r"step(\d+)", p).group(1))
        if step == 0: continue
        ev = [json.loads(l) for l in open(p)]
        row = {}
        for s in ("forget", "retain"):
            rows = [r for r in ev if r["set"] == s and r["modality"] == "txt" and r["pair_id"] in dc]
            c = collections.Counter(); ok = 0
            for r in rows:
                st = style(r["response"], names.get(r["profile"], "")); good = n_(r["response"]) == n_(r["correct_text"])
                ok += good; c[st] += 1; c["shortW"] += (st == "short" and not good)
            row[s] = (ok / len(rows), c["bio"] / len(rows), c["refusal"] / len(rows), c["shortW"] / len(rows))
        near = min(lr, key=lambda k: abs(k - step)) if lr else None
        f, rt = row["forget"], row["retain"]
        print(f"{tag:20s} {step:4d} | {f[0]:19.2f} {f[1]:4.2f} {f[2]:5.2f} {f[3]:6.2f} | {rt[0]:12.2f} {rt[1]:4.2f} | {lr.get(near, float('nan')):8.1f}")
