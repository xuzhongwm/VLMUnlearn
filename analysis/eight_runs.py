"""Exploratory (DEV-022 follow-up): what the eight UMU unlearning runs on this machine did, from existing outputs only
(evals/step*.jsonl generations, checkpoints.jsonl, training logs). No model is loaded."""
import collections, glob, json, os, re, sys
sys.path.insert(0, "/scratch/xuzhong/VLMUnlearn")
import numpy as np
import pandas as pd
from vlmu.umu import parse

V = "/scratch/xuzhong/vlm_unlearn"
RUNS = {"run2": "umu_npo_rt_txt_lr5e-05_s0", "run4": "umu_npo_rt_txt_lr1e-04_s0_rw3n200",
        "G1": "umu_npo_rt_txt_lr1e-04_s0_gate_rw3n200", "G2": "umu_npo_rt_txt_lr2e-04_s0_gate_rw3n200",
        "G4": "umu_npo_rt_txt_lr1e-04_s0_gate_rw3n300", "G3s1": "umu_npo_rt_txt_lr2e-04_s1_gate_rw5n200",
        "G3s0": "umu_npo_rt_txt_lr2e-04_s0_gate_rw5n200", "G3s2": "umu_npo_rt_txt_lr2e-04_s2_gate_rw5n200"}
names, opts = {}, {}
for f in glob.glob(f"{V}/data/raw/umu/*/train-*.parquet"):
    for _, r in pd.read_parquet(f, columns=["ID", "Biography"]).iterrows():
        names[str(r["ID"]).zfill(3)] = parse(r["Biography"]).get("Name") or ""
REF = re.compile(r"^(i'm|i am|i's|i don't|i do not|i cannot|i can't|it's|it is|as an ai|sorry)\b", re.I)
n_ = lambda t: t.strip().rstrip(".").strip()


def style(t, nm):
    if len(t.split()) < 6:
        return "short"
    if REF.match(t):
        return "refusal"
    return "bio" if nm and t.lower().startswith(nm.split()[0].lower()) else "other"


def load(rid):
    rd = f"{V}/runs/llava/umu_unlearn/{rid}"
    dc = set(json.load(open(f"{rd}/baseline_dc.json"))["pair_ids"])
    steps = sorted(int(re.search(r"step(\d+)", p).group(1)) for p in glob.glob(f"{rd}/evals/step*.jsonl"))
    ev = {s: [json.loads(l) for l in open(f"{rd}/evals/step{s}.jsonl")] for s in steps}
    ck = {json.loads(l)["step"]: json.loads(l) for l in open(f"{rd}/checkpoints.jsonl")}
    log = collections.defaultdict(dict)
    for line in open(f"{rd}/log.txt"):
        m = re.search(r"step (\d+)/\d+ .*?(\{.*\})", line)
        if m:
            log[int(m.group(1))] = eval(m.group(2), {"__builtins__": {}})   # the training log's own dict literal
    return rd, dc, steps, ev, ck, log


def txt_rows(ev, dc, s, setname, modality="txt"):
    return [r for r in ev[s] if r["set"] == setname and r["modality"] == modality and r["pair_id"] in dc]


data = {k: load(v) for k, v in RUNS.items()}
final = {k: max(d[2]) for k, d in data.items()}

print("\n=== 1. Final checkpoint: forget facts by attribute (txt form): correct / bio / short-wrong, n ===")
fields = ["Height", "Annual_Salary", "Birth", "Born", "Residence", "Education", "Occupation", "Medical_Conditions"]
print(f"{'run':6s} " + " ".join(f"{f[:11]:>16s}" for f in fields))
for k, (rd, dc, steps, ev, ck, log) in data.items():
    rows = txt_rows(ev, dc, final[k], "forget")
    cells = []
    for fld in fields:
        rr = [r for r in rows if r["field"] == fld]
        if not rr:
            cells.append(f"{'-':>16s}"); continue
        c = collections.Counter()
        for r in rr:
            st = style(r["response"], names.get(r["profile"], "")); ok = n_(r["response"]) == n_(r["correct_text"])
            c["ok"] += ok; c["bio"] += st == "bio"; c["sw"] += (st == "short" and not ok)
        cells.append(f"{c['ok']/len(rr):.2f}/{c['bio']/len(rr):.2f}/{c['sw']/len(rr):.2f} n{len(rr):<2d}")
    print(f"{k:6s} " + " ".join(f"{x:>16s}" for x in cells))

print("\n=== 2. Final checkpoint: what the wrong SHORT answers are (forget, txt) ===")
for k, (rd, dc, steps, ev, ck, log) in data.items():
    rows = [r for r in txt_rows(ev, dc, final[k], "forget")
            if style(r["response"], names.get(r["profile"], "")) == "short" and n_(r["response"]) != n_(r["correct_text"])]
    if not rows:
        print(f"{k:6s} no wrong short answers"); continue
    # option lists come from the img rows of the same pair (vlmu/umu.py: temp is in the prompt only); use step-0 responses of other people as a proxy pool
    vals = collections.Counter(n_(r["response"]) for r in rows)
    top = vals.most_common(4)
    van = {r["pair_id"]: n_(r["response"]) for r in txt_rows(ev, dc, 0, "forget")} if 0 in ev else {}
    other_people_answer = sum(1 for r in rows if n_(r["response"]) in {n_(x["correct_text"]) for x in txt_rows(ev, dc, final[k], "forget") if x["pair_id"] != r["pair_id"] and x["field"] == r["field"]})
    print(f"{k:6s} n={len(rows):2d}; distinct {len(vals):2d}; equal to another forget person's true answer for that attribute: {other_people_answer:2d}; most frequent: {top}")

print("\n=== 3. Portrait route and name+portrait over training (forget dc acc: txt / img / txt+img; retain img) ===")
for k, (rd, dc, steps, ev, ck, log) in data.items():
    line = []
    for s in steps:
        if s == 0:
            continue
        c = ck.get(s, {})
        line.append(f"{s}:{c.get('forget_dc_txt', float('nan')):.2f}/{c.get('forget_dc_img', float('nan')):.2f}/{(c.get('forget_dc_txtimg') or float('nan')):.2f}")
    print(f"{k:6s} " + "  ".join(line))
    print(f"{'':6s} final retain dc img {ck[final[k]]['retain_dc_img']:.3f}; min forget img over training {min(ck[s]['forget_dc_img'] for s in steps if s):.3f}")

print("\n=== 4. Is the style per person? (switch runs, final step: P(both facts of a person are bio) vs independence) ===")
for k, (rd, dc, steps, ev, ck, log) in data.items():
    rows = txt_rows(ev, dc, final[k], "forget")
    by = collections.defaultdict(list)
    for r in rows:
        by[r["profile"]].append(style(r["response"], names.get(r["profile"], "")) == "bio")
    pairs = [v for v in by.values() if len(v) == 2]
    p = np.mean([x for v in by.values() for x in v])
    both = np.mean([all(v) for v in pairs]) if pairs else float("nan")
    mixed = np.mean([v[0] != v[1] for v in pairs]) if pairs else float("nan")
    if p > 0.05:
        print(f"{k:6s} bio share {p:.2f}; people with 2 facts: {len(pairs)}; both bio {both:.2f} (independent {p*p:.2f}); one bio one not {mixed:.2f} (independent {2*p*(1-p):.2f})")

print("\n=== 5. Training losses (mean over the logged steps in each quarter of training): npo_loss / log_ratio / retain_nll ===")
for k, (rd, dc, steps, ev, ck, log) in data.items():
    ss = sorted(log)
    if not ss:
        print(k, "no log"); continue
    qs = np.array_split(np.array(ss), 4)
    parts = []
    for q in qs:
        g = lambda key: np.mean([log[s].get(key, np.nan) for s in q])  # noqa: E731
        parts.append(f"{g('npo_loss'):.2f}/{g('log_ratio'):6.1f}/{g('retain_nll'):.2f}")
    peak = max(ss, key=lambda s: log[s].get("retain_nll", 0))
    print(f"{k:6s} " + " | ".join(parts) + f"   peak retain_nll {log[peak].get('retain_nll'):.2f} at step {peak}")
