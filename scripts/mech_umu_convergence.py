"""UMU track, DEV-024: where do the name route and the portrait route converge? Identity swap across routes.

The portrait reaches a person's facts without passing the text-only unlearning edit (DEV-020, DEV-021). This script
maps where, in the vanilla, "the person named X" and "the person in this portrait" start to share one representation
in the question stream, and on which side of that point the unlearning signal sits.

Prompts for a fact (person P, attribute; the readout's unimodal question q):
  N(P)  name route   "USER: {q}\\nASSISTANT:"
  F(P)  face route   "USER: <image>\\n{q with P's name -> 'this person'}\\nASSISTANT:" + P's portrait
Both share the tokens after the referent ("'s occupation?\\nASSISTANT:"). Each fact B is paired with the next fact A
(sorted) of the same attribute with a different answer.

Part 1, map (theta_0 = UMU's vanilla, no LoRA). At one layer L the residual stream at the shared suffix positions is
copied from a source run into a target run (end-aligned, as tools.cross_route_patch); the target is scored as a
two-way forced choice between A's and B's answer texts (summed log p, teacher forced). Conditions (source -> target):
  X1 F(A) -> N(B), X2 N(A) -> F(B)            cross-route (main)
  W1 N(A) -> N(B), W2 F(A) -> F(B)            within-route reference
  S1 F(B) -> N(B), S2 N(B) -> F(B)            same person, control
Swap share = share of pairs whose patched target prefers the source person's answer (S1/S2: keeps B's answer), on
pairs whose unpatched targets prefer B's answer in both routes.
Reading (DEV-024): convergence found if some layer has swap share >= 0.5 in both X1 and X2 (L* = the shallowest); no
shared format if max_L min(X1, X2) <= 0.2 while W1 and W2 reach >= 0.8; otherwise partial. Not interpretable if S1/S2
keep B in < 0.9 of pairs at any layer, or W1/W2 never reach 0.8.

Part 2, unlearned (theta_u, same person, forget facts; descriptive). The name route's suffix states N(P) are copied
into the face route F(P) at layer L; suppression share S(L) = (lp_F - lp_F,patched) / (lp_F - lp_N) on the correct
answer. theta_0 under the same copy is the control.

  # part 1, split over GPUs (one process per condition group; separate output folders):
  python scripts/mech_umu_convergence.py --config configs/llava/umu_npo_rt.yaml --part map --conds X1,W1,S1
  python scripts/mech_umu_convergence.py --config configs/llava/umu_npo_rt.yaml --part map --conds X2,W2,S2
  # part 2, per unlearned checkpoint:
  python scripts/mech_umu_convergence.py --config configs/llava/umu_npo_rt.yaml --part unlearned \\
      --run-id umu_npo_rt_txt_lr2e-04_s0_gate_rw5n200 --step 300
  # merge, verdict, figure:
  python scripts/mech_umu_convergence.py --config configs/llava/umu_npo_rt.yaml --report
Outputs: results/llava/mech_umu/<dc run>/step0/convergence/ (part 1 and the report) and
results/llava/mech_umu/<run>/step<N>/convergence_transfer/ (part 2).
"""
import contextlib
import io
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np  # noqa: E402
import torch  # noqa: E402
from PIL import Image  # noqa: E402

from scripts.mech_umu_form_sweep import answer_logprobs, facts_for  # noqa: E402
from vlmu.config import base_parser, resolve  # noqa: E402
from vlmu.mech import tools  # noqa: E402
from vlmu.mech.hooks import lm_parts, patch_outputs, record_outputs  # noqa: E402
from vlmu.models import get_adapter  # noqa: E402
from vlmu.runs import free, model_dir, results_dir  # noqa: E402
from vlmu.umu import encode_teacher_forced, load_umu  # noqa: E402
from vlmu.utils import Logger, atomic_write_json, read_json, read_jsonl, write_jsonl  # noqa: E402

DC_RUN = "umu_npo_rt_txt_lr2e-04_s0_gate_rw5n200"     # supplies the doubly-correct facts (66 forget + 122 retain)
CONDS = {"X1": ("F", "A", "N"), "X2": ("N", "A", "F"), "W1": ("N", "A", "N"), "W2": ("F", "A", "F"),
         "S1": ("F", "B", "N"), "S2": ("N", "B", "F")}          # (source route, source person, target route); target = B
FOUND, NONE_MAX, W_MIN, S_MIN = 0.5, 0.2, 0.8, 0.9               # DEV-024
SELF_TOL = 0.05


_PROMPT = None     # DEV-032: adapter.prompt for non-LLaVA families; None keeps the LLaVA strings of DEV-024/027


def name_prompt(f):
    return _PROMPT(f["q_uni"], False) if _PROMPT else f"USER: {f['q_uni']}\nASSISTANT:"


def face_prompt(f):
    q = f["q_uni"].replace(f["name"], "this person")
    return _PROMPT(q, True) if _PROMPT else f"USER: <image>\n{q}\nASSISTANT:"


def make_pairs(facts):
    """[(A, B)]: B paired with the next fact of the same attribute (sorted, cyclic) whose answer differs."""
    by = {}
    for f in facts:
        if f["name"] and f["name"] in f["q_uni"]:
            by.setdefault(f["field"], []).append(f)
    pairs = []
    for fs in by.values():
        fs = sorted(fs, key=lambda f: f["pair_id"])
        for i, b in enumerate(fs):
            for j in range(1, len(fs)):
                a = fs[(i + j) % len(fs)]
                if a["answer_text"].strip() != b["answer_text"].strip() and a["profile"] != b["profile"]:
                    pairs.append((a, b))
                    break
    return pairs


def prompt_of(route, f):
    return (name_prompt(f), None) if route == "N" else (face_prompt(f), f["img"])


def encode_prompts(adapter, items):
    imgs = [Image.open(io.BytesIO(i)).convert("RGB") for _, i in items if i is not None]
    assert len(imgs) in (0, len(items))
    return adapter.to_device(adapter._process([p for p, _ in items], imgs or None, "left"))


def common_suffix(a, b):
    k = 0
    while k < min(len(a), len(b)) and a[-1 - k] == b[-1 - k]:
        k += 1
    return k


def post_referent_len(adapter, f):
    """Number of final prompt tokens after the referent: the common suffix of N(f) and F(f) as text. Every condition
    copies only these positions (DEV-024: 'the question's tokens after the referent'), also when source and target
    share more (W2: two face prompts are identical as text)."""
    ids = lambda s: adapter.tok(s, add_special_tokens=False)["input_ids"]  # noqa: E731
    return common_suffix(ids(name_prompt(f)), ids(face_prompt(f)))


def align(src, tgt, max_k):
    """(rows, tgt_pos, src_pos): the common suffix of each source prompt (left-padded) and the target's prompt part
    (right-padded, answer appended), end-aligned and cut to max_k[i] tokens (the tokens after the referent)."""
    labels = tgt["labels"]
    Ts = src["input_ids"].shape[1]
    b, tp, sp, ks = [], [], [], []
    for i in range(labels.shape[0]):
        plen = int((labels[i] != -100).nonzero()[0])
        t = tgt["input_ids"][i, :plen].tolist()
        s = src["input_ids"][i, src["attention_mask"][i].bool()].tolist()
        k = min(common_suffix(s, t), max_k[i])
        assert k >= 3, f"row {i}: the routes share only {k} final tokens"
        ks.append(k)
        b += [i] * k
        tp += list(range(plen - k, plen))
        sp += list(range(Ts - k, Ts))
    dev = labels.device
    return tuple(torch.tensor(x, device=dev) for x in (b, tp, sp)), ks


@torch.no_grad()
def scored(adapter, model, tgt, store=None, index=None, layer=None, both=False):
    """Summed answer log p per row (DEV-024); both=True: (summed, first answer token) (DEV-032)."""
    ctx = patch_outputs(model, "resid", {layer: store[layer]}, index) if layer is not None else contextlib.nullcontext()
    if not both:
        with ctx:
            return answer_logprobs(adapter, model, tgt)[0]
    with ctx:                                   # as answer_logprobs, plus top-1 of the first answer token
        t = tools.answer_targets(tgt)
        logits = tools.forward_logits(adapter, model, tgt)
    lp = tools.target_logprob(logits[t["b"], t["pos"]], t["tok"])
    f = tools._first(t)
    first = torch.full((t["n_rows"],), float("nan"), device=lp.device)
    top = torch.zeros(t["n_rows"], dtype=torch.bool, device=lp.device)
    first[t["b"][f]] = lp[f]
    top[t["b"][f]] = logits[t["b"][f], t["pos"][f]].argmax(-1) == t["tok"][f]
    return (tools._per_row(lp, t).float().cpu().tolist(), first.float().cpu().tolist(), top.cpu().tolist())


def first_token(adapter, f, answer):
    """Id of the answer's first token after the prompt (the name and face prompts end identically: 'ASSISTANT:')."""
    p = name_prompt(f)
    full = adapter.tok(p + adapter.answer_suffix(answer), add_special_tokens=True)["input_ids"]
    return full[len(adapter.tok(p, add_special_tokens=True)["input_ids"])]


@torch.no_grad()
def first_logprobs(adapter, model, tgt, store=None, index=None, layer=None):
    """[B, V] log-probabilities of the first answer token (at the last prompt position), optionally patched."""
    ctx = patch_outputs(model, "resid", {layer: store[layer]}, index) if layer is not None else contextlib.nullcontext()
    with ctx:
        logits = tools.forward_logits(adapter, model, tgt)
    plen = [int((tgt["labels"][i] != -100).nonzero()[0]) for i in range(tgt["labels"].shape[0])]
    return torch.stack([logits[i, n - 1].float().log_softmax(-1) for i, n in enumerate(plen)]).cpu()


def run_map_first(adapter, model, pairs, conds, layers, args, log):
    """DEV-027 part A: the DEV-024 suffix swap, scored on the first answer token only."""
    recs, checked = [], False
    for c in conds:
        sroute, sperson, troute = CONDS[c]
        bs = args.batch_img if "F" in (sroute, troute) else args.batch_txt
        for i in range(0, len(pairs), bs):
            chunk = pairs[i:i + bs]
            src = encode_prompts(adapter, [prompt_of(sroute, a if sperson == "A" else b) for a, b in chunk])
            store = {}
            with torch.no_grad(), record_outputs(model, "resid", layers, store):
                tools.forward_logits(adapter, model, src)
            tgt = encode_teacher_forced(adapter, [tf_rows(troute, b, b["answer_text"]) for a, b in chunk])
            if not checked:
                self_patch_check(adapter, model, tgt, 16 if 16 in layers else layers[-1], log)
                checked = True
            index, ks = align(src, tgt, [post_referent_len(adapter, b) for _, b in chunk])
            if i == 0:
                log(f"  {c}: copied positions per row (first batch) {ks}")
            res = {"base": first_logprobs(adapter, model, tgt),
                   **{L: first_logprobs(adapter, model, tgt, store, index, L) for L in layers}}
            for j, (a, b) in enumerate(chunk):
                ta, tb = a["_tokA"], b["_tokB"]
                cell = lambda lp: {"A": float(lp[j, ta]), "B": float(lp[j, tb]), "top1": int(lp[j].argmax())}  # noqa: E731
                recs.append({"cond": c, "B": b["pair_id"], "A": a["pair_id"], "field": b["field"], "set": b["set"],
                             "tokA": ta, "tokB": tb, "base": cell(res["base"]),
                             "layers": {str(L): cell(res[L]) for L in layers}})
            store = None
        log(f"  {c}: {len(pairs)} pairs done")
    return recs


def run_imageswap(adapter, model, pairs, layers, args, log):
    """DEV-027 part B (after Cohen et al., ACL 2025): at one layer, the image-token states of F(A) replace those of
    F(B) (same positions: the two face prompts are the same text); first answer token scored."""
    img_id = lm_parts(model)[2]
    recs, checked = [], False
    for i in range(0, len(pairs), args.batch_img):
        chunk = pairs[i:i + args.batch_img]
        src = encode_teacher_forced(adapter, [tf_rows("F", a, b["answer_text"]) for a, b in chunk])
        tgt = encode_teacher_forced(adapter, [tf_rows("F", b, b["answer_text"]) for a, b in chunk])
        assert torch.equal(src["input_ids"], tgt["input_ids"]), "face prompts of A and B must tokenise identically"
        store = {}
        with torch.no_grad(), record_outputs(model, "resid", layers, store):
            tools.forward_logits(adapter, model, src)
        rows, pos = (tgt["input_ids"] == img_id).nonzero(as_tuple=True)
        index = (rows, pos, pos)
        if not checked:
            own = {}
            with torch.no_grad(), record_outputs(model, "resid", [layers[0]], own):
                tools.forward_logits(adapter, model, tgt)
            err = (first_logprobs(adapter, model, tgt, own, index, layers[0]) - first_logprobs(adapter, model, tgt)).abs().max()
            log(f"  image-swap self check: max |change in log p| = {float(err):.2e} (tolerance {SELF_TOL}); "
                f"{int((rows == 0).sum())} image positions per row")
            if err > SELF_TOL:
                raise RuntimeError("patching the image tokens with their own states changed log p")
            checked = True
        res = {"base": first_logprobs(adapter, model, tgt), **{L: first_logprobs(adapter, model, tgt, store, index, L) for L in layers}}
        for j, (a, b) in enumerate(chunk):
            ta, tb = a["_tokA"], b["_tokB"]
            cell = lambda lp: {"A": float(lp[j, ta]), "B": float(lp[j, tb]), "top1": int(lp[j].argmax())}  # noqa: E731
            recs.append({"cond": "IMG", "B": b["pair_id"], "A": a["pair_id"], "field": b["field"], "tokA": ta, "tokB": tb,
                         "base": cell(res["base"]), "layers": {str(L): cell(res[L]) for L in layers}})
        store = None
        log(f"  {min(i + args.batch_img, len(pairs))}/{len(pairs)} pairs")
    return recs


def summarize_transfer(rs, layers):
    """DEV-032: S(L) = (lp_F - lp_F,patched) / (theta_u's lp_F - lp_N), for theta_u and the theta_0 control, on the summed
    answer log p (as DEV-024) and on the first answer token; top-1 of the correct first token in the patched face route."""
    out = {}
    for key, (lf, ln, pt) in {"sum": ("lp_F", "lp_N", "patched"), "first": ("lp1_F", "lp1_N", "patched1")}.items():
        gap = float(np.mean([r["u"][lf] - r["u"][ln] for r in rs]))
        out[key] = {"gap_nats": gap, **{t: {str(L): float(np.mean([r[t][lf] - r[t][pt][str(L)] for r in rs]) / gap)
                                            for L in layers} for t in ("u", "0")}}
    out["top1"] = {t: {"N": float(np.mean([r[t]["top1_N"] for r in rs])), "F": float(np.mean([r[t]["top1_F"] for r in rs])),
                       **{str(L): float(np.mean([r[t]["patched_top1"][str(L)] for r in rs])) for L in layers}} for t in ("u", "0")}
    out["n_facts"] = len(rs)
    return out


def summarize_first(recs, layers):
    """DEV-027: swap share (pairwise first-token preference for the source person) and top-1 share per condition."""
    base_ok = {}
    for r in recs:
        if r["cond"] == "IMG":
            continue
        tr = CONDS[r["cond"]][2]
        base_ok.setdefault((r["B"], tr), r["base"]["B"] > r["base"]["A"])
    keep = {b for (b, tr), ok in base_ok.items() if ok and base_ok.get((b, "N" if tr == "F" else "F"), False)}
    out = {"n_pairs_kept": len(keep), "n_pairs": len({r["B"] for r in recs}), "curves": {}, "top1": {}}
    for c in list(CONDS) + ["IMG"]:
        rows = [r for r in recs if r["cond"] == c and (r["B"] in keep or (c == "IMG" and r["base"]["B"] > r["base"]["A"]))]
        if not rows:
            continue
        want, other = ("B", "A") if c.startswith("S") else ("A", "B")
        out["curves"][c] = {str(L): float(np.mean([r["layers"][str(L)][want] > r["layers"][str(L)][other] for r in rows])) for L in layers}
        tok = "tokB" if want == "B" else "tokA"
        out["top1"][c] = {str(L): float(np.mean([r["layers"][str(L)]["top1"] == r[tok] for r in rows])) for L in layers}
        out[f"n_{c}"] = len(rows)
    cur = out["curves"]
    first = lambda c: next((L for L in layers if cur[c][str(L)] >= FOUND), None)  # noqa: E731
    out["L_star"] = {c: first(c) for c in ("X1", "X2", "W1", "W2") if c in cur}
    if "IMG" in cur:
        out["L_c_image"] = max((L for L in layers if cur["IMG"][str(L)] >= FOUND), default=None)
    word = "incomplete"
    if all(k in cur for k in CONDS):
        valid = (min(min(cur["S1"].values()), min(cur["S2"].values())) >= S_MIN
                 and max(cur["W1"].values()) >= W_MIN and max(cur["W2"].values()) >= W_MIN)
        ls = out["L_star"]
        if not valid:
            word = "NOT INTERPRETABLE"
        elif None in (ls["X1"], ls["X2"]) or abs(ls["X1"] - ls["W1"]) > 4 or abs(ls["X2"] - ls["W2"]) > 4:
            word = "not supported"
        elif (abs(ls["X1"] - ls["W1"]) <= 2 and abs(ls["X2"] - ls["W2"]) <= 2
              and max(cur["X1"].values()) >= W_MIN and max(cur["X2"].values()) >= W_MIN):
            word = "convergence supported"
        else:
            word = "partial"
    out["verdict"] = word
    return out


def tf_rows(route, person, answer):
    p, img = prompt_of(route, person)
    return {"uid": f"{route}/{person['pair_id']}/{answer}", "prompt": p, "correct_text": answer, "image_bytes": img}


@torch.no_grad()
def self_patch_check(adapter, model, tgt, layer, log):
    store = {}
    with record_outputs(model, "resid", [layer], store):
        base = answer_logprobs(adapter, model, tgt)[0]
    n = tgt["input_ids"].shape[0]
    rows = torch.arange(n, device=tgt["input_ids"].device).repeat_interleave(3)
    pos = torch.stack([torch.arange(3) for _ in range(n)]).flatten().to(rows.device) + 1
    with patch_outputs(model, "resid", store, (rows, pos, pos)):
        again = answer_logprobs(adapter, model, tgt)[0]
    err = max(abs(a - b) for a, b in zip(base, again))
    log(f"  self-patch check (layer {layer}): max |change in log p| = {err:.2e} nats (tolerance {SELF_TOL})")
    if err > SELF_TOL:
        raise RuntimeError("patching a target with its own states changed log p: the hooks are not neutral")


def run_map(adapter, model, pairs, conds, layers, args, log):
    recs, checked = [], False
    for c in conds:
        sroute, sperson, troute = CONDS[c]
        bs = args.batch_img if "F" in (sroute, troute) else args.batch_txt
        for i in range(0, len(pairs), bs):
            chunk = pairs[i:i + bs]
            src = encode_prompts(adapter, [prompt_of(sroute, a if sperson == "A" else b) for a, b in chunk])
            store = {}
            with torch.no_grad(), record_outputs(model, "resid", layers, store):
                tools.forward_logits(adapter, model, src)
            res = {}
            for cand in ("A", "B"):
                tgt = encode_teacher_forced(adapter, [tf_rows(troute, b, (a if cand == "A" else b)["answer_text"])
                                                      for a, b in chunk])
                if not checked:
                    self_patch_check(adapter, model, tgt, 16 if 16 in layers else layers[-1], log)
                    checked = True
                index, ks = align(src, tgt, [post_referent_len(adapter, b) for _, b in chunk])
                if i == 0 and cand == "A":
                    log(f"  {c}: copied positions per row (first batch) {ks}")
                res[cand] = {"base": scored(adapter, model, tgt),
                             **{L: scored(adapter, model, tgt, store, index, L) for L in layers}}
            for j, (a, b) in enumerate(chunk):
                recs.append({"cond": c, "B": b["pair_id"], "A": a["pair_id"], "field": b["field"], "set": b["set"],
                             "base": {k: res[k]["base"][j] for k in "AB"},
                             "layers": {str(L): {k: res[k][L][j] for k in "AB"} for L in layers}})
            store = None
        log(f"  {c}: {len(pairs)} pairs done")
    return recs


def run_transfer(adapter, model, facts, layers, args, log):
    """Part 2: theta_u's (and theta_0's) name-route suffix states copied into the face route, same person."""
    recs = []
    for i in range(0, len(facts), args.batch_img):
        chunk = facts[i:i + args.batch_img]
        out = {}
        for tag, off in (("u", False), ("0", True)):
            ctx = model.disable_adapter() if off else contextlib.nullcontext()
            with ctx:
                src = encode_prompts(adapter, [prompt_of("N", f) for f in chunk])
                store = {}
                with torch.no_grad(), record_outputs(model, "resid", layers, store):
                    tools.forward_logits(adapter, model, src)
                tn = encode_teacher_forced(adapter, [tf_rows("N", f, f["answer_text"]) for f in chunk])
                tf = encode_teacher_forced(adapter, [tf_rows("F", f, f["answer_text"]) for f in chunk])
                index, _ = align(src, tf, [post_referent_len(adapter, f) for f in chunk])
                out[tag] = {"lp_N": scored(adapter, model, tn, both=True), "lp_F": scored(adapter, model, tf, both=True),
                            **{L: scored(adapter, model, tf, store, index, L, both=True) for L in layers}}
        for j, f in enumerate(chunk):
            recs.append({"pair_id": f["pair_id"], "field": f["field"],
                         **{t: {"lp_N": out[t]["lp_N"][0][j], "lp_F": out[t]["lp_F"][0][j],
                                "patched": {str(L): out[t][L][0][j] for L in layers},
                                "lp1_N": out[t]["lp_N"][1][j], "lp1_F": out[t]["lp_F"][1][j],
                                "patched1": {str(L): out[t][L][1][j] for L in layers},
                                "top1_N": out[t]["lp_N"][2][j], "top1_F": out[t]["lp_F"][2][j],
                                "patched_top1": {str(L): out[t][L][2][j] for L in layers}} for t in ("u", "0")}})
        log(f"  {min(i + args.batch_img, len(facts))}/{len(facts)} facts")
    return recs


def summarize_map(recs, layers):
    base_pref = {}
    for r in recs:
        tr = CONDS[r["cond"]][2]
        base_pref.setdefault((r["B"], tr), r["base"]["B"] > r["base"]["A"])
    keep = {b for (b, tr), ok in base_pref.items() if ok and base_pref.get((b, "N" if tr == "F" else "F"), False)}
    out = {"n_pairs_kept": len(keep), "n_pairs": len({r["B"] for r in recs}), "curves": {}}
    for c in CONDS:
        rows = [r for r in recs if r["cond"] == c and r["B"] in keep]
        if not rows:
            continue
        want = "B" if c.startswith("S") else "A"
        other = "A" if want == "B" else "B"
        out["curves"][c] = {str(L): float(np.mean([r["layers"][str(L)][want] > r["layers"][str(L)][other] for r in rows]))
                            for L in layers}
    cur = out["curves"]
    word, lstar = "incomplete", None
    if all(k in cur for k in CONDS):
        both = {L: min(cur["X1"][str(L)], cur["X2"][str(L)]) for L in layers}
        valid = (min(min(cur["S1"].values()), min(cur["S2"].values())) >= S_MIN
                 and max(cur["W1"].values()) >= W_MIN and max(cur["W2"].values()) >= W_MIN)
        hits = [L for L in layers if cur["X1"][str(L)] >= FOUND and cur["X2"][str(L)] >= FOUND]
        if not valid:
            word = "NOT INTERPRETABLE"
        elif hits:
            word, lstar = "convergence found", min(hits)
        elif max(both.values()) <= NONE_MAX:
            word = "no shared format in the question stream"
        else:
            word = "partial"
        out["max_min_X1_X2"] = max(both.values())
    out["verdict"], out["L_star"] = word, lstar
    return out


def plot(summary, transfer, layers, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    INK2, MUTED, GRID = "#52514e", "#898781", "#e1e0d9"
    col = {"X1": "#2a78d6", "X2": "#eb6834", "W1": "#1baf7a", "W2": "#eda100", "S1": "#e87ba4", "S2": "#6250d6"}
    lab = {"X1": "X1 face(A) → name(B)", "X2": "X2 name(A) → face(B)", "W1": "W1 name(A) → name(B)",
           "W2": "W2 face(A) → face(B)", "S1": "S1 face(B) → name(B), keeps B", "S2": "S2 name(B) → face(B), keeps B"}
    n = 2 if transfer else 1
    fig, axes = plt.subplots(1, n, figsize=(5.6 * n, 3.4), dpi=200, squeeze=False)
    ax = axes[0][0]
    for c, cur in summary["curves"].items():
        ax.plot(layers, [cur[str(L)] for L in layers], color=col[c], lw=2, ls="-" if c[0] == "X" else (":" if c[0] == "S" else "--"),
                label=lab[c])
    ax.axhline(0.5, color=MUTED, lw=0.8, ls=(0, (3, 2)))
    if summary.get("L_star") is not None:
        ax.axvline(summary["L_star"], color=MUTED, lw=0.8)
    ax.set_xlabel("layer of the copy", fontsize=7, color=INK2)
    ax.set_ylabel("share of pairs", fontsize=7, color=INK2)
    ax.set_title(f"θ0: identity swap at the shared question tokens (n={summary['n_pairs_kept']})", fontsize=8, loc="left")
    ax.legend(frameon=False, fontsize=5.5, loc="center right")
    if transfer:
        ax2 = axes[0][1]
        tcol = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100"]
        for k, (name, s) in enumerate(transfer.items()):
            ax2.plot(layers, [s["u"][str(L)] for L in layers], color=tcol[k % 4], lw=2, label=f"{name}: θu")
            ax2.plot(layers, [s["0"][str(L)] for L in layers], color=tcol[k % 4], lw=1, ls=":", label=f"{name}: θ0 control")
        if summary.get("L_star") is not None:
            ax2.axvline(summary["L_star"], color=MUTED, lw=0.8)
        ax2.set_xlabel("layer of the copy", fontsize=7, color=INK2)
        ax2.set_ylabel("suppression share S(L)", fontsize=7, color=INK2)
        ax2.set_title("name-route states copied into the face route, same person (forget)", fontsize=8, loc="left")
        ax2.legend(frameon=False, fontsize=5.5)
    for a in axes[0]:
        a.grid(color=GRID, lw=0.5)
        a.tick_params(labelsize=6.5, colors=MUTED)
        for side in ("top", "right"):
            a.spines[side].set_visible(False)
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def plot_first(summary, layers, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    INK2, MUTED, GRID = "#52514e", "#898781", "#e1e0d9"
    col = {"X1": "#2a78d6", "X2": "#eb6834", "W1": "#1baf7a", "W2": "#eda100", "S1": "#e87ba4", "S2": "#6250d6", "IMG": "#52514e"}
    lab = {"X1": "X1 face(A) → name(B)", "X2": "X2 name(A) → face(B)", "W1": "W1 name(A) → name(B)",
           "W2": "W2 face(A) → face(B)", "S1": "S1 face(B) → name(B), keeps B", "S2": "S2 name(B) → face(B), keeps B",
           "IMG": "image tokens of A → face(B) (Cohen et al.)"}
    fig, ax = plt.subplots(figsize=(6.4, 3.6), dpi=200)
    for c, cur in summary["curves"].items():
        ls = "-" if c[0] == "X" else (":" if c[0] == "S" else ("-." if c == "IMG" else "--"))
        ax.plot(layers, [cur[str(L)] for L in layers], color=col[c], lw=2, ls=ls, label=lab[c])
    ax.axhline(0.5, color=MUTED, lw=0.8, ls=(0, (3, 2)))
    ax.set_xlabel("layer of the copy", fontsize=7, color=INK2)
    ax.set_ylabel("share preferring the swapped-in person (first answer token)", fontsize=6.5, color=INK2)
    ax.set_title(f"θ0: identity swap, first-token readout (n={summary['n_pairs_kept']} pairs)", fontsize=8, loc="left")
    ax.legend(frameon=False, fontsize=5.5, loc="center left", bbox_to_anchor=(1.0, 0.5))
    ax.grid(color=GRID, lw=0.5)
    ax.tick_params(labelsize=6.5, colors=MUTED)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def main():
    p = base_parser(__doc__)
    p.add_argument("--part", choices=["map", "unlearned", "mapfirst", "imageswap"], default=None,
                   help="mapfirst / imageswap: DEV-027 (first-token readout; image-token swap after Cohen et al.)")
    p.add_argument("--report", action="store_true")
    p.add_argument("--report-first", action="store_true", help="DEV-027 report: mapfirst + imageswap")
    p.add_argument("--conds", default="X1,X2,W1,W2,S1,S2")
    p.add_argument("--run-id", default=None, help="part 2: the unlearned run")
    p.add_argument("--dc-run", default=DC_RUN, help="run whose baseline_dc.json supplies the facts (DEV-032: a Qwen run)")
    p.add_argument("--step", type=int, default=300)
    p.add_argument("--layers", default="0,2,4,6,8,10,12,14,16,18,20,22,24,26,28,30,31")
    p.add_argument("--max-pairs", type=int, default=0)
    p.add_argument("--batch-txt", type=int, default=16)
    p.add_argument("--batch-img", type=int, default=4)
    args = p.parse_args()
    cfg = resolve(args)
    layers = [int(x) for x in args.layers.split(",")]
    dc_dir = model_dir(cfg) / "umu_unlearn" / args.dc_run
    n_eval = read_json(dc_dir / "config.json")["umu"]["n_retain_eval"]
    base_out = results_dir(cfg) / "mech_umu" / args.dc_run / "step0" / "convergence"
    if args.report_first:
        log = Logger(base_out / "report_first_log.txt")
        recs = [r for d in sorted(base_out.glob("part1first_*")) for r in read_jsonl(d / "items.jsonl")]
        recs += read_jsonl(base_out / "imageswap" / "items.jsonl")
        s = summarize_first(recs, layers)
        for d in sorted((results_dir(cfg) / "mech_umu").glob("*/step*/convergence_transfer")):
            rs = read_jsonl(d / "items.jsonl")
            if "lp1_F" in rs[0]["u"]:            # DEV-032 items; DEV-024's (answer sum only) are left to --report
                s.setdefault("transfer", {})[f"{d.parent.parent.name}/{d.parent.name}"] = summarize_transfer(rs, layers)
        for name, t in s.get("transfer", {}).items():
            for key in ("sum", "first"):
                log(f"transfer {name} [{key}] (theta_u route gap {t[key]['gap_nats']:.2f} nats): θu "
                    + " ".join(f"L{L}:{t[key]['u'][str(L)]:+.2f}" for L in layers))
                log(f"{'':{len(name) + 25}s}θ0 " + " ".join(f"L{L}:{t[key]['0'][str(L)]:+.2f}" for L in layers))
            log(f"transfer {name} top-1 θu: N {t['top1']['u']['N']:.2f} F {t['top1']['u']['F']:.2f} patched "
                + " ".join(f"L{L}:{t['top1']['u'][str(L)]:.2f}" for L in layers))
        atomic_write_json(base_out / "summary_first.json", s)
        log(f"DEV-027: {s['n_pairs_kept']}/{s['n_pairs']} pairs kept; verdict: {s['verdict']}; first layer >= 0.5: {s['L_star']}; "
            f"image-token swap: last layer >= 0.5 = {s.get('L_c_image')}")
        for c, cur in s["curves"].items():
            log(f"  {c} (n={s[f'n_{c}']}): swap " + " ".join(f"L{L}:{cur[str(L)]:.2f}" for L in layers))
            log(f"  {'':{len(c) + 8}s}top-1 " + " ".join(f"L{L}:{s['top1'][c][str(L)]:.2f}" for L in layers))
        plot_first(s, layers, base_out / "convergence_first.png")
        log(f"wrote {base_out / 'summary_first.json'} and convergence_first.png")
        return
    if args.report:
        log = Logger(base_out / "report_log.txt")
        recs = [r for d in sorted(base_out.glob("part1_*")) for r in read_jsonl(d / "items.jsonl")]
        summary = summarize_map(recs, layers)
        transfer = {}
        for d in sorted((results_dir(cfg) / "mech_umu").glob("*/step*/convergence_transfer")):
            rs = read_jsonl(d / "items.jsonl")
            name = d.parent.parent.name.replace("umu_npo_rt_txt_lr2e-04_", "").replace("_gate_rw5n200", "")
            s = {}
            for t in ("u", "0"):
                gap = np.mean([r["u"]["lp_F"] - r["u"]["lp_N"] for r in rs])     # theta_u's route gap, both rows
                s[t] = {str(L): float(np.mean([r[t]["lp_F"] - r[t]["patched"][str(L)] for r in rs]) / gap) for L in layers}
            s["gap_nats"] = float(gap)
            transfer[name] = s
        summary["transfer"] = transfer
        atomic_write_json(base_out / "summary.json", summary)
        log(f"part 1: {summary['n_pairs_kept']}/{summary['n_pairs']} pairs kept; verdict: {summary['verdict']}; L* = {summary['L_star']}")
        for c, cur in summary["curves"].items():
            log(f"  {c}: " + " ".join(f"L{L}:{cur[str(L)]:.2f}" for L in layers))
        for name, s in transfer.items():
            log(f"part 2 {name} (theta_u route gap {s['gap_nats']:.2f} nats): θu " + " ".join(f"L{L}:{s['u'][str(L)]:+.2f}" for L in layers))
            log(f"{'':{len(name) + 7}s}θ0 " + " ".join(f"L{L}:{s['0'][str(L)]:+.2f}" for L in layers))
        plot(summary, transfer, layers, base_out / "convergence.png")
        log(f"wrote {base_out / 'summary.json'} and convergence.png")
        return
    dc = set(read_json(dc_dir / "baseline_dc.json")["pair_ids"])
    dfs = load_umu(cfg["paths"]["root"], n_eval, 0, print, cfg["umu"]["forget_split"])
    facts = sorted(facts_for(dfs, dc), key=lambda f: (f["set"], f["pair_id"]))
    adapter = get_adapter({**cfg["model"], "id": cfg["umu"]["ckpt"], "processor_id": cfg["model"]["id"]})
    global _PROMPT
    _PROMPT = None if cfg["model"]["family"] == "llava" else adapter.prompt
    if args.part in ("mapfirst", "imageswap"):
        pairs = make_pairs(facts)
        for f in {id(x): x for pr in pairs for x in pr}.values():
            f["_tokA"] = f["_tokB"] = first_token(adapter, f, f["answer_text"])
        n_all = len(pairs)
        pairs = [(a, b) for a, b in pairs if a["_tokA"] != b["_tokB"]]
        if args.max_pairs:
            pairs = pairs[:args.max_pairs]
        conds = args.conds.split(",")
        out = base_out / (f"part1first_{'_'.join(conds)}" if args.part == "mapfirst" else "imageswap")
        log = Logger(out / "log.txt")
        log(f"DEV-027 {args.part}: {len(pairs)} of {n_all} pairs have different first answer tokens; layers {layers}"
            + (f"; conditions {conds}" if args.part == "mapfirst" else ""))
        model = adapter.load_model()
        model.eval()
        recs = (run_map_first(adapter, model, pairs, conds, layers, args, log) if args.part == "mapfirst"
                else run_imageswap(adapter, model, pairs, layers, args, log))
        write_jsonl(out / "items.jsonl", recs)
        log(f"wrote {out / 'items.jsonl'}")
        model = None
        free()
        return
    if args.part == "map":
        conds = args.conds.split(",")
        out = base_out / f"part1_{'_'.join(conds)}"
        log = Logger(out / "log.txt")
        pairs = make_pairs(facts)
        if args.max_pairs:
            pairs = pairs[:args.max_pairs]
        log(f"part 1 (theta_0), conditions {conds}: {len(pairs)} pairs from {len(facts)} dc facts of {args.dc_run}; layers {layers}")
        a, b = pairs[0]
        log(f"  e.g. N(B) {name_prompt(b)!r}\n       F(A) {face_prompt(a)!r}; answers A {a['answer_text']!r} / B {b['answer_text']!r}")
        model = adapter.load_model()
        model.eval()
        recs = run_map(adapter, model, pairs, conds, layers, args, log)
        write_jsonl(out / "items.jsonl", recs)
        log(f"wrote {out / 'items.jsonl'}")
    else:
        from peft import PeftModel
        run_dir = model_dir(cfg) / "umu_unlearn" / args.run_id
        out = results_dir(cfg) / "mech_umu" / args.run_id / f"step{args.step}" / "convergence_transfer"
        log = Logger(out / "log.txt")
        run_dc = set(read_json(run_dir / "baseline_dc.json")["pair_ids"])
        fs = [f for f in facts if f["set"] == "forget" and f["pair_id"] in run_dc and f["name"] in f["q_uni"]]
        if args.max_pairs:
            fs = fs[:args.max_pairs]
        model = PeftModel.from_pretrained(adapter.load_model(), str(run_dir / "ckpt" / f"step{args.step}"), is_trainable=False)
        model.eval()
        log(f"part 2 {args.run_id} step {args.step}: {len(fs)} forget facts; layers {layers}")
        recs = run_transfer(adapter, model, fs, layers, args, log)
        write_jsonl(out / "items.jsonl", recs)
        log(f"wrote {out / 'items.jsonl'}")
    model = None
    free()


if __name__ == "__main__":
    main()
