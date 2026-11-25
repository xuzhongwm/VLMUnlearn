"""UMU track, DEV-052: can a forget person be recognised behind the portrait from text alone, and does acting there stop
the leak? An activation-level test on the vanilla (a hook; no weight changes; not an unlearning method).

At one layer L after the merge, at the last prompt token:
  codes     forget person   = mean of the centred states of that person's NAME questions only (plain UM_QA questions); for a
                              query about attribute a, the code leaves attribute a out
            competitor      = a known retain person (retain profiles 101-170): one code from the name questions, one from the
                              portrait questions
            centring        = the mean over the competitors for each (route, attribute, prompt format)
  gate      fires when the nearest code (cosine) to the centred query state is a forget person's and the cosine is above tau;
            tau = per prompt format, the smallest value at which the gate fires on <= 5% of the questions of the calibration
            people (retain profiles 171-200, unknown to the gate)
  action    the state at the last prompt token is replaced by the centroid of its (route, attribute, format): at layer L
            only (gate, all), or at every layer from L to the last (gate_deep, all_deep; the verdict uses these)
Conditions: off (hook installed, never fires), gate / gate_deep, all / all_deep (fires on every question; unseen retain only).
Readout: UMU protocol (greedy generation, exact match) on the dc facts, forms txt and img, forget and unseen retain
(retain profiles 1-100); the share of questions on which the gate fires. Reading and thresholds: DEVIATIONS.md DEV-052.

  python scripts/mech_umu_merge_gate.py --config <model config> --layer <L> --dc-run <run id> [--set umu.ckpt=...]
Writes results/<model>/mech_umu/vanilla_<ckpt name>/merge_gate/{summary.json,items.jsonl,log.txt}.
"""
import io
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np  # noqa: E402
import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402
from PIL import Image  # noqa: E402

from scripts.mech_umu_identity_code import ATTRS  # noqa: E402
from vlmu.config import base_parser, resolve  # noqa: E402
from vlmu.mech.hooks import _hidden, _replace, component_modules, record_outputs  # noqa: E402
from vlmu.models import get_adapter  # noqa: E402
from vlmu.runs import free, model_dir, results_dir  # noqa: E402
from vlmu.umu import _response, image_bytes, load_umu, parse, profile_name, protocol_items  # noqa: E402
from vlmu.utils import Logger, atomic_write_json, read_json, write_jsonl  # noqa: E402

SUCCESS, NONE, RETAIN_MIN, FORGET_TXT_MAX, OFF_MIN, ALL_MAX, FP_MAX = 0.50, 0.80, 0.90, 0.70, 0.98, 0.60, 0.05      # DEV-052
TAUS = [round(0.05 * i, 2) for i in range(11)]
MODES = ("off", "gate", "gate_deep", "all", "all_deep")
N_COMP, N_KNOWN, N_EVAL = 70, 100, 100
ROUTE = {"txt": "N", "img": "F"}


def plain_rows(df, adapter, route):
    """One prompt-only row per (person, attribute): the plain UM_QA name question (N) or MM_QA portrait question (F)."""
    rows = []
    for pi, (_, row) in enumerate(df.iterrows()):
        qa, img = parse(row["UM_QA" if route == "N" else "MM_QA"]), image_bytes(row["image"])
        for ai, a in enumerate(ATTRS):
            q = qa["question"][a]
            assert (profile_name(row) in q) == (route == "N"), q
            rows.append({"person": pi, "attr": ai, "prompt": adapter.prompt(q, route == "F"), "image_bytes": img if route == "F" else None})
    return rows


def batch_of(adapter, rows):
    imgs = [Image.open(io.BytesIO(r["image_bytes"])).convert("RGB") for r in rows if r["image_bytes"] is not None]
    assert len(imgs) in (0, len(rows)), "batches must be homogeneous in image presence"
    return adapter.to_device(adapter._process([r["prompt"] for r in rows], imgs or None, "left"))


@torch.no_grad()
def last_states(adapter, model, rows, layers, bs):
    """Residual states after `layers` at the last prompt token, [n, len(layers), D] float32 on cpu (left-padded prompt-only
    batches); [n, D] for a single layer given as an int."""
    one = isinstance(layers, int)
    ls = [layers] if one else list(layers)
    out = []
    for i in range(0, len(rows), bs):
        batch, store = batch_of(adapter, rows[i:i + bs]), {}
        assert bool(batch["attention_mask"][:, -1].all())
        with record_outputs(model, "resid", ls, store), adapter.autocast():
            adapter.forward(model, batch, last_only=True)
        out.append(torch.stack([store[x][:, -1].float().cpu() for x in ls], 1))
    out = torch.cat(out)
    return out[:, 0] if one else out


class Gate:
    """Forward hook on decoder layer L. On the prompt pass of a generation it scores the last position against the codes and,
    where the gate fires, replaces the state by the centroid."""

    def __init__(self, keys, n_forget):
        self.keys, self.n_forget = keys, n_forget          # keys [A, K, D] normalised; the first n_forget are forget codes
        self.mode, self.tau, self.ctx, self.out, self.act = "off", 0.0, None, None, None

    def score(self, x, centroid, attr):
        q = F.normalize(x.float() - centroid, dim=-1)
        top, arg = torch.einsum("bd,bkd->bk", q, self.keys[attr]).max(1)
        return top, arg

    def __call__(self, mod, inp, out):
        h = _hidden(out)
        if self.ctx is None or h.shape[1] == 1:
            return None
        assert self.out is None and h.shape[1] == self.ctx["T"], "the gate must see the prompt pass exactly once"
        top, arg = self.score(h[:, -1], self.ctx["centroid"], self.ctx["attr"])
        fire = (arg < self.n_forget) & (top > self.tau)
        self.out = {"top": top.cpu(), "arg": arg.cpu(), "would_fire": fire.cpu()}
        self.act = fire if self.mode.startswith("gate") else torch.full_like(fire, self.mode.startswith("all"))
        if not bool(self.act.any()):
            return None
        h = h.clone()
        h[self.act, -1] = self.ctx["centroid"][self.act].to(h.dtype)
        return _replace(out, h)

    def follower(self, layer):
        """Hook for a layer above L: in the deep conditions, the same rows get that layer's centroid (prompt pass only)."""
        def hook(mod, inp, out):
            h = _hidden(out)
            if self.ctx is None or h.shape[1] == 1 or not self.mode.endswith("_deep") or not bool(self.act.any()):
                return None
            h = h.clone()
            h[self.act, -1] = self.ctx["deep"][layer][self.act].to(h.dtype)
            return _replace(out, h)
        return hook


def main():
    p = base_parser(__doc__)
    p.add_argument("--layer", type=int, required=True)
    p.add_argument("--dc-run", required=True, help="run id of the same vanilla whose baseline_dc.json defines the dc facts")
    p.add_argument("--batch-txt", type=int, default=32)
    p.add_argument("--batch-img", type=int, default=8)
    p.add_argument("--tag", default=None)
    p.add_argument("--smoke", action="store_true", help="few people and facts; output under merge_gate_smoke")
    args = p.parse_args()
    cfg = resolve(args)
    tag = args.tag or Path(cfg["umu"]["ckpt"]).name
    out = results_dir(cfg) / "mech_umu" / f"vanilla_{tag}" / ("merge_gate_smoke" if args.smoke else "merge_gate")
    log = Logger(out / "log.txt")
    L, dev = args.layer, None
    adapter = get_adapter({**cfg["model"], "id": cfg["umu"]["ckpt"], "processor_id": cfg["model"]["id"]})
    dfs = load_umu(cfg["paths"]["root"], N_EVAL + N_KNOWN, 0, log, cfg["umu"]["forget_split"])
    forget, unseen, known = dfs["forget"], dfs["retain"].iloc[:N_EVAL], dfs["retain"].iloc[N_EVAL:N_EVAL + N_KNOWN]
    n_comp = N_COMP
    if args.smoke:
        forget, unseen, known, n_comp = forget.head(10), unseen.head(10), known.head(16), 10
    assert not set(known["ID"]) & set(unseen["ID"]) and not set(forget["ID"]) & (set(known["ID"]) | set(unseen["ID"]))
    dc = set(read_json(model_dir(cfg) / "umu_unlearn" / args.dc_run / "baseline_dc.json")["pair_ids"])
    prompt_fn = None if cfg["model"]["family"] == "llava" else adapter.prompt
    proto = lambda d, s: [r for r in protocol_items({s: d}, sets=(s,), forms={s: ("img", "txt")}, prompt_fn=prompt_fn)  # noqa: E731
                          if r["field"] in ATTRS]
    model = adapter.load_model()
    model.eval()
    dev = next(model.parameters()).device
    bs = {"N": args.batch_txt, "F": args.batch_img}
    P_f, P_k, A = len(forget), len(known), len(ATTRS)
    log(f"vanilla {cfg['umu']['ckpt']} ({cfg['model']['id']}), layer {L}: {P_f} forget people, {n_comp} competitors + {P_k - n_comp} "
        f"calibration people (known retain), {len(unseen)} unseen retain people")

    # ---- fit: states of the plain questions. Forget people: the name route only.
    st = lambda df, route: last_states(adapter, model, plain_rows(df, adapter, route), L, bs[route]).view(len(df), A, -1)  # noqa: E731
    f_rows = plain_rows(forget, adapter, "N")
    assert all(r["image_bytes"] is None for r in f_rows), "no portrait of a forget person may enter the codes"
    SF_N = last_states(adapter, model, f_rows, L, bs["N"]).view(P_f, A, -1)
    SK = {r: st(known, r) for r in ("N", "F")}
    log("  states of the plain questions recorded (forget: name route only; known retain: both routes)")
    cen_plain = {r: SK[r][:n_comp].mean(0) for r in ("N", "F")}                                     # [A, D]
    kp = proto(known, "retain")
    n_layers = len(component_modules(model, "resid"))
    deep = list(range(L, n_layers))
    kp_deep = {m: last_states(adapter, model, [r for r in kp if r["modality"] == m], deep, bs[ROUTE[m]]) for m in ("txt", "img")}
    kp_state = {m: v[:, 0] for m, v in kp_deep.items()}
    comp_ids = set(known["ID"].iloc[:n_comp])
    cen_proto, cen_deep, n_cen = {}, {}, {}
    for m in ("txt", "img"):
        rows_m = [r for r in kp if r["modality"] == m]
        is_comp = torch.tensor([r["profile"] in comp_ids for r in rows_m])
        route_mean = kp_deep[m][is_comp].mean(0)
        cen = []
        for a in ATTRS:
            sel = is_comp & torch.tensor([r["field"] == a for r in rows_m])
            n_cen[f"{m}/{a}"] = int(sel.sum())
            cen.append(kp_deep[m][sel].mean(0) if int(sel.sum()) >= 5 else route_mean)      # attributes without protocol items: the route mean
        cen_deep[ROUTE[m]] = torch.stack(cen)                                               # [A, n deep layers, D]
        cen_proto[ROUTE[m]] = cen_deep[ROUTE[m]][:, 0]
    log(f"  protocol centroids from {int(is_comp.sum())} items per form; items per attribute {sorted(set(n_cen.values()))}")
    # codes
    cF = SF_N - cen_plain["N"][None]                                                               # [P_f, A, D]
    loo = (cF.sum(1, keepdim=True) - cF) / (A - 1)                                                 # [P_f, A, D]: code without attribute a
    comp = torch.cat([(SK[r][:n_comp] - cen_plain[r][None]).mean(1) for r in ("N", "F")])          # [2 * n_comp, D]
    keys = F.normalize(torch.cat([loo.transpose(0, 1), comp[None].expand(A, -1, -1)], 1), dim=-1).to(dev)      # [A, P_f + 2 n_comp, D]
    gate = Gate(keys, P_f)

    def fires(states, centroid, attr):
        top, arg = gate.score(states.to(dev), centroid.to(dev), attr.to(dev))
        return top.cpu(), arg.cpu()

    # ---- calibration of tau on people the gate does not know (plain and protocol, both routes)
    cal = {}
    ar = torch.arange(A).repeat(P_k - n_comp)
    for r in ("N", "F"):
        cal[f"plain/{r}"] = fires(SK[r][n_comp:].flatten(0, 1), cen_plain[r][ar], ar)
    cal_ids = set(known["ID"].iloc[n_comp:])
    for m in ("txt", "img"):
        rows_m = [r for r in kp if r["modality"] == m]
        sel = torch.tensor([r["profile"] in cal_ids for r in rows_m])
        at = torch.tensor([ATTRS.index(r["field"]) for r in rows_m])[sel]
        cal[f"protocol/{ROUTE[m]}"] = fires(kp_state[m][sel], cen_proto[ROUTE[m]][at], at)
    fp = {t: {k: ((arg < P_f) & (top > t)).float().mean().item() for k, (top, arg) in cal.items()} for t in TAUS}
    taus = {fmt: next((t for t in TAUS if max(v for k, v in fp[t].items() if k.startswith(fmt)) <= FP_MAX), TAUS[-1])
            for fmt in ("plain", "protocol")}
    tau, gate.tau = taus["plain"], taus["protocol"]            # the readout's prompts are in the protocol format
    log("  calibration (share of the calibration people's questions on which the gate fires): " + "; ".join(
        f"tau {t}: " + ", ".join(f"{k} {v:.2f}" for k, v in fp[t].items()) for t in TAUS if t <= max(taus.values()) + 0.051))
    log(f"  tau: plain {taus['plain']}, protocol {taus['protocol']}")

    # ---- the recogniser on the plain questions: forget people by portrait (never used above) and by name
    SF_F = st(forget, "F")
    arf, own = torch.arange(A).repeat(P_f), torch.arange(P_f).repeat_interleave(A)
    plain_fire = {}
    for r, S in (("N", SF_N), ("F", SF_F)):
        top, arg = fires(S.flatten(0, 1), cen_plain[r][arf], arf)
        plain_fire[f"forget/{r}"] = {"fires": ((arg < P_f) & (top > tau)).float().mean().item(), "nearest_is_own": (arg == own).float().mean().item(),
                                     "nearest_is_forget": (arg < P_f).float().mean().item(), "cos_nearest": top.mean().item()}
    for r in ("N", "F"):
        top, arg = cal[f"plain/{r}"]
        plain_fire[f"calibration/{r}"] = {"fires": ((arg < P_f) & (top > tau)).float().mean().item(),
                                          "nearest_is_forget": (arg < P_f).float().mean().item(), "cos_nearest": top.mean().item()}
    log("  plain questions: " + "; ".join(f"{k}: fires {v['fires']:.2f} (nearest code is a forget person's {v['nearest_is_forget']:.2f}"
                                          + (f", the own person's {v['nearest_is_own']:.2f}" if "nearest_is_own" in v else "") + ")"
                                          for k, v in plain_fire.items()))

    # ---- readout: UMU protocol generation with the hook
    rows = [r for s, d in (("forget", forget), ("retain", unseen)) for r in proto(d, s) if r["pair_id"] in dc]
    f_index = {pid: i for i, pid in enumerate(forget["ID"])}
    mods = component_modules(model, "resid")
    hds = [mods[L].register_forward_hook(gate)] + [mods[x].register_forward_hook(gate.follower(x)) for x in deep[1:]]
    recs = []
    for mode in MODES:
        gate.mode = mode
        todo = rows if not mode.startswith("all") else [r for r in rows if r["set"] == "retain"]
        for m in ("txt", "img"):
            group = sorted([r for r in todo if r["modality"] == m], key=lambda r: len(r["prompt"]))
            route = ROUTE[m]
            for i in range(0, len(group), bs[route]):
                chunk = group[i:i + bs[route]]
                batch = batch_of(adapter, chunk)
                attr = torch.tensor([ATTRS.index(r["field"]) for r in chunk], device=dev)
                cd = cen_deep[route].to(dev)[attr]                                           # [B, n deep layers, D]
                gate.ctx = {"centroid": cd[:, 0], "deep": {x: cd[:, k] for k, x in enumerate(deep)}, "attr": attr,
                            "T": batch["input_ids"].shape[1]}
                gate.out = gate.act = None
                with torch.no_grad():
                    gen = model.generate(**batch, max_new_tokens=50, do_sample=False, num_beams=1, use_cache=True,
                                         pad_token_id=adapter.tok.pad_token_id)
                texts = adapter.tok.batch_decode(gen[:, batch["input_ids"].shape[1]:], skip_special_tokens=True)
                assert gate.out is not None, "the gate did not see the prompt pass"
                for j, (r, t) in enumerate(zip(chunk, texts)):
                    resp, arg = _response(t), int(gate.out["arg"][j])
                    recs.append({"cond": mode, "uid": r["uid"], "set": r["set"], "modality": m, "profile": r["profile"], "field": r["field"],
                                 "pair_id": r["pair_id"], "correct_text": r["correct_text"], "response": resp, "correct": resp == r["correct_text"],
                                 "would_fire": bool(gate.out["would_fire"][j]), "cos_nearest": float(gate.out["top"][j]),
                                 "nearest": "forget" if arg < P_f else "retain_name" if arg < P_f + n_comp else "retain_face",
                                 "nearest_is_own": bool(r["set"] == "forget" and arg == f_index[r["profile"]])})
        log(f"  generated condition {mode}: {len(todo)} items")
    for hd in hds:
        hd.remove()
    gate.ctx = None
    model = None
    free()
    write_jsonl(out / "items.jsonl", recs)

    acc, fire, n = {}, {}, {}
    for mode in MODES:
        for s in ("forget", "retain"):
            for m in ("txt", "img"):
                rs = [x for x in recs if x["cond"] == mode and x["set"] == s and x["modality"] == m]
                if rs:
                    acc[f"{mode}/{s}/{m}"] = float(np.mean([x["correct"] for x in rs]))
                    fire[f"{mode}/{s}/{m}"] = float(np.mean([x["would_fire"] for x in rs]))
                    n[f"{s}/{m}"] = len(rs)
    own_rate = {m: float(np.mean([x["nearest_is_own"] for x in recs if x["cond"] == "off" and x["set"] == "forget" and x["modality"] == m]))
                for m in ("txt", "img")}
    off_ok = min(acc[f"off/{s}/{m}"] for s in ("forget", "retain") for m in ("txt", "img")) >= OFF_MIN
    readable = acc["all_deep/retain/img"] <= ALL_MAX
    qualifies = (acc["gate_deep/retain/txt"] >= RETAIN_MIN and acc["gate_deep/retain/img"] >= RETAIN_MIN
                 and acc["gate_deep/forget/txt"] <= FORGET_TXT_MAX)
    g = acc["gate_deep/forget/img"]
    verdict = ("not readable" if not readable else "not qualifying" if not qualifies else
               "success" if g <= SUCCESS else "no effect" if g >= NONE else "partial")
    summary = {"model": cfg["model"]["id"], "ckpt": cfg["umu"]["ckpt"], "layer": L, "dc_run": args.dc_run, "tau": taus, "calibration": fp,
               "protocol_centroid_items": n_cen, "plain_questions": plain_fire, "n": n, "accuracy": acc, "fires": fire,
               "forget_nearest_code_is_own": own_rate, "off_reproduces_vanilla": off_ok, "readable": readable, "qualifies": qualifies,
               "verdict": verdict}
    for mode in MODES:
        log(f"  {mode:>9}: " + " | ".join(f"{k.split('/', 1)[1]} acc {v:.2f} (gate would fire {fire[k]:.2f})" for k, v in acc.items()
                                         if k.startswith(mode + "/")))
    log(f"  forget questions whose nearest code is the own person's: txt {own_rate['txt']:.2f}, img {own_rate['img']:.2f}")
    atomic_write_json(out / "summary.json", summary)
    if not off_ok and not args.smoke:
        raise RuntimeError("condition off does not reproduce the vanilla on the dc facts")
    log(f"verdict: {verdict} (forget dc img under gate_deep {g:.2f}; readable {readable}; qualifies {qualifies})")
    log(f"wrote {out / 'summary.json'}")


if __name__ == "__main__":
    main()
