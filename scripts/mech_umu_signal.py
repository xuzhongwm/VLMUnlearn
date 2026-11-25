"""UMU track, DEV-044: what the text-only edit writes at the name, where it is carried, and whether it is shared.

On the forget dc facts of an unlearned checkpoint (name question, no image; first answer token, teacher forced;
theta_0 = the same model with the LoRA off; F(T) = lp_theta0 - lp_thetau):
  (A) state patching by position group and layer. At one layer the residual stream at a group of positions is exchanged
      between theta_u and theta_0 on the same prompt. Groups: name tokens / tokens after the name / last prompt token /
      tokens before the name / all prompt tokens.
        restore  theta_0's states into theta_u   R = 1 - F(patched) / F(T)
        inject   theta_u's states into theta_0   I = F(injected) / F(T)
  (B) is the written signal shared across people? delta = mean over a fact's name tokens of (h_u - h_0) at a layer.
        cosine between the deltas of different people;
        sub_mean    theta_u, minus the mean delta of the OTHER forget people at the name tokens -> recovery
        sub_rand    theta_u, minus a random vector of the same norm                            -> recovery (control)
        add_forget  theta_0, plus that mean delta at a forget person's name                    -> induced share
        add_retain  theta_0, plus the forget people's mean delta at a RETAIN person's name     -> induced share, top-1
  (C) does the name still point to the person? theta_u's state at the last name token against the theta_0 prototypes
      of all people (cosine after centring): share of people whose nearest prototype is their own.
Reading: DEVIATIONS.md DEV-044. Validity: patching theta_u with its own states changes log p by <= 0.05 nats.
--form img (DEV-046): the portrait-only question ("this person" + the portrait) instead of the name question; the
referent group is then the image tokens (still stored under the key "name"), and the output folder is signal_img.

  python scripts/mech_umu_signal.py --config <model config> --run-id <run> --step <N> --set umu.n_retain_eval=100 [umu.ckpt=...]
Writes results/<model>/mech_umu/<run>/step<N>/signal/{summary.json,items.jsonl,log.txt}.
"""
import contextlib
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np  # noqa: E402
import torch  # noqa: E402

from scripts.mech_umu_form_sweep import facts_for  # noqa: E402
from vlmu.config import base_parser, resolve  # noqa: E402
from vlmu.mech import tools  # noqa: E402
from vlmu.mech.hooks import _hidden, _replace, component_modules, decoder_layers, lm_parts, patch_outputs, \
    record_outputs  # noqa: E402
from vlmu.models import get_adapter  # noqa: E402
from vlmu.runs import free, model_dir, results_dir  # noqa: E402
from vlmu.umu import encode_teacher_forced, first_target, load_umu, token_groups_umu  # noqa: E402
from vlmu.utils import Logger, atomic_write_json, read_json, write_jsonl  # noqa: E402

TOL = 0.05
GROUPS = ("name", "after", "last", "before", "all")


@contextlib.contextmanager
def add_at(model, layer, rows, pos, vec):
    """Residual stream after `layer`: h[rows, pos] += vec (vec [n, D], aligned with rows / pos)."""
    mod = component_modules(model, "resid")[layer]

    def hook(m, inp, out):
        h = _hidden(out).clone()
        h[rows, pos] += vec.to(h.dtype)
        return _replace(out, h)
    hd = mod.register_forward_hook(hook)
    try:
        yield
    finally:
        hd.remove()


@torch.no_grad()
def first_lp(adapter, model, batch, t, vanilla=False, ctx=None):
    with contextlib.ExitStack() as st:
        if vanilla:
            st.enter_context(model.disable_adapter())
        if ctx is not None:
            st.enter_context(ctx)
        at = tools.forward_logits(adapter, model, batch)[t["b"], t["pos"]].float()
    return at.log_softmax(-1).gather(-1, t["tok"][:, None]).squeeze(-1).cpu(), (at.argmax(-1) == t["tok"]).cpu()


@torch.no_grad()
def states(adapter, model, batch, layers, vanilla):
    store = {}
    with contextlib.ExitStack() as st:
        if vanilla:
            st.enter_context(model.disable_adapter())
        st.enter_context(record_outputs(model, "resid", layers, store))
        tools.forward_logits(adapter, model, batch)
    return store


def group_index(name, last):
    """{group: (rows, positions)} over a right-padded batch; `last` = position of the last prompt token per row."""
    B, T = name.shape
    ar = torch.arange(T, device=name.device)[None].expand(B, T)
    first_n = torch.where(name, ar, T).min(1).values[:, None]
    last_n = torch.where(name, ar, -1).max(1).values[:, None]
    masks = {"name": name, "after": (ar > last_n) & (ar < last[:, None]), "last": ar == last[:, None], "before": ar < first_n,
             "all": ar <= last[:, None]}
    return {g: m.nonzero(as_tuple=True) for g, m in masks.items()}, last_n.squeeze(1)


def main():
    p = base_parser(__doc__)
    p.add_argument("--run-id", required=True)
    p.add_argument("--step", type=int, required=True)
    p.add_argument("--layers", default=None, help="default: every second layer and the last")
    p.add_argument("--dir-layers", default=None, help="layers of the direction tests (B); default: 4 layers spread over the first 60%%")
    p.add_argument("--batch", type=int, default=16)
    p.add_argument("--max-facts", type=int, default=0, help="smoke test")
    p.add_argument("--form", choices=["txt", "img"], default="txt", help="img: portrait-only question, referent = image tokens")
    args = p.parse_args()
    if args.form == "img" and args.batch == 16:
        args.batch = 4
    cfg = resolve(args)
    run_dir = model_dir(cfg) / "umu_unlearn" / args.run_id
    out = results_dir(cfg) / "mech_umu" / args.run_id / f"step{args.step}" / ("signal" if args.form == "txt" else "signal_img")
    log = Logger(out / "log.txt")
    from peft import PeftModel
    dc = set(read_json(run_dir / "baseline_dc.json")["pair_ids"])
    dfs = load_umu(cfg["paths"]["root"], cfg["umu"]["n_retain_eval"], 0, log, cfg["umu"]["forget_split"])
    facts = [f for f in sorted(facts_for(dfs, dc), key=lambda f: f["pair_id"])
             if args.form == "img" or (f["name"] and f["name"] in f["q_uni"])]
    sets = {s: [f for f in facts if f["set"] == s] for s in ("forget", "retain")}
    if args.max_facts:
        sets = {s: v[:args.max_facts] for s, v in sets.items()}
    adapter = get_adapter({**cfg["model"], "id": cfg["umu"]["ckpt"], "processor_id": cfg["model"]["id"]})
    model = PeftModel.from_pretrained(adapter.load_model(), str(run_dir / "ckpt" / f"step{args.step}"), is_trainable=False)
    model.eval()
    n_layers = len(decoder_layers(model))
    image_id = lm_parts(model)[2]
    layers = [int(x) for x in args.layers.split(",")] if args.layers else sorted(set(range(0, n_layers, 2)) | {n_layers - 1})
    dir_layers = ([int(x) for x in args.dir_layers.split(",")] if args.dir_layers
                  else [L for L in layers if L in {int(round(n_layers * q / 2)) * 2 for q in (0.15, 0.3, 0.45, 0.6)}])
    log(f"{args.run_id} step {args.step} ({cfg['model']['id']}): {len(sets['forget'])} forget and {len(sets['retain'])} retain dc "
        f"facts; layers {layers}; direction layers {dir_layers}")

    def batches(fs):
        for i in range(0, len(fs), args.batch):
            chunk = fs[i:i + args.batch]
            img = args.form == "img"
            rows = [{"pair_id": f["pair_id"], "uid": f"{f['pair_id']}/{args.form}", "correct_text": f["answer_text"], "name": f["name"],
                     "prompt": adapter.prompt(f["q_mul"] if img else f["q_uni"], img), "image_bytes": f["img"] if img else None}
                    for f in chunk]
            batch = encode_teacher_forced(adapter, rows)
            t = first_target(tools.answer_targets(batch))
            if img:
                ref = batch["input_ids"] == image_id
                if not bool((ref.sum(1) > 0).all()):
                    raise RuntimeError("image tokens not found in every row")
            else:
                groups, mism = token_groups_umu(adapter, model, rows, batch, t)
                ref = groups["name"]
                if mism or not bool((ref.sum(1) > 0).all()):
                    raise RuntimeError(f"name tokens not found in every row ({mism} mismatches)")
            yield chunk, batch, t, ref

    # ---- pass 1: baselines, the patching map (A), deltas and last-name-token states ---------------------------------------
    recs, delta, proto, checked = [], {}, {"0": {}, "u": {}}, False
    for set_name in ("forget", "retain"):
        for chunk, batch, t, name in batches(sets[set_name]):
            idx, last_n = group_index(name, t["pos"])
            lp0, top0 = first_lp(adapter, model, batch, t, vanilla=True)
            lpu, topu = first_lp(adapter, model, batch, t)
            s0, su = states(adapter, model, batch, layers, True), states(adapter, model, batch, layers, False)
            rows_b = torch.arange(name.shape[0], device=name.device)
            for j, f in enumerate(chunk):
                recs.append({"pair_id": f["pair_id"], "set": set_name, "profile": f["profile"], "field": f["field"],
                             "lp0": lp0[j].item(), "lpu": lpu[j].item(), "top0": bool(top0[j]), "topu": bool(topu[j])})
                for tag, s in (("0", s0), ("u", su)):
                    proto[tag].setdefault(f["profile"], []).append(torch.stack([s[L][j, last_n[j]].float().cpu() for L in layers]))
            if set_name != "forget":
                continue
            nm = name.float()[..., None]
            for j, f in enumerate(chunk):       # delta [n_layers, D]: mean over the row's name tokens of h_u - h_0
                delta[f["pair_id"]] = torch.stack([(((su[L][j] - s0[L][j]).float() * nm[j]).sum(0) / nm[j].sum()).cpu() for L in layers])
            if not checked:
                L0 = layers[len(layers) // 2]
                same, _ = first_lp(adapter, model, batch, t, ctx=patch_outputs(model, "resid", {L0: su[L0]}, idx["all"] + (idx["all"][1],)))
                err = (same - lpu).abs().max().item()
                log(f"  check: theta_u patched with its own states at layer {L0}: max |change| {err:.2e} nats (tolerance {TOL}); "
                    f"name tokens per row {name.sum(1).tolist()}")
                if err > TOL:
                    raise RuntimeError("the patch hook is not neutral")
                checked = True
            base = len(recs) - len(chunk)
            for L in layers:
                for g in GROUPS:
                    r, pos = idx[g]
                    a, at = first_lp(adapter, model, batch, t, ctx=patch_outputs(model, "resid", {L: s0[L]}, (r, pos, pos)))
                    b, bt = first_lp(adapter, model, batch, t, vanilla=True, ctx=patch_outputs(model, "resid", {L: su[L]}, (r, pos, pos)))
                    for j in range(len(chunk)):
                        recs[base + j][f"restore/{g}/{L}"] = a[j].item()
                        recs[base + j][f"inject/{g}/{L}"] = b[j].item()
                        recs[base + j][f"restore_top/{g}/{L}"] = bool(at[j])
                        recs[base + j][f"inject_top/{g}/{L}"] = bool(bt[j])
            log(f"  map: {sum(r['set'] == 'forget' for r in recs)}/{len(sets['forget'])} forget facts")
    fg = [r for r in recs if r["set"] == "forget"]
    rt = [r for r in recs if r["set"] == "retain"]
    m = lambda rs, k: float(np.mean([r[k] for r in rs]))  # noqa: E731
    FT = m(fg, "lp0") - m(fg, "lpu")
    summary = {"run_id": args.run_id, "step": args.step, "model": cfg["model"]["id"], "n_forget": len(fg), "n_retain": len(rt),
               "form": args.form, "referent": "name tokens" if args.form == "txt" else "image tokens",
               "layers": layers, "F_T": FT, "top1": {"forget_u": m(fg, "topu"), "forget_0": m(fg, "top0"), "retain_u": m(rt, "topu"),
                                                    "retain_0": m(rt, "top0")},
               "restore": {g: {L: 1 - (m(fg, "lp0") - m(fg, f"restore/{g}/{L}")) / FT for L in layers} for g in GROUPS},
               "inject": {g: {L: (m(fg, "lp0") - m(fg, f"inject/{g}/{L}")) / FT for L in layers} for g in GROUPS},
               "restore_top1": {g: {L: m(fg, f"restore_top/{g}/{L}") for L in layers} for g in GROUPS},
               "inject_top1": {g: {L: m(fg, f"inject_top/{g}/{L}") for L in layers} for g in GROUPS}}
    log(f"F(T) = {FT:.2f} nats; top-1 forget θu {summary['top1']['forget_u']:.2f} θ0 {summary['top1']['forget_0']:.2f}; retain θu "
        f"{summary['top1']['retain_u']:.2f} θ0 {summary['top1']['retain_0']:.2f}")
    for kind in ("restore", "inject"):
        for g in GROUPS:
            log(f"  (A) {kind:7s} {g:6s} " + " ".join(f"L{L}:{summary[kind][g][L]:+.2f}" for L in layers))
    for g in ("name", "last"):
        log(f"  (A) restore top-1 {g:6s} " + " ".join(f"L{L}:{summary['restore_top1'][g][L]:.2f}" for L in layers))

    # ---- (C) does the name still point to the person? ---------------------------------------------------------------------
    people = sorted(proto["0"])
    P0 = torch.stack([torch.stack(proto["0"][q]).mean(0) for q in people])           # [people, n_layers, D]
    Pu = torch.stack([torch.stack(proto["u"][q]).mean(0) for q in people])
    is_f = torch.tensor([q in {f["profile"] for f in sets["forget"]} for q in people])
    summary["identity"] = {}
    for li, L in enumerate(layers):
        mu = P0[:, li].mean(0, keepdim=True)
        a = torch.nn.functional.normalize(Pu[:, li] - mu, dim=1) @ torch.nn.functional.normalize(P0[:, li] - mu, dim=1).t()
        own = a.argmax(1) == torch.arange(len(people))
        summary["identity"][L] = {"forget_own_nearest": own[is_f].float().mean().item(), "retain_own_nearest": own[~is_f].float().mean().item(),
                                  "forget_cos_own": a.diag()[is_f].mean().item(), "retain_cos_own": a.diag()[~is_f].mean().item()}
    log("  (C) share whose nearest θ0 prototype is their own, forget: " + " ".join(f"L{L}:{summary['identity'][L]['forget_own_nearest']:.2f}" for L in layers))
    log("  (C)                                              retain: " + " ".join(f"L{L}:{summary['identity'][L]['retain_own_nearest']:.2f}" for L in layers))
    log("  (C) cosine to the own θ0 prototype, forget:               " + " ".join(f"L{L}:{summary['identity'][L]['forget_cos_own']:.2f}" for L in layers))

    # ---- (B) is the signal shared? -----------------------------------------------------------------------------------------
    prof = {f["pair_id"]: f["profile"] for f in sets["forget"]}
    ids = list(delta)
    D = torch.stack([delta[i] for i in ids])                                       # [facts, n_layers, D]
    owner = [prof[i] for i in ids]
    per_person = {q: D[[k for k, o in enumerate(owner) if o == q]].mean(0) for q in sorted(set(owner))}
    PP = torch.stack(list(per_person.values()))
    summary["delta"] = {}
    for li, L in enumerate(layers):
        x = torch.nn.functional.normalize(PP[:, li], dim=1)
        c = x @ x.t()
        off = c[~torch.eye(len(x), dtype=torch.bool)]
        summary["delta"][L] = {"cos_between_people": off.mean().item(), "norm": PP[:, li].norm(dim=1).mean().item(),
                               "norm_h0": P0[:, li].norm(dim=1).mean().item()}
    log("  (B) cosine of delta between people: " + " ".join(f"L{L}:{summary['delta'][L]['cos_between_people']:.2f}" for L in layers))
    log("  (B) |delta| / |h_0| at the name:    " + " ".join(f"L{L}:{summary['delta'][L]['norm'] / summary['delta'][L]['norm_h0']:.2f}" for L in layers))
    gen = torch.Generator().manual_seed(0)
    dir_recs = {}
    for L in dir_layers:
        li = layers.index(L)
        allmean = D[:, li].mean(0)
        for set_name in ("forget", "retain"):
            for chunk, batch, t, name in batches(sets[set_name]):
                idx, _ = group_index(name, t["pos"])
                r, pos = idx["name"]
                if set_name == "forget":
                    loo = torch.stack([D[[k for k, o in enumerate(owner) if o != f["profile"]], li].mean(0) for f in chunk])   # [B, D]
                    v = loo[r.cpu()].to(name.device)
                    rnd = torch.randn(loo.shape, generator=gen)
                    rnd = (rnd / rnd.norm(dim=1, keepdim=True) * loo.norm(dim=1, keepdim=True))[r.cpu()].to(name.device)
                    res = {"sub_mean": first_lp(adapter, model, batch, t, ctx=add_at(model, L, r, pos, -v)),
                           "sub_rand": first_lp(adapter, model, batch, t, ctx=add_at(model, L, r, pos, -rnd)),
                           "add_forget": first_lp(adapter, model, batch, t, vanilla=True, ctx=add_at(model, L, r, pos, v))}
                else:
                    v = allmean[None].expand(len(r), -1).to(name.device)
                    rnd = torch.randn(allmean.shape, generator=gen)
                    rnd = (rnd / rnd.norm() * allmean.norm())[None].expand(len(r), -1).to(name.device)
                    res = {"add_retain": first_lp(adapter, model, batch, t, vanilla=True, ctx=add_at(model, L, r, pos, v)),
                           "add_retain_rand": first_lp(adapter, model, batch, t, vanilla=True, ctx=add_at(model, L, r, pos, rnd))}
                for k, (lp, top) in res.items():
                    for j, f in enumerate(chunk):
                        dir_recs.setdefault((L, k), []).append((f["pair_id"], lp[j].item(), bool(top[j])))
    base = {r["pair_id"]: r for r in recs}
    summary["direction"] = {}
    for L in dir_layers:
        z = {}
        for k in ("sub_mean", "sub_rand"):
            rs = dir_recs[(L, k)]
            z[k] = 1 - float(np.mean([base[i]["lp0"] - lp for i, lp, _ in rs])) / FT
            z[k + "_top1"] = float(np.mean([top for _, _, top in rs]))
        for k in ("add_forget", "add_retain", "add_retain_rand"):
            rs = dir_recs[(L, k)]
            z[k] = float(np.mean([base[i]["lp0"] - lp for i, lp, _ in rs])) / FT
            z[k + "_top1"] = float(np.mean([top for _, _, top in rs]))
        summary["direction"][L] = z
        log(f"  (B) L{L}: sub_mean R {z['sub_mean']:+.2f} (top-1 {z['sub_mean_top1']:.2f}) | sub_rand R {z['sub_rand']:+.2f} | add_forget I "
            f"{z['add_forget']:+.2f} (top-1 {z['add_forget_top1']:.2f}) | add_retain I {z['add_retain']:+.2f} (top-1 {z['add_retain_top1']:.2f}; "
            f"random vector {z['add_retain_rand']:+.2f}, top-1 {z['add_retain_rand_top1']:.2f})")
    model = None
    free()
    write_jsonl(out / "items.jsonl", recs)
    R, inj = summary["restore"], summary["inject"]
    early = [L for L in layers if 4 <= L <= 12]
    summary["verdict"] = {"M1": max(R["name"][L] for L in early) >= 0.8, "M2": max(R["name"][L] for L in layers if L >= 22) <= 0.2,
                          "M3": min(R["last"][L] for L in layers if L >= 24) >= 0.8, "M4": max(inj["name"][L] for L in early) >= 0.5}
    best = max(dir_layers, key=lambda L: summary["direction"][L]["sub_mean"])
    z = summary["direction"][best]
    summary["verdict"]["direction_layer"] = best
    summary["verdict"]["shared"] = ("shared" if z["sub_mean"] >= 0.5 and z["sub_rand"] <= 0.1 else
                                    "person-specific" if z["sub_mean"] < 0.2 else "partly shared")
    summary["verdict"]["generic_flag"] = bool(z["sub_mean"] >= 0.5 and z["sub_rand"] <= 0.1
                                              and summary["top1"]["retain_0"] - z["add_retain_top1"] >= 0.3)
    log(f"DEV-044: {summary['verdict']}")
    atomic_write_json(out / "summary.json", summary)
    log(f"wrote {out / 'summary.json'}")


if __name__ == "__main__":
    main()
