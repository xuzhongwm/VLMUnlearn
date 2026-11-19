"""UMU track: why does the unlearning signal not form when an image is in the prompt? (DEVIATIONS DEV-018)

DEV-015 to DEV-017 found that, on the text-only name question, unlearning builds a vector in the last prompt token's
state (by about layer 16) that switches the response from the short answer to a long sentence about the person. With an
image before the question the answer survives. This script asks how the image prevents the vector from forming.

Inputs, same fact and answer throughout (teacher-forced; readout = first answer token):
  text      USER: {name question}\nASSISTANT:                     the training form
  filler    USER: {576 filler text tokens}\n{name question}\n...  same length as an image, text only
  blank     USER: <image>\n{name question}\n...                   grey image
  own       USER: <image>\n{name question}\n...                   the person's portrait and the name
  portrait  USER: <image>\n{portrait question}\n...               the benchmark's image question, no name

A. Measurement. signal(layer) = projection of (h_u - h_0) at the last prompt token on the unit vector of the text
   form (mean change over other people's facts, as DEV-016), divided by the same projection on the text form: the
   share of the text-form signal that is present in each input.
B. Supplying the signal. theta_u with the text-form vector added at layer 16: does the answer fail on image inputs,
   including the real portrait question?
C. Attention knockout (the causal test). Hypothesis "dilution": tokens after the image read mostly image tokens, whose
   states the LoRA hardly changes, so the change does not accumulate. In the given layers the tokens after the image
   (question, "ASSISTANT:", answer) are stopped from attending to the image tokens (for `filler`: to the filler
   tokens). If dilution is the cause, the signal comes back and the answer fails under theta_u, while theta_0 under the
   same knockout still answers. Run on blank and own (the name still identifies the person); not on portrait, where
   the image is the only source of identity.

Readouts per input: top-1 accuracy of the first answer token, share of first tokens that are the first token of the
person's name (the long-sentence style), forgetting in nats, and the signal.

Reading (thresholds fixed in DEV-018 before the run; blank image, knockout in all layers, forget facts, layer 16):
  dilution supported   signal >= 0.5 and theta_u's top-1 falls at least half-way from its value with the image to its
                       value on the text form, with theta_0's top-1 under knockout >= 0.6
  dilution rejected    signal <= 0.2 (theta_0 control met)
  not interpretable    theta_0's top-1 under knockout < 0.6 (the knockout itself breaks the model)
  otherwise            partial

Default: the doubly-correct forget facts; about 8 minutes on an L4; not resumable. Opt-in: --with-retain,
--filler-knockout.

  python scripts/mech_umu_crossmodal.py --config configs/llava/umu_npo_rt.yaml \
    --run-id umu_npo_rt_txt_lr5e-05_s0 --step 150
Writes results/llava/mech_umu/<run>/step<N>/crossmodal/{summary.json,items.jsonl,crossmodal.png,log.txt}.
"""
import contextlib
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np  # noqa: E402
import torch  # noqa: E402

from scripts.mech_umu_form_sweep import blank_image_bytes, facts_for, filler_string  # noqa: E402
from scripts.mech_umu_signal_transfer import held_out_means  # noqa: E402
from vlmu.config import base_parser, resolve  # noqa: E402
from vlmu.mech import tools  # noqa: E402
from vlmu.mech.hooks import decoder_layers, lm_parts, patch_outputs, record_outputs  # noqa: E402
from vlmu.models import get_adapter  # noqa: E402
from vlmu.runs import free, model_dir, results_dir  # noqa: E402
from vlmu.umu import encode_teacher_forced, first_target, load_umu, umu_run_id  # noqa: E402
from vlmu.utils import Logger, atomic_write_json, read_json, tracked_run, write_jsonl  # noqa: E402

LGRID = (4, 8, 12, 16, 20, 24)
PRIMARY = 16
CONTEXTS = ("filler", "blank", "own", "portrait")
BANDS = {"all": (0, 99), "0-15": (0, 16)}
CHECK_TOL = 0.05                                   # nats; an explicit causal mask must reproduce the unhooked run
SIGNAL_MIN, SIGNAL_REJECT, VANILLA_MIN = 0.5, 0.2, 0.6   # DEV-018


def make_rows(fact, blank, filler):
    shared = {"pair_id": fact["pair_id"], "set": fact["set"], "profile": fact["profile"], "correct_text": fact["answer_text"]}
    q, qm = fact["q_uni"], fact["q_mul"]
    spec = {"text": (f"USER: {q}\nASSISTANT:", None), "filler": (f"USER: {filler}\n{q}\nASSISTANT:", None),
            "blank": (f"USER: <image>\n{q}\nASSISTANT:", blank), "own": (f"USER: <image>\n{q}\nASSISTANT:", fact["img"]),
            "portrait": (f"USER: <image>\n{qm}\nASSISTANT:", fact["img"])}
    return {k: {**shared, "uid": f"{fact['pair_id']}/{k}", "prompt": p, "image_bytes": img} for k, (p, img) in spec.items()}


@contextlib.contextmanager
def block_attention(model, layer_ids, rows, cols):
    """In the given decoder layers, the tokens at positions `rows` cannot attend to the tokens at positions `cols`.
    The layer's additive attention mask is replaced by a copy with those entries at the minimum (a causal mask is built
    when the layer received none). Full forward passes only (no generation cache)."""
    layers = [m for _, m in decoder_layers(model)]

    def pre(_module, args, kwargs):
        h = args[0] if args else kwargs["hidden_states"]
        n = h.shape[1]
        low = torch.finfo(h.dtype).min
        mask = kwargs.get("attention_mask")
        if mask is None:
            mask = torch.full((n, n), low, device=h.device, dtype=h.dtype).triu(1)[None, None].expand(h.shape[0], 1, n, n)
        if mask.dtype == torch.bool:
            raise RuntimeError("expected an additive attention mask in the decoder layers, got a boolean one")
        mask = mask.clone()
        if len(rows) and len(cols):
            mask[:, :, rows[:, None], cols[None, :]] = low
        kwargs["attention_mask"] = mask
        return args, kwargs
    handles = [layers[i].register_forward_pre_hook(pre, with_kwargs=True) for i in layer_ids]
    try:
        yield
    finally:
        for handle in handles:
            handle.remove()


@torch.no_grad()
def run(adapter, model, batch, t, name_tok, vanilla, add=None, block=None):
    """One forward pass. Returns (state of the last prompt token at LGRID, [B, n_layers, D] float32 on the CPU;
    per row: log p of the first answer token, whether it is the argmax, whether the argmax is the name's first token).
    add = (layer, [B, D] state to put at the last prompt token); block = (layer_ids, rows, cols)."""
    store = {}
    with contextlib.ExitStack() as stack:
        if vanilla:
            stack.enter_context(model.disable_adapter())
        stack.enter_context(record_outputs(model, "resid", LGRID, store, select=(t["b"], t["pos"])))
        if add is not None:
            layer, state = add
            n_rows, n_tok = batch["input_ids"].shape
            cached = torch.zeros(n_rows, n_tok, state.shape[-1], device=t["b"].device, dtype=adapter.dtype)
            cached[t["b"], t["pos"]] = state.to(device=t["b"].device, dtype=adapter.dtype)
            stack.enter_context(patch_outputs(model, "resid", {layer: cached}, (t["b"], t["pos"], t["pos"])))
        if block is not None:
            stack.enter_context(block_attention(model, *block))
        at = tools.forward_logits(adapter, model, batch)[t["b"], t["pos"]].float()
    lp = at.log_softmax(-1).gather(-1, t["tok"][:, None]).squeeze(-1)
    arg = at.argmax(-1)
    states = torch.stack([store[layer].float().cpu() for layer in LGRID], 1)
    return states, {"lp": lp.cpu().tolist(), "top1": (arg == t["tok"]).cpu().tolist(), "name": (arg == name_tok.to(arg.device)).cpu().tolist()}


def prefix_positions(batch, t, context, image_id, n_open, newline_id):
    """(rows, cols) for the knockout: cols = the image tokens (blank / own) or the filler tokens (filler); rows = every
    position after them. Checked to be the same in every row of the batch."""
    ids = batch["input_ids"]
    if context == "filler":
        first_nl = [int((ids[i] == newline_id).nonzero()[0]) for i in range(ids.shape[0])]
        if len(set(first_nl)) != 1:
            raise ValueError("the filler prefix does not end at the same position in every row")
        cols = torch.arange(n_open, first_nl[0], device=ids.device)
    else:
        is_img = ids == image_id
        if not bool((is_img == is_img[0]).all()) or not bool(is_img[0].any()):
            raise ValueError("the image tokens are not at the same positions in every row")
        cols = is_img[0].nonzero().squeeze(-1)
    rows = torch.arange(int(cols[-1]) + 1, ids.shape[1], device=ids.device)
    if int(t["pos"].min()) <= int(cols[-1]):
        raise ValueError("the last prompt token is not after the blocked tokens")
    return rows, cols


def mean_of(recs, key):
    vals = [r[key] for r in recs if r.get(key) is not None]
    return float(np.mean([float(v) for v in vals])) if vals else None


def signal_share(recs, ctx_key, k):
    """Mean projection on the text-form unit vector in one input / the same on the text form, at layer index k."""
    rows = [r for r in recs if ctx_key in r["proj"]]
    den = float(np.mean([r["proj"]["text"][k] for r in rows])) if rows else 0.0
    return float(np.mean([r["proj"][ctx_key][k] for r in rows]) / den) if rows and abs(den) > 1e-8 else None


def summarize(recs_all, contexts, ko_contexts):
    out = {}
    for set_name in ("forget", "retain"):
        recs = [r for r in recs_all if r["set"] == set_name]
        if not recs:
            continue
        s = {"n_facts": len(recs), "layers": list(LGRID), "inputs": {}}
        for c in ("text",) + tuple(contexts):
            d = {"top1_u": mean_of(recs, f"{c}/top1_u"), "top1_0": mean_of(recs, f"{c}/top1_0"),
                 "name_u": mean_of(recs, f"{c}/name_u"), "name_0": mean_of(recs, f"{c}/name_0"),
                 "forgetting": float(np.mean([r[f"{c}/lp_0"] - r[f"{c}/lp_u"] for r in recs])),
                 "signal": [signal_share(recs, c, k) for k in range(len(LGRID))]}
            if c != "text":
                d["add_vector"] = {"top1_u": mean_of(recs, f"{c}/add/top1_u"), "name_u": mean_of(recs, f"{c}/add/name_u")}
            if c in ko_contexts:
                d["knockout"] = {}
                for band in BANDS:
                    key = f"{c}/ko:{band}"
                    d["knockout"][band] = {
                        "top1_u": mean_of(recs, f"{key}/top1_u"), "top1_0": mean_of(recs, f"{key}/top1_0"),
                        "name_u": mean_of(recs, f"{key}/name_u"), "name_0": mean_of(recs, f"{key}/name_0"),
                        "forgetting": float(np.mean([r[f"{key}/lp_0"] - r[f"{key}/lp_u"] for r in recs])),
                        "signal": [signal_share(recs, key, k) for k in range(len(LGRID))]}
            s["inputs"][c] = d
        out[set_name] = s
    return out


def fmt(v):
    return "    -" if v is None else f"{v:5.2f}"


def verdict(summary):
    f = summary.get("forget")
    if not f or "knockout" not in f["inputs"].get("blank", {}):
        return ["no knockout on the blank-image input: no verdict"]
    k = LGRID.index(PRIMARY)
    text, blank = f["inputs"]["text"], f["inputs"]["blank"]
    ko = blank["knockout"]["all"]
    sig, tu, t0 = ko["signal"][k], ko["top1_u"], ko["top1_0"]
    half = (blank["top1_u"] + text["top1_u"]) / 2
    head = (f"blank image, attention to the image blocked in all layers: signal {fmt(sig)} of the text form (unblocked "
            f"{fmt(blank['signal'][k])}); top-1 unlearned {fmt(blank['top1_u'])} -> {fmt(tu)} (text form {fmt(text['top1_u'])}, "
            f"half-way {half:.2f}); vanilla {fmt(blank['top1_0'])} -> {fmt(t0)}")
    if t0 is None or t0 < VANILLA_MIN:
        return [head + " -> not interpretable: the knockout breaks the vanilla"]
    if sig is not None and sig >= SIGNAL_MIN and tu <= half:
        return [head + " -> dilution supported"]
    if sig is not None and sig <= SIGNAL_REJECT:
        return [head + " -> dilution rejected"]
    return [head + " -> partial, no verdict under DEV-018"]


def plot(summary, contexts, ko_contexts, title, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    INK2, MUTED, GRID, AXIS = "#52514e", "#898781", "#e1e0d9", "#c3c2b7"
    colors = {"text": "#111111", "filler": "#898781", "blank": "#2a78d6", "own": "#1baf7a", "portrait": "#eda100"}
    names = {"text": "name question", "filler": "576 text tokens + name question", "blank": "blank image + name question",
             "own": "portrait + name question", "portrait": "portrait question (no name)"}
    f = summary["forget"]["inputs"]
    num = lambda xs: [np.nan if v is None else v for v in xs]  # noqa: E731
    fig, axes = plt.subplots(1, 3, figsize=(14.5, 3.8), dpi=200)
    ax = axes[0]
    for c in ("text",) + tuple(contexts):
        ax.plot(LGRID, num(f[c]["signal"]), color=colors[c], lw=1.8, marker="o", ms=3, label=names[c])
    ax.set_title("A. How much of the text-form signal is present\n(last prompt token, share of the text form)", loc="left",
                 fontsize=8, color=INK2)
    ax.set_xlabel("decoder layer", color=INK2, fontsize=8)
    ax.legend(frameon=False, fontsize=6.5)
    ax = axes[1]
    for c in ko_contexts:
        ax.plot(LGRID, num(f[c]["signal"]), color=colors[c], lw=1.4, ls=":", marker="o", ms=3, label=f"{names[c]}")
        for band, ls in (("all", "-"), ("0-15", "--")):
            if band in f[c]["knockout"]:
                ax.plot(LGRID, num(f[c]["knockout"][band]["signal"]), color=colors[c], lw=1.8, ls=ls, marker="o", ms=3,
                        label=f"  attention to the prefix blocked, layers {band}")
    ax.set_title("C. Signal when the tokens after the image\ncannot attend to it", loc="left", fontsize=8, color=INK2)
    ax.set_xlabel("decoder layer", color=INK2, fontsize=8)
    ax.legend(frameon=False, fontsize=6)
    for a in axes[:2]:
        for y in (0, 1):
            a.axhline(y, color=AXIS, lw=0.9)
    ax = axes[2]
    labels, vals, cols = [], [], []
    for c in ("text",) + tuple(contexts):
        labels.append(names[c].replace(" + ", "\n+ ").replace(" (", "\n("))
        vals.append(f[c]["top1_u"])
        cols.append(colors[c])
        if c != "text":
            labels.append("+ vector")
            vals.append(f[c]["add_vector"]["top1_u"])
            cols.append("#e34948")
        if c in ko_contexts:
            labels.append("attention\nblocked")
            vals.append(f[c]["knockout"]["all"]["top1_u"])
            cols.append("#6a4fd8")
    ax.bar(range(len(vals)), [0 if v is None else v for v in vals], color=cols, width=0.72)
    ax.set_xticks(range(len(vals)), labels, fontsize=5.5)
    ax.set_ylim(0, 1)
    ax.set_title("B, C. Unlearned model: correct first answer token is the most likely one", loc="left", fontsize=8, color=INK2)
    for a in axes:
        for side in ("top", "right"):
            a.spines[side].set_visible(False)
        a.grid(axis="y", color=GRID, lw=0.6)
        a.set_axisbelow(True)
        a.tick_params(colors=MUTED, labelsize=8)
    axes[2].tick_params(axis="x", labelsize=5.5)
    fig.suptitle(title, x=0.01, ha="left", fontsize=9, color=INK2)
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def main():
    p = base_parser(__doc__)
    p.add_argument("--run-id", default=None)
    p.add_argument("--step", type=int, required=True)
    p.add_argument("--with-retain", action="store_true", help="also score the doubly-correct retain facts")
    p.add_argument("--filler-knockout", action="store_true", help="also block attention to the filler text tokens")
    p.add_argument("--max-facts", type=int, default=0, help="0 = all selected facts")
    p.add_argument("--batch-txt", type=int, default=16)
    p.add_argument("--batch-img", type=int, default=4)
    args = p.parse_args()
    cfg = resolve(args)
    run_id = args.run_id or umu_run_id(cfg)
    run_dir = model_dir(cfg) / "umu_unlearn" / run_id
    ckpt = run_dir / "ckpt" / f"step{args.step}"
    if not (ckpt / "adapter_config.json").exists():
        raise SystemExit(f"no adapter at {ckpt}")
    out = results_dir(cfg) / "mech_umu" / run_id / f"step{args.step}" / "crossmodal"
    log = Logger(out / "log.txt")
    with tracked_run(results_dir(cfg), out, f"mech_umu_crossmodal_{run_id}_step{args.step}", "mech_umu_crossmodal"):
        from peft import PeftModel
        dc = set(read_json(run_dir / "baseline_dc.json")["pair_ids"])
        dfs = load_umu(cfg["paths"]["root"], cfg["umu"]["n_retain_eval"], 0, log, cfg["umu"]["forget_split"])
        all_facts = sorted(facts_for(dfs, dc), key=lambda f: (f["set"], f["pair_id"]))     # the vector needs all forget facts
        sets = ("forget", "retain") if args.with_retain else ("forget",)
        facts = [f for f in all_facts if f["set"] in sets]
        if args.max_facts:
            facts = facts[:args.max_facts]
        adapter = get_adapter({**cfg["model"], "id": cfg["umu"]["ckpt"], "processor_id": cfg["model"]["id"]})
        model = PeftModel.from_pretrained(adapter.load_model(), str(ckpt), is_trainable=False)
        model.eval()
        n_layers = len(decoder_layers(model))
        tok = adapter.tok
        image_id = lm_parts(model)[2]
        newline_id = tok.convert_tokens_to_ids("<0x0A>")
        n_open = len(tok("USER:", add_special_tokens=True)["input_ids"])
        blank, filler = blank_image_bytes(), filler_string(tok, 576)
        rows = {f["pair_id"]: make_rows(f, blank, filler) for f in all_facts}
        name_tok = {f["pair_id"]: tok(f["name"], add_special_tokens=False)["input_ids"][0] for f in all_facts}
        ko_contexts = ("blank", "own") + (("filler",) if args.filler_knockout else ())
        kP = LGRID.index(PRIMARY)
        log(f"{run_id} step {args.step}: {len(facts)} facts ({', '.join(sets)}); inputs text + {CONTEXTS}; knockout on {ko_contexts}, "
            f"bands {list(BANDS)}; signal at layers {LGRID}")

        def names_of(chunk):
            return torch.tensor([name_tok[f["pair_id"]] for f in chunk])

        # the text form on all forget facts (and the selected ones): the vector and the reference
        rec, delta = {}, {}
        selected = {f["pair_id"] for f in facts}
        base = sorted((f for f in all_facts if f["set"] == "forget" or f["pair_id"] in selected),
                      key=lambda f: len(rows[f["pair_id"]]["text"]["prompt"]))
        for i in range(0, len(base), args.batch_txt):
            chunk = base[i:i + args.batch_txt]
            batch = encode_teacher_forced(adapter, [rows[f["pair_id"]]["text"] for f in chunk])
            t = first_target(tools.answer_targets(batch))
            h_u, r_u = run(adapter, model, batch, t, names_of(chunk), vanilla=False)
            h_0, r_0 = run(adapter, model, batch, t, names_of(chunk), vanilla=True)
            for j, f in enumerate(chunk):
                pid = f["pair_id"]
                delta[pid] = h_u[j] - h_0[j]
                rec[pid] = {"pair_id": pid, "set": f["set"], "profile": f["profile"], "proj": {},
                            "text/lp_u": r_u["lp"][j], "text/lp_0": r_0["lp"][j], "text/top1_u": r_u["top1"][j],
                            "text/top1_0": r_0["top1"][j], "text/name_u": r_u["name"][j], "text/name_0": r_0["name"][j]}
        vec, _ = held_out_means([f for f in all_facts if f["pair_id"] in delta], delta)
        unit = {pid: v / v.norm(dim=-1, keepdim=True).clamp_min(1e-8) for pid, v in vec.items()}
        for pid in rec:
            rec[pid]["proj"]["text"] = [round(float(x), 4) for x in (delta[pid] * unit[pid]).sum(-1)]

        checked = False
        for c in CONTEXTS:
            bs = args.batch_txt // 2 if c == "filler" else args.batch_img
            for i in range(0, len(facts), bs):
                chunk = facts[i:i + bs]
                pids = [f["pair_id"] for f in chunk]
                batch = encode_teacher_forced(adapter, [rows[pid][c] for pid in pids])
                t = first_target(tools.answer_targets(batch))
                nt = names_of(chunk)
                u_hat = torch.stack([unit[pid] for pid in pids])                        # [B, n_layers, D]
                h_u, r_u = run(adapter, model, batch, t, nt, vanilla=False)
                h_0, r_0 = run(adapter, model, batch, t, nt, vanilla=True)
                proj = ((h_u - h_0) * u_hat).sum(-1)
                _, r_add = run(adapter, model, batch, t, nt, vanilla=False,
                               add=(PRIMARY, h_u[:, kP] + torch.stack([vec[pid][kP] for pid in pids])))
                for j, pid in enumerate(pids):
                    rec[pid]["proj"][c] = [round(float(x), 4) for x in proj[j]]
                    rec[pid].update({f"{c}/lp_u": r_u["lp"][j], f"{c}/lp_0": r_0["lp"][j], f"{c}/top1_u": r_u["top1"][j],
                                     f"{c}/top1_0": r_0["top1"][j], f"{c}/name_u": r_u["name"][j], f"{c}/name_0": r_0["name"][j],
                                     f"{c}/add/top1_u": r_add["top1"][j], f"{c}/add/name_u": r_add["name"][j]})
                if c in ko_contexts:
                    rws, cls = prefix_positions(batch, t, c, image_id, n_open, newline_id)
                    if not checked:   # an explicit causal mask with nothing blocked must reproduce the unhooked run
                        empty = torch.zeros(0, dtype=torch.long, device=rws.device)
                        _, r_chk = run(adapter, model, batch, t, nt, vanilla=False, block=(range(n_layers), empty, empty))
                        err = max(abs(a - b) for a, b in zip(r_chk["lp"], r_u["lp"]))
                        log(f"  mask check ({c}): max |change in log p| with nothing blocked = {err:.2e} nats (tolerance {CHECK_TOL}); "
                            f"{len(cls)} blocked source tokens, {len(rws)} reading positions")
                        if err > CHECK_TOL:
                            raise RuntimeError(f"replacing the attention mask without blocking anything changed log p by {err:.3f} "
                                               "nats: the hook does not reproduce the model's own mask")
                        checked = True
                    for band, (a, b) in BANDS.items():
                        ids = range(a, min(b, n_layers))
                        k_u, q_u = run(adapter, model, batch, t, nt, vanilla=False, block=(ids, rws, cls))
                        k_0, q_0 = run(adapter, model, batch, t, nt, vanilla=True, block=(ids, rws, cls))
                        pk = ((k_u - k_0) * u_hat).sum(-1)
                        key = f"{c}/ko:{band}"
                        for j, pid in enumerate(pids):
                            rec[pid]["proj"][key] = [round(float(x), 4) for x in pk[j]]
                            rec[pid].update({f"{key}/lp_u": q_u["lp"][j], f"{key}/lp_0": q_0["lp"][j], f"{key}/top1_u": q_u["top1"][j],
                                             f"{key}/top1_0": q_0["top1"][j], f"{key}/name_u": q_u["name"][j],
                                             f"{key}/name_0": q_0["name"][j]})
            log(f"  input '{c}' done")
        model = None
        free()

        recs = [rec[f["pair_id"]] for f in facts]
        write_jsonl(out / "items.jsonl", recs)
        summary = summarize(recs, CONTEXTS, ko_contexts)
        lines = verdict(summary)
        atomic_write_json(out / "summary.json", {
            "run_id": run_id, "step": args.step, "layers": list(LGRID), "primary_layer": PRIMARY, "inputs": ["text"] + list(CONTEXTS),
            "knockout_inputs": list(ko_contexts), "bands": {k: list(v) for k, v in BANDS.items()},
            "readout": "first answer token: top1 = correct token is the argmax; name = argmax is the first token of the person's name; "
                       "signal = projection of (h_u - h_0) at the last prompt token on the text-form unit vector / the same on the text form",
            "thresholds": {"signal_min": SIGNAL_MIN, "signal_reject": SIGNAL_REJECT, "vanilla_top1_min": VANILLA_MIN, "mask_check_tol": CHECK_TOL},
            "verdict": lines, "summary": summary})
        for set_name, s in summary.items():
            log(f"{set_name}: n={s['n_facts']}")
            log(f"  {'input':34s} {'top-1 u':>8s} {'top-1 0':>8s} {'name u':>7s} {'name 0':>7s} {'forget':>7s}  signal at layers {list(LGRID)}")
            for c, d in s["inputs"].items():
                log(f"  {c:34s} {fmt(d['top1_u']):>8s} {fmt(d['top1_0']):>8s} {fmt(d['name_u']):>7s} {fmt(d['name_0']):>7s} "
                    f"{d['forgetting']:7.2f}  " + " ".join(fmt(v) for v in d["signal"]))
                if "add_vector" in d:
                    log(f"  {'  + text-form vector at layer ' + str(PRIMARY):34s} {fmt(d['add_vector']['top1_u']):>8s} {'':>8s} "
                        f"{fmt(d['add_vector']['name_u']):>7s}")
                for band, k in d.get("knockout", {}).items():
                    log(f"  {'  attention blocked, layers ' + band:34s} {fmt(k['top1_u']):>8s} {fmt(k['top1_0']):>8s} {fmt(k['name_u']):>7s} "
                        f"{fmt(k['name_0']):>7s} {k['forgetting']:7.2f}  " + " ".join(fmt(v) for v in k["signal"]))
        for line in lines:
            log(line)
        if "forget" in summary:
            plot(summary, CONTEXTS, ko_contexts, f"{run_id} step {args.step}: why the unlearning signal does not form behind an image",
                 out / "crossmodal.png")
        log(f"wrote {out / 'summary.json'}")


if __name__ == "__main__":
    main()
