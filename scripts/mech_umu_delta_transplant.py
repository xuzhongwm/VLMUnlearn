"""UMU track, DEV-049: is the perturbation that unlearning writes at a forget person's name transferable to another name?

delta of a forget fact = mean over its name tokens of (h_theta_u - h_theta_0) at one layer (name question, no image).
On each retain dc fact, with a donor forget fact of the same attribute where one exists:
  u+donor / 0+donor   theta_u / theta_0 (LoRA off), plus the donor's delta at the retain person's name tokens
  u+rand  / 0+rand    a random vector of the same norm (control)
References on the forget facts:  0+own (theta_0 plus the fact's own delta),  u-own (theta_u minus it).
Readout: first answer token, log p and top-1. Reading: DEVIATIONS.md DEV-049.

  python scripts/mech_umu_delta_transplant.py --config <model config> --run-id <run> --step <N> --layers 10,12 --set umu.n_retain_eval=100 [umu.ckpt=...]
Writes results/<model>/mech_umu/<run>/step<N>/delta_transplant/{summary.json,items.jsonl,log.txt}.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np  # noqa: E402
import torch  # noqa: E402

from scripts.mech_umu_form_sweep import facts_for  # noqa: E402
from scripts.mech_umu_signal import add_at, first_lp, group_index, states  # noqa: E402
from vlmu.config import base_parser, resolve  # noqa: E402
from vlmu.mech import tools  # noqa: E402
from vlmu.models import get_adapter  # noqa: E402
from vlmu.runs import free, model_dir, results_dir  # noqa: E402
from vlmu.umu import encode_teacher_forced, first_target, load_umu, token_groups_umu  # noqa: E402
from vlmu.utils import Logger, atomic_write_json, read_json, write_jsonl  # noqa: E402

FALL_MAX, RAND_MARGIN, FALL_TRANSFER, OWN_MIN = 0.15, 0.10, 0.50, 0.3      # DEV-049


def main():
    p = base_parser(__doc__)
    p.add_argument("--run-id", required=True)
    p.add_argument("--step", type=int, required=True)
    p.add_argument("--layers", default="10,12")
    p.add_argument("--batch", type=int, default=16)
    args = p.parse_args()
    cfg = resolve(args)
    run_dir = model_dir(cfg) / "umu_unlearn" / args.run_id
    out = results_dir(cfg) / "mech_umu" / args.run_id / f"step{args.step}" / "delta_transplant"
    log = Logger(out / "log.txt")
    layers = [int(x) for x in args.layers.split(",")]
    from peft import PeftModel
    dc = set(read_json(run_dir / "baseline_dc.json")["pair_ids"])
    dfs = load_umu(cfg["paths"]["root"], cfg["umu"]["n_retain_eval"], 0, log, cfg["umu"]["forget_split"])
    facts = [f for f in sorted(facts_for(dfs, dc), key=lambda f: f["pair_id"]) if f["name"] and f["name"] in f["q_uni"]]
    sets = {s: [f for f in facts if f["set"] == s] for s in ("forget", "retain")}
    adapter = get_adapter({**cfg["model"], "id": cfg["umu"]["ckpt"], "processor_id": cfg["model"]["id"]})
    model = PeftModel.from_pretrained(adapter.load_model(), str(run_dir / "ckpt" / f"step{args.step}"), is_trainable=False)
    model.eval()
    log(f"{args.run_id} step {args.step} ({cfg['model']['id']}): {len(sets['forget'])} forget and {len(sets['retain'])} retain dc facts; "
        f"layers {layers}")

    def batches(fs):
        for i in range(0, len(fs), args.batch):
            chunk = fs[i:i + args.batch]
            rows = [{"pair_id": f["pair_id"], "uid": f"{f['pair_id']}/T", "correct_text": f["answer_text"], "name": f["name"],
                     "prompt": adapter.prompt(f["q_uni"], False), "image_bytes": None} for f in chunk]
            batch = encode_teacher_forced(adapter, rows)
            t = first_target(tools.answer_targets(batch))
            groups, mism = token_groups_umu(adapter, model, rows, batch, t)
            if mism or not bool((groups["name"].sum(1) > 0).all()):
                raise RuntimeError(f"name tokens not found in every row ({mism} mismatches)")
            yield chunk, batch, t, groups["name"]

    # pass 1: the delta of every forget fact, and the references on the forget facts
    delta, recs = {}, []
    for chunk, batch, t, name in batches(sets["forget"]):
        idx, _ = group_index(name, t["pos"])
        r, pos = idx["name"]
        s0, su = states(adapter, model, batch, layers, True), states(adapter, model, batch, layers, False)
        nm = name.float()[..., None]
        d = {L: ((su[L] - s0[L]).float() * nm).sum(1) / nm.sum(1) for L in layers}          # [B, D]
        lp0, top0 = first_lp(adapter, model, batch, t, vanilla=True)
        lpu, topu = first_lp(adapter, model, batch, t)
        res = {L: {"0+own": first_lp(adapter, model, batch, t, vanilla=True, ctx=add_at(model, L, r, pos, d[L][r])),
                   "u-own": first_lp(adapter, model, batch, t, ctx=add_at(model, L, r, pos, -d[L][r]))} for L in layers}
        for j, f in enumerate(chunk):
            delta[f["pair_id"]] = {L: d[L][j].cpu() for L in layers}
            rec = {"pair_id": f["pair_id"], "set": "forget", "field": f["field"], "lp0": lp0[j].item(), "lpu": lpu[j].item(),
                   "top0": bool(top0[j]), "topu": bool(topu[j])}
            for L in layers:
                for k, (lp, top) in res[L].items():
                    rec[f"{k}/{L}/lp"], rec[f"{k}/{L}/top"] = lp[j].item(), bool(top[j])
            recs.append(rec)
    by_field = {}
    for f in sets["forget"]:
        by_field.setdefault(f["field"], []).append(f["pair_id"])
    all_ids = [f["pair_id"] for f in sets["forget"]]
    # pass 2: transplant onto retain names
    gen = torch.Generator().manual_seed(0)
    count = {}
    for chunk, batch, t, name in batches(sets["retain"]):
        idx, _ = group_index(name, t["pos"])
        r, pos = idx["name"]
        donors = []
        for f in chunk:                                        # same attribute where possible, cycling through the donors
            pool = by_field.get(f["field"]) or all_ids
            k = count.get(f["field"], 0)
            count[f["field"]] = k + 1
            donors.append(pool[k % len(pool)])
        lp0, top0 = first_lp(adapter, model, batch, t, vanilla=True)
        lpu, topu = first_lp(adapter, model, batch, t)
        res = {}
        for L in layers:
            d = torch.stack([delta[i][L] for i in donors])                         # [B, D]
            rnd = torch.randn(d.shape, generator=gen)
            rnd = rnd / rnd.norm(dim=1, keepdim=True) * d.norm(dim=1, keepdim=True)
            d, rnd = d[r.cpu()].to(name.device), rnd[r.cpu()].to(name.device)
            res[L] = {"u+donor": first_lp(adapter, model, batch, t, ctx=add_at(model, L, r, pos, d)),
                      "0+donor": first_lp(adapter, model, batch, t, vanilla=True, ctx=add_at(model, L, r, pos, d)),
                      "u+rand": first_lp(adapter, model, batch, t, ctx=add_at(model, L, r, pos, rnd)),
                      "0+rand": first_lp(adapter, model, batch, t, vanilla=True, ctx=add_at(model, L, r, pos, rnd))}
        for j, f in enumerate(chunk):
            rec = {"pair_id": f["pair_id"], "set": "retain", "field": f["field"], "donor": donors[j], "lp0": lp0[j].item(),
                   "lpu": lpu[j].item(), "top0": bool(top0[j]), "topu": bool(topu[j])}
            for L in layers:
                for k, (lp, top) in res[L].items():
                    rec[f"{k}/{L}/lp"], rec[f"{k}/{L}/top"] = lp[j].item(), bool(top[j])
            recs.append(rec)
    model = None
    free()
    write_jsonl(out / "items.jsonl", recs)
    fg, rt = [x for x in recs if x["set"] == "forget"], [x for x in recs if x["set"] == "retain"]
    m = lambda rs, k: float(np.mean([x[k] for x in rs]))  # noqa: E731
    FT = m(fg, "lp0") - m(fg, "lpu")
    summary = {"run_id": args.run_id, "step": args.step, "model": cfg["model"]["id"], "n_forget": len(fg), "n_retain": len(rt), "F_T": FT,
               "top1": {"forget_0": m(fg, "top0"), "forget_u": m(fg, "topu"), "retain_0": m(rt, "top0"), "retain_u": m(rt, "topu")},
               "layers": {}}
    log(f"F(T) = {FT:.2f} nats; top-1 forget θ0 {summary['top1']['forget_0']:.2f} θu {summary['top1']['forget_u']:.2f}; retain θ0 "
        f"{summary['top1']['retain_0']:.2f} θu {summary['top1']['retain_u']:.2f}")
    for L in layers:
        z = {"own_induced": (m(fg, "lp0") - m(fg, f"0+own/{L}/lp")) / FT, "own_induced_top1": m(fg, f"0+own/{L}/top"),
             "own_removed_recovery": 1 - (m(fg, "lp0") - m(fg, f"u-own/{L}/lp")) / FT, "own_removed_top1": m(fg, f"u-own/{L}/top")}
        for k, base in (("u+donor", "lpu"), ("0+donor", "lp0"), ("u+rand", "lpu"), ("0+rand", "lp0")):
            z[f"{k}_nats"] = m(rt, base) - m(rt, f"{k}/{L}/lp")
            z[f"{k}_top1"] = m(rt, f"{k}/{L}/top")
        fall, fall_r = summary["top1"]["retain_u"] - z["u+donor_top1"], summary["top1"]["retain_u"] - z["u+rand_top1"]
        z["readable"] = z["own_removed_recovery"] >= OWN_MIN
        z["verdict"] = ("transferable" if fall >= FALL_TRANSFER else
                        "not transferable" if fall <= FALL_MAX and fall <= fall_r + RAND_MARGIN else "partial")
        summary["layers"][L] = z
        log(f"  L{L} forget facts: θ0 + own delta induces {z['own_induced']:+.2f} of F(T) (top-1 {z['own_induced_top1']:.2f}); θu − own delta "
            f"recovers {z['own_removed_recovery']:+.2f} (top-1 {z['own_removed_top1']:.2f})")
        log(f"  L{L} retain facts: θu + donor delta: top-1 {z['u+donor_top1']:.2f} ({z['u+donor_nats']:+.2f} nats) | θu + random: "
            f"{z['u+rand_top1']:.2f} ({z['u+rand_nats']:+.2f}) | θ0 + donor: {z['0+donor_top1']:.2f} ({z['0+donor_nats']:+.2f}) | θ0 + random: "
            f"{z['0+rand_top1']:.2f} ({z['0+rand_nats']:+.2f}) -> {z['verdict']}{'' if z['readable'] else ' (not readable: own delta recovers < 0.3)'}")
    atomic_write_json(out / "summary.json", summary)
    log(f"wrote {out / 'summary.json'}")


if __name__ == "__main__":
    main()
