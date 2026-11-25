"""UMU track, DEV-050: what is the perturbation that text-only unlearning writes at a forget person's name?

delta of a fact = mean over its name tokens of (h_theta_u - h_theta_0) at one layer (name question, no image, no options).
  (1) structure     share of the squared norm in the mean over people; PCA of the person-level delta; cosine between people;
                    collapse test (is the name pushed towards the average name?); the same delta at retain names.
  (2) --compare     cosine between the deltas of checkpoints that share a vanilla (mean, and per person after centring).
  (3) "unknown person" direction u = in theta_0, the mean name-token state of UNKNOWN names minus that of the retain
      people's names in the same questions. Two kinds of unknown name, each a full name that none of the 500 profiles has:
        recomb  the person's first name + another retain person's last name (the pre-registered kind)
        novel   a first name and a last name that no profile has (added after the smoke test, before the full run)
        cosine(mean delta, u)
        0+u / u+u     theta_0 / theta_u, plus u at a retain person's name tokens, at the norm of delta and at u's own norm;
                      controls: a random vector (0+rand, u+rand); reference: the forget mean delta (0+mean, u+mean)
        u-proj        theta_u, minus the component of the fact's delta along u, at a forget person's name tokens
                      (references: u-own = minus the whole delta, u-mean = minus the forget mean delta)
        first token   what theta_0 says first for unknown names, against what theta_u says first for forget names
Readout: first answer token, log p and top-1; short greedy generations to read. Reading and thresholds: DEVIATIONS.md DEV-050.
Validity (abort): name tokens found in every row; no unknown name is a profile's name; F(T) >= 1 nat; adding a zero
vector changes log p by <= 0.05 nats.

  python scripts/mech_umu_delta_what.py --config <model config> --run-id <run> --step <N> --set umu.n_retain_eval=100 [umu.ckpt=...]
  python scripts/mech_umu_delta_what.py --config <model config> --compare <run>:<step> <run>:<step> ...
Writes results/<model>/mech_umu/<run>/step<N>/delta_what/{summary.json,items.jsonl,gen_*.jsonl,delta.pt,log.txt}.
"""
import collections
import itertools
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np  # noqa: E402
import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402

from scripts.mech_umu_form_sweep import facts_for  # noqa: E402
from scripts.mech_umu_signal import add_at, first_lp, group_index, states  # noqa: E402
from vlmu.config import base_parser, resolve  # noqa: E402
from vlmu.mech import tools  # noqa: E402
from vlmu.models import get_adapter  # noqa: E402
from vlmu.runs import free, model_dir, results_dir  # noqa: E402
from vlmu.umu import encode_teacher_forced, first_target, load_umu, profile_name, token_groups_umu  # noqa: E402
from vlmu.utils import Logger, atomic_write_json, read_json, write_jsonl  # noqa: E402

COS_IS, DROP_IS, RAND_MAX, COS_NOT, DROP_NOT, TOL, FT_MIN = 0.5, 0.3, 0.1, 0.2, 0.1, 0.05, 1.0     # DEV-050
N_PROFILES = 450                                                              # retain_90: with forget_10, all 500 profiles
KINDS = ("recomb", "novel")
NOVEL_FIRST = ("Bartholomew Winifred Thaddeus Ottoline Cordelia Ignatius Philippa Barnaby Henrietta Cornelius Millicent Algernon "
               "Rosalind Montgomery Clementine Fitzgerald Gwendolyn Ambrose Wilhelmina Reginald Humphrey Marguerite").split()
NOVEL_LAST = ("Thistlewood Pennington Ashcombe Featherstone Quimby Hargreaves Wetherby Colquhoun Ravenscroft Dunmore Fairweather "
              "Oakenshaw Trevelyan Underhill Lockridge Marchbanks Stanhope Winterbourne Greystoke Halloran Petherick Rookwood "
              "Ellingham").split()


def unknown_names(retain_names, all_names):
    """{name: the person's first name + the last name of another retain person}, a full name no profile has."""
    split = {n: (n.split()[0], " ".join(n.split()[1:])) for n in retain_names}
    out, names = {}, list(retain_names)
    for i, n in enumerate(names):
        for k in range(1, len(names)):
            cand = f"{split[n][0]} {split[names[(i + k) % len(names)]][1]}"
            if cand not in all_names and split[names[(i + k) % len(names)]][1] != split[n][1]:
                out[n] = cand
                break
        else:
            raise RuntimeError(f"no unknown name for {n}")
    return out


def novel_names(retain_names, all_names):
    """{name: a full name whose first and last names are in none of the profiles}; all different."""
    used = {w for n in all_names for w in n.split()}
    fs, ls = [x for x in NOVEL_FIRST if x not in used], [x for x in NOVEL_LAST if x not in used]
    out = {n: f"{fs[i % len(fs)]} {ls[(i // len(fs) + i) % len(ls)]}" for i, n in enumerate(retain_names)}
    if len(set(out.values())) != len(out):
        raise RuntimeError("novel names are not all different")
    return out


def cos(a, b):
    return F.cosine_similarity(a.float(), b.float(), dim=-1)


def compare(args, cfg):
    runs = [(r.rsplit(":", 1)[0], int(r.rsplit(":", 1)[1])) for r in args.compare]
    data = {r: torch.load(results_dir(cfg) / "mech_umu" / r[0] / f"step{r[1]}" / "delta_what" / "delta.pt") for r in runs}
    out = results_dir(cfg) / "mech_umu" / "_compare" / "delta_what"
    log = Logger(out / "log.txt")
    layers = data[runs[0]]["layers"]
    res = []
    for a, b in itertools.combinations(runs, 2):
        da, db = data[a], data[b]
        common = [p for p in da["people"] if p in db["people"]]
        ia, ib = [da["people"].index(p) for p in common], [db["people"].index(p) for p in common]
        z = {"a": f"{a[0]}:{a[1]}", "b": f"{b[0]}:{b[1]}", "n_people": len(common), "layers": {}}
        for li, L in enumerate(layers):
            A, B = da["person"][ia, li].float(), db["person"][ib, li].float()
            z["layers"][L] = {"cos_mean": cos(A.mean(0), B.mean(0)).item(), "cos_person_raw": cos(A, B).mean().item(),
                              "cos_person_centred": cos(A - A.mean(0), B - B.mean(0)).mean().item(),
                              "cos_person_centred_shuffled": cos(A - A.mean(0), (B - B.mean(0)).roll(1, 0)).mean().item(),
                              "cos_u": cos(da["u"][li], db["u"][li]).item()}
        res.append(z)
        log(f"{z['a']}  vs  {z['b']}  ({len(common)} people): " + " | ".join(
            f"L{L}: mean {v['cos_mean']:.2f}, person {v['cos_person_raw']:.2f}, centred {v['cos_person_centred']:.2f} "
            f"(shuffled {v['cos_person_centred_shuffled']:.2f})" for L, v in z["layers"].items()))
    atomic_write_json(out / f"{args.compare_tag}.json", res)
    log(f"wrote {out / (args.compare_tag + '.json')}")


def main():
    p = base_parser(__doc__)
    p.add_argument("--run-id")
    p.add_argument("--step", type=int)
    p.add_argument("--layers", default="8,10,12")
    p.add_argument("--batch", type=int, default=16)
    p.add_argument("--gen-tokens", type=int, default=24)
    p.add_argument("--max-facts", type=int, default=0, help="smoke test: output under delta_what_smoke")
    p.add_argument("--compare", nargs="+", default=None, metavar="RUN:STEP")
    p.add_argument("--compare-tag", default="compare")
    args = p.parse_args()
    cfg = resolve(args)
    if args.compare:
        return compare(args, cfg)
    run_dir = model_dir(cfg) / "umu_unlearn" / args.run_id
    out = results_dir(cfg) / "mech_umu" / args.run_id / f"step{args.step}" / ("delta_what_smoke" if args.max_facts else "delta_what")
    log = Logger(out / "log.txt")
    layers = [int(x) for x in args.layers.split(",")]
    from peft import PeftModel
    dc = set(read_json(run_dir / "baseline_dc.json")["pair_ids"])
    dfs = load_umu(cfg["paths"]["root"], N_PROFILES, 0, log, cfg["umu"]["forget_split"])
    all_names = {profile_name(r) for s in ("forget", "retain") for _, r in dfs[s].iterrows()}
    dfs["retain"] = dfs["retain"].head(cfg["umu"]["n_retain_eval"])
    facts = [f for f in sorted(facts_for(dfs, dc), key=lambda f: f["pair_id"]) if f["name"] and f["name"] in f["q_uni"]]
    sets = {s: [f for f in facts if f["set"] == s] for s in ("forget", "retain")}
    if args.max_facts:
        sets = {s: v[:args.max_facts] for s, v in sets.items()}
    retain_names = list(dict.fromkeys(f["name"] for f in sets["retain"]))
    unk = {"recomb": unknown_names(retain_names, all_names), "novel": novel_names(retain_names, all_names)}
    if any(v in all_names for d in unk.values() for v in d.values()):
        raise RuntimeError("an unknown name is the name of a profile")
    adapter = get_adapter({**cfg["model"], "id": cfg["umu"]["ckpt"], "processor_id": cfg["model"]["id"]})
    model = PeftModel.from_pretrained(adapter.load_model(), str(run_dir / "ckpt" / f"step{args.step}"), is_trainable=False)
    model.eval()
    tok = adapter.tok
    log(f"{args.run_id} step {args.step} ({cfg['model']['id']}): {len(sets['forget'])} forget and {len(sets['retain'])} retain dc facts; "
        f"layers {layers}; {len(all_names)} profile names; unknown names e.g. " + "; ".join(
            f"{k}: {list(v.items())[:2]}" for k, v in unk.items()))

    def batches(fs, unknown=None):
        for i in range(0, len(fs), args.batch):
            chunk = fs[i:i + args.batch]
            rows = []
            for f in chunk:
                name = unk[unknown][f["name"]] if unknown else f["name"]
                q = f["q_uni"].replace(f["name"], name)
                rows.append({"pair_id": f["pair_id"], "uid": f"{f['pair_id']}/T", "correct_text": f["answer_text"], "name": name,
                             "prompt": adapter.prompt(q, False), "image_bytes": None})
            batch = encode_teacher_forced(adapter, rows)
            t = first_target(tools.answer_targets(batch))
            groups, mism = token_groups_umu(adapter, model, rows, batch, t)
            if mism or not bool((groups["name"].sum(1) > 0).all()):
                raise RuntimeError(f"name tokens not found in every row ({mism} mismatches)")
            yield chunk, rows, batch, t, groups["name"]

    def name_mean(s, name):
        nm = name.float()[..., None]
        return {L: ((s[L].float() * nm).sum(1) / nm.sum(1)).cpu() for L in layers}                # [B, D]

    @torch.no_grad()
    def first_tok(batch, t, vanilla):
        import contextlib
        with (model.disable_adapter() if vanilla else contextlib.nullcontext()):
            at = tools.forward_logits(adapter, model, batch)[t["b"], t["pos"]]
        return [tok.decode([i]).strip() for i in at.argmax(-1).tolist()]

    @torch.no_grad()
    def generate(rows, vanilla):
        import contextlib
        b = adapter.to_device(adapter._process([r["prompt"] for r in rows], None, "left"))
        with (model.disable_adapter() if vanilla else contextlib.nullcontext()), adapter.autocast():
            return adapter.generate(model, b, args.gen_tokens)

    # ---- pass A: forget facts. delta, the theta_0 name state, baselines, theta_u's first token and generations
    D_f, H0_f, recs_f, gen_f, zero_checked = {L: [] for L in layers}, {L: [] for L in layers}, [], [], False
    for chunk, rows, batch, t, name in batches(sets["forget"]):
        s0, su = states(adapter, model, batch, layers, True), states(adapter, model, batch, layers, False)
        h0, hu = name_mean(s0, name), name_mean(su, name)
        lp0, top0 = first_lp(adapter, model, batch, t, vanilla=True)
        lpu, topu = first_lp(adapter, model, batch, t)
        if not zero_checked:
            idx, _ = group_index(name, t["pos"])
            r, pos = idx["name"]
            lpz, _ = first_lp(adapter, model, batch, t, ctx=add_at(model, layers[0], r, pos, torch.zeros(len(r), h0[layers[0]].shape[1],
                                                                                                         device=name.device)))
            dz = (lpz - lpu).abs().max().item()
            log(f"validity: adding a zero vector changes log p by at most {dz:.4f} nats")
            if dz > TOL:
                raise RuntimeError(f"adding a zero vector changes log p by {dz:.3f} nats")
            zero_checked = True
        fu, g = first_tok(batch, t, False), generate(rows, False)
        for L in layers:
            D_f[L].append(hu[L] - h0[L]), H0_f[L].append(h0[L])
        for j, f in enumerate(chunk):
            recs_f.append({"pair_id": f["pair_id"], "set": "forget", "profile": f["profile"], "field": f["field"], "lp0": lp0[j].item(),
                           "lpu": lpu[j].item(), "top0": bool(top0[j]), "topu": bool(topu[j]), "first_u": fu[j]})
            gen_f.append({"pair_id": f["pair_id"], "field": f["field"], "question": f["q_uni"], "answer": f["answer_text"], "theta_u": g[j]})
    D_f, H0_f = {L: torch.cat(v) for L, v in D_f.items()}, {L: torch.cat(v) for L, v in H0_f.items()}
    m = lambda rs, k: float(np.mean([x[k] for x in rs]))  # noqa: E731
    FT = m(recs_f, "lp0") - m(recs_f, "lpu")
    log(f"forget facts: F(T) = {FT:.2f} nats; top-1 θ0 {m(recs_f, 'top0'):.2f}, θu {m(recs_f, 'topu'):.2f}")
    if FT < FT_MIN and not args.max_facts:
        raise RuntimeError(f"F(T) = {FT:.2f} nats: the checkpoint does not forget")
    write_jsonl(out / "gen_forget.jsonl", gen_f)

    # ---- pass B: retain facts. known and unknown names in theta_0 (-> u), the delta at retain names, first tokens, generations
    H0_r, D_r, recs_r = {L: [] for L in layers}, {L: [] for L in layers}, []
    HK_r, gen_u = {k: {L: [] for L in layers} for k in KINDS}, {k: [] for k in KINDS}
    for chunk, rows, batch, t, name in batches(sets["retain"]):
        s0, su = states(adapter, model, batch, layers, True), states(adapter, model, batch, layers, False)
        h0, hu = name_mean(s0, name), name_mean(su, name)
        lp0, top0 = first_lp(adapter, model, batch, t, vanilla=True)
        lpu, topu = first_lp(adapter, model, batch, t)
        f0 = first_tok(batch, t, True)
        for L in layers:
            H0_r[L].append(h0[L]), D_r[L].append(hu[L] - h0[L])
        base = len(recs_r)
        for j, f in enumerate(chunk):
            recs_r.append({"pair_id": f["pair_id"], "set": "retain", "profile": f["profile"], "field": f["field"], "lp0": lp0[j].item(),
                           "lpu": lpu[j].item(), "top0": bool(top0[j]), "topu": bool(topu[j]), "first_0": f0[j]})
        for kind in KINDS:                                   # the same chunk with the unknown name, in theta_0
            _, rows_k, batch_k, t_k, name_k = next(itertools.islice(batches(sets["retain"][base:base + len(chunk)], kind), 0, 1))
            hk = name_mean(states(adapter, model, batch_k, layers, True), name_k)
            lpk, topk = first_lp(adapter, model, batch_k, t_k, vanilla=True)     # scored on the original person's answer
            fk, g = first_tok(batch_k, t_k, True), generate(rows_k, True)
            for L in layers:
                HK_r[kind][L].append(hk[L])
            for j, f in enumerate(chunk):
                recs_r[base + j].update({f"name_{kind}": rows_k[j]["name"], f"first_0_{kind}": fk[j], f"lp0_{kind}": lpk[j].item(),
                                         f"top0_{kind}": bool(topk[j])})
                gen_u[kind].append({"pair_id": f["pair_id"], "field": f["field"], "question": rows_k[j]["prompt"],
                                    "answer_of_original": f["answer_text"], "theta_0": g[j]})
    H0_r, D_r = ({L: torch.cat(v) for L, v in x.items()} for x in (H0_r, D_r))
    HK_r = {k: {L: torch.cat(v) for L, v in x.items()} for k, x in HK_r.items()}
    for kind in KINDS:
        write_jsonl(out / f"gen_unknown_{kind}.jsonl", gen_u[kind])
    U = {k: {L: HK_r[k][L].mean(0) - H0_r[L].mean(0) for L in layers} for k in KINDS}
    half = torch.arange(len(recs_r)) % 2 == 0
    mean_f = {L: D_f[L].mean(0) for L in layers}
    norm_d = {L: D_f[L].norm(dim=1).mean().item() for L in layers}

    # ---- pass C: forget facts, removing parts of the delta in theta_u
    gen = torch.Generator().manual_seed(0)
    k = 0
    for chunk, rows, batch, t, name in batches(sets["forget"]):
        idx, _ = group_index(name, t["pos"])
        r, pos = idx["name"]
        for L in layers:
            d = D_f[L][k:k + len(chunk)]
            vecs = {"u-own": d, "u-mean": mean_f[L][None].expand_as(d)}
            for kind in KINDS:
                uh = U[kind][L] / U[kind][L].norm()
                vecs[f"u-proj_{kind}"] = (d @ uh)[:, None] * uh[None]
            for key, v in vecs.items():
                lp, top = first_lp(adapter, model, batch, t, ctx=add_at(model, L, r, pos, -v[r.cpu()].to(name.device)))
                for j in range(len(chunk)):
                    recs_f[k + j][f"{key}/{L}/lp"], recs_f[k + j][f"{key}/{L}/top"] = lp[j].item(), bool(top[j])
        k += len(chunk)

    # ---- pass D: retain facts, adding u / a random vector / the forget mean delta at the name tokens
    k = 0
    for chunk, rows, batch, t, name in batches(sets["retain"]):
        idx, _ = group_index(name, t["pos"])
        r, pos = idx["name"]
        for L in layers:
            B = len(chunk)
            rnd = torch.randn(B, mean_f[L].shape[0], generator=gen)
            rnd = rnd / rnd.norm(dim=1, keepdim=True)
            vecs = {"rand@d": rnd * norm_d[L], "rand@u": rnd * U["recomb"][L].norm(), "mean": mean_f[L][None].expand(B, -1)}
            for kind in KINDS:
                uh = (U[kind][L] / U[kind][L].norm())[None].expand(B, -1)
                vecs[f"u_{kind}@d"], vecs[f"u_{kind}@u"] = uh * norm_d[L], uh * U[kind][L].norm()
            for key, v in vecs.items():
                v = v[r.cpu()].to(name.device)
                for base, van in (("0", True), ("u", False)):
                    lp, top = first_lp(adapter, model, batch, t, vanilla=van, ctx=add_at(model, L, r, pos, v))
                    for j in range(B):
                        recs_r[k + j][f"{base}+{key}/{L}/lp"], recs_r[k + j][f"{base}+{key}/{L}/top"] = lp[j].item(), bool(top[j])
        k += len(chunk)
    model = None
    free()
    write_jsonl(out / "items.jsonl", recs_f + recs_r)

    # ---- summary
    people_f = list(dict.fromkeys(x["profile"] for x in recs_f))
    people_r = list(dict.fromkeys(x["profile"] for x in recs_r))
    pf = torch.tensor([people_f.index(x["profile"]) for x in recs_f])
    pr = torch.tensor([people_r.index(x["profile"]) for x in recs_r])
    by_person = lambda X, idx, n: torch.stack([X[idx == i].mean(0) for i in range(n)])  # noqa: E731
    cnt = lambda xs: collections.Counter(xs).most_common(5)  # noqa: E731
    fields = sorted({x["field"] for x in recs_f + recs_r})
    first = {"theta_u_forget": cnt([x["first_u"] for x in recs_f]), "theta_0_retain": cnt([x["first_0"] for x in recs_r]),
             **{f"theta_0_{k}": cnt([x[f"first_0_{k}"] for x in recs_r]) for k in KINDS},
             "by_field": {fld: {"theta_u_forget": cnt([x["first_u"] for x in recs_f if x["field"] == fld])[:2],
                                **{f"theta_0_{k}": cnt([x[f"first_0_{k}"] for x in recs_r if x["field"] == fld])[:2] for k in KINDS}}
                          for fld in fields}}
    same_first = {k: first["theta_u_forget"][0][0] == first[f"theta_0_{k}"][0][0] for k in KINDS}
    summary = {"run_id": args.run_id, "step": args.step, "model": cfg["model"]["id"], "n_forget": len(recs_f), "n_retain": len(recs_r),
               "n_forget_people": len(people_f), "n_retain_people": len(people_r), "F_T": FT,
               "top1": {"forget_0": m(recs_f, "top0"), "forget_u": m(recs_f, "topu"), "retain_0": m(recs_r, "top0"),
                        "retain_u": m(recs_r, "topu"), **{f"{k}_0_original_answer": m(recs_r, f"top0_{k}") for k in KINDS}},
               "first_token": first, "first_token_same": same_first, "layers": {}}
    log(f"retain facts: top-1 θ0 {summary['top1']['retain_0']:.2f}, θu {summary['top1']['retain_u']:.2f}; θ0 with an unknown name still gives "
        f"the original person's answer first: " + ", ".join(f"{k} {summary['top1'][f'{k}_0_original_answer']:.2f}" for k in KINDS))
    log(f"first token: θu on forget names {first['theta_u_forget'][:3]} | " + " | ".join(
        f"θ0 on {k} names {first[f'theta_0_{k}'][:3]} ({'same' if same_first[k] else 'different'})" for k in KINDS))
    P_f, P_r = {}, {}
    for L in layers:
        Pf, Pr = by_person(D_f[L], pf, len(people_f)), by_person(D_r[L], pr, len(people_r))
        P_f[L], P_r[L] = Pf, Pr
        c0f, c0r = by_person(H0_f[L], pf, len(people_f)), by_person(H0_r[L], pr, len(people_r))
        centre = torch.cat([c0f, c0r]).mean(0)
        sv = torch.linalg.svdvals(Pf - Pf.mean(0)) ** 2
        ev = (sv.cumsum(0) / sv.sum()).tolist()
        n = F.normalize(Pf, dim=-1)
        pair = ((n @ n.T).sum() - len(Pf)) / (len(Pf) * (len(Pf) - 1))
        z = {"norm_delta": norm_d[L], "norm_h0_name": H0_f[L].norm(dim=1).mean().item(),
             "norm_delta_retain": D_r[L].norm(dim=1).mean().item(),
             "shared_share": (Pf.mean(0).norm() ** 2 / (Pf.norm(dim=1) ** 2).mean()).item(), "cos_between_people": pair.item(),
             "pca_var": {k: ev[k - 1] for k in (1, 2, 5, 10) if k <= len(ev)},
             "collapse_cos_forget": cos(Pf, centre[None] - c0f).mean().item(), "collapse_cos_retain": cos(Pr, centre[None] - c0r).mean().item(),
             "cos_mean_forget_retain": cos(Pf.mean(0), Pr.mean(0)).item(), "cos_u_recomb_novel": cos(U["recomb"][L], U["novel"][L]).item()}
        for key in ("u-own", "u-mean") + tuple(f"u-proj_{k}" for k in KINDS):
            z[f"{key}_recovery"] = 1 - (m(recs_f, "lp0") - m(recs_f, f"{key}/{L}/lp")) / FT
            z[f"{key}_top1"] = m(recs_f, f"{key}/{L}/top")
        for base, ref, t1 in (("0", "lp0", "retain_0"), ("u", "lpu", "retain_u")):
            for key in ("rand@d", "rand@u", "mean") + tuple(f"u_{k}@{w}" for k in KINDS for w in "du"):
                z[f"{base}+{key}_top1"] = m(recs_r, f"{base}+{key}/{L}/top")
                z[f"{base}+{key}_drop"] = summary["top1"][t1] - z[f"{base}+{key}_top1"]
                z[f"{base}+{key}_nats"] = m(recs_r, ref) - m(recs_r, f"{base}+{key}/{L}/lp")
        log(f"  L{L} structure: |Δ| {z['norm_delta']:.1f} (|h0| {z['norm_h0_name']:.1f}, retain |Δ| {z['norm_delta_retain']:.1f}); shared share "
            f"{z['shared_share']:.2f}; cosine between people {z['cos_between_people']:.2f}; PCA "
            f"{ {k: round(v, 2) for k, v in z['pca_var'].items()} }; collapse cosine forget {z['collapse_cos_forget']:+.2f} / retain "
            f"{z['collapse_cos_retain']:+.2f}; cos(mean Δ forget, retain) {z['cos_mean_forget_retain']:+.2f}")
        log(f"  L{L} forget facts, θu minus: own Δ R {z['u-own_recovery']:+.2f} (top-1 {z['u-own_top1']:.2f}) | mean Δ {z['u-mean_recovery']:+.2f} "
            f"({z['u-mean_top1']:.2f}); retain facts plus the forget mean Δ: θ0 {z['0+mean_top1']:.2f}, θu {z['u+mean_top1']:.2f}; random at |Δ|: "
            f"θ0 {z['0+rand@d_top1']:.2f}, θu {z['u+rand@d_top1']:.2f}")
        for kind in KINDS:
            u = U[kind][L]
            q = {"norm_u": u.norm().item(), "cos_mean_u": cos(mean_f[L], u).item(), "cos_person_u": cos(Pf, u[None]).mean().item(),
                 "proj_mean_on_u_over_norm_u": (mean_f[L] @ u / u.norm() ** 2).item(),
                 "u_split_half_cos": cos(HK_r[kind][L][half].mean(0) - H0_r[L][half].mean(0),
                                         HK_r[kind][L][~half].mean(0) - H0_r[L][~half].mean(0)).item()}
            d0, du, r0, ru = (z[f"0+u_{kind}@d_drop"], z[f"u+u_{kind}@d_drop"], z["0+rand@d_drop"], z["u+rand@d_drop"])
            is_u = q["cos_mean_u"] >= COS_IS and d0 >= DROP_IS and r0 <= RAND_MAX and same_first[kind]
            q["verdict"] = "unknown-person direction" if is_u else "not" if q["cos_mean_u"] <= COS_NOT and d0 <= DROP_NOT else "partial"
            q["marker"] = bool(du >= DROP_IS and ru <= RAND_MAX and d0 < DROP_IS)
            z[kind] = q
            log(f"  L{L} unknown direction ({kind}): |u| {q['norm_u']:.1f}; cos(mean Δ, u) {q['cos_mean_u']:+.2f} (per person "
                f"{q['cos_person_u']:+.2f}; split-half cosine of u {q['u_split_half_cos']:.2f}; projection / |u| "
                f"{q['proj_mean_on_u_over_norm_u']:+.2f}) | θu minus the part along u: R {z[f'u-proj_{kind}_recovery']:+.2f} (top-1 "
                f"{z[f'u-proj_{kind}_top1']:.2f}) | retain top-1 at |Δ|: 0+u {z[f'0+u_{kind}@d_top1']:.2f}, u+u {z[f'u+u_{kind}@d_top1']:.2f}; at "
                f"|u|: 0+u {z[f'0+u_{kind}@u_top1']:.2f}, u+u {z[f'u+u_{kind}@u_top1']:.2f} -> {q['verdict']}"
                f"{' (marker reading holds)' if q['marker'] else ''}")
        z["verdict"] = z["recomb"]["verdict"]                                  # the pre-registered kind
        summary["layers"][L] = z
    vs = [summary["layers"][L]["verdict"] for L in layers]
    summary["verdict"] = ("unknown-person direction" if "unknown-person direction" in vs else "not" if all(v == "not" for v in vs)
                          else "partial")
    log(f"verdict: {summary['verdict']}")
    atomic_write_json(out / "summary.json", summary)
    torch.save({"layers": layers, "people": people_f, "person": torch.stack([P_f[L] for L in layers], 1).half(),
                "people_retain": people_r, "person_retain": torch.stack([P_r[L] for L in layers], 1).half(),
                "u": torch.stack([U["recomb"][L] for L in layers]).half(), "u_novel": torch.stack([U["novel"][L] for L in layers]).half(), "facts": [x["pair_id"] for x in recs_f],
                "fact": torch.stack([D_f[L] for L in layers], 1).half()}, out / "delta.pt")
    log(f"wrote {out / 'summary.json'}")


if __name__ == "__main__":
    main()
