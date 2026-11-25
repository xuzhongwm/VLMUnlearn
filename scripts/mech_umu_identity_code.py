"""UMU track, DEV-051: is there a person code shared by the name route and the face route? (the vanilla only)

For every person and attribute, two prompts without options:
  N  the UM_QA name question, no image          F  the MM_QA "this person" question with the portrait
The residual stream at the last prompt token is recorded at a set of layers. For every (attribute, route, layer) the mean
over people is subtracted, so what is left varies with the person only.
  leave-one-attribute-out retrieval (main readout): query = one person's state on one attribute in one route; keys = every
      person's code in the other route, a code being the mean over that person's OTHER attributes; hit = the nearest key
      (cosine) is the same person. F->N, N->F, and within a route (N->N, F->F) as a ceiling.
  whole-code retrieval: each person's mean over all attributes, F against N.
  same-attribute retrieval: query and key from the same attribute (the shared answer helps; a reference).
Chance = 1 / number of people. Reading and thresholds: DEVIATIONS.md DEV-051.
Validity (abort): every person has every question in both routes; the name is in every N question and in no F question;
a batched state equals the state of the same prompt run alone (cosine >= 0.99); retrieval with shuffled people <= 0.05.

  python scripts/mech_umu_identity_code.py --config <model config> --merge-layer <L*> [--set umu.ckpt=...]
Writes results/<model>/mech_umu/vanilla_<ckpt name>/identity_code/{summary.json,items.jsonl,log.txt,states.pt,identity_code.png}.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np  # noqa: E402
import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402

from vlmu.config import base_parser, resolve  # noqa: E402
from vlmu.mech import tools  # noqa: E402
from vlmu.mech.hooks import decoder_layers, record_outputs  # noqa: E402
from vlmu.models import get_adapter  # noqa: E402
from vlmu.runs import free, results_dir  # noqa: E402
from vlmu.umu import encode_teacher_forced, first_target, image_bytes, load_umu, parse, profile_name  # noqa: E402
from vlmu.utils import Logger, atomic_write_json, write_jsonl  # noqa: E402

ATTRS = ("Born", "Birth", "Gender", "Occupation", "Height", "Education", "Annual_Salary", "Residence", "Medical_Conditions")
SHARED_MIN, NONE_MAX, SHUFFLE_MAX, PAD_COS = 0.5, 0.1, 0.05, 0.99           # DEV-051
ROUTES = ("N", "F")


def people_rows(dfs, adapter, n_forget, n_retain):
    """[{set, id, name, rows: {route: [row per attribute]}}]; aborts on a missing question or a name in a face question."""
    people = []
    for set_name, n in (("forget", n_forget), ("retain", n_retain)):
        for _, row in dfs[set_name].head(n).iterrows():
            name, um, mm, img = profile_name(row), parse(row["UM_QA"]), parse(row["MM_QA"]), image_bytes(row["image"])
            rows = {"N": [], "F": []}
            for a in ATTRS:
                for route, qa in (("N", um), ("F", mm)):
                    q, ans = qa.get("question", {}).get(a), qa.get("answer", {}).get(a)
                    if q is None or ans is None:
                        raise RuntimeError(f"{set_name}/{row['ID']}: no {a} question in route {route}")
                    if route == "N" and name not in q:
                        raise RuntimeError(f"{set_name}/{row['ID']}/{a}: the name is not in the name question: {q}")
                    if route == "F" and any(w in q for w in [name] + name.split()):
                        raise RuntimeError(f"{set_name}/{row['ID']}/{a}: the name is in the face question: {q}")
                    rows[route].append({"uid": f"{set_name}/{row['ID']}/{a}/{route}", "prompt": adapter.prompt(q, route == "F"),
                                        "correct_text": str(ans), "image_bytes": img if route == "F" else None})
            people.append({"set": set_name, "id": row["ID"], "name": name, "rows": rows})
    return people


@torch.no_grad()
def run(adapter, model, rows, layers):
    """States at the last prompt token [B, n_layers, D] (float32, cpu), log p and top-1 of the first answer token."""
    batch = encode_teacher_forced(adapter, rows)
    t = first_target(tools.answer_targets(batch))
    assert torch.equal(t["b"], torch.arange(len(rows), device=t["b"].device)), "one first answer token per row, in order"
    store = {}
    with record_outputs(model, "resid", layers, store, select=(t["b"], t["pos"])):
        at = tools.forward_logits(adapter, model, batch)[t["b"], t["pos"]].float()
    h = torch.stack([store[L].float().cpu() for L in layers], 1)
    return h, at.log_softmax(-1).gather(-1, t["tok"][:, None]).squeeze(-1).cpu(), (at.argmax(-1) == t["tok"]).cpu()


def loao(Q, K):
    """[P, A] hits: query Q[p, a] against the keys mean over a' != a of K[p', a'] (Q, K centred, [P, A, D])."""
    P, A, _ = Q.shape
    tot, hits = K.sum(1), torch.zeros(P, A)
    for a in range(A):
        sim = F.normalize(Q[:, a], dim=-1) @ F.normalize((tot - K[:, a]) / (A - 1), dim=-1).T
        hits[:, a] = (sim.argmax(1) == torch.arange(P)).float()
    return hits


def same_attr(Q, K):
    P, A, _ = Q.shape
    return torch.stack([((F.normalize(Q[:, a], dim=-1) @ F.normalize(K[:, a], dim=-1).T).argmax(1) == torch.arange(P)).float()
                        for a in range(A)], 1)


def whole(Q, K):
    return ((F.normalize(Q.mean(1), dim=-1) @ F.normalize(K.mean(1), dim=-1).T).argmax(1) == torch.arange(Q.shape[0])).float()


def main():
    p = base_parser(__doc__)
    p.add_argument("--merge-layer", type=int, required=True, help="L* of the model (LLaVA 20, Qwen 22)")
    p.add_argument("--layers", default=None, help="default: 12, 14, ... and the last layer")
    p.add_argument("--n-forget", type=int, default=50)
    p.add_argument("--n-retain", type=int, default=100)
    p.add_argument("--batch-txt", type=int, default=32)
    p.add_argument("--batch-img", type=int, default=8)
    p.add_argument("--tag", default=None, help="output folder suffix (default: the name of umu.ckpt)")
    p.add_argument("--smoke", action="store_true", help="12 + 12 people, output under identity_code_smoke")
    args = p.parse_args()
    cfg = resolve(args)
    tag = args.tag or Path(cfg["umu"]["ckpt"]).name
    out = results_dir(cfg) / "mech_umu" / f"vanilla_{tag}" / ("identity_code_smoke" if args.smoke else "identity_code")
    log = Logger(out / "log.txt")
    if args.smoke:
        args.n_forget = args.n_retain = 12
    adapter = get_adapter({**cfg["model"], "id": cfg["umu"]["ckpt"], "processor_id": cfg["model"]["id"]})
    dfs = load_umu(cfg["paths"]["root"], args.n_retain, 0, log, cfg["umu"]["forget_split"])
    people = people_rows(dfs, adapter, args.n_forget, args.n_retain)
    P, A = len(people), len(ATTRS)
    model = adapter.load_model()
    model.eval()
    n_layers = len(decoder_layers(model))
    layers = [int(x) for x in args.layers.split(",")] if args.layers else sorted(set(range(12, n_layers, 2)) | {n_layers - 1})
    log(f"vanilla {cfg['umu']['ckpt']} ({cfg['model']['id']}): {P} people ({args.n_forget} forget + {args.n_retain} retain) x {A} "
        f"attributes x 2 routes = {2 * P * A} prompts; layers {layers}; L* = {args.merge_layer}; chance {1 / P:.3f}")

    X, lp, top = {}, {}, {}
    for route, bs in (("N", args.batch_txt), ("F", args.batch_img)):
        flat = [r for person in people for r in person["rows"][route]]
        hs, lps, tops = [], [], []
        for i in range(0, len(flat), bs):
            h, l, t = run(adapter, model, flat[i:i + bs], layers)
            hs.append(h), lps.append(l), tops.append(t)
            if (i // bs) % 20 == 0:
                log(f"  route {route}: {i + len(h)}/{len(flat)}")
        X[route] = torch.cat(hs).view(P, A, len(layers), -1)
        lp[route], top[route] = torch.cat(lps).view(P, A), torch.cat(tops).view(P, A).float()
        # validity: a batched state against the same prompt run alone
        worst = 1.0
        for j in np.linspace(0, len(flat) - 1, 4).astype(int).tolist():
            h1, _, _ = run(adapter, model, [flat[j]], layers)
            worst = min(worst, F.cosine_similarity(h1[0], X[route].view(P * A, len(layers), -1)[j], dim=-1).min().item())
        log(f"  route {route}: batched vs alone, lowest cosine over 4 prompts and all layers {worst:.4f}; first-token top-1 "
            f"{top[route].mean():.3f}")
        if worst < PAD_COS:
            raise RuntimeError(f"route {route}: a batched state differs from the same prompt run alone (cosine {worst:.4f})")
    model = None
    free()

    is_f = torch.tensor([person["set"] == "forget" for person in people])
    perm = torch.randperm(P, generator=torch.Generator().manual_seed(0))
    summary = {"model": cfg["model"]["id"], "ckpt": cfg["umu"]["ckpt"], "n_people": P, "n_forget": int(is_f.sum()), "attrs": list(ATTRS),
               "merge_layer": args.merge_layer, "chance": 1 / P,
               "top1": {r: {"all": top[r].mean().item(), "forget": top[r][is_f].mean().item(), "retain": top[r][~is_f].mean().item(),
                            "by_attr": {a: top[r][:, k].mean().item() for k, a in enumerate(ATTRS)}} for r in ROUTES},
               "layers": {}}
    hits_fn = {}
    for li, L in enumerate(layers):
        C = {r: X[r][:, :, li] - X[r][:, :, li].mean(0, keepdim=True) for r in ROUTES}         # centred per (attribute, route)
        h = {"F->N": loao(C["F"], C["N"]), "N->F": loao(C["N"], C["F"]), "N->N": loao(C["N"], C["N"]), "F->F": loao(C["F"], C["F"])}
        hits_fn[L] = h["F->N"]
        shuffled = (F.normalize(C["F"][:, 0], dim=-1) @ F.normalize(C["N"][perm][:, 1:].mean(1), dim=-1).T).argmax(1)
        z = {k: v.mean().item() for k, v in h.items()}
        z.update({"F->N forget": h["F->N"][is_f].mean().item(), "F->N retain": h["F->N"][~is_f].mean().item(),
                  "N->F forget": h["N->F"][is_f].mean().item(), "N->F retain": h["N->F"][~is_f].mean().item(),
                  "F->N by_attr": {a: h["F->N"][:, k].mean().item() for k, a in enumerate(ATTRS)},
                  "whole F->N": whole(C["F"], C["N"]).mean().item(), "whole N->F": whole(C["N"], C["F"]).mean().item(),
                  "same_attr F->N": same_attr(C["F"], C["N"]).mean().item(),
                  "shuffled F->N": (shuffled == torch.arange(P)).float().mean().item()})
        summary["layers"][L] = z
        log(f"  L{L:>2}: leave-one-attribute-out F->N {z['F->N']:.3f}  N->F {z['N->F']:.3f} | within N {z['N->N']:.3f}  within F "
            f"{z['F->F']:.3f} | whole F->N {z['whole F->N']:.3f}  N->F {z['whole N->F']:.3f} | same attribute F->N "
            f"{z['same_attr F->N']:.3f} | shuffled {z['shuffled F->N']:.3f}")
        if z["shuffled F->N"] > SHUFFLE_MAX and not args.smoke:
            raise RuntimeError(f"L{L}: retrieval with shuffled people is {z['shuffled F->N']:.3f} > {SHUFFLE_MAX}")
    after = {L: summary["layers"][L]["F->N"] for L in layers if L >= args.merge_layer}
    best = max(after, key=after.get)
    every = [summary["layers"][L]["F->N"] for L in layers]
    summary["best_layer_after_merge"], summary["best_F->N"] = best, after[best]
    summary["verdict"] = ("shared person code" if after[best] >= SHARED_MIN else "none" if max(every) <= NONE_MAX else "partial")
    log(f"F->N leave-one-attribute-out: best layer >= L* is L{best} with {after[best]:.3f} (chance {1 / P:.3f}) -> {summary['verdict']}")
    atomic_write_json(out / "summary.json", summary)
    write_jsonl(out / "items.jsonl", [
        {"set": person["set"], "id": person["id"], "attr": a, "lp_N": lp["N"][i, k].item(), "lp_F": lp["F"][i, k].item(),
         "top_N": bool(top["N"][i, k]), "top_F": bool(top["F"][i, k]), **{f"hit_F->N/{L}": bool(hits_fn[L][i, k]) for L in layers}}
        for i, person in enumerate(people) for k, a in enumerate(ATTRS)])
    torch.save({"layers": layers, "attrs": list(ATTRS), "people": [(q["set"], q["id"], q["name"]) for q in people],
                "N": X["N"].half(), "F": X["F"].half()}, out / "states.pt")

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(6.4, 3.8))
    for key, lab, st in (("F->N", "face query → name keys", "o-"), ("N->F", "name query → face keys", "s-"),
                         ("N->N", "within the name route", "^:"), ("F->F", "within the face route", "v:"),
                         ("whole F->N", "whole code, face → name", "x--")):
        ax.plot(layers, [summary["layers"][L][key] for L in layers], st, label=lab, ms=4)
    ax.axvline(args.merge_layer, color="grey", lw=0.8, ls="--")
    ax.axhline(1 / P, color="k", lw=0.6)
    ax.set_xlabel("layer (state at the last prompt token)"), ax.set_ylabel("same person retrieved")
    ax.set_title(f"Person code across routes, vanilla ({cfg['model']['name']}; {P} people, chance {1 / P:.3f})", fontsize=9)
    ax.set_ylim(-0.02, 1.02), ax.legend(fontsize=7)
    fig.tight_layout(), fig.savefig(out / "identity_code.png", dpi=180)
    log(f"wrote {out / 'summary.json'}")


if __name__ == "__main__":
    main()
