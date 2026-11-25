"""UMU track, DEV-041: a MEND-style editor for request-level unlearning that needs no portrait of the person.

MEND (Mitchell et al., ICLR 2022) trains a small network that turns the gradient of one edit into a weight update that
carries over to rephrasings and leaves unrelated inputs alone. Here the "rephrasing" is the other route to the person:
  edit      for one person, text only: the gradient of -log p(refusal | name question) with respect to the MLP
            down-projections of the decoder layers from --min-layer on (after the name and face routes merge, DEV-027/035)
  editor    per token and edited layer: (u, delta) -> (u~, delta~); the update is dW_l = -alpha_l * mean_tokens delta~ u~^T
            (u = the layer's input, delta = the gradient at its output; one shared low-rank residual MLP with a scale and
            shift per layer, initialised to the identity; the base model is frozen)
  trained   on people who are NOT to be forgotten and whose portraits may be used. After a person's text-only edit:
              generality  the person's PORTRAIT questions give the refusal        c_edit * NLL
              edit        the person's name questions give the refusal             c_edit * NLL
              locality    other people's name and portrait answers are unchanged   KL(before || after), answer tokens
  used      on a forget person: text gradient -> editor -> update. No portrait of a forget person is read during training
            or editing; their portraits appear only in the readout.

People (retain_90 sorted by ID): [0:100] readout retain (as every UMU run), [100:400] editor training, [400:450] held-out
validation people. Readout: UMU protocol (generation, exact match) under one person's edit at a time, on that person's
facts (txt, img, txt+img) and on the facts of --n-loc readout-retain people; doubly-correct facts of the unedited model.
Also: all forget people edited at once (the factors of the 50 edits summed), full protocol readout.

  python scripts/unlearn_umu_mend.py --config <model config> --tag <name> [--min-layer 20] [--episodes 3000] [--set umu.ckpt=...]
  --episodes 0: the untrained editor (a rescaled gradient step) as a control; --calibrate a,b,c: choose alpha0 first.
Writes runs/<model>/umu_mend/<tag>/{editor.pt, train.jsonl, evals/, summary.json, log.txt}.
"""
import contextlib
import math
import random
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np  # noqa: E402
import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402

from vlmu.config import base_parser, resolve  # noqa: E402
from vlmu.models import get_adapter  # noqa: E402
from vlmu.runs import model_dir  # noqa: E402
from vlmu.umu import DESC_FIELDS, fact_table, generate_protocol, load_umu, parse, protocol_items, save_image  # noqa: E402
from vlmu.utils import Logger, append_jsonl, atomic_write_json  # noqa: E402

REFUSAL = "I'm not sure."
ATTRS = DESC_FIELDS + ("Medical_Conditions",)


def person_rows(row, form, answer=None):
    """Training rows of one person: UM_QA (txt) or MM_QA + portrait (img); answer=None keeps the true answers."""
    qa = parse(row["UM_QA" if form == "txt" else "MM_QA"])
    return [{"uid": f"mend/{row['ID']}/{f}/{form}", "form": form, "question": qa["question"][f],
             "answer": answer or str(qa["answer"][f]), "image": None if form == "txt" else f"{row['ID']}.jpg"}
            for f in ATTRS if qa.get("answer", {}).get(f) is not None and qa.get("question", {}).get(f)]


def edit_modules(model, min_layer):
    pat = re.compile(r"(^|\.)layers\.(\d+)\.mlp\.down_proj$")
    out = []
    for n, m in model.named_modules():
        h = pat.search(n)
        if h and not re.search(r"visual|vision", n) and int(h.group(2)) >= min_layer:
            out.append((int(h.group(2)), n, m))
    assert out, "no MLP down-projection found"
    return sorted(out, key=lambda x: x[0])


class Editor(torch.nn.Module):
    def __init__(self, d_in, d_out, n_mod, rank, alpha0):
        super().__init__()
        self.d_in, x = d_in, d_in + d_out
        self.U, self.V = torch.nn.Linear(x, rank), torch.nn.Linear(rank, x)
        torch.nn.init.zeros_(self.V.weight)
        torch.nn.init.zeros_(self.V.bias)
        self.scale = torch.nn.Parameter(torch.ones(n_mod, rank))
        self.shift = torch.nn.Parameter(torch.zeros(n_mod, rank))
        self.log_alpha = torch.nn.Parameter(torch.full((n_mod,), math.log(alpha0)))
        for name, d in (("u", d_in), ("d", d_out)):
            self.register_buffer(f"mu_{name}", torch.zeros(n_mod, d))
            self.register_buffer(f"var_{name}", torch.ones(n_mod, d))
        self.register_buffer("seen", torch.zeros(n_mod))

    @torch.no_grad()
    def update_stats(self, i, u, d, momentum=0.99):
        m = 0.0 if self.seen[i] == 0 else momentum
        for name, x in (("u", u), ("d", d)):
            getattr(self, f"mu_{name}")[i].mul_(m).add_((1 - m) * x.mean(0))
            getattr(self, f"var_{name}")[i].mul_(m).add_((1 - m) * x.var(0, unbiased=False))
        self.seen[i] += 1

    def forward(self, i, u, d):
        un = (u - self.mu_u[i]) / (self.var_u[i].sqrt() + 1e-6)
        dn = (d - self.mu_d[i]) / (self.var_d[i].sqrt() + 1e-6)
        x = torch.cat([un, dn], 1)
        y = x + self.V(F.relu(self.U(x) * self.scale[i] + self.shift[i]))
        return y[:, :self.d_in], y[:, self.d_in:], self.log_alpha[i].exp()


@contextlib.contextmanager
def capture(mods, store):
    """Per edited module: its input u [B, T, d_in] and the gradient at its output delta [B, T, d_out]."""
    def mk(i):
        def hook(mod, inp, out):
            store[i] = {"u": inp[0].detach()}
            out.register_hook(lambda g: store[i].__setitem__("d", g.detach()))
        return hook
    for _, _, m in mods:
        m.weight.requires_grad_(True)
    hs = [m.register_forward_hook(mk(i)) for i, (_, _, m) in enumerate(mods)]
    try:
        yield
    finally:
        for h in hs:
            h.remove()
        for _, _, m in mods:
            m.weight.requires_grad_(False)
            m.weight.grad = None


@contextlib.contextmanager
def applied(mods, edits):
    """The edited model: out += -alpha * (x . u~_j) delta~_j summed over the edit's tokens j, per edited module."""
    def mk(i):
        ut, dt, a = edits[i]

        def hook(mod, inp, out):
            coef = inp[0].float() @ ut.t()
            return out + ((-a / ut.shape[0]) * (coef @ dt)).to(out.dtype)      # mean over the edit's tokens
        return hook
    hs = [m.register_forward_hook(mk(i)) for i, (_, _, m) in enumerate(mods)]
    try:
        yield
    finally:
        for h in hs:
            h.remove()


def nll(adapter, model, batch):
    tok_lp, _, _ = adapter.answer_logprobs(model, batch)
    return -tok_lp.mean()


def answer_logp(adapter, model, batch):
    mask = batch["labels"][:, 1:] != -100
    return adapter.forward(model, batch)[:, :-1][mask].float().log_softmax(-1)


def main():
    p = base_parser(__doc__)
    p.add_argument("--tag", required=True)
    p.add_argument("--min-layer", type=int, required=True)
    p.add_argument("--episodes", type=int, default=3000)
    p.add_argument("--accum", type=int, default=4)
    p.add_argument("--lr", type=float, default=1e-5)
    p.add_argument("--lr-alpha", type=float, default=1e-3)
    p.add_argument("--alpha0", type=float, default=1e-4)
    p.add_argument("--rank", type=int, default=1920)
    p.add_argument("--c-edit", type=float, default=0.1)
    p.add_argument("--n-per", type=int, default=4, help="questions per batch in the outer loss")
    p.add_argument("--n-loc", type=int, default=10, help="readout-retain people scored under every single edit")
    p.add_argument("--n-val", type=int, default=50)
    p.add_argument("--log-every", type=int, default=50)
    p.add_argument("--smoke", action="store_true", help="8 episodes, 2 people per readout")
    p.add_argument("--calibrate", default=None, help="comma-separated alpha0 values: 16 untrained episodes each, then exit")
    args = p.parse_args()
    cfg = resolve(args)
    run_dir = model_dir(cfg) / "umu_mend" / args.tag
    log = Logger(run_dir / "log.txt")
    seed = cfg["unlearn"].get("seed", 0)
    random.seed(seed)
    torch.manual_seed(seed)
    dfs = load_umu(cfg["paths"]["root"], 450, 0, log, cfg["umu"]["forget_split"])
    forget, ret = dfs["forget"], dfs["retain"]
    loc_eval, train_people, val_people = ret.iloc[:100], ret.iloc[100:400], ret.iloc[400:450].head(args.n_val)
    if args.smoke:
        forget, val_people, args.episodes, args.n_loc, args.log_every = forget.head(2), val_people.head(2), 8, 2, 2
        loc_eval = loc_eval.head(4)
    image_root = run_dir / "images"
    for df in (train_people,):                       # portraits used in training: the editor-training people only
        for _, row in df.iterrows():
            save_image(row["image"], image_root / f"{row['ID']}.jpg")
    assert not set(forget["ID"]) & {f.stem for f in image_root.glob("*.jpg")}, "a forget portrait is in the training image root"
    adapter = get_adapter({**cfg["model"], "id": cfg["umu"]["ckpt"], "processor_id": cfg["model"]["id"]})
    model = adapter.load_model()
    model.eval()
    for prm in model.parameters():
        prm.requires_grad_(False)
    mods = edit_modules(model, args.min_layer)
    d_out, d_in = mods[0][2].weight.shape
    editor = Editor(d_in, d_out, len(mods), args.rank, args.alpha0).to(adapter.device)
    opt = torch.optim.Adam([{"params": [q for n, q in editor.named_parameters() if n != "log_alpha"], "lr": args.lr},
                            {"params": [editor.log_alpha], "lr": args.lr_alpha}])
    log(f"{cfg['model']['id']} ({cfg['umu']['ckpt']}): editing mlp.down_proj of layers {[m[0] for m in mods]} "
        f"({d_in} -> {d_out}); editor {sum(q.numel() for q in editor.parameters()) / 1e6:.1f}M parameters, rank {args.rank}; "
        f"{len(train_people)} training people, {len(val_people)} validation people, {len(forget)} forget people")

    def factors(row, train):
        """Text-only edit gradient of one person -> per-module (u, delta) over the real tokens, then the editor."""
        batch = adapter.encode_train(person_rows(row, "txt", REFUSAL), None)
        store = {}
        with capture(mods, store), torch.enable_grad():
            loss = nll(adapter, model, batch)
            loss.backward()
        mask = batch["attention_mask"].bool()
        edits = []
        for i in range(len(mods)):
            u, d = store[i]["u"][mask].float(), store[i]["d"][mask].float()
            if train:
                editor.update_stats(i, u, d)
            edits.append(editor(i, u, d))
        return edits, loss.item()

    # ---- editor training ------------------------------------------------------------------------------------------------
    train_rows = [row for _, row in train_people.iterrows()]
    if args.calibrate:          # the untrained editor (identity): how large may the first step be?
        for a0 in [float(x) for x in args.calibrate.split(",")]:
            with torch.no_grad():
                editor.log_alpha.fill_(math.log(a0))
            random.seed(seed)
            out = []
            for _ in range(16):
                row = random.choice(train_rows)
                other = random.choice([r for r in train_rows if r["ID"] != row["ID"]])
                bs = [adapter.encode_train(random.sample(person_rows(row, "img", REFUSAL), 4), image_root),
                      adapter.encode_train(random.sample(person_rows(row, "img"), 4), image_root),
                      adapter.encode_train(random.sample(person_rows(other, "txt"), 4), None),
                      adapter.encode_train(random.sample(person_rows(other, "img"), 4), image_root)]
                with torch.no_grad():
                    before = [answer_logp(adapter, model, b) for b in bs[2:]]
                    edits, _ = factors(row, train=True)
                    with applied(mods, edits):
                        kl = sum(F.kl_div(answer_logp(adapter, model, b), ref, reduction="batchmean", log_target=True)
                                 for b, ref in zip(bs[2:], before)) / 2
                        out.append((nll(adapter, model, bs[0]).item(), nll(adapter, model, bs[1]).item(), kl.item()))
            h = np.mean(out, 0)
            log(f"calibrate alpha0 {a0:.0e}: portrait NLL of the refusal {h[0]:.3f} | portrait NLL of the true answer {h[1]:.3f} | "
                f"locality KL {h[2]:.4f}")
        return
    editor_path = run_dir / "editor.pt"
    if editor_path.exists() and not args.smoke:
        editor.load_state_dict(torch.load(editor_path, map_location=adapter.device))
        log(f"loaded {editor_path}")
    else:
        with torch.no_grad():                      # input statistics of the editor, before any update (and for --episodes 0)
            for row in random.sample(train_rows, min(64, len(train_rows))):
                factors(row, train=True)
        hist, t0 = [], __import__("time").time()
        for ep in range(args.episodes):
            row = random.choice(train_rows)
            others = random.sample([r for r in train_rows if r["ID"] != row["ID"]], 2)
            pick = lambda rows: random.sample(rows, min(args.n_per, len(rows)))  # noqa: E731
            b_gen = adapter.encode_train(pick(person_rows(row, "img", REFUSAL)), image_root)
            b_edit = adapter.encode_train(pick(person_rows(row, "txt", REFUSAL)), None)
            b_loct = adapter.encode_train(pick(person_rows(others[0], "txt") + person_rows(others[1], "txt")), None)
            b_loci = adapter.encode_train(pick(person_rows(others[0], "img") + person_rows(others[1], "img")), image_root)
            with torch.no_grad():
                before = [answer_logp(adapter, model, b) for b in (b_loct, b_loci)]
            edits, l_in = factors(row, train=True)
            with applied(mods, edits):
                l_gen, l_edit = nll(adapter, model, b_gen), nll(adapter, model, b_edit)
                l_loc = sum(F.kl_div(answer_logp(adapter, model, b), ref, reduction="batchmean", log_target=True)
                            for b, ref in zip((b_loct, b_loci), before)) / 2
            loss = args.c_edit * (l_gen + l_edit) / 2 + l_loc
            if not torch.isfinite(loss):
                raise RuntimeError(f"episode {ep}: non-finite loss (gen {l_gen.item()}, edit {l_edit.item()}, loc {l_loc.item()})")
            (loss / args.accum).backward()
            hist.append((l_gen.item(), l_edit.item(), l_loc.item(), l_in))
            if (ep + 1) % args.accum == 0:
                torch.nn.utils.clip_grad_norm_(editor.parameters(), 100.0)
                opt.step()
                opt.zero_grad(set_to_none=True)
            if (ep + 1) % args.log_every == 0:
                h = np.mean(hist[-args.log_every:], 0)
                rec = {"episode": ep + 1, "gen_nll": h[0], "edit_nll": h[1], "loc_kl": h[2], "inner_nll": h[3],
                       "alpha": editor.log_alpha.exp().tolist(), "s_per_episode": (__import__("time").time() - t0) / (ep + 1)}
                append_jsonl(run_dir / "train.jsonl", [rec])
                log(f"episode {ep + 1}/{args.episodes}: portrait NLL of the refusal {h[0]:.3f} | name NLL {h[1]:.3f} | locality KL "
                    f"{h[2]:.4f} | unedited name NLL {h[3]:.3f} | alpha {min(rec['alpha']):.1e}..{max(rec['alpha']):.1e} "
                    f"({rec['s_per_episode']:.1f} s/episode)")
        torch.save(editor.state_dict(), editor_path)
    editor.eval()

    # ---- readout ------------------------------------------------------------------------------------------------------
    forms3 = {"forget": ("img", "txt", "txt+img")}
    pf = adapter.prompt if cfg["model"]["family"] != "llava" else None
    gen_bs = {"img": 8, "txt": 32}

    def protocol(rows, name):
        return generate_protocol(model, adapter.processor, rows, run_dir / "evals" / f"{name}.jsonl", lambda *a: None,
                                 batch_size=gen_bs, dtype=adapter.dtype)

    loc_small = loc_eval.head(args.n_loc)
    base = protocol(protocol_items({"forget": forget, "retain": loc_eval}, forms=forms3, prompt_fn=pf)
                    + [dict(r, set="val", uid=r["uid"].replace("/forget/", "/val/"))
                       for r in protocol_items({"forget": val_people}, sets=("forget",), forms=forms3, prompt_fn=pf)], "unedited")
    dc = {k[1] for k, v in fact_table(base).items() if v["txt"] and v["img"]}
    hit = {(r["pair_id"], r["modality"]): r["correct_exact"] for r in base}
    log(f"unedited model: {len(dc)} doubly-correct facts over the forget, validation and readout-retain people")
    summary = {"tag": args.tag, "model": cfg["model"]["id"], "min_layer": args.min_layer, "n_dc": len(dc)}
    loc_rows = protocol_items({"retain": loc_small}, sets=("retain",), prompt_fn=pf)
    for set_name, people in (("val", val_people), ("forget", forget)):
        per = []
        for _, row in people.iterrows():
            with torch.no_grad():
                edits, _ = factors(row, train=False)
            own = protocol_items({"forget": people[people["ID"] == row["ID"]]}, sets=("forget",), forms=forms3, prompt_fn=pf)
            with applied(mods, edits), torch.no_grad():
                res = protocol(own + loc_rows, f"{set_name}_{row['ID']}")
            for r in res:
                if r["pair_id"] in dc:
                    per.append({"person": row["ID"], "own": r["set"] == "forget", "modality": r["modality"], "hit": r["correct_exact"],
                                "response": r["response"]})
        acc = lambda own, m: float(np.mean([x["hit"] for x in per if x["own"] == own and x["modality"] == m] or [float("nan")]))  # noqa: E731
        summary[f"single/{set_name}"] = {"own_txt": acc(True, "txt"), "own_img": acc(True, "img"), "own_txt+img": acc(True, "txt+img"),
                                         "loc_txt": acc(False, "txt"), "loc_img": acc(False, "img"),
                                         "n_own_facts": sum(x["own"] and x["modality"] == "txt" for x in per), "n_edits": len(people),
                                         "refusals_own_img": float(np.mean([x["response"].strip() == REFUSAL for x in per
                                                                            if x["own"] and x["modality"] == "img"] or [float("nan")]))}
        z = summary[f"single/{set_name}"]
        log(f"one person edited at a time, {set_name} people ({z['n_edits']} edits, {z['n_own_facts']} dc facts): the person's facts "
            f"txt {z['own_txt']:.2f} img {z['own_img']:.2f} txt+img {z['own_txt+img']:.2f} (portrait answers that are the refusal: "
            f"{z['refusals_own_img']:.2f}) | other people's facts txt {z['loc_txt']:.2f} img {z['loc_img']:.2f}")
    # all forget people at once: the factors of the single edits concatenated per module
    with torch.no_grad():
        alls = [factors(row, train=False)[0] for _, row in forget.iterrows()]
    edits = [(torch.cat([e[i][0] for e in alls]), torch.cat([e[i][1] for e in alls]), alls[0][i][2]) for i in range(len(mods))]
    with applied(mods, edits), torch.no_grad():
        res = protocol(protocol_items({"forget": forget, "retain": loc_eval}, forms=forms3, prompt_fn=pf), "all_at_once")
    a = lambda s, m: float(np.mean([r["correct_exact"] for r in res if r["set"] == s and r["modality"] == m and r["pair_id"] in dc]))  # noqa: E731
    summary["all_at_once"] = {"forget_dc_txt": a("forget", "txt"), "forget_dc_img": a("forget", "img"), "forget_dc_txt+img": a("forget", "txt+img"),
                              "retain_dc_txt": a("retain", "txt"), "retain_dc_img": a("retain", "img")}
    z = summary["all_at_once"]
    log(f"all {len(forget)} forget people edited at once: forget dc txt {z['forget_dc_txt']:.2f} img {z['forget_dc_img']:.2f} txt+img "
        f"{z['forget_dc_txt+img']:.2f} | retain dc txt {z['retain_dc_txt']:.2f} img {z['retain_dc_img']:.2f}")
    summary["unedited_all"] = {m: float(np.mean([v for (pid, mm), v in hit.items() if mm == m])) for m in ("txt", "img")}
    atomic_write_json(run_dir / "summary.json", summary)
    log(f"wrote {run_dir / 'summary.json'}")


if __name__ == "__main__":
    main()
