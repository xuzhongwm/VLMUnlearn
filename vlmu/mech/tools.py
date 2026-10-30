"""Intervention tools for the H1 / H1' / H2 analyses (research.md, "Hypotheses & Predictions").

Models: theta_u = vanilla theta_0 + an unlearning LoRA (PEFT). theta_0 is the same model with the
adapter disabled, so both are read from one loaded model.

Targets: one (row, position, token) triple per scored token, as dict(b=, pos=, tok=) of
LongTensors. mc_targets scores the correct option letter at the last position of a
left-padded MC batch (adapter.encode_mc); answer_targets scores the teacher-forced answer
tokens of an adapter.encode_train batch. Per-row results sum over that row's targets.

- layer_logit_diff: logit lens (final norm + unembedding on every decoder-layer output) ->
  summed target log-prob per layer, theta_u - theta_0. Read the difference, not the early-layer
  absolute values.
- component_attribution: exact split of the final target-logit change theta_u - theta_0 into the
  layer-0 input (embeddings) and every layer's attention and MLP output. With the final RMSNorm
  scale taken from each run, logit = sum_c <c, w * W_U[tok]> / rms, so the parts sum to the
  change (up to bf16 rounding of the residual adds).
- delta_wx_norm: per LoRA module, ||Delta W x|| = ||d(x)|| averaged over a token group (name /
  image / template / last), plus, for the residual writers o_proj and down_proj, the projection
  of d(x) on the unit answer direction w * W_U[tok] (first target of the row).
- lora_layer_mask: keep the unlearning LoRA only on chosen layers (others disabled).
- cross_model_patch: replace component outputs of one model (theta_u / theta_0) by the other's,
  same input, chosen layers and positions.
- cross_route_patch: replace component outputs of a target run (e.g. image route) by those of a
  source run (e.g. text route) of the same fact. ALIGNMENT RULE: the two prompts differ in length
  (image tokens vs name tokens), so only end-aligned positions are patched: the longest common
  suffix of the two token-id sequences (the question text after the referent, the options, the
  instruction and the final position), or only the final position (positions="last"). Both
  batches must be left-padded (encode_mc). describe_alignment prints the patched tokens.
"""
import contextlib
import re

import torch

from ..data.fields import referent_from_stem
from .hooks import (_hooks, component_modules, decoder_layers, disable_locally, layer_index, lm_parts, lora_delta,
                    lora_modules, patch_outputs, record_outputs)


# ---------------------------------------------------------------------------------------- basics
def forward_logits(adapter, model, batch):
    with torch.no_grad(), adapter.autocast():
        return adapter.forward(model, batch)


def mc_targets(adapter, rows, batch):
    assert bool(batch["attention_mask"][:, -1].all()), "MC batches must be left-padded"
    B, T = batch["input_ids"].shape
    dev = batch["input_ids"].device
    return {"b": torch.arange(B, device=dev), "pos": torch.full((B,), T - 1, device=dev),
            "tok": torch.tensor([adapter.letter_ids[r["answer"]] for r in rows], device=dev), "n_rows": B}


def answer_targets(batch):
    lab = batch["labels"][:, 1:]
    b, pos = (lab != -100).nonzero(as_tuple=True)
    return {"b": b, "pos": pos, "tok": lab[b, pos], "n_rows": batch["labels"].shape[0]}


def _per_row(values, targets):
    return torch.zeros(targets["n_rows"], device=values.device, dtype=values.dtype).index_add(0, targets["b"], values)


def _first(targets):
    """Index (into the target list) of each row's first target."""
    first = {}
    for k, b in enumerate(targets["b"].tolist()):
        first.setdefault(b, k)
    return torch.tensor([first[b] for b in range(targets["n_rows"])], device=targets["b"].device)


def target_logprob(logits_at, tok):
    return logits_at.float().log_softmax(-1).gather(-1, tok[:, None]).squeeze(-1)


@contextlib.contextmanager
def _theta0(model, on):
    """theta_0 (adapter disabled) when on, else theta_u."""
    if on:
        with model.disable_adapter():
            yield
    else:
        yield


# ---------------------------------------------------------------------------------------- logit lens
@torch.no_grad()
def lens_logprobs(adapter, model, batch, targets):
    """[n_layers, n_rows]: summed target log-prob read from each decoder layer's output via the
    final norm + unembedding. The last layer reproduces the model's own output."""
    norm, head, _ = lm_parts(model)
    n = len(decoder_layers(model))
    store = {}
    with record_outputs(model, "resid", range(n), store, select=(targets["b"], targets["pos"])):
        forward_logits(adapter, model, batch)
    out = []
    with adapter.autocast():
        for i in range(n):
            out.append(_per_row(target_logprob(head(norm(store[i])), targets["tok"]), targets))
    return torch.stack(out)


def lens_logits(adapter, model, batch, targets, layer=-1):
    """Full-vocabulary logit-lens logits at the target positions for one layer (for checks)."""
    norm, head, _ = lm_parts(model)
    i = range(len(decoder_layers(model)))[layer]
    store = {}
    with record_outputs(model, "resid", [i], store, select=(targets["b"], targets["pos"])):
        forward_logits(adapter, model, batch)
    with torch.no_grad(), adapter.autocast():
        return head(norm(store[i]))


@torch.no_grad()
def layer_logit_diff(adapter, model, batch, targets):
    lp_u = lens_logprobs(adapter, model, batch, targets)
    with model.disable_adapter():
        lp_0 = lens_logprobs(adapter, model, batch, targets)
    return {"lp_u": lp_u.cpu(), "lp_0": lp_0.cpu(), "diff": (lp_u - lp_0).cpu()}


# ---------------------------------------------------------------------------------------- attribution
@torch.no_grad()
def _components(adapter, model, batch, targets):
    """Layer-0 input, per-layer attn / mlp outputs and the final residual at the target positions (fp32)."""
    sel = (targets["b"], targets["pos"])
    n = len(decoder_layers(model))
    attn, mlp, resid, emb = {}, {}, {}, {}

    def pre(mod, args, kwargs):
        h = args[0] if args else kwargs["hidden_states"]
        emb["h"] = h.detach()[sel].clone()
    with _hooks([decoder_layers(model)[0][1].register_forward_pre_hook(pre, with_kwargs=True)]), \
            record_outputs(model, "attn", range(n), attn, select=sel), \
            record_outputs(model, "mlp", range(n), mlp, select=sel), \
            record_outputs(model, "resid", [n - 1], resid, select=sel):
        logits = forward_logits(adapter, model, batch)
    logit = logits[sel].float().gather(-1, targets["tok"][:, None]).squeeze(-1)
    f = lambda d: torch.stack([d[i].float() for i in range(n)])  # noqa: E731  [L, n_t, D]
    return {"embed": emb["h"].float(), "attn": f(attn), "mlp": f(mlp), "final": resid[n - 1].float(), "logit": logit}


def _contrib(parts, direction, eps):
    s = torch.rsqrt(parts["final"].pow(2).mean(-1) + eps)  # [n_t] final RMSNorm scale of this run
    return {"embed": (parts["embed"] * direction).sum(-1) * s,
            "attn": (parts["attn"] * direction).sum(-1) * s, "mlp": (parts["mlp"] * direction).sum(-1) * s}


@torch.no_grad()
def component_attribution(adapter, model, batch, targets):
    """Per row: change of the summed target logit (theta_u - theta_0) split into embed / attn[l] / mlp[l]."""
    norm, head, _ = lm_parts(model)
    eps = getattr(norm, "variance_epsilon", getattr(norm, "eps", 1e-6))
    direction = head.weight[targets["tok"]].float() * norm.weight.float()  # [n_t, D]
    pu = _components(adapter, model, batch, targets)
    with model.disable_adapter():
        p0 = _components(adapter, model, batch, targets)
    cu, c0 = _contrib(pu, direction, eps), _contrib(p0, direction, eps)
    row = lambda v: _per_row(v, targets)  # noqa: E731
    embed = row(cu["embed"] - c0["embed"])
    attn = torch.stack([row(x) for x in cu["attn"] - c0["attn"]])
    mlp = torch.stack([row(x) for x in cu["mlp"] - c0["mlp"]])
    return {"embed": embed.cpu(), "attn": attn.cpu(), "mlp": mlp.cpu(),
            "total_attribution": (embed + attn.sum(0) + mlp.sum(0)).cpu(),
            "delta_logit": row(pu["logit"] - p0["logit"]).cpu(),
            "logit_scale": torch.maximum(row(pu["logit"].abs()), row(p0["logit"].abs())).cpu()}


# ---------------------------------------------------------------------------------------- ||dW x||
def mc_texts(adapter, rows, batch):
    has_image = any(k in batch for k in ("pixel_values",))
    return [adapter.mc_text(r, has_image) for r in rows]


def token_groups(adapter, model, rows, batch, texts, targets):
    """Boolean [B, T] masks: image (placeholder tokens), name (tokens overlapping the profile
    name in text rows; needs row['name']), last (each row's first target position) and
    template (every other non-padding token: chat frame, question, options, instruction).
    The name is row['name'] if present, else parsed from the paired text stem."""
    ids, valid = batch["input_ids"], batch["attention_mask"].bool()
    _, _, img_id = lm_parts(model)
    image = (ids == img_id) & valid if img_id is not None else torch.zeros_like(valid)
    name = torch.zeros_like(valid)
    for i, (r, text) in enumerate(zip(rows, texts)):
        who = r.get("name") or (referent_from_stem(r["field"], r["question"]) if r.get("source") == "paired" else None)
        if not who or bool(image[i].any()) or who.startswith("the pictured"):
            continue
        enc = adapter.tok(text, add_special_tokens=True, return_offsets_mapping=True)
        start = int(valid[i].nonzero()[0])
        row_ids = ids[i, valid[i]].tolist()
        if enc["input_ids"] != row_ids:
            raise ValueError(f"re-tokenised text does not match the batch for row {r.get('uid')}")
        spans = [(m.start(), m.end()) for m in re.finditer(re.escape(who), text)]
        for j, (a, b) in enumerate(enc["offset_mapping"]):
            if b > a and any(a < e and b > s for s, e in spans):
                name[i, start + j] = True
    last = torch.zeros_like(valid)
    f = _first(targets)
    last[targets["b"][f], targets["pos"][f]] = True
    template = valid & ~image & ~name & ~last
    return {"name": name, "image": image, "template": template, "last": last}


_RESID_WRITERS = ("o_proj", "down_proj")


@torch.no_grad()
def delta_wx_norm(adapter, model, batch, targets, groups):
    """Per layer and module kind: mean ||d(x)|| over each token group ([L, B], NaN where the group
    is empty), and for o_proj / down_proj the mean projection of d(x) on the unit answer direction."""
    norm, head, _ = lm_parts(model)
    f = _first(targets)
    u = head.weight[targets["tok"][f]].float() * norm.weight.float()
    u = u / u.norm(dim=-1, keepdim=True)  # [B, D]
    mods = lora_modules(model)
    n_layers = len(decoder_layers(model))
    out = {"norm": {}, "proj": {}}

    def gmean(v):  # v [B, T] -> {group: [B]}
        res = {}
        for g, m in groups.items():
            cnt = m.sum(1)
            res[g] = torch.where(cnt > 0, (v * m).sum(1) / cnt.clamp_min(1), torch.full_like(cnt, float("nan"), dtype=v.dtype))
        return res

    def mk(name):
        kind = name.rsplit(".", 1)[-1]
        li = layer_index(name)

        def hook(mod, inp, o):
            d = lora_delta(mod, inp[0]).float()
            for g, v in gmean(d.norm(dim=-1)).items():
                out["norm"].setdefault(kind, {}).setdefault(g, torch.full((n_layers, d.shape[0]), float("nan")))[li] = v.cpu()
            if kind in _RESID_WRITERS:
                for g, v in gmean((d * u[:, None]).sum(-1)).items():
                    out["proj"].setdefault(kind, {}).setdefault(g, torch.full((n_layers, d.shape[0]), float("nan")))[li] = v.cpu()
        return hook
    with _hooks([m.register_forward_hook(mk(n)) for n, m in mods.items()]):
        forward_logits(adapter, model, batch)
    return out


# ---------------------------------------------------------------------------------------- masks / patching
def lora_layer_mask(model, keep_layers):
    """Context: the LoRA stays active only on decoder layers in keep_layers."""
    keep = set(keep_layers)
    return disable_locally(model, [n for n in lora_modules(model) if layer_index(n) not in keep])


@torch.no_grad()
def cross_model_patch(adapter, model, batch, layers, kind="resid", source="vanilla", positions=None):
    """Run the other model with component outputs at `layers` taken from `source` ('vanilla' =
    theta_0 patched into theta_u; 'unlearned' = theta_u patched into theta_0). positions: None (all)
    or a bool [B, T] mask. Returns the patched run's logits."""
    store = {}
    with _theta0(model, source == "vanilla"), record_outputs(model, kind, layers, store):
        forward_logits(adapter, model, batch)
    index = None
    if positions is not None:
        b, p = positions.nonzero(as_tuple=True)
        index = (b, p, p)
    with _theta0(model, source != "vanilla"), patch_outputs(model, kind, store, index):
        return forward_logits(adapter, model, batch)


def route_alignment(src_batch, tgt_batch, positions="suffix"):
    """End-aligned positions (see module docstring): per row the longest common suffix of the
    token ids (or 1 for positions='last'). Returns (rows, tgt_pos, src_pos) and the per-row lengths."""
    for bt in (src_batch, tgt_batch):
        assert bool(bt["attention_mask"][:, -1].all()), "cross-route patching needs left-padded batches"
    S, T = src_batch["input_ids"], tgt_batch["input_ids"]
    assert S.shape[0] == T.shape[0], "source and target batches must hold the same facts in the same order"
    bs, tp, sp, ks = [], [], [], []
    for i in range(S.shape[0]):
        s = S[i, src_batch["attention_mask"][i].bool()].tolist()
        t = T[i, tgt_batch["attention_mask"][i].bool()].tolist()
        k = 0
        while k < min(len(s), len(t)) and s[-1 - k] == t[-1 - k]:
            k += 1
        if positions == "last":
            k = min(k, 1)
        assert k >= 1, f"row {i}: the two routes do not share the final token"
        ks.append(k)
        bs += [i] * k
        tp += list(range(T.shape[1] - k, T.shape[1]))
        sp += list(range(S.shape[1] - k, S.shape[1]))
    dev = T.device
    return tuple(torch.tensor(x, device=dev) for x in (bs, tp, sp)), ks


def describe_alignment(adapter, src_batch, tgt_batch, index, row=0):
    b, tp, sp = index
    sel = (b == row).nonzero().squeeze(-1).tolist()
    return [{"tgt_pos": int(tp[j]), "src_pos": int(sp[j]),
             "tgt_token": adapter.tok.decode([int(tgt_batch["input_ids"][row, tp[j]])]),
             "src_token": adapter.tok.decode([int(src_batch["input_ids"][row, sp[j]])])} for j in sel]


@torch.no_grad()
def cross_route_patch(adapter, model, src_batch, tgt_batch, layers, kind="resid", positions="suffix"):
    """Target run (e.g. image route) with component outputs at `layers` taken from the source run
    (e.g. text route) of the same facts, at end-aligned positions. Same model for both runs.
    Returns (patched target logits, alignment index, per-row aligned lengths)."""
    index, ks = route_alignment(src_batch, tgt_batch, positions)
    store = {}
    with record_outputs(model, kind, layers, store):
        forward_logits(adapter, model, src_batch)
    with patch_outputs(model, kind, store, index):
        return forward_logits(adapter, model, tgt_batch), index, ks


__all__ = ["answer_targets", "component_attribution", "component_modules", "cross_model_patch", "cross_route_patch",
           "delta_wx_norm", "describe_alignment", "forward_logits", "layer_logit_diff", "lens_logits", "lens_logprobs",
           "lora_layer_mask", "mc_targets", "mc_texts", "route_alignment", "token_groups"]
