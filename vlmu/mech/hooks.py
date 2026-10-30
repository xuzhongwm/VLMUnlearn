"""Hook utilities for adapter-contribution analyses (Milestone J; later window removal and
contribution replacement y = F_U(H_I) - d(H_I) + d(H_T)).

- lora_modules / decoder_layers: locate LoRA-wrapped linear layers and LM decoder layers
- lora_delta: the adapter's contribution d(x) = scaling * B(A(x)) of one LoRA module
- remove_contributions: hook-based removal of d(x) at chosen modules
- disable_locally: PEFT-native disabling of the adapter at chosen modules (reference)
- record_inputs / replace_contributions: cache module inputs from a source pass, then in a
  target pass replace out -> out - d(x_target) + d(x_source) (all positions or the last one)
- record_layer_outputs / patch_layer_outputs: cache / patch decoder-layer hidden states
- record_outputs / patch_outputs: the same for kind = "resid" (decoder-layer output), "attn"
  (self-attention output) or "mlp" (MLP output), optionally at selected (row, position) pairs
- lm_parts: the LM's final norm and unembedding, and the image placeholder token id
"""
import contextlib
import re

import torch

_LAYER_IDX = re.compile(r"layers\.(\d+)\.")


def lora_modules(model) -> dict:
    from peft.tuners.lora import LoraLayer
    return {n: m for n, m in model.named_modules() if isinstance(m, LoraLayer) and hasattr(m, "base_layer")}


def layer_index(name: str):
    m = _LAYER_IDX.search(name)
    return int(m.group(1)) if m else None


def decoder_layers(model) -> list:
    pat = re.compile(r"(^|\.)layers\.(\d+)$")
    return [(n, m) for n, m in model.named_modules() if pat.search(n) and not re.search(r"visual|vision", n)]


def lora_delta(mod, x):
    """d(x) for the active adapter of one LoRA module (eval mode: dropout is identity)."""
    a = mod.active_adapters[0]
    A, B = mod.lora_A[a], mod.lora_B[a]
    return (B(A(mod.lora_dropout[a](x.to(A.weight.dtype)))) * mod.scaling[a])


@contextlib.contextmanager
def _hooks(handles):
    try:
        yield
    finally:
        for h in handles:
            h.remove()


def remove_contributions(model, names):
    mods = lora_modules(model)

    def hook(mod, inp, out):
        return out - lora_delta(mod, inp[0]).to(out.dtype)
    return _hooks([mods[n].register_forward_hook(hook) for n in names])


@contextlib.contextmanager
def disable_locally(model, names):
    mods = lora_modules(model)
    for n in names:
        mods[n].enable_adapters(False)
    try:
        yield
    finally:
        for n in names:
            mods[n].enable_adapters(True)


def record_inputs(model, names, store: dict):
    mods = lora_modules(model)

    def mk(n):
        def hook(mod, inp, out):
            store[n] = inp[0].detach().clone()
        return hook
    return _hooks([mods[n].register_forward_hook(mk(n)) for n in names])


def replace_contributions(model, names, source_inputs: dict, positions="all"):
    """out -> out - d(x_target) + d(x_source). positions='last' edits only the final position
    (left-padded batches: the token whose logits are scored)."""
    mods = lora_modules(model)

    def mk(n):
        def hook(mod, inp, out):
            x_t, x_s = inp[0], source_inputs[n]
            if positions == "last":
                out = out.clone()
                out[:, -1] = out[:, -1] - lora_delta(mod, x_t[:, -1]).to(out.dtype) + lora_delta(mod, x_s[:, -1]).to(out.dtype)
                return out
            return out - lora_delta(mod, x_t).to(out.dtype) + lora_delta(mod, x_s).to(out.dtype)
        return hook
    return _hooks([mods[n].register_forward_hook(mk(n)) for n in names])


def _hidden(out):
    return out[0] if isinstance(out, tuple) else out


def _replace(out, h):
    return (h,) + tuple(out[1:]) if isinstance(out, tuple) else h


_KIND = {"attn": "self_attn", "mlp": "mlp"}


def component_modules(model, kind) -> list:
    """Per decoder layer: the layer itself (resid), its self-attention (attn) or its MLP (mlp)."""
    layers = [m for _, m in decoder_layers(model)]
    return layers if kind == "resid" else [getattr(m, _KIND[kind]) for m in layers]


def record_outputs(model, kind, idxs, store: dict, select=None):
    """store[i] = output of component i ([B, T, D]), or only out[select] for select = (rows, positions)."""
    mods = component_modules(model, kind)

    def mk(i):
        def hook(mod, inp, out):
            h = _hidden(out).detach()
            store[i] = h[select].clone() if select is not None else h.clone()
        return hook
    return _hooks([mods[i].register_forward_hook(mk(i)) for i in idxs])


def patch_outputs(model, kind, cached: dict, index=None):
    """Replace component outputs by cached ones: everywhere (index=None, same shapes), or at
    index = (rows, target_positions, source_positions): out[b, tp] = cached[b, sp]."""
    mods = component_modules(model, kind)

    def mk(i):
        def hook(mod, inp, out):
            h = _hidden(out)
            if index is None:
                return _replace(out, cached[i].to(h.dtype))
            b, tp, sp = index
            h = h.clone()
            h[b, tp] = cached[i][b, sp].to(h.dtype)
            return _replace(out, h)
        return hook
    return _hooks([mods[i].register_forward_hook(mk(i)) for i in cached])


def record_layer_outputs(model, idxs, store: dict):
    return record_outputs(model, "resid", idxs, store)


def patch_layer_outputs(model, cached: dict, positions="all"):
    if positions == "all":
        return patch_outputs(model, "resid", cached)
    B, T = next(iter(cached.values())).shape[:2]
    last = torch.full((B,), T - 1, device=next(iter(cached.values())).device)
    return patch_outputs(model, "resid", cached, (torch.arange(B, device=last.device), last, last))


def _one(model, pattern, what):
    found = [m for n, m in model.named_modules() if re.search(pattern, n) and not re.search(r"visual|vision", n)]
    assert len(found) == 1, f"expected one {what}, found {len(found)}"
    return found[0]


def lm_parts(model):
    """(final RMSNorm, unembedding, image placeholder token id) of the language model."""
    norm, head = _one(model, r"(^|\.)norm$", "final norm"), _one(model, r"(^|\.)lm_head$", "lm_head")
    assert getattr(norm, "bias", None) is None and getattr(head, "bias", None) is None, "expects RMSNorm + bias-free head"
    cfg = model.config
    img = next((v for k in ("image_token_id", "image_token_index") if (v := getattr(cfg, k, None)) is not None), None)
    return norm, head, img
