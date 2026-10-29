"""Hook utilities for adapter-contribution analyses (Milestone J; later window removal and
contribution replacement y = F_U(H_I) - d(H_I) + d(H_T)).

- lora_modules / decoder_layers: locate LoRA-wrapped linear layers and LM decoder layers
- lora_delta: the adapter's contribution d(x) = scaling * B(A(x)) of one LoRA module
- remove_contributions: hook-based removal of d(x) at chosen modules
- disable_locally: PEFT-native disabling of the adapter at chosen modules (reference)
- record_inputs / replace_contributions: cache module inputs from a source pass, then in a
  target pass replace out -> out - d(x_target) + d(x_source) (all positions or the last one)
- record_layer_outputs / patch_layer_outputs: cache / patch decoder-layer hidden states
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


def record_layer_outputs(model, idxs, store: dict):
    layers = decoder_layers(model)

    def mk(i):
        def hook(mod, inp, out):
            store[i] = _hidden(out).detach().clone()
        return hook
    return _hooks([layers[i][1].register_forward_hook(mk(i)) for i in idxs])


def patch_layer_outputs(model, cached: dict, positions="all"):
    layers = decoder_layers(model)

    def mk(i):
        def hook(mod, inp, out):
            h = _hidden(out)
            new = cached[i] if positions == "all" else torch.cat([h[:, :-1], cached[i][:, -1:]], dim=1)
            return (new,) + tuple(out[1:]) if isinstance(out, tuple) else new
        return hook
    return _hooks([layers[i][1].register_forward_hook(mk(i)) for i in cached])
