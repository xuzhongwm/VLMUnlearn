"""Milestone J: correctness checks for the hook machinery, on a TRAINED adapter.

(i)   adapter-off reproduces the vanilla logits
(ii)  hook-based contribution removal (subtracting d(x) = scaling*B(A(x))) matches PEFT-native
      local adapter disabling, for a layer window and for all modules (the latter also = vanilla)
(iii) self-replacement reproduces the native logits:
      (a) decoder-layer outputs cached from a pass and patched back in (all / last position)
      (b) contribution replacement y = F_U(H) - d(H) + d(H_src) with H_src = H (all / last)
Tools of vlmu/mech/tools.py (research.md task D, tests J1-J6):
J1  passive hooks (recording, ||dW x||, logit lens) leave the logits unchanged
J2  identity patching: a model's own resid / attn / mlp outputs patched back in, and
    cross_route_patch with source = target, leave the logits unchanged; cross_model_patch of the
    last layer's residual output reproduces the source model (theta_0 -> theta_u gives theta_0)
J3  the logit lens of the last layer equals the model's logits
J4  lora_layer_mask: all layers kept = theta_u, none kept = theta_0 (vanilla)
J5  component attributions sum to the total target-logit change theta_u - theta_0
J6  cross_route_patch alignment: patched source and target tokens are identical; the patched
    tokens of the first pair are logged for a manual check
Runs in float32 (mech.dtype) with relative tolerance mech.tol_rel_cuda_fp32 (GPU) / tol_rel_cpu;
mech.tol_rel_half applies when mech.dtype is a half type. Logits are compared over every non-padding position of image+text and text-only MC prompts
(J3, J5: at the target positions). Each check is a function of the shared setup, so
scripts/check_hooks.py and tests/test_milestone_j.py run the same code.
"""
from pathlib import Path

import torch

from ..evaluate import CHECKPOINT_SUITE, load_items
from ..runs import model_dir, processed_dir, results_dir, vanilla_adapter_dir
from ..train import BatchPlan, Ctx, STEP_FNS
from ..models import get_adapter
from ..utils import Logger, atomic_write_json, read_jsonl, seed_all, tracked_run
from . import hooks, tools


def _logits(adapter, model, batch):
    return tools.forward_logits(adapter, model, batch).float()[batch["attention_mask"].bool()]  # [n_valid, vocab]


def _cmp(a, b, tol_rel):
    d = (a.float() - b.float()).abs()
    rel = (d.max() / b.float().abs().max().clamp_min(1e-12)).item()
    return {"max_abs_diff": d.max().item(), "rel_diff": rel, "top1_agreement": (a.argmax(-1) == b.argmax(-1)).float().mean().item(),
            "tol_rel": tol_rel, "passed": rel <= tol_rel}


class J:
    """Shared state: vanilla + a briefly trained fresh adapter, paired img/txt MC batches."""

    def __init__(self, **kw):
        self.__dict__.update(kw)


def setup(cfg, log=None):
    out_dir = model_dir(cfg) / "milestone_j"
    log = log or Logger(out_dir / "log.txt")
    mc = cfg.get("mech", {})
    processed = processed_dir(cfg, log)
    seed_all(0)
    # float32 by default: in bf16, (base + d) - d != base after rounding, and the error compounds over
    # layers (smoke test 2026-10-29: rel. diff up to 6.7e-2 on Qwen2-VL-2B), which tests the arithmetic,
    # not the hooks. LLaVA-7B in float32 needs ~30 GB (A100); on smaller GPUs set mech.dtype=bfloat16.
    adapter = get_adapter({**cfg["model"], "dtype": mc.get("dtype", "float32")})
    if adapter.dtype != torch.float32:
        tol = mc.get("tol_rel_half", 1e-2)
    else:
        tol = mc.get("tol_rel_cuda_fp32", 1e-4) if adapter.device.type == "cuda" else mc.get("tol_rel_cpu", 1e-5)
    n = mc.get("n_items", 4)
    items = load_items(processed, CHECKPOINT_SUITE[:1])
    txt = {r["pair_id"]: r for r in items if r["modality"] == "txt"}
    img = {r["pair_id"]: r for r in items if r["modality"] == "img"}
    pairs = sorted(set(txt) & set(img))[:n]
    rows = {"img": [img[p] for p in pairs], "txt": [txt[p] for p in pairs]}
    batches = {m: adapter.encode_mc(r, processed) for m, r in rows.items()}
    model = adapter.load_model(vanilla_adapter_dir(cfg))
    model.eval()
    van = {m: _logits(adapter, model, b) for m, b in batches.items()}

    # train a fresh adapter for a few steps so B != 0 (a TRAINED adapter)
    model = adapter.add_lora(model, cfg["lora"], seed=0)
    train_rows = read_jsonl(processed / "unlearn_forget.jsonl")[: 4 * mc.get("train_steps", 8)]
    ctx = Ctx(model=model, adapter=adapter, cfg=cfg, plan=BatchPlan(train_rows, 4, 0), retain_plan=None, image_root=processed,
              loss="ga", seed=0, ref_cache=None, run_dir=out_dir)
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=mc.get("lr", 1e-4))
    model.train()
    for step in range(ctx.plan.steps_per_epoch):
        with adapter.autocast():
            STEP_FNS["ga"](ctx, step)
        opt.step()
        opt.zero_grad(set_to_none=True)
    model.eval()
    lora = sorted(hooks.lora_modules(model))
    layers = sorted({hooks.layer_index(x) for x in lora})
    window = [x for x in lora if layers[len(layers) // 3] <= hooks.layer_index(x) < layers[2 * len(layers) // 3] + 1]
    on = {m: _logits(adapter, model, b) for m, b in batches.items()}
    targets = {m: tools.mc_targets(adapter, rows[m], b) for m, b in batches.items()}
    return J(cfg=cfg, log=log, out_dir=out_dir, adapter=adapter, model=model, tol=tol, rows=rows, batches=batches,
             van=van, on=on, targets=targets, lora=lora, window=window, n_dec=len(hooks.decoder_layers(model)))


# ---------------------------------------------------------------------------------------- (i)-(iii)
def check_adapter_changes_logits(j):
    out = {}
    for m in j.batches:
        rel = _cmp(j.on[m], j.van[m], j.tol)["rel_diff"]
        out[f"{m}/adapter_changes_logits"] = {"rel_diff_on_vs_vanilla": rel, "passed": rel > j.tol}
    return out


def check_i_adapter_off(j):
    out = {}
    for m, b in j.batches.items():
        with j.model.disable_adapter():
            out[f"{m}/(i)_adapter_off_equals_vanilla"] = _cmp(_logits(j.adapter, j.model, b), j.van[m], j.tol)
    return out


def check_ii_removal(j):
    out = {}
    for m, b in j.batches.items():
        for label, names in (("window", j.window), ("all", j.lora)):
            with hooks.remove_contributions(j.model, names):
                removed = _logits(j.adapter, j.model, b)
            with hooks.disable_locally(j.model, names):
                disabled = _logits(j.adapter, j.model, b)
            out[f"{m}/(ii)_removal_equals_local_disable[{label}]"] = _cmp(removed, disabled, j.tol)
            if label == "all":
                out[f"{m}/(ii)_removal_all_equals_vanilla"] = _cmp(removed, j.van[m], j.tol)
    return out


def check_iii_self_replacement(j):
    out = {}
    for m, b in j.batches.items():
        cache = {}
        idxs = sorted({0, j.n_dec // 2, j.n_dec - 1})
        with hooks.record_layer_outputs(j.model, idxs, cache):
            _logits(j.adapter, j.model, b)
        for pos in ("all", "last"):
            with hooks.patch_layer_outputs(j.model, cache, positions=pos):
                out[f"{m}/(iii-a)_layer_self_patch[{pos}]"] = _cmp(_logits(j.adapter, j.model, b), j.on[m], j.tol)
        src = {}
        with hooks.record_inputs(j.model, j.lora, src):
            _logits(j.adapter, j.model, b)
        for pos in ("all", "last"):
            with hooks.replace_contributions(j.model, j.lora, src, positions=pos):
                out[f"{m}/(iii-b)_contribution_self_replacement[{pos}]"] = _cmp(_logits(j.adapter, j.model, b), j.on[m], j.tol)
    return out


# ---------------------------------------------------------------------------------------- J1-J6
def check_j1_passive_hooks(j):
    out = {}
    all_layers = range(j.n_dec)
    for m, b in j.batches.items():
        store = {k: {} for k in ("resid", "attn", "mlp")}
        with hooks.record_outputs(j.model, "resid", all_layers, store["resid"]), \
                hooks.record_outputs(j.model, "attn", all_layers, store["attn"]), \
                hooks.record_outputs(j.model, "mlp", all_layers, store["mlp"]):
            out[f"{m}/J1_recording_hooks_passive"] = _cmp(_logits(j.adapter, j.model, b), j.on[m], j.tol)
        groups = tools.token_groups(j.adapter, j.model, j.rows[m], b, tools.mc_texts(j.adapter, j.rows[m], b), j.targets[m])
        tools.delta_wx_norm(j.adapter, j.model, b, j.targets[m], groups)  # hooks removed on exit
        out[f"{m}/J1_after_delta_wx_norm_unchanged"] = _cmp(_logits(j.adapter, j.model, b), j.on[m], j.tol)
        n_name = int(groups["name"].sum())
        expect_name = m == "txt"
        out[f"{m}/J1_token_groups"] = {"n_name": n_name, "n_image": int(groups["image"].sum()),
                                       "n_template": int(groups["template"].sum()), "n_last": int(groups["last"].sum()),
                                       "passed": (n_name > 0) == expect_name and int(groups["last"].sum()) == len(j.rows[m])
                                       and (int(groups["image"].sum()) > 0) == (m == "img")}
    return out


def check_j2_identity_patch(j):
    out = {}
    layers = sorted({0, j.n_dec // 2, j.n_dec - 1})
    last = [j.n_dec - 1]
    for m, b in j.batches.items():
        for kind in ("resid", "attn", "mlp"):  # theta_u's own outputs patched back into theta_u
            store = {}
            with hooks.record_outputs(j.model, kind, layers, store):
                _logits(j.adapter, j.model, b)
            with hooks.patch_outputs(j.model, kind, store):
                out[f"{m}/J2_self_patch[{kind}]"] = _cmp(_logits(j.adapter, j.model, b), j.on[m], j.tol)
        # cross_model_patch at the last layer's residual output reproduces the source model
        lg = tools.cross_model_patch(j.adapter, j.model, b, last, source="vanilla")
        out[f"{m}/J2_cross_model_patch_last_layer[vanilla->unlearned]"] = _cmp(lg.float()[b["attention_mask"].bool()], j.van[m], j.tol)
        lg = tools.cross_model_patch(j.adapter, j.model, b, last, source="unlearned")
        out[f"{m}/J2_cross_model_patch_last_layer[unlearned->vanilla]"] = _cmp(lg.float()[b["attention_mask"].bool()], j.on[m], j.tol)
        lg, _, _ = tools.cross_route_patch(j.adapter, j.model, b, b, layers)
        out[f"{m}/J2_cross_route_patch_source_eq_target"] = _cmp(lg.float()[b["attention_mask"].bool()], j.on[m], j.tol)
    return out


def check_j3_logit_lens_last(j):
    out = {}
    for m, b in j.batches.items():
        t = j.targets[m]
        lens = tools.lens_logits(j.adapter, j.model, b, t, layer=-1)
        real = tools.forward_logits(j.adapter, j.model, b)[t["b"], t["pos"]]
        out[f"{m}/J3_last_layer_lens_equals_logits"] = _cmp(lens, real, j.tol)
    return out


def check_j4_lora_layer_mask(j):
    out = {}
    for m, b in j.batches.items():
        with tools.lora_layer_mask(j.model, range(j.n_dec)):
            out[f"{m}/J4_mask_all_kept_equals_unlearned"] = _cmp(_logits(j.adapter, j.model, b), j.on[m], j.tol)
        with tools.lora_layer_mask(j.model, []):
            out[f"{m}/J4_mask_none_kept_equals_vanilla"] = _cmp(_logits(j.adapter, j.model, b), j.van[m], j.tol)
    return out


def check_j5_attribution_sum(j):
    out = {}
    for m, b in j.batches.items():
        a = tools.component_attribution(j.adapter, j.model, b, j.targets[m])
        err = (a["total_attribution"] - a["delta_logit"]).abs().max().item()
        scale = a["logit_scale"].max().item()
        out[f"{m}/J5_attributions_sum_to_delta_logit"] = {
            "max_abs_err": err, "logit_scale": scale, "rel_err": err / max(scale, 1e-12),
            "delta_logit": [round(x, 4) for x in a["delta_logit"].tolist()],
            "total_attribution": [round(x, 4) for x in a["total_attribution"].tolist()],
            "tol_rel": j.tol, "passed": err / max(scale, 1e-12) <= j.tol}
    return out


def check_j6_route_alignment(j):
    src, tgt = j.batches["txt"], j.batches["img"]
    index, ks = tools.route_alignment(src, tgt)
    b, tp, sp = index
    same = bool((tgt["input_ids"][b, tp] == src["input_ids"][b, sp]).all())
    last_in = all(((b == i) & (tp == tgt["input_ids"].shape[1] - 1)).any().item() for i in range(len(ks)))
    shown = tools.describe_alignment(j.adapter, src, tgt, index, row=0)
    j.log("J6 cross-route alignment, pair", j.rows["txt"][0]["pair_id"], f"({ks[0]} end-aligned positions):")
    j.log("  txt prompt:", repr(tools.mc_texts(j.adapter, j.rows["txt"][:1], src)[0]))
    j.log("  patched tokens (tgt_pos <- src_pos):", " | ".join(f"{x['tgt_pos']}<-{x['src_pos']} {x['tgt_token']!r}" for x in shown))
    return {"txt->img/J6_aligned_tokens_identical": {"aligned_lengths": ks, "last_position_patched": last_in,
                                                     "first_pair_tokens": [x["tgt_token"] for x in shown],
                                                     "passed": same and last_in and min(ks) >= 1}}


CHECKS = {
    "adapter_changes_logits": check_adapter_changes_logits,
    "i_adapter_off": check_i_adapter_off,
    "ii_removal": check_ii_removal,
    "iii_self_replacement": check_iii_self_replacement,
    "j1_passive_hooks": check_j1_passive_hooks,
    "j2_identity_patch": check_j2_identity_patch,
    "j3_logit_lens_last": check_j3_logit_lens_last,
    "j4_lora_layer_mask": check_j4_lora_layer_mask,
    "j5_attribution_sum": check_j5_attribution_sum,
    "j6_route_alignment": check_j6_route_alignment,
}


def run_milestone_j(cfg):
    out_dir = model_dir(cfg) / "milestone_j"
    log = Logger(out_dir / "log.txt")
    with tracked_run(results_dir(cfg), out_dir, f"{cfg['model']['name']}_milestone_j", "milestone_j"):
        j = setup(cfg, log)
        res = {"device": j.adapter.device.type, "tol_rel": j.tol, "n_lora_modules": len(j.lora),
               "n_window_modules": len(j.window), "n_decoder_layers": j.n_dec, "checks": {}}
        C = res["checks"]
        for fn in CHECKS.values():
            C.update(fn(j))
        res["passed"] = all(v["passed"] for v in C.values())
        atomic_write_json(Path(out_dir) / "milestone_j.json", res)
        atomic_write_json(results_dir(cfg) / "milestone_j.json", res)
        for k, v in C.items():
            log(f"{'PASS' if v['passed'] else 'FAIL'} {k}: " + ", ".join(f"{a}={b:.2e}" if isinstance(b, float) else f"{a}={b}"
                                                                         for a, b in v.items() if a != "passed"))
        log("MILESTONE J", "PASSED" if res["passed"] else "FAILED")
        if not res["passed"]:
            raise RuntimeError("milestone J checks failed")
        return res
