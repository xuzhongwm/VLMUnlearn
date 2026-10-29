"""Milestone J: correctness checks for the hook machinery, on a TRAINED adapter.

(i)   adapter-off reproduces the vanilla logits
(ii)  hook-based contribution removal (subtracting d(x) = scaling*B(A(x))) matches PEFT-native
      local adapter disabling, for a layer window and for all modules (the latter also = vanilla)
(iii) self-replacement reproduces the native logits:
      (a) decoder-layer outputs cached from a pass and patched back in (all / last position)
      (b) contribution replacement y = F_U(H) - d(H) + d(H_src) with H_src = H (all / last)
Logits are compared over every non-padding position of image+text and text-only MC prompts.
"""
from pathlib import Path

import torch

from ..evaluate import CHECKPOINT_SUITE, load_items
from ..runs import model_dir, processed_dir, results_dir, vanilla_adapter_dir
from ..train import BatchPlan, Ctx, STEP_FNS
from ..models import get_adapter
from ..utils import Logger, atomic_write_json, read_jsonl, seed_all, tracked_run
from . import hooks


def _logits(adapter, model, batch):
    with torch.no_grad(), torch.autocast(device_type=adapter.device.type, dtype=adapter.dtype,
                                         enabled=adapter.device.type == "cuda"):
        lg = adapter.forward(model, batch).float()
    return lg[batch["attention_mask"].bool()]  # [n_valid_positions, vocab]


def _cmp(a, b, tol_rel, letter_ids=None):
    d = (a - b).abs()
    rel = (d.max() / b.abs().max().clamp_min(1e-12)).item()
    out = {"max_abs_diff": d.max().item(), "rel_diff": rel, "top1_agreement": (a.argmax(-1) == b.argmax(-1)).float().mean().item(),
           "tol_rel": tol_rel, "passed": rel <= tol_rel}
    return out


def run_milestone_j(cfg):
    out_dir = model_dir(cfg) / "milestone_j"
    log = Logger(out_dir / "log.txt")
    mc = cfg.get("mech", {})
    with tracked_run(results_dir(cfg), out_dir, f"{cfg['model']['name']}_milestone_j", "milestone_j"):
        processed = processed_dir(cfg, log)
        seed_all(0)
        adapter = get_adapter(cfg["model"])
        tol = mc.get("tol_rel_cuda", 1e-2) if adapter.device.type == "cuda" else mc.get("tol_rel_cpu", 1e-5)
        n = mc.get("n_items", 4)
        items = load_items(processed, CHECKPOINT_SUITE[:1])
        batches = {m: adapter.encode_mc([r for r in items if r["modality"] == m][:n], processed) for m in ("img", "txt")}
        model = adapter.load_model(vanilla_adapter_dir(cfg))
        model.eval()
        van = {m: _logits(adapter, model, b) for m, b in batches.items()}

        # train a fresh adapter for a few steps so B != 0 (a TRAINED adapter)
        model = adapter.add_lora(model, cfg["lora"], seed=0)
        rows = read_jsonl(processed / "unlearn_forget.jsonl")[: 4 * mc.get("train_steps", 8)]
        ctx = Ctx(model=model, adapter=adapter, cfg=cfg, plan=BatchPlan(rows, 4, 0), retain_plan=None, image_root=processed,
                  loss="ga", seed=0, ref_cache=None, run_dir=out_dir)
        opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=mc.get("lr", 1e-4))
        model.train()
        for step in range(ctx.plan.steps_per_epoch):
            with torch.autocast(device_type=adapter.device.type, dtype=adapter.dtype, enabled=adapter.device.type == "cuda"):
                STEP_FNS["ga"](ctx, step)
            opt.step()
            opt.zero_grad(set_to_none=True)
        model.eval()
        lora = sorted(hooks.lora_modules(model))
        layers = sorted({hooks.layer_index(x) for x in lora})
        window = [x for x in lora if layers[len(layers) // 3] <= hooks.layer_index(x) < layers[2 * len(layers) // 3] + 1]
        n_dec = len(hooks.decoder_layers(model))
        res = {"device": adapter.device.type, "tol_rel": tol, "n_lora_modules": len(lora), "n_window_modules": len(window),
               "n_decoder_layers": n_dec, "checks": {}}
        C = res["checks"]
        for m, b in batches.items():
            on = _logits(adapter, model, b)
            C[f"{m}/adapter_changes_logits"] = {"rel_diff_on_vs_vanilla": _cmp(on, van[m], tol)["rel_diff"],
                                                "passed": _cmp(on, van[m], tol)["rel_diff"] > tol}
            with model.disable_adapter():
                C[f"{m}/(i)_adapter_off_equals_vanilla"] = _cmp(_logits(adapter, model, b), van[m], tol)
            for label, names in (("window", window), ("all", lora)):
                with hooks.remove_contributions(model, names):
                    removed = _logits(adapter, model, b)
                with hooks.disable_locally(model, names):
                    disabled = _logits(adapter, model, b)
                C[f"{m}/(ii)_removal_equals_local_disable[{label}]"] = _cmp(removed, disabled, tol)
                if label == "all":
                    C[f"{m}/(ii)_removal_all_equals_vanilla"] = _cmp(removed, van[m], tol)
            cache = {}
            idxs = sorted({0, n_dec // 2, n_dec - 1})
            with hooks.record_layer_outputs(model, idxs, cache):
                _logits(adapter, model, b)
            for pos in ("all", "last"):
                with hooks.patch_layer_outputs(model, cache, positions=pos):
                    C[f"{m}/(iii-a)_layer_self_patch[{pos}]"] = _cmp(_logits(adapter, model, b), on, tol)
            src = {}
            with hooks.record_inputs(model, lora, src):
                _logits(adapter, model, b)
            for pos in ("all", "last"):
                with hooks.replace_contributions(model, lora, src, positions=pos):
                    C[f"{m}/(iii-b)_contribution_self_replacement[{pos}]"] = _cmp(_logits(adapter, model, b), on, tol)
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
