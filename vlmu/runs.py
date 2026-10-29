"""Job orchestration: pretrained eval, vanilla fine-tune (+gate), unlearning runs (grid /
rerun), selection at matched in-modality forgetting, and seed reruns."""
import copy
import gc
import glob
from pathlib import Path

import torch
import yaml

from .config import REPO, apply_overrides, deep_merge, load_yaml, localize
from .evaluate import CHECKPOINT_SUITE, FULL_SUITE, load_items, score, summarize
from .metrics import nd
from .models import get_adapter
from .train import BatchPlan, Ctx, clear_resume, save_adapter, train
from .utils import Logger, append_jsonl, atomic_write_json, is_done, mark_done, read_json, read_jsonl, \
    seed_all, tracked_run


# ---------------------------------------------------------------------------------------- paths
def model_dir(cfg):
    return Path(cfg["paths"]["runs"]) / cfg["model"]["name"]


def vanilla_adapter_dir(cfg):
    return Path(cfg["paths"]["models"]) / f"{cfg['model']['name']}_vanilla"


def vanilla_eval_dir(cfg):
    return model_dir(cfg) / "vanilla" / "full_eval"


def results_dir(cfg):
    return Path(cfg["paths"]["results"]) / cfg["model"]["name"]


def processed_dir(cfg, log):
    p = Path(cfg["paths"]["processed"])
    if not is_done(p):
        raise RuntimeError(f"processed data missing at {p}: run scripts/prepare_data.py"
                           + (" --debug" if cfg["debug"] else "") + " first")
    return Path(localize(p, log))


def free():
    """Release GPU memory after the caller has dropped its model reference."""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


# ---------------------------------------------------------------------------------------- eval
def full_eval(cfg, adapter, model, processed, out_dir, log):
    out_dir = Path(out_dir)
    if is_done(out_dir):
        log(f"full eval already done: {out_dir}")
        return read_json(out_dir / "summary.json")
    items = load_items(processed, FULL_SUITE, cfg["eval"].get("max_per_group"))
    res = score(adapter, model, items, processed, cfg["eval"], out_dir / "items.jsonl", log)
    summ = summarize(res)
    atomic_write_json(out_dir / "summary.json", summ)
    mark_done(out_dir)
    for k, v in summ.items():
        log(f"  {k:28s} acc={v['acc']:5.1f}  n={v['n']:4d}  chance={v['chance']:.0f}  "
            f"p_letters={v['p_letters']:.2f}")
    return summ


def run_pretrained_eval(cfg):
    out = model_dir(cfg) / "pretrained"
    log = Logger(out / "log.txt")
    with tracked_run(results_dir(cfg), out, f"{cfg['model']['name']}_pretrained", "eval_pretrained"):
        processed = processed_dir(cfg, log)
        adapter = get_adapter(cfg["model"])
        model = adapter.load_model()
        full_eval(cfg, adapter, model, processed, out / "full_eval", log)
        mark_done(out)


# ---------------------------------------------------------------------------------------- vanilla
def run_vanilla(cfg):
    run_dir = model_dir(cfg) / "vanilla"
    log = Logger(run_dir / "log.txt")
    name = cfg["model"]["name"]
    if is_done(run_dir):
        log(f"vanilla already DONE: {read_json(run_dir / 'gate.json')}")
        return
    with tracked_run(results_dir(cfg), run_dir, f"{name}_vanilla", "vanilla_ft"):
        atomic_write_json(run_dir / "config.json", cfg)
        processed = processed_dir(cfg, log)
        tcfg = cfg["train"]
        seed_all(tcfg["seed"])
        adapter = get_adapter(cfg["model"])
        out_adapter = vanilla_adapter_dir(cfg)
        if not (out_adapter / "adapter_config.json").exists():
            rows = read_jsonl(processed / "train_vanilla.jsonl")
            if cfg["debug"]:
                n = cfg.get("debug_rows", 24)
                rows = [r for r in rows if r["form"] == "img"][:n // 2] + [r for r in rows if r["form"] == "txt"][:n // 2]
            plan = BatchPlan(rows, tcfg["batch_size"], tcfg["seed"])
            total = tcfg.get("max_steps") or plan.steps_per_epoch * tcfg["epochs"]
            save_every = tcfg.get("save_every_steps", 500)
            epoch_steps = {plan.steps_per_epoch * e for e in range(1, tcfg["epochs"] + 1)} if not cfg["debug"] else set()
            ckpts = set(range(save_every, total, save_every)) | epoch_steps | {total}
            log(f"vanilla FT: {len(rows)} rows, {plan.steps_per_epoch} steps/epoch, {total} steps")
            model = adapter.add_lora(adapter.load_model(), cfg["lora"], tcfg["seed"])
            ckpt_suite = load_items(processed, CHECKPOINT_SUITE, cfg["eval"].get("max_per_group"))

            def on_ckpt(step):
                if step in epoch_steps or step == total:
                    out = run_dir / "evals" / f"step{step}.jsonl"
                    summ = summarize(score(adapter, model, ckpt_suite, processed, cfg["eval"], out, log))
                    log(f"step {step} " + " ".join(f"{k}={v['acc']:.1f}" for k, v in summ.items()))
                    append_jsonl(run_dir / "checkpoints.jsonl", [{"step": step, **{k: v["acc"] for k, v in summ.items()}}])
                return {}

            ctx = Ctx(model=model, adapter=adapter, cfg=cfg, run_dir=run_dir, plan=plan, retain_plan=None,
                      image_root=processed, loss="nll", seed=tcfg["seed"], ref_cache=None)
            train(ctx, total, total, ckpts, on_ckpt, log)
            save_adapter(model, out_adapter, dtype=None)
            model = ctx = None  # release before free()
            free()
        # Evaluate the merged model exactly as unlearning runs will load it.
        model = adapter.load_model(out_adapter)
        summ = full_eval(cfg, adapter, model, processed, vanilla_eval_dir(cfg), log)
        g = cfg["gate"]["min_forget_acc"]
        accs = {m: summ[f"forget/paired/{m}"]["acc"] for m in ("img", "txt")}
        gate = {"passed": all(a >= g for a in accs.values()), "threshold": g, "forget_paired_acc": accs,
                "retain_paired_acc": {m: summ[f"retain/paired/{m}"]["acc"] for m in ("img", "txt")}}
        atomic_write_json(run_dir / "gate.json", gate)
        log(f"GATE {'PASSED' if gate['passed'] else 'FAILED'}: {gate}")
        clear_resume(run_dir)
        mark_done(run_dir, {"gate_passed": gate["passed"]})


def require_gate(cfg, log):
    gate_file = model_dir(cfg) / "vanilla" / "gate.json"
    if not gate_file.exists():
        raise RuntimeError(f"no vanilla model yet ({gate_file}); run scripts/finetune_vanilla.py first")
    gate = read_json(gate_file)
    if not gate["passed"]:
        if cfg["debug"] or cfg.get("allow_failed_gate"):
            log(f"WARNING: vanilla gate failed {gate} — continuing (debug/allow_failed_gate)")
        else:
            raise RuntimeError(f"vanilla gate failed: {gate}. Unlearning is blocked (D2).")
    return read_json(vanilla_eval_dir(cfg) / "summary.json")


# ---------------------------------------------------------------------------------------- unlearning
def checkpoint_record(cfg, summ, vanilla, step, spe):
    u, s = cfg["unlearn"], cfg["selection"]
    rec = {"run_id": cfg["run_id"], "method": u["method"], "modality": u["modality"], "lr": cfg["train"]["lr"],
           "seed": u["seed"], "step": step, "epoch": round(step / spe, 3)}
    for set_name in ("forget", "retain"):
        for m in ("img", "txt"):
            k = f"{set_name}/paired/{m}"
            rec[f"{set_name}_{m}_acc"] = summ[k]["acc"]
            rec[f"{set_name}_{m}_vanilla"] = vanilla[k]["acc"]
            rec[f"{set_name}_{m}_p_letters"] = summ[k]["p_letters"]
    for m in ("img", "txt"):
        rec[f"nd_{m}"] = nd(vanilla[f"forget/paired/{m}"]["acc"], summ[f"forget/paired/{m}"]["acc"])
        rec[f"retain_drop_{m}"] = vanilla[f"retain/paired/{m}"]["acc"] - summ[f"retain/paired/{m}"]["acc"]
    rec["nd_in"] = rec[f"nd_{u['modality']}"]
    rec["feasible"] = max(rec["retain_drop_img"], rec["retain_drop_txt"]) <= s["retain_margin"]
    rec["distance"] = abs(rec["nd_in"] - s["target_nd"])
    rec["matched"] = rec["feasible"] and rec["distance"] <= s["tol"]
    rec["stop"] = rec["nd_in"] >= s["stop_nd"] or max(rec["retain_drop_img"], rec["retain_drop_txt"]) > s["stop_retain_drop"]
    return rec


def run_unlearn(cfg):
    """mode 'grid': checkpoint evals, keep the closest feasible adapter, early stop.
    mode 'rerun': train to unlearn.stop_at_step, save adapter, full eval."""
    u = cfg["unlearn"]
    run_dir = model_dir(cfg) / "unlearn" / cfg["run_id"]
    log = Logger(run_dir / "log.txt")
    if is_done(run_dir):
        log(f"{cfg['run_id']} already DONE")
        return
    with tracked_run(results_dir(cfg), run_dir, cfg["run_id"], f"unlearn_{u.get('mode', 'grid')}"):
        atomic_write_json(run_dir / "config.json", cfg)
        processed = processed_dir(cfg, log)
        vanilla = require_gate(cfg, log)
        tcfg, mode = cfg["train"], u.get("mode", "grid")
        seed_all(u["seed"])
        adapter = get_adapter(cfg["model"])
        model = adapter.add_lora(adapter.load_model(vanilla_adapter_dir(cfg)), cfg["lora"], u["seed"])
        forget = [r for r in read_jsonl(processed / "unlearn_forget.jsonl") if r["form"] == u["modality"]]
        retain = [r for r in read_jsonl(processed / "unlearn_retain.jsonl") if r["form"] == u["modality"]]
        if cfg["debug"]:
            forget, retain = forget[:cfg.get("debug_rows", 16)], retain[:cfg.get("debug_rows", 16)]
        plan = BatchPlan(forget, tcfg["batch_size"], u["seed"])
        rplan = BatchPlan(retain, tcfg["batch_size"], u["seed"] + 7919) if u["method"] == "ga_diff" else None
        spe = plan.steps_per_epoch
        total = tcfg.get("max_steps") or spe * tcfg["max_epochs"]
        every = tcfg.get("eval_every_steps") or max(1, round(spe * tcfg["eval_every_epochs"]))
        ckpts = set(range(every, total + 1, every)) | {total}
        stop_at = u.get("stop_at_step") or total
        log(f"{cfg['run_id']}: {len(forget)} forget rows, {spe} steps/epoch, schedule {total} steps, "
            f"stop_at {stop_at}, mode {mode}")
        ckpt_items = load_items(processed, CHECKPOINT_SUITE, cfg["eval"].get("max_per_group"))
        records = {r["step"]: r for r in read_jsonl(run_dir / "checkpoints.jsonl")}

        def on_ckpt(step):
            if mode != "grid":
                return {}
            if step in records:
                return {"stop": records[step]["stop"]}
            res = score(adapter, model, ckpt_items, processed, cfg["eval"], run_dir / "evals" / f"step{step}.jsonl", log)
            rec = checkpoint_record(cfg, summarize(res), vanilla, step, spe)
            best_f = run_dir / "best.json"
            best = read_json(best_f) if best_f.exists() else None
            if rec["feasible"] and (best is None or rec["distance"] < best["distance"]):
                save_adapter(model, run_dir / "best_adapter")
                atomic_write_json(best_f, rec)
                rec["saved_adapter"] = True
            append_jsonl(run_dir / "checkpoints.jsonl", [rec])
            records[step] = rec
            log(f"ckpt step {step}: ND img={rec['nd_img']:.2f} txt={rec['nd_txt']:.2f} | retain drop "
                f"img={rec['retain_drop_img']:.1f} txt={rec['retain_drop_txt']:.1f} | feasible={rec['feasible']} "
                f"stop={rec['stop']}")
            return {"stop": rec["stop"]}

        ctx = Ctx(model=model, adapter=adapter, cfg=cfg, run_dir=run_dir, plan=plan, retain_plan=rplan,
                  image_root=processed, loss=u["method"], seed=u["seed"], ref_cache={} if u["method"] == "npo" else None)
        final = train(ctx, total, stop_at, ckpts, on_ckpt, log)
        if mode == "rerun":
            if final != stop_at:
                raise RuntimeError(f"rerun stopped at {final}, expected {stop_at}")
            save_adapter(model, run_dir / "final_adapter")
            full_eval(cfg, adapter, model, processed, run_dir / "full_eval", log)
        clear_resume(run_dir)
        mark_done(run_dir, {"final_step": final})
        model = ctx = None  # release before free()
        free()


def eval_adapter(cfg, adapter_dir, out_dir, log):
    """Full eval of vanilla + an unlearning LoRA adapter."""
    from peft import PeftModel
    if is_done(out_dir):
        return read_json(Path(out_dir) / "summary.json")
    processed = processed_dir(cfg, log)
    adapter = get_adapter(cfg["model"])
    model = PeftModel.from_pretrained(adapter.load_model(vanilla_adapter_dir(cfg)), str(adapter_dir))
    summ = full_eval(cfg, adapter, model, processed, out_dir, log)
    del model
    free()
    return summ


# ---------------------------------------------------------------------------------------- grid / selection
def run_configs(model_name):
    return sorted(glob.glob(str(REPO / "configs" / model_name / "runs" / "*.yaml")))


def load_run_cfg(path, base_cfg):
    cfg = load_yaml(path)
    if base_cfg["debug"]:
        cfg = deep_merge(cfg, cfg.get("debug_overrides", {}))
    cfg = apply_overrides(cfg, base_cfg.get("cli_overrides", []))  # CLI wins
    cfg["cli_overrides"] = base_cfg.get("cli_overrides", [])
    cfg["debug"] = base_cfg["debug"]
    cfg["paths"] = base_cfg["paths"]
    cfg["config_file"] = str(path)
    return cfg


def select(cfg, log):
    """Matched-forgetting selection over all seed-0 grid checkpoints (same rule everywhere)."""
    s = cfg["selection"]
    recs, incomplete = [], []
    for path in run_configs(cfg["model"]["name"]):
        rc = load_run_cfg(path, cfg)
        rd = model_dir(rc) / "unlearn" / rc["run_id"]
        if not is_done(rd):
            incomplete.append(rc["run_id"])
        recs += read_jsonl(rd / "checkpoints.jsonl")
    out = {"rule": s, "incomplete_runs": incomplete, "cells": {}}
    for method in cfg["grid"]["methods"]:
        for modality in cfg["grid"]["modalities"]:
            cand = [r for r in recs if r["method"] == method and r["modality"] == modality and r["seed"] == cfg["grid"]["seed"]]
            feas = [r for r in cand if r["feasible"]]
            key = f"{method}/{modality}"
            if not feas:
                out["cells"][key] = {"status": "no_feasible_checkpoint", "n_checkpoints": len(cand)}
                continue
            best = min(feas, key=lambda r: (r["distance"], r["step"], r["lr"]))
            out["cells"][key] = {"status": "matched" if best["matched"] else "unmatched", "n_checkpoints": len(cand),
                                 "n_feasible": len(feas), **{k: best[k] for k in ("run_id", "lr", "step", "epoch",
                                                                                   "nd_in", "nd_img", "nd_txt", "distance",
                                                                                   "retain_drop_img", "retain_drop_txt")}}
    return out, recs


def rerun_cfg(cfg, cell, method, modality, seed):
    rc = load_run_cfg(next(p for p in run_configs(cfg["model"]["name"]) if Path(p).stem == cell["run_id"]), cfg)
    rc = copy.deepcopy(rc)
    rc["run_id"] = f"{cell['run_id'].rsplit('_s', 1)[0]}_s{seed}_step{cell['step']}"
    rc["unlearn"].update({"seed": seed, "stop_at_step": cell["step"], "mode": "rerun", "method": method,
                          "modality": modality})
    return rc


def save_generated_cfg(cfg, rc):
    """Generated (seed-rerun) configs are saved next to the results so every run stays fully defined."""
    out = results_dir(cfg) / "generated_configs" / f"{rc['run_id']}.yaml"
    out.parent.mkdir(parents=True, exist_ok=True)
    keep = {k: v for k, v in rc.items() if k not in ("paths",)}
    out.write_text(yaml.safe_dump(keep, sort_keys=False))
    return out

