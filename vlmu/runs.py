"""Job orchestration: pretrained eval, vanilla fine-tune (+gate, frozen doubly-correct set),
unlearning runs with calibration-only checkpoint evals, checkpoint/lr selection, full evals.

Selection (PROTOCOL §4.3 as specified in the 2026-10-29 instructions; see DEVIATIONS.md):
  eligible checkpoint = in-modality accuracy on the calibration doubly-correct forget facts in
  [window_lo, window_hi] AND retain drop <= retain_margin (calibration retain, doubly-correct);
  per run, the EARLIEST eligible checkpoint is selected (training stops there).
"""
import collections
import gc
import glob
from pathlib import Path

import torch

from .config import REPO, apply_overrides, deep_merge, load_yaml, localize
from .evaluate import CHECKPOINT_SUITE, FULL_SUITE, load_items, permutation_check, score, summarize
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


def dc_file(cfg):
    return model_dir(cfg) / "vanilla" / "doubly_correct.json"


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
    """All sets, both splits, plus the 3 extra cyclic shifts of every paired primary item."""
    out_dir = Path(out_dir)
    if is_done(out_dir):
        log(f"full eval already done: {out_dir}")
        return read_json(out_dir / "summary.json")
    items = load_items(processed, FULL_SUITE, cfg["eval"].get("max_per_group"), shifts=True)
    res = score(adapter, model, items, processed, cfg["eval"], out_dir / "items.jsonl", log)
    summ = {"all_items": summarize(res), "permutation_all_items": permutation_check(res)}
    if dc_file(cfg).exists():
        dc = set(read_json(dc_file(cfg))["pair_ids"])
        summ["doubly_correct"] = summarize(res, dc)
        summ["permutation_doubly_correct"] = permutation_check(res, dc)
    atomic_write_json(out_dir / "summary.json", summ)
    mark_done(out_dir)
    for k, v in summ["all_items"].items():
        if "/shift" not in k:
            log(f"  {k:40s} acc={v['acc']:.3f} n={v['n']:4d} margin={v['margin']:+.2f} p_letters={v['p_letters']:.2f}")
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
def freeze_doubly_correct(cfg, results, log):
    """Facts the vanilla answers correctly in BOTH primary forms (training portrait + text, original
    option order). Written once per model and never overwritten (frozen before any unlearning)."""
    f = dc_file(cfg)
    by_pair = collections.defaultdict(dict)
    for r in results:
        if r["source"] == "paired" and r["shift"] == 0 and r["modality"] in ("img", "txt"):
            by_pair[r["pair_id"]][r["modality"]] = r
    dc = sorted(p for p, m in by_pair.items() if len(m) == 2 and m["img"]["correct"] and m["txt"]["correct"])
    cov = {"by_split": collections.defaultdict(lambda: [0, 0]), "by_entity": collections.defaultdict(lambda: [0, 0]),
           "by_attribute": collections.defaultdict(lambda: [0, 0])}
    dcs = set(dc)
    for p, m in by_pair.items():
        r = m.get("txt") or m.get("img")
        for key, k in (("by_split", f"{r['set']}/{r['split']}"), ("by_entity", f"{r['set']}/{r['split']}/{r['profile']}"),
                       ("by_attribute", f"{r['set']}/{r['split']}/{r['field']}")):
            cov[key][k][1] += 1
            cov[key][k][0] += int(p in dcs)
    out = {"pair_ids": dc, "definition": "vanilla correct on img (training portrait) AND txt, original option order",
           "coverage": {k: {kk: {"n_doubly_correct": a, "n_pairs": b, "coverage": round(a / b, 4)}
                            for kk, (a, b) in sorted(v.items())} for k, v in cov.items()}}
    if f.exists():
        old = read_json(f)
        if old["pair_ids"] != dc:
            raise RuntimeError(f"doubly-correct set already frozen at {f} and differs from this vanilla eval")
        return old
    atomic_write_json(f, out)
    log("doubly-correct coverage:", {k: v["coverage"] for k, v in out["coverage"]["by_split"].items()})
    return out


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
            ckpt_items = load_items(processed, CHECKPOINT_SUITE, cfg["eval"].get("max_per_group"))

            def on_ckpt(step):
                if step in epoch_steps or step == total:
                    out = run_dir / "evals" / f"step{step}.jsonl"
                    summ = summarize(score(adapter, model, ckpt_items, processed, cfg["eval"], out, log))
                    log(f"step {step} " + " ".join(f"{k}={v['acc']:.3f}" for k, v in summ.items()))
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
        full_eval(cfg, adapter, model, processed, vanilla_eval_dir(cfg), log)
        res = read_jsonl(vanilla_eval_dir(cfg) / "items.jsonl")
        prim = [r for r in res if r["set"] == "forget" and r["source"] == "paired" and r["shift"] == 0]
        accs = {m: float(sum(r["correct"] for r in prim if r["modality"] == m) /
                         max(1, sum(r["modality"] == m for r in prim))) for m in ("img", "txt")}
        g = cfg["gate"]["min_forget_acc"]
        gate = {"passed": all(a >= g for a in accs.values()), "threshold": g,
                "forget_paired_acc_primary_forms": accs, "n_entities": len({r["profile"] for r in prim})}
        dc = freeze_doubly_correct(cfg, res, log)
        gate["doubly_correct_coverage"] = {k: v["coverage"] for k, v in dc["coverage"]["by_split"].items()}
        atomic_write_json(run_dir / "gate.json", gate)
        log(f"GATE {'PASSED' if gate['passed'] else 'FAILED'}: {gate}")
        clear_resume(run_dir)
        mark_done(run_dir, {"gate_passed": gate["passed"]})


def require_gate(cfg, log):
    gate_file = model_dir(cfg) / "vanilla" / "gate.json"
    if not gate_file.exists() or not dc_file(cfg).exists():
        raise RuntimeError(f"no vanilla model / doubly-correct set yet ({gate_file}); run scripts/finetune_vanilla.py first")
    gate = read_json(gate_file)
    if not gate["passed"]:
        if cfg["debug"] or cfg.get("allow_failed_gate"):
            log(f"WARNING: vanilla gate failed {gate} — continuing (debug/allow_failed_gate)")
        else:
            raise RuntimeError(f"vanilla gate failed: {gate}. Unlearning is blocked.")
    return set(read_json(dc_file(cfg))["pair_ids"])


# ---------------------------------------------------------------------------------------- unlearning
def checkpoint_record(cfg, summ, step, spe):
    """Calibration metrics on the doubly-correct set (vanilla accuracy = 1 there by construction)."""
    u, s = cfg["unlearn"], cfg["selection"]
    rec = {"run_id": cfg["run_id"], "method": u["method"], "modality": u["modality"], "lr": cfg["train"]["lr"],
           "seed": u["seed"], "step": step, "epoch": round(step / spe, 3)}
    for set_name in ("forget", "retain"):
        for m in ("img", "txt"):
            g = summ.get(f"{set_name}/calibration/paired/{m}")
            rec[f"{set_name}_cal_{m}_acc"] = g["acc"] if g else None
            rec[f"{set_name}_cal_{m}_n"] = g["n"] if g else 0
            rec[f"{set_name}_cal_{m}_margin"] = g["margin"] if g else None
    missing = [k for k in rec if k.endswith("_acc") and rec[k] is None]
    if missing:
        raise RuntimeError(f"no doubly-correct calibration items for {missing}")
    for m in ("img", "txt"):
        rec[f"retain_drop_{m}"] = 1.0 - rec[f"retain_cal_{m}_acc"]
    rec["acc_in"] = rec[f"forget_cal_{u['modality']}_acc"]
    rec["retain_drop_max"] = max(rec["retain_drop_img"], rec["retain_drop_txt"])
    rec["in_window"] = s["window_lo"] <= rec["acc_in"] <= s["window_hi"]
    rec["eligible"] = rec["in_window"] and rec["retain_drop_max"] <= s["retain_margin"]
    if rec["eligible"]:
        rec["stop"], rec["stop_reason"] = True, "earliest_eligible"
    elif rec["acc_in"] < s["window_lo"]:
        rec["stop"], rec["stop_reason"] = True, "passed_window_without_eligible_checkpoint"
    elif rec["retain_drop_max"] > s["stop_retain_drop"]:
        rec["stop"], rec["stop_reason"] = True, "retain_drop_exceeded"
    else:
        rec["stop"], rec["stop_reason"] = False, None
    return rec


def run_unlearn(cfg):
    """Fresh LoRA on the frozen vanilla; checkpoint evals on calibration entities only; stops at
    the earliest eligible checkpoint and saves its adapter (selected_adapter/)."""
    u = cfg["unlearn"]
    run_dir = model_dir(cfg) / "unlearn" / cfg["run_id"]
    log = Logger(run_dir / "log.txt")
    if is_done(run_dir):
        log(f"{cfg['run_id']} already DONE")
        return
    with tracked_run(results_dir(cfg), run_dir, cfg["run_id"], "unlearn"):
        atomic_write_json(run_dir / "config.json", cfg)
        processed = processed_dir(cfg, log)
        dc = require_gate(cfg, log)
        tcfg = cfg["train"]
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
        log(f"{cfg['run_id']}: {len(forget)} forget rows, {spe} steps/epoch, schedule {total} steps, eval every {every}")
        ckpt_items = load_items(processed, CHECKPOINT_SUITE, cfg["eval"].get("max_per_group"), pair_ids=dc)
        if not ckpt_items:
            raise RuntimeError("no doubly-correct calibration items to evaluate")
        records = {r["step"]: r for r in read_jsonl(run_dir / "checkpoints.jsonl")}

        def on_ckpt(step):
            if step in records:
                return {"stop": records[step]["stop"]}
            res = score(adapter, model, ckpt_items, processed, cfg["eval"], run_dir / "evals" / f"step{step}.jsonl", log)
            rec = checkpoint_record(cfg, summarize(res), step, spe)
            if rec["eligible"]:
                save_adapter(model, run_dir / "selected_adapter")
                atomic_write_json(run_dir / "selected.json", rec)
            append_jsonl(run_dir / "checkpoints.jsonl", [rec])
            records[step] = rec
            log(f"ckpt step {step}: forget-cal acc img={rec['forget_cal_img_acc']:.3f} txt={rec['forget_cal_txt_acc']:.3f}"
                f" | retain drop img={rec['retain_drop_img']:.3f} txt={rec['retain_drop_txt']:.3f}"
                f" | eligible={rec['eligible']} stop={rec['stop_reason']}")
            return {"stop": rec["stop"]}

        ctx = Ctx(model=model, adapter=adapter, cfg=cfg, run_dir=run_dir, plan=plan, retain_plan=rplan,
                  image_root=processed, loss=u["method"], seed=u["seed"], ref_cache={} if u["method"] == "npo" else None)
        final = train(ctx, total, total, ckpts, on_ckpt, log)
        last = records[max(records)] if records else None
        status = "eligible" if (run_dir / "selected.json").exists() else \
            (last["stop_reason"] if last and last["stop_reason"] else "window_not_reached")
        clear_resume(run_dir)
        mark_done(run_dir, {"final_step": final, "status": status})
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
    model = None
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


def run_status(rd):
    if not is_done(rd):
        return {"status": "incomplete"}
    done = read_json(rd / "DONE")
    out = {"status": done.get("status"), "final_step": done.get("final_step")}
    if (rd / "selected.json").exists():
        sel = read_json(rd / "selected.json")
        out.update({k: sel[k] for k in ("step", "epoch", "acc_in", "forget_cal_img_acc", "forget_cal_txt_acc",
                                        "retain_drop_img", "retain_drop_txt")})
    return out


def select(cfg, log):
    """Per (method, unlearning modality): the lowest lr whose runs have an eligible checkpoint in
    EVERY seed (calibration only); for each seed, that run's earliest eligible checkpoint.
    All runs and their statuses are reported (ineligible runs are not dropped)."""
    seeds = cfg["grid"]["seeds"]
    runs, recs = {}, []
    for path in run_configs(cfg["model"]["name"]):
        rc = load_run_cfg(path, cfg)
        rd = model_dir(rc) / "unlearn" / rc["run_id"]
        u = rc["unlearn"]
        runs[(u["method"], u["modality"], rc["train"]["lr"], u["seed"])] = {"run_id": rc["run_id"], "dir": str(rd),
                                                                            **run_status(rd)}
        recs += read_jsonl(rd / "checkpoints.jsonl")
    out = {"rule": cfg["selection"], "seeds": seeds, "runs": {v["run_id"]: v for v in runs.values()}, "cells": {}}
    for method in cfg["grid"]["methods"]:
        for modality in cfg["grid"]["modalities"]:
            key = f"{method}/{modality}"
            lrs = sorted({k[2] for k in runs if k[0] == method and k[1] == modality})
            chosen = next((lr for lr in lrs if all(runs.get((method, modality, lr, s), {}).get("status") == "eligible"
                                                   for s in seeds)), None)
            if chosen is None:
                out["cells"][key] = {"status": "no_lr_eligible_in_all_seeds"}
                continue
            out["cells"][key] = {"status": "selected", "lr": chosen,
                                 "seeds": {str(s): runs[(method, modality, chosen, s)] for s in seeds}}
    return out, recs
