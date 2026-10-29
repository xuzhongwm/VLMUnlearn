"""Resumable LoRA training loop shared by vanilla fine-tuning and unlearning.

Batches are a deterministic function of (seed, epoch), homogeneous in modality, so a run
resumes exactly from its saved step (adapter + optimizer + scheduler state).
"""
import math
import os
import shutil
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from .utils import atomic_write_json, read_json


class BatchPlan:
    def __init__(self, rows, batch_size, seed):
        self.rows, self.bs, self.seed = rows, batch_size, seed
        self.groups = {}
        for i, r in enumerate(rows):
            self.groups.setdefault(r["form"], []).append(i)
        self._cache = {}
        self.steps_per_epoch = sum(math.ceil(len(v) / batch_size) for v in self.groups.values())

    def epoch(self, e):
        if e not in self._cache:
            rng = np.random.default_rng(self.seed * 1000 + e)
            batches = []
            for form in sorted(self.groups):
                idx = rng.permutation(self.groups[form])
                batches += [idx[i:i + self.bs].tolist() for i in range(0, len(idx), self.bs)]
            order = rng.permutation(len(batches))
            self._cache = {e: [batches[o] for o in order]}
        return self._cache[e]

    def batch(self, step):
        e, b = divmod(step, self.steps_per_epoch)
        return [self.rows[i] for i in self.epoch(e)[b]]


# ---------------------------------------------------------------------------------------
# Losses. Each step function runs forward+backward itself (GA_Diff backprops the two terms
# separately to halve activation memory) and returns scalars for logging.

def step_nll(ctx, step):
    batch = ctx.adapter.encode_train(ctx.plan.batch(step), ctx.image_root)
    tok_lp, _, _ = ctx.adapter.answer_logprobs(ctx.model, batch)
    loss = -tok_lp.mean()
    loss.backward()
    return {"loss": loss.item()}


def step_ga(ctx, step):
    batch = ctx.adapter.encode_train(ctx.plan.batch(step), ctx.image_root)
    tok_lp, _, _ = ctx.adapter.answer_logprobs(ctx.model, batch)
    nll = -tok_lp.mean()
    (-nll).backward()
    return {"forget_nll": nll.item()}


def step_ga_diff(ctx, step):
    out = step_ga(ctx, step)
    rb = ctx.adapter.encode_train(ctx.retain_plan.batch(step), ctx.image_root)
    tok_lp, _, _ = ctx.adapter.answer_logprobs(ctx.model, rb)
    rnll = -tok_lp.mean()
    (ctx.cfg["unlearn"].get("retain_weight", 1.0) * rnll).backward()
    return {**out, "retain_nll": rnll.item()}


def step_npo(ctx, step):
    """NPO (Zhang et al. 2024): L = -(2/beta) E[log sigmoid(-beta * (log pi_theta(y|x) - log pi_ref(y|x)))],
    log pi = sum of answer-token log-probs; reference = vanilla (adapter disabled), cached per sample."""
    rows = ctx.plan.batch(step)
    batch = ctx.adapter.encode_train(rows, ctx.image_root)
    missing = [i for i, r in enumerate(rows) if r["uid"] not in ctx.ref_cache]
    if missing:
        with torch.no_grad(), ctx.model.disable_adapter():
            _, ref, _ = ctx.adapter.answer_logprobs(ctx.model, batch)
        for i in missing:
            ctx.ref_cache[rows[i]["uid"]] = ref[i].item()
    ref = torch.tensor([ctx.ref_cache[r["uid"]] for r in rows], device=ctx.adapter.device)
    _, lp, _ = ctx.adapter.answer_logprobs(ctx.model, batch)
    beta = ctx.cfg["unlearn"]["npo_beta"]
    loss = -(2.0 / beta) * F.logsigmoid(-beta * (lp - ref)).mean()
    loss.backward()
    return {"npo_loss": loss.item(), "log_ratio": (lp - ref).mean().item()}


STEP_FNS = {"nll": step_nll, "ga": step_ga, "ga_diff": step_ga_diff, "npo": step_npo}


class Ctx:
    def __init__(self, **kw):
        self.__dict__.update(kw)


def lora_state(model):
    from peft import get_peft_model_state_dict
    return {k: v.detach().cpu().clone() for k, v in get_peft_model_state_dict(model).items()}


def save_adapter(model, out_dir, dtype=torch.bfloat16):
    """Save the LoRA adapter (bf16 by default to halve storage)."""
    out_dir = Path(out_dir)
    tmp = out_dir.with_name(out_dir.name + ".tmp")
    if tmp.exists():
        shutil.rmtree(tmp)
    model.save_pretrained(str(tmp))
    if dtype is not None:
        from safetensors.torch import load_file, save_file
        f = tmp / "adapter_model.safetensors"
        save_file({k: v.to(dtype) for k, v in load_file(str(f)).items()}, str(f))
    if out_dir.exists():
        shutil.rmtree(out_dir)
    os.replace(tmp, out_dir)


def train(ctx, total_steps, stop_at, checkpoint_steps, on_checkpoint, log):
    """Run steps [resume_step, stop_at). Calls on_checkpoint(step) at checkpoint_steps
    (idempotent; may return {'stop': True}). Returns the final step."""
    from peft import set_peft_model_state_dict

    model, cfg, run_dir = ctx.model, ctx.cfg, Path(ctx.run_dir)
    tcfg = cfg["train"]
    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=tcfg["lr"], weight_decay=tcfg.get("weight_decay", 0.0))
    warm = int(tcfg.get("warmup_ratio", 0.0) * total_steps)
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: (s + 1) / max(1, warm) if s < warm else max(0.0, (total_steps - s) / max(1, total_steps - warm)))
    resume = run_dir / "resume" / "state.pt"
    step = 0
    if resume.exists():
        st = torch.load(resume, map_location="cpu", weights_only=False)
        set_peft_model_state_dict(model, st["lora"])
        opt.load_state_dict(st["optimizer"])
        sched.load_state_dict(st["scheduler"])
        step = st["step"]
        if ctx.ref_cache is not None and (run_dir / "npo_ref.json").exists():
            ctx.ref_cache.update(read_json(run_dir / "npo_ref.json"))
        log(f"resumed at step {step}")
        if step in checkpoint_steps and on_checkpoint(step).get("stop"):
            return step
    step_fn = STEP_FNS[ctx.loss]
    model.train()
    t0, logs = time.time(), []
    while step < stop_at:
        torch.manual_seed(ctx.seed * 100003 + step)  # LoRA dropout reproducible across resumes
        with torch.autocast(device_type=ctx.adapter.device.type, dtype=ctx.adapter.dtype,
                            enabled=ctx.adapter.device.type == "cuda"):
            logs.append(step_fn(ctx, step))
        torch.nn.utils.clip_grad_norm_(params, tcfg.get("max_grad_norm", 1.0))
        opt.step()
        sched.step()
        opt.zero_grad(set_to_none=True)
        step += 1
        if step % tcfg.get("log_every", 20) == 0 or step == stop_at:
            mean = {k: round(float(np.mean([x[k] for x in logs])), 4) for k in logs[0]}
            log(f"step {step}/{stop_at} lr={sched.get_last_lr()[0]:.2e} {mean} "
                f"({(time.time() - t0) / len(logs):.2f}s/step)")
            t0, logs = time.time(), []
        if step in checkpoint_steps or step == stop_at:
            tmp = resume.with_name("state.pt.tmp")
            resume.parent.mkdir(parents=True, exist_ok=True)
            torch.save({"step": step, "lora": lora_state(model), "optimizer": opt.state_dict(),
                        "scheduler": sched.state_dict()}, tmp)
            os.replace(tmp, resume)
            if ctx.ref_cache:
                atomic_write_json(run_dir / "npo_ref.json", ctx.ref_cache)
            if step in checkpoint_steps:
                model.eval()
                res = on_checkpoint(step)
                model.train()
                if res.get("stop"):
                    break
    return step


def clear_resume(run_dir):
    shutil.rmtree(Path(run_dir) / "resume", ignore_errors=True)
