"""T->M unlearning on UMU-Bench (DEV-011): fresh LoRA on UMU's released vanilla, text-only NPO (or GA)
on the forget split's UM_QA, UMU-protocol readout (generation + exact match) at step 0 and every
eval_every_epochs. Resumable (training state, ref cache, per-checkpoint generations).

Per checkpoint (checkpoints.jsonl, printed as a table at the end):
  forget/all    Acc_txt, Acc_img on all forget facts
  forget/dc     the same on the run's doubly-correct forget facts (both forms correct at step 0);
                G = Acc_img - Acc_txt there (the leak: the image route still answers what the text route forgot)
  retain/dc     Acc_txt, Acc_img on the doubly-correct retain facts (1.0 at step 0 by construction)
LoRA adapters are saved at every checkpoint (ckpt/step<N>/, ~80 MB each) for the mechanism analyses.

  python scripts/unlearn_umu.py --config configs/llava/umu_npo.yaml [--set train.lr=2e-5] [--debug]
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from vlmu.config import base_parser, resolve  # noqa: E402
from vlmu.models import get_adapter  # noqa: E402
from vlmu.runs import free, model_dir, results_dir  # noqa: E402
from vlmu.train import NPO_METHODS, BatchPlan, Ctx, clear_resume, save_adapter, train  # noqa: E402
from vlmu.umu import fact_table, generate_protocol, load_umu, protocol_items, protocol_summary, \
    unlearn_rows  # noqa: E402
from vlmu.utils import Logger, append_jsonl, atomic_write_json, is_done, mark_done, read_json, read_jsonl, \
    seed_all, tracked_run  # noqa: E402

METHODS = ("npo", "ga")   # forget-only objectives (UMU has no separate unlearning-retain set wired up here)


def fmt(x, spec="+.3f"):
    return "-" if x is None else format(x, spec)


def record(res, dc, step, spe):
    allf = protocol_summary(res)
    dcs = protocol_summary(res, dc)
    g = lambda d, k, m: d.get(f"correct_exact/{k}", {}).get(m)  # noqa: E731
    rec = {"step": step, "epoch": round(step / spe, 3),
           "forget_all_txt": g(allf, "forget", "uni"), "forget_all_img": g(allf, "forget", "mul"),
           "forget_dc_txt": g(dcs, "forget", "uni"), "forget_dc_img": g(dcs, "forget", "mul"),
           "retain_dc_txt": g(dcs, "retain", "uni"), "retain_dc_img": g(dcs, "retain", "mul"),
           "n_forget_dc": g(dcs, "forget", "n_facts"), "n_retain_dc": g(dcs, "retain", "n_facts"),
           "summary_all": allf}
    if rec["forget_dc_img"] is not None:
        rec["G"] = rec["forget_dc_img"] - rec["forget_dc_txt"]
    return rec


def print_table(records, log):
    log(f"{'step':>5s} {'epoch':>5s} | forget all: {'txt':>5s} {'img':>5s} | forget dc: {'txt':>5s} {'img':>5s} "
        f"{'G':>6s} | retain dc: {'txt':>5s} {'img':>5s}")
    f = lambda x: f"{x:5.2f}" if x is not None else "    -"  # noqa: E731
    for s in sorted(records):
        r = records[s]
        log(f"{s:5d} {r['epoch']:5.2f} |             {f(r['forget_all_txt'])} {f(r['forget_all_img'])} |"
            f"            {f(r['forget_dc_txt'])} {f(r['forget_dc_img'])} {fmt(r.get('G'), '+6.2f'):>6s} |"
            f"            {f(r['retain_dc_txt'])} {f(r['retain_dc_img'])}")


def main():
    args = base_parser(__doc__).parse_args()
    cfg = resolve(args)
    u, tcfg, ucfg = cfg["unlearn"], cfg["train"], cfg["umu"]
    assert u["method"] in METHODS and u["modality"] == "txt", f"supported: {METHODS} x txt"
    run_id = f"umu_{u['method']}_{u['modality']}_lr{tcfg['lr']:.0e}_s{u['seed']}"
    run_dir = model_dir(cfg) / "umu_unlearn" / run_id
    log = Logger(run_dir / "log.txt")
    if is_done(run_dir):
        log(f"{run_id} already DONE")
        print_table({r["step"]: r for r in read_jsonl(run_dir / "checkpoints.jsonl")}, log)
        return
    with tracked_run(results_dir(cfg), run_dir, run_id, "unlearn_umu"):
        atomic_write_json(run_dir / "config.json", cfg)
        seed_all(u["seed"])
        dfs = load_umu(cfg["paths"]["root"], ucfg["n_retain_eval"], 0, log, ucfg["forget_split"])
        if cfg["debug"]:
            dfs = {k: v.head(2) for k, v in dfs.items()}
        forget = unlearn_rows(dfs["forget"], u["modality"])
        if cfg["debug"]:
            forget = forget[:cfg.get("debug_rows", 16)]
        assert all(r["image"] is None for r in forget), "text-only unlearning data contains images"
        eval_rows = protocol_items(dfs)

        # UMU vanilla as the base; processor of the public base (its processor_config lacks
        # num_additional_image_tokens; the tokenizer is the same)
        adapter = get_adapter({**cfg["model"], "id": ucfg["ckpt"], "processor_id": cfg["model"]["id"]})
        model = adapter.add_lora(adapter.load_model(), cfg["lora"], u["seed"])
        plan = BatchPlan(forget, tcfg["batch_size"], u["seed"])
        ctx = Ctx(model=model, adapter=adapter, cfg=cfg, run_dir=run_dir, plan=plan, retain_plan=None,
                  image_root=None, loss=u["method"], seed=u["seed"],
                  ref_cache={} if u["method"] in NPO_METHODS else None, text_only=True)
        spe = plan.steps_per_epoch
        total = tcfg.get("max_steps") or spe * tcfg["max_epochs"]
        every = tcfg.get("eval_every_steps") or max(1, round(spe * tcfg["eval_every_epochs"]))
        ckpts = set(range(every, total + 1, every)) | {total}
        log(f"{run_id}: {len(forget)} forget rows (txt), {spe} steps/epoch, {total} steps, eval every {every}; "
            f"{len(eval_rows)} eval generations per checkpoint")
        gen_bs = cfg["eval"].get("gen_batch", 8)
        records = {r["step"]: r for r in read_jsonl(run_dir / "checkpoints.jsonl")}

        def evaluate(step):
            return generate_protocol(model, adapter.processor, eval_rows, run_dir / "evals" / f"step{step}.jsonl",
                                     log, batch_size=gen_bs, dtype=adapter.dtype)

        # step 0 = this run's baseline (same model object, dtype and batching as every later checkpoint)
        dc_path = run_dir / "baseline_dc.json"
        if not dc_path.exists():
            facts = fact_table(evaluate(0))
            atomic_write_json(dc_path, {"pair_ids": sorted(pid for (s, pid), v in facts.items() if v["txt"] and v["img"]),
                                        "n_by_set": {s: sum(1 for (ss, _), v in facts.items()
                                                            if ss == s and v["txt"] and v["img"]) for s in ("forget", "retain")}})
        dc = set(read_json(dc_path)["pair_ids"])
        if 0 not in records:
            records[0] = record(evaluate(0), dc, 0, spe)
            append_jsonl(run_dir / "checkpoints.jsonl", [records[0]])
        log(f"baseline (step 0): {read_json(dc_path)['n_by_set']} doubly-correct facts; forget all txt "
            f"{fmt(records[0]['forget_all_txt'])} img {fmt(records[0]['forget_all_img'])} (UMU protocol, batched bf16; "
            f"batch-1 fp16 reference from 5c-2: txt 0.750 img 0.810)")

        def on_ckpt(step):
            if step not in records:
                save_adapter(model, run_dir / "ckpt" / f"step{step}")
                rec = record(evaluate(step), dc, step, spe)
                append_jsonl(run_dir / "checkpoints.jsonl", [rec])
                records[step] = rec
                log(f"ckpt step {step} (epoch {rec['epoch']}): forget dc txt={fmt(rec['forget_dc_txt'])} "
                    f"img={fmt(rec['forget_dc_img'])} G={fmt(rec.get('G'))} | retain dc txt={fmt(rec['retain_dc_txt'])} "
                    f"img={fmt(rec['retain_dc_img'])}")
            return {}

        final = train(ctx, total, total, ckpts, on_ckpt, log)
        print_table(records, log)
        atomic_write_json(run_dir / "summary.json", {"run_id": run_id, "final_step": final, "steps_per_epoch": spe,
                                                    "checkpoints": [records[s] for s in sorted(records)]})
        clear_resume(run_dir)
        mark_done(run_dir, {"final_step": final})
        model = ctx = None
        free()


if __name__ == "__main__":
    main()
