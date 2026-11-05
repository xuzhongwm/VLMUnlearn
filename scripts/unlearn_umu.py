"""T->M unlearning on UMU-Bench (DEV-011, DEV-012): fresh LoRA on UMU's released vanilla, text-only
unlearning on the forget split's UM_QA, UMU-protocol readout (generation + exact match) at step 0 and
every eval_every_epochs. Resumable (training state, ref cache, per-checkpoint generations).

Methods: npo / ga (forget only) and npo_rt / ga_diff (+ retain NLL on the UM_QA of retain_90 profiles
that are NOT in the retain readout: umu.n_retain_unlearn profiles after the first umu.n_retain_eval).
Forget-only NPO at lr 5e-5 collapsed every text-only input to 'PAPAPA...' from epoch 2 (DEV-012).

Per checkpoint (checkpoints.jsonl, printed as a table at the end):
  forget/all    Acc_txt, Acc_img on all forget facts
  forget/dc     on the run's doubly-correct forget facts (txt and img correct at step 0): txt, img,
                txt+img (name + portrait, if eval.forms has it) and G = Acc_img - Acc_txt
  retain/dc     txt, img on the doubly-correct retain facts (1.0 at step 0 by construction)
  degen         share of the most frequent text-only response (1.0 = one answer to everything)
LoRA adapters are saved at every checkpoint (ckpt/step<N>/, ~80 MB each) for the mechanism analyses.

  python scripts/unlearn_umu.py --config configs/llava/umu_npo_rt.yaml [--set train.lr=1e-5] [--set run_tag=rw3 ...] [--debug]
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from vlmu.config import base_parser, resolve  # noqa: E402
from vlmu.models import get_adapter  # noqa: E402
from vlmu.runs import free, model_dir, results_dir  # noqa: E402
from vlmu.train import NPO_METHODS, RETAIN_METHODS, BatchPlan, Ctx, clear_resume, save_adapter, train  # noqa: E402
from vlmu.umu import degeneracy, fact_table, generate_protocol, load_umu, protocol_items, protocol_summary, \
    umu_run_id, unlearn_rows  # noqa: E402
from vlmu.utils import Logger, append_jsonl, atomic_write_json, is_done, mark_done, read_json, read_jsonl, \
    seed_all, tracked_run  # noqa: E402

METHODS = ("npo", "ga", "npo_rt", "ga_diff")


def fmt(x, spec="+.3f"):
    return "-" if x is None else format(x, spec)


def record(res, dc, step, spe):
    allf = protocol_summary(res)
    dcs = protocol_summary(res, dc)
    g = lambda d, k, m: d.get(f"correct_exact/{k}", {}).get(m)  # noqa: E731
    rec = {"step": step, "epoch": round(step / spe, 3),
           "forget_all_txt": g(allf, "forget", "uni"), "forget_all_img": g(allf, "forget", "mul"),
           "forget_dc_txt": g(dcs, "forget", "uni"), "forget_dc_img": g(dcs, "forget", "mul"),
           "forget_dc_txtimg": g(dcs, "forget", "uni_img"),
           "retain_dc_txt": g(dcs, "retain", "uni"), "retain_dc_img": g(dcs, "retain", "mul"),
           "seen_dc_txt": g(dcs, "retain_seen", "uni"), "seen_dc_img": g(dcs, "retain_seen", "mul"),
           "n_forget_dc": g(dcs, "forget", "n_facts"), "n_retain_dc": g(dcs, "retain", "n_facts"),
           "n_seen_dc": g(dcs, "retain_seen", "n_facts"),
           "degen_txt": degeneracy(res, "txt"), "summary_all": allf}
    if rec["forget_dc_img"] is not None:
        rec["G"] = rec["forget_dc_img"] - rec["forget_dc_txt"]
    return rec


def print_table(records, log):
    log(f"{'step':>5s} {'epoch':>5s} | forget all {'txt':>5s} {'img':>5s} | forget dc {'txt':>5s} {'img':>5s} "
        f"{'t+i':>5s} {'G':>6s} | retain dc {'txt':>5s} {'img':>5s} | seen dc {'txt':>5s} {'img':>5s} | {'degen':>5s}")
    f = lambda x: f"{x:5.2f}" if x is not None else "    -"  # noqa: E731
    for s in sorted(records):
        r = records[s]
        log(f"{s:5d} {r['epoch']:5.2f} |            {f(r['forget_all_txt'])} {f(r['forget_all_img'])} |"
            f"           {f(r['forget_dc_txt'])} {f(r['forget_dc_img'])} {f(r.get('forget_dc_txtimg'))} "
            f"{fmt(r.get('G'), '+6.2f'):>6s} |           {f(r['retain_dc_txt'])} {f(r['retain_dc_img'])} |"
            f"         {f(r.get('seen_dc_txt'))} {f(r.get('seen_dc_img'))} | {f(r.get('degen_txt'))}")


def main():
    args = base_parser(__doc__).parse_args()
    cfg = resolve(args)
    u, tcfg, ucfg = cfg["unlearn"], cfg["train"], cfg["umu"]
    assert u["method"] in METHODS and u["modality"] == "txt", f"supported: {METHODS} x txt"
    run_id = umu_run_id(cfg)
    run_dir = model_dir(cfg) / "umu_unlearn" / run_id
    log = Logger(run_dir / "log.txt")
    if is_done(run_dir):
        log(f"{run_id} already DONE")
        print_table({r["step"]: r for r in read_jsonl(run_dir / "checkpoints.jsonl")}, log)
        return
    with tracked_run(results_dir(cfg), run_dir, run_id, "unlearn_umu"):
        atomic_write_json(run_dir / "config.json", cfg)
        seed_all(u["seed"])
        n_eval, n_ru = ucfg["n_retain_eval"], ucfg.get("n_retain_unlearn", 0) if u["method"] in RETAIN_METHODS else 0
        dfs = load_umu(cfg["paths"]["root"], n_eval + n_ru, 0, log, ucfg["forget_split"])
        retain_unlearn_df, dfs["retain"] = dfs["retain"].iloc[n_eval:], dfs["retain"].iloc[:n_eval]
        if cfg["debug"]:
            dfs = {k: v.head(2) for k, v in dfs.items()}
            retain_unlearn_df = retain_unlearn_df.head(2)
        forget = unlearn_rows(dfs["forget"], u["modality"])
        retain = unlearn_rows(retain_unlearn_df, u["modality"]) if n_ru else []
        if cfg["debug"]:
            forget, retain = forget[:cfg.get("debug_rows", 16)], retain[:cfg.get("debug_rows", 16)]
        assert all(r["image"] is None for r in forget + retain), "text-only unlearning data contains images"
        assert not set(retain_unlearn_df["ID"]) & set(dfs["retain"]["ID"]), "retain-unlearn overlaps the retain readout"
        # 'retain_seen': people the retain term trains on (the first n of the retain-unlearn set). Comparing them with the
        # unseen readout people separates "the retain term protects only the people it was shown" from "forgetting
        # became person-specific" (DEV-012).
        n_seen = min(ucfg.get("n_retain_seen_eval", 20), len(retain_unlearn_df)) if retain else 0
        if n_seen:
            dfs["retain_seen"] = retain_unlearn_df.head(n_seen)
        eval_rows = protocol_items(dfs, sets=("forget", "retain", "retain_seen"),
                                   forms={k: tuple(v) for k, v in cfg["eval"].get("forms", {}).items()})

        # UMU vanilla as the base; processor of the public base (its processor_config lacks
        # num_additional_image_tokens; the tokenizer is the same)
        adapter = get_adapter({**cfg["model"], "id": ucfg["ckpt"], "processor_id": cfg["model"]["id"]})
        model = adapter.add_lora(adapter.load_model(), cfg["lora"], u["seed"])
        plan = BatchPlan(forget, tcfg["batch_size"], u["seed"])
        rplan = BatchPlan(retain, tcfg["batch_size"], u["seed"] + 7919) if retain else None
        ctx = Ctx(model=model, adapter=adapter, cfg=cfg, run_dir=run_dir, plan=plan, retain_plan=rplan,
                  image_root=None, loss=u["method"], seed=u["seed"],
                  ref_cache={} if u["method"] in NPO_METHODS else None, text_only=True)
        spe = plan.steps_per_epoch
        total = tcfg.get("max_steps") or spe * tcfg["max_epochs"]
        every = tcfg.get("eval_every_steps") or max(1, round(spe * tcfg["eval_every_epochs"]))
        ckpts = set(range(every, total + 1, every)) | {total}
        log(f"{run_id}: {len(forget)} forget rows (txt), {len(retain)} retain-unlearn rows "
            f"({len(retain_unlearn_df)} profiles, disjoint from the {len(dfs['retain'])} readout profiles), {spe} steps/epoch, "
            f"{total} steps, eval every {every}; {len(eval_rows)} eval generations per checkpoint")
        gen_bs = cfg["eval"].get("gen_batch", 8)
        records = {r["step"]: r for r in read_jsonl(run_dir / "checkpoints.jsonl")}

        def evaluate(step):
            return generate_protocol(model, adapter.processor, eval_rows, run_dir / "evals" / f"step{step}.jsonl",
                                     log, batch_size=gen_bs, dtype=adapter.dtype)

        # step 0 = this run's baseline (same model object, dtype and batching as every later checkpoint).
        # Generations of identical step-0 items (same uid = same prompt; same base model, dtype, batching per
        # modality) can be taken from an earlier run; only the missing items are generated.
        src = ucfg.get("reuse_step0_from")
        step0 = run_dir / "evals" / "step0.jsonl"
        if src and not step0.exists() and (run_dir.parent / src / "evals" / "step0.jsonl").exists():
            want = {r["uid"] for r in eval_rows}
            prev = [r for r in read_jsonl(run_dir.parent / src / "evals" / "step0.jsonl") if r["uid"] in want]
            append_jsonl(step0, prev)
            log(f"step 0: reused {len(prev)} generations from {src}")
        dc_path = run_dir / "baseline_dc.json"
        if not dc_path.exists():
            facts = fact_table(evaluate(0))
            atomic_write_json(dc_path, {"pair_ids": sorted(pid for (s, pid), v in facts.items() if v["txt"] and v["img"]),
                                        "n_by_set": {s: sum(1 for (ss, _), v in facts.items()
                                                            if ss == s and v["txt"] and v["img"]) for s in ("forget", "retain", "retain_seen")}})
        dc = set(read_json(dc_path)["pair_ids"])
        if 0 not in records:
            records[0] = record(evaluate(0), dc, 0, spe)
            append_jsonl(run_dir / "checkpoints.jsonl", [records[0]])
        log(f"baseline (step 0): {read_json(dc_path)['n_by_set']} doubly-correct facts; forget all txt "
            f"{fmt(records[0]['forget_all_txt'])} img {fmt(records[0]['forget_all_img'])}; forget dc txt+img "
            f"{fmt(records[0].get('forget_dc_txtimg'))} (UMU protocol, batched bf16)")

        def on_ckpt(step):
            if step not in records:
                save_adapter(model, run_dir / "ckpt" / f"step{step}")
                rec = record(evaluate(step), dc, step, spe)
                append_jsonl(run_dir / "checkpoints.jsonl", [rec])
                records[step] = rec
                log(f"ckpt step {step} (epoch {rec['epoch']}): forget dc txt={fmt(rec['forget_dc_txt'])} "
                    f"img={fmt(rec['forget_dc_img'])} txt+img={fmt(rec.get('forget_dc_txtimg'))} G={fmt(rec.get('G'))} | "
                    f"retain dc txt={fmt(rec['retain_dc_txt'])} img={fmt(rec['retain_dc_img'])} | "
                    f"seen dc txt={fmt(rec.get('seen_dc_txt'))} img={fmt(rec.get('seen_dc_img'))} | "
                    f"degen txt={fmt(rec.get('degen_txt'), '.2f')}")
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
