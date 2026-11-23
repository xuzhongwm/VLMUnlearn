"""UMU track, second model family (DEV-030): fine-tune a UMU vanilla for a model UMU did not release (Qwen2-VL-7B).

UMU's released vanilla (`chengyewang/llava_smu_ft`) is a LoRA fine-tune of LLaVA-1.5-7B on the UMU profiles (their
finetune/finetune.py: LoRA r8/a16, lr 2e-5, 5 epochs, MM_QA with the portrait; the UM_QA loop is commented out there,
yet the released model answers name questions). Here, so that the model knows every person by name AND by face:
  data    all 500 profiles (forget_10 + retain_90), both forms: UM_QA (name question, no image) and MM_QA (portrait +
          "this person" question); answer tokens only in the loss (Adapter.encode_train)
  model   LoRA on the LM linear layers (vision tower and merger frozen), the repo's training loop (vlmu/train.py)
  readout after every epoch: UMU protocol (vlmu/umu.py, the model's own chat template) on the 50 forget profiles and the
          first 100 retain_90 profiles, txt and img; the epoch's adapter is saved
  select  the epoch with the highest min(forget all txt, forget all img); accepted if both >= 0.70 (DEV-030)
  merge   the selected adapter is merged into the base and saved as a full model under $VLMU_ROOT/models/<name>_umu_vanilla
          (the start model of the unlearning runs: --set umu.ckpt=<that path>)

  python scripts/finetune_umu_vanilla.py --config configs/qwen2vl7b/umu_vanilla.yaml [--debug]
"""
import ast
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from vlmu.config import base_parser, resolve  # noqa: E402
from vlmu.models import get_adapter  # noqa: E402
from vlmu.runs import free, model_dir  # noqa: E402
from vlmu.train import BatchPlan, Ctx, save_adapter, train  # noqa: E402
from vlmu.umu import fact_table, generate_protocol, load_umu, protocol_items, protocol_summary, save_image, \
    unlearn_rows  # noqa: E402
from vlmu.utils import Logger, append_jsonl, atomic_write_json, is_done, mark_done, read_jsonl  # noqa: E402

ACCEPT = 0.70   # DEV-030


def image_rows(df):
    rows = []
    for _, row in df.iterrows():
        qa = ast.literal_eval(row["MM_QA"]) if isinstance(row["MM_QA"], str) else row["MM_QA"]
        for field, q in qa.get("question", {}).items():
            a = qa.get("answer", {}).get(field)
            if a is not None:
                rows.append({"uid": f"umu_vanilla/{row['ID']}/{field}/img", "profile": row["ID"], "field": field,
                             "form": "img", "question": q, "answer": str(a), "image": f"{row['ID']}.jpg"})
    return rows


def main():
    p = base_parser(__doc__)
    p.add_argument("--merge-epoch", type=int, default=0, help="merge this epoch instead of the selected one")
    p.add_argument("--extra-epochs", type=int, default=0,
                   help="DEV-030 amendment: continue from the last epoch's adapter for N more epochs (own lr schedule)")
    p.add_argument("--extra-lr", type=float, default=1e-4)
    args = p.parse_args()
    cfg = resolve(args)
    tag = f"_{cfg['run_tag']}" if cfg.get("run_tag") else ""          # DEV-031: e.g. umu_vanilla_vis
    run_dir = model_dir(cfg) / f"umu_vanilla{tag}"
    log = Logger(run_dir / "log.txt")
    tcfg, ucfg = cfg["train"], cfg["umu"]
    dfs = load_umu(cfg["paths"]["root"], 450, 0, log, ucfg["forget_split"])        # 50 + 450 = all 500 profiles
    if cfg["debug"]:
        dfs = {k: v.head(2) for k, v in dfs.items()}
    image_root = run_dir / "images"
    for df in dfs.values():
        for _, row in df.iterrows():
            save_image(row["image"], image_root / f"{row['ID']}.jpg")
    rows = []
    for df in dfs.values():
        rows += unlearn_rows(df, "txt") + image_rows(df)
    adapter = get_adapter(cfg["model"])
    eval_dfs = {"forget": dfs["forget"], "retain": dfs["retain"].head(ucfg.get("n_retain_eval", 100))}
    eval_rows = protocol_items(eval_dfs, sets=("forget", "retain"), prompt_fn=adapter.prompt)
    log(f"{cfg['model']['id']}: {sum(r['form'] == 'txt' for r in rows)} txt + {sum(r['form'] == 'img' for r in rows)} img "
        f"training rows from {sum(len(d) for d in dfs.values())} profiles; {len(eval_rows)} readout items per epoch")
    log("example prompt: " + repr(adapter.prompt("<question>", True)))
    epochs_path = run_dir / "epochs.jsonl"

    def make_on_ckpt(model, spe, offset):
        """Readout after an epoch: save the adapter as ckpt/epoch<offset + k>, score, append to epochs.jsonl."""
        done = {r["epoch"] for r in read_jsonl(epochs_path)}

        def on_ckpt(step):
            ep = offset + max(1, round(step / spe))
            if ep in done:
                return {}
            save_adapter(model, run_dir / "ckpt" / f"epoch{ep}")
            res = generate_protocol(model, adapter.processor, eval_rows, run_dir / "evals" / f"epoch{ep}.jsonl", log,
                                    batch_size=cfg["eval"].get("gen_batch", 8), dtype=adapter.dtype)
            summ = protocol_summary(res)["correct_exact/forget"], protocol_summary(res)["correct_exact/retain"]
            facts = fact_table(res)
            n_dc = {s: sum(1 for (ss, _), v in facts.items() if ss == s and v["txt"] and v["img"]) for s in ("forget", "retain")}
            rec = {"step": step, "epoch": ep, "forget_txt": summ[0]["uni"], "forget_img": summ[0]["mul"],
                   "retain_txt": summ[1]["uni"], "retain_img": summ[1]["mul"], "n_dc": n_dc}
            append_jsonl(epochs_path, [rec])
            log(f"epoch {ep}: forget txt {rec['forget_txt']:.3f} img {rec['forget_img']:.3f} | retain txt "
                f"{rec['retain_txt']:.3f} img {rec['retain_img']:.3f} | doubly-correct {n_dc}")
            return {}
        return on_ckpt

    if not is_done(run_dir):
        model = adapter.add_lora(adapter.load_model(), cfg["lora"], tcfg["seed"])
        plan = BatchPlan(rows, tcfg["batch_size"], tcfg["seed"])
        ctx = Ctx(model=model, adapter=adapter, cfg=cfg, run_dir=run_dir, plan=plan, retain_plan=None,
                  image_root=image_root, loss="nll", seed=tcfg["seed"], ref_cache=None, text_only=False)
        spe = plan.steps_per_epoch
        total = tcfg.get("max_steps") or spe * tcfg["epochs"]
        ckpts = {min(spe * e, total) for e in range(1, tcfg["epochs"] + 1)}
        log(f"{spe} steps/epoch, {total} steps, readout after steps {sorted(ckpts)}")
        train(ctx, total, total, ckpts, make_on_ckpt(model, spe, 0), log)
        model = ctx = None
        free()
        mark_done(run_dir, {"final_step": total})
    extra_dir = run_dir / "extra"
    if args.extra_epochs and not is_done(extra_dir):
        from peft import PeftModel
        last = max(r["epoch"] for r in read_jsonl(epochs_path) if r["epoch"] <= tcfg["epochs"])
        base = adapter.load_model()
        if cfg["lora"].get("gradient_checkpointing", True):
            base.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
            base.enable_input_require_grads()
        model = PeftModel.from_pretrained(base, str(run_dir / "ckpt" / f"epoch{last}"), is_trainable=True)
        ecfg = {**cfg, "train": {**tcfg, "lr": args.extra_lr, "warmup_ratio": 0.0}}
        plan = BatchPlan(rows, tcfg["batch_size"], tcfg["seed"] + 1)
        ctx = Ctx(model=model, adapter=adapter, cfg=ecfg, run_dir=extra_dir, plan=plan, retain_plan=None,
                  image_root=image_root, loss="nll", seed=tcfg["seed"] + 1, ref_cache=None, text_only=False)
        spe = plan.steps_per_epoch
        total = spe * args.extra_epochs
        ckpts = {spe * e for e in range(1, args.extra_epochs + 1)}
        log(f"extra epochs (DEV-030 amendment): from epoch {last}, {args.extra_epochs} more at lr {args.extra_lr} "
            f"(linear decay), readout after steps {sorted(ckpts)}")
        train(ctx, total, total, ckpts, make_on_ckpt(model, spe, last), log)
        model = ctx = None
        free()
        mark_done(extra_dir, {"final_step": total})
    recs = read_jsonl(epochs_path)
    best = max(recs, key=lambda r: min(r["forget_txt"], r["forget_img"]))
    ep = args.merge_epoch or best["epoch"]
    chosen = next(r for r in recs if r["epoch"] == ep)
    ok = min(chosen["forget_txt"], chosen["forget_img"]) >= ACCEPT
    log(f"selected epoch {ep} (forget txt {chosen['forget_txt']:.3f}, img {chosen['forget_img']:.3f}); "
        f"{'ACCEPTED' if ok else 'NOT accepted'} (both >= {ACCEPT})")
    atomic_write_json(run_dir / "selected.json", {"epoch": ep, "accepted": ok, "record": chosen})
    if cfg["debug"]:
        return
    from peft import PeftModel
    out = Path(cfg["paths"]["root"]) / "models" / f"{cfg['model']['name']}_umu_vanilla{tag}"
    base = adapter.load_model()
    merged = PeftModel.from_pretrained(base, str(run_dir / "ckpt" / f"epoch{ep}")).merge_and_unload()
    merged.save_pretrained(out, safe_serialization=True)
    adapter.processor.save_pretrained(out)
    log(f"merged epoch {ep} into {cfg['model']['id']} -> {out}")


if __name__ == "__main__":
    main()
