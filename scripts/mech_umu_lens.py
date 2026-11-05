"""UMU track, research.md Section 6 step 2: logit-lens curves of an unlearning checkpoint vs UMU's vanilla.

theta_0 = UMU's released vanilla, theta_u = theta_0 + the run's LoRA at --step (one loaded model; theta_0 =
adapter disabled). Items = the run's doubly-correct facts, as the UMU-protocol prompt + the correct option
text, teacher-forced (DEV-011: targets are answer-text tokens, not option letters). Routes:
  forget: txt (name, no image), img (portrait + "this person"), txt+img (name + portrait)
  retain: txt, img (control: what the unlearning does to facts it should keep)
Per row and decoder layer, read through the final norm + unembedding (logit lens):
  first   log p(first answer token)              -- the token that decides the generated answer
  answer  sum of log p over the answer tokens (end-of-sequence excluded)
for theta_u and theta_0; diff = theta_u - theta_0 (research.md: read the difference, not early-layer values).
Each row also carries whether that route answered correctly at --step (evals/step<N>.jsonl), so curves can
be split into forgotten / not forgotten.

Reading (research.md, Hypotheses & Predictions): the image route's diff dips in middle layers and recovers
-> H2 (suppression then compensation); stays ~0 throughout -> H1 / H1' (next: lora_layer_mask + delta_wx_norm).
txt+img tracking txt -> the name gates the forgetting; tracking img -> the missing image does.

  python scripts/mech_umu_lens.py --config configs/llava/umu_npo_rt.yaml --step 75 [--run-id ...]
Writes results/llava/mech_umu/<run_id>/step<N>/lens.json (+ a quick-look lens.png).
"""
import collections
import io
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np  # noqa: E402
import torch  # noqa: E402

from vlmu.config import base_parser, resolve  # noqa: E402
from vlmu.mech import tools  # noqa: E402
from vlmu.mech.hooks import decoder_layers  # noqa: E402
from vlmu.models import get_adapter  # noqa: E402
from vlmu.runs import free, model_dir, results_dir  # noqa: E402
from vlmu.umu import load_umu, protocol_items  # noqa: E402
from vlmu.utils import Logger, atomic_write_json, read_json, read_jsonl, tracked_run  # noqa: E402

FORMS = {"forget": ("txt", "img", "txt+img"), "retain": ("txt", "img")}


def umu_run_id(cfg):
    u, t = cfg["unlearn"], cfg["train"]
    return f"umu_{u['method']}_{u['modality']}_lr{t['lr']:.0e}_s{u['seed']}"


def encode_tf(adapter, rows):
    """Right-padded teacher-forced batch: UMU-protocol prompt + ' {correct option text}</s>'; labels = answer
    tokens only (as Adapter.encode_train, with the protocol prompt instead of adapter.prompt)."""
    from PIL import Image
    imgs = [Image.open(io.BytesIO(r["image_bytes"])).convert("RGB") for r in rows if r["image_bytes"] is not None]
    assert len(imgs) in (0, len(rows)), "batches must be homogeneous in image presence"
    imgs = imgs or None
    prompts = [r["prompt"] for r in rows]
    full = [p + adapter.answer_suffix(r["correct_text"]) for p, r in zip(prompts, rows)]
    b = adapter._process(full, imgs, "right")
    pb = adapter._process(prompts, imgs, "right")
    labels = b["input_ids"].clone()
    labels[b["attention_mask"] == 0] = -100
    for i, n in enumerate(pb["attention_mask"].sum(1).tolist()):
        if not torch.equal(b["input_ids"][i, :n], pb["input_ids"][i, :n]):
            raise ValueError(f"prompt is not a token prefix of prompt+answer: {rows[i]['uid']}")
        labels[i, :n] = -100
    labels[labels == adapter.tok.eos_token_id] = -100       # score the answer text, not the end of turn
    b["labels"] = labels
    return adapter.to_device(b)


def first_only(t):
    keep = tools._first(t)
    return {"b": t["b"][keep], "pos": t["pos"][keep], "tok": t["tok"][keep], "n_rows": t["n_rows"]}


def plot(rows, n_layers, title, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from vlmu.report import INK2, SURFACE
    colors = {"txt": "#2a78d6", "img": "#eb6834", "txt+img": "#1baf7a"}
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.8), dpi=150, sharey=True)
    fig.patch.set_facecolor(SURFACE)
    for ax, s in zip(axes, ("forget", "retain")):
        ax.set_facecolor(SURFACE)
        ax.axhline(0, color=INK2, lw=0.8)
        for m in FORMS[s]:
            d = np.array([r["first"]["diff"] for r in rows if r["set"] == s and r["modality"] == m])
            if not len(d):
                continue
            mean, sem = d.mean(0), d.std(0, ddof=1) / np.sqrt(len(d)) if len(d) > 1 else 0 * d.mean(0)
            ax.plot(range(n_layers), mean, color=colors[m], lw=2, label=f"{m} (n={len(d)})")
            ax.fill_between(range(n_layers), mean - sem, mean + sem, color=colors[m], alpha=0.15, lw=0)
        ax.set_title(f"{s} (doubly-correct facts)", fontsize=10)
        ax.set_xlabel("decoder layer")
        ax.legend(frameon=False, fontsize=8)
    axes[0].set_ylabel("Δ log p(first answer token)\nθu − θ0, logit lens")
    fig.suptitle(title, fontsize=10)
    fig.tight_layout()
    fig.savefig(path, facecolor=SURFACE)
    plt.close(fig)


def main():
    p = base_parser(__doc__)
    p.add_argument("--step", type=int, required=True)
    p.add_argument("--run-id", default=None, help="default: derived from the config (as scripts/unlearn_umu.py)")
    p.add_argument("--batch", type=int, default=8)
    args = p.parse_args()
    cfg = resolve(args)
    run_id = args.run_id or umu_run_id(cfg)
    run_dir = model_dir(cfg) / "umu_unlearn" / run_id
    out = results_dir(cfg) / "mech_umu" / run_id / f"step{args.step}"
    log = Logger(out / "log.txt")
    adapter_dir = run_dir / "ckpt" / f"step{args.step}"
    if not (adapter_dir / "adapter_config.json").exists():
        raise SystemExit(f"no adapter at {adapter_dir}")
    with tracked_run(results_dir(cfg), out, f"mech_umu_lens_{run_id}_step{args.step}", "mech_umu_lens"):
        from peft import PeftModel
        dc = set(read_json(run_dir / "baseline_dc.json")["pair_ids"])
        dfs = load_umu(cfg["paths"]["root"], cfg["umu"]["n_retain_eval"], 0, log, cfg["umu"]["forget_split"])
        items = [r for r in protocol_items(dfs, forms=FORMS) if r["pair_id"] in dc]
        at_step = {r["uid"]: r["correct_exact"] for r in read_jsonl(run_dir / "evals" / f"step{args.step}.jsonl")}
        log(f"{run_id} step {args.step}: {len(items)} rows over {len(dc)} doubly-correct facts "
            f"{dict(collections.Counter((r['set'], r['modality']) for r in items))}")

        adapter = get_adapter({**cfg["model"], "id": cfg["umu"]["ckpt"], "processor_id": cfg["model"]["id"]})
        model = PeftModel.from_pretrained(adapter.load_model(), str(adapter_dir), is_trainable=False)
        model.eval()
        n_layers = len(decoder_layers(model))
        results = []
        for m in sorted({r["modality"] for r in items}):
            group = sorted([r for r in items if r["modality"] == m], key=lambda r: len(r["prompt"]))
            for i in range(0, len(group), args.batch):
                chunk = group[i:i + args.batch]
                batch = encode_tf(adapter, chunk)
                t_all = tools.answer_targets(batch)
                res = {"answer": tools.layer_logit_diff(adapter, model, batch, t_all),
                       "first": tools.layer_logit_diff(adapter, model, batch, first_only(t_all))}
                for j, r in enumerate(chunk):
                    results.append({k: r[k] for k in ("uid", "set", "modality", "profile", "field", "pair_id", "correct_text")}
                                   | {"correct_at_step": at_step.get(r["uid"]),
                                      **{t: {k: [round(float(x), 4) for x in v[:, j]] for k, v in res[t].items()}
                                         for t in res}})
            log(f"  {m}: {len(group)} rows")
        model = None
        free()
        atomic_write_json(out / "lens.json", {"run_id": run_id, "step": args.step, "n_layers": n_layers,
                                              "base": cfg["umu"]["ckpt"], "adapter": str(adapter_dir),
                                              "targets": {"first": "first answer token", "answer": "sum over answer tokens, EOS excluded"},
                                              "rows": results})
        for s in FORMS:
            for m in FORMS[s]:
                d = np.array([r["first"]["diff"] for r in results if r["set"] == s and r["modality"] == m])
                if len(d):
                    mean = d.mean(0)
                    log(f"  {s}/{m:7s} n={len(d):3d}  mean diff (first token) by layer 0,8,16,24,last: "
                        + " ".join(f"{mean[k]:+.2f}" for k in (0, 8, 16, 24, n_layers - 1)))
        plot(results, n_layers, f"{run_id} step {args.step}: logit lens, θu − θ0", out / "lens.png")
        log(f"wrote {out / 'lens.json'} and lens.png")


if __name__ == "__main__":
    main()
