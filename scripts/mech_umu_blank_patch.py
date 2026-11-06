"""Causal bridge from the UMU form sweep: why does a blank image bypass text-only unlearning?

For the same name question and answer, compare no image (text) with a blank image. At each chosen
decoder layer, replace ONLY the last prompt token's residual state with the state from the other
route. Run both directions under the unlearned model and its adapter-off vanilla control.
Teacher forcing supplies an answer suffix, but the patched position is before its first token;
causal attention prevents any answer-token information entering the recorded state.

The main readout is log p of the first correct answer token. ``specific_effect`` is the patch
effect under the unlearned model minus the same patch effect under vanilla. For text -> blank,
a negative effect means the blank-image route became more suppressed specifically after unlearning.
For blank -> text, a positive effect means the image route restored the text answer.

  python scripts/mech_umu_blank_patch.py --config configs/llava/umu_npo_rt.yaml \
    --run-id umu_npo_rt_txt_lr5e-05_s0 --step 150

Default: 12 facts in stable pair-id order and a coarse layer grid. If the form-sweep generation
files exist, restrict to facts that are wrong without an image and right with a blank image.
Use --max-facts 0 for all eligible facts and --layers all
for every decoder layer. Output: results/llava/mech_umu/<run>/step<N>/blank_patch.json and .png.
"""
import contextlib
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np  # noqa: E402
import torch  # noqa: E402

from scripts.mech_umu_form_sweep import blank_image_bytes, facts_for  # noqa: E402
from vlmu.config import base_parser, resolve  # noqa: E402
from vlmu.mech import tools  # noqa: E402
from vlmu.mech.hooks import component_modules, decoder_layers, record_outputs  # noqa: E402
from vlmu.models import get_adapter  # noqa: E402
from vlmu.runs import free, model_dir, results_dir  # noqa: E402
from vlmu.umu import encode_teacher_forced, first_target, load_umu, umu_run_id  # noqa: E402
from vlmu.utils import Logger, atomic_write_json, read_json, read_jsonl, tracked_run  # noqa: E402


def routes(fact, blank):
    q = fact["q_uni"]
    shared = {"set": fact["set"], "profile": fact["profile"], "field": fact["field"],
              "pair_id": fact["pair_id"], "correct_text": fact["answer_text"]}
    return {
        "text": {**shared, "uid": fact["pair_id"] + "/text", "prompt": f"USER: {q}\nASSISTANT:", "image_bytes": None},
        "blank": {**shared, "uid": fact["pair_id"] + "/blank", "prompt": f"USER: <image>\n{q}\nASSISTANT:",
                  "image_bytes": blank},
    }


def first_position(batch):
    t = first_target(tools.answer_targets(batch))
    assert t["n_rows"] == 1 and len(t["pos"]) == 1
    return int(t["pos"][0]), int(t["tok"][0])


def layer_grid(spec, n):
    if spec == "all":
        return list(range(n))
    if spec == "coarse":
        return sorted(set([0, n - 1] + list(range(4, n, 4))))
    out = sorted(set(int(x) for x in spec.split(",")))
    if not out or out[0] < 0 or out[-1] >= n:
        raise ValueError(f"layers must be in 0..{n-1}")
    return out


def token_logp(logits, pos, tok):
    return float(tools.target_logprob(logits[:, pos], torch.tensor([tok], device=logits.device))[0])


@contextlib.contextmanager
def patch_last_token(module, pos, state):
    def hook(_module, _input, output):
        hidden = output[0] if isinstance(output, tuple) else output
        updated = hidden.clone()
        updated[:, pos] = state.to(device=updated.device, dtype=updated.dtype)
        return (updated,) + output[1:] if isinstance(output, tuple) else updated

    handle = module.register_forward_hook(hook)
    try:
        yield
    finally:
        handle.remove()


def evaluate_pair(adapter, model, batches, layers):
    """Score both directions under theta_u and theta_0, returning per-layer raw and control effects."""
    mods = component_modules(model, "resid")
    positions = {k: first_position(b) for k, b in batches.items()}
    if positions["text"][1] != positions["blank"][1]:
        raise ValueError("The two routes have different first answer tokens")
    for k, b in batches.items():
        p, _ = positions[k]
        if b["labels"][0, p] != -100 or b["labels"][0, p + 1] == -100:
            raise ValueError(f"{k}: patch position is not the last prompt token")
    # The matched suffix must include the final prompt token. The source state is causal and
    # cannot contain the teacher-forced answer. Never align on the batch's final answer token.
    if batches["text"]["input_ids"][0, positions["text"][0]] != batches["blank"]["input_ids"][0, positions["blank"][0]]:
        raise ValueError("The two routes do not share the final prompt token")

    result = {}
    for tag in ("u", "0"):
        ctx = model.disable_adapter() if tag == "0" else contextlib.nullcontext()
        with ctx:
            cached, base = {}, {}
            for route, batch in batches.items():
                pos, tok = positions[route]
                store = {}
                sel = (torch.tensor([0], device=batch["input_ids"].device),
                       torch.tensor([pos], device=batch["input_ids"].device))
                with record_outputs(model, "resid", range(len(mods)), store, select=sel):
                    logits = tools.forward_logits(adapter, model, batch)
                base[route] = token_logp(logits, pos, tok)
                cached[route] = store
            patched = {}
            for source, target in (("text", "blank"), ("blank", "text")):
                pos, tok = positions[target]
                key = f"{source}_to_{target}"
                patched[key] = {}
                for layer in layers:
                    with patch_last_token(mods[layer], pos, cached[source][layer]):
                        logits = tools.forward_logits(adapter, model, batches[target])
                    patched[key][str(layer)] = token_logp(logits, pos, tok)
            result[tag] = {"base": base, "patched": patched}
    for direction in ("text_to_blank", "blank_to_text"):
        target = direction.split("_to_")[1]
        for layer in layers:
            key = str(layer)
            du = result["u"]["patched"][direction][key] - result["u"]["base"][target]
            d0 = result["0"]["patched"][direction][key] - result["0"]["base"][target]
            result.setdefault("effects", {}).setdefault(direction, {})[key] = {
                "unlearned": du, "vanilla": d0, "specific": du - d0}
    return result


def plot(summary, layers, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(7, 3.8), dpi=180)
    for direction, color, label in (("text_to_blank", "#d35432", "text → blank"),
                                    ("blank_to_text", "#2769b2", "blank → text")):
        vals = [summary[direction][str(k)] for k in layers]
        ax.plot(layers, [v["mean"] for v in vals], marker="o", ms=4, color=color, label=label)
        ax.fill_between(layers, [v["lo"] for v in vals], [v["hi"] for v in vals], color=color, alpha=.15)
    ax.axhline(0, color="#999", lw=.8)
    ax.set(xlabel="decoder layer patched at last prompt token", ylabel="unlearning-specific change in log p(first answer token)")
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def main():
    p = base_parser(__doc__)
    p.add_argument("--run-id", default=None)
    p.add_argument("--step", type=int, required=True)
    p.add_argument("--max-facts", type=int, default=12, help="0 = all eligible facts")
    p.add_argument("--layers", default="coarse", help="coarse, all, or comma-separated indices")
    args = p.parse_args()
    cfg = resolve(args)
    run_id = args.run_id or umu_run_id(cfg)
    run_dir = model_dir(cfg) / "umu_unlearn" / run_id
    adapter_dir = run_dir / "ckpt" / f"step{args.step}"
    if not (adapter_dir / "adapter_config.json").exists():
        raise SystemExit(f"no adapter at {adapter_dir}")
    out = results_dir(cfg) / "mech_umu" / run_id / f"step{args.step}"
    log = Logger(out / "blank_patch_log.txt")
    with tracked_run(results_dir(cfg), out, f"mech_umu_blank_patch_{run_id}_step{args.step}", "mech_umu_blank_patch"):
        from peft import PeftModel

        dc = set(read_json(run_dir / "baseline_dc.json")["pair_ids"])
        dfs = load_umu(cfg["paths"]["root"], cfg["umu"]["n_retain_eval"], 0, log, cfg["umu"]["forget_split"])
        facts = sorted((f for f in facts_for(dfs, dc) if f["set"] == "forget"), key=lambda x: x["pair_id"])
        base_gen = out / "form_sweep_gen_base_u.jsonl"
        blank_gen = out / "form_sweep_gen_blank_image_u.jsonl"
        selection = "all doubly-correct forget facts"
        if base_gen.exists() and blank_gen.exists():
            base_ok = {r["uid"]: r["correct"] for r in read_jsonl(base_gen)}
            blank_ok = {r["uid"]: r["correct"] for r in read_jsonl(blank_gen)}
            facts = [f for f in facts if base_ok.get(f"forget/{f['pair_id']}/base") is False and
                     blank_ok.get(f"forget/{f['pair_id']}/blank_image") is True]
            selection = "base wrong and blank-image right under the same prompts (form-sweep generation)"
        log(f"selection: {selection}; eligible facts={len(facts)}")
        adapter = get_adapter({**cfg["model"], "id": cfg["umu"]["ckpt"], "processor_id": cfg["model"]["id"]})
        model = PeftModel.from_pretrained(adapter.load_model(), str(adapter_dir), is_trainable=False)
        model.eval()
        layers = layer_grid(args.layers, len(decoder_layers(model)))
        blank = blank_image_bytes()
        per = {}
        for fact in facts:
            rows = routes(fact, blank)
            batches = {k: encode_teacher_forced(adapter, [v]) for k, v in rows.items()}
            per[fact["pair_id"]] = evaluate_pair(adapter, model, batches, layers)
            log(f"  {len(per)}: {fact['pair_id']} complete")
            if args.max_facts and len(per) >= args.max_facts:
                break
        if not per:
            raise RuntimeError("No eligible facts found")
        summary = {}
        rng = np.random.default_rng(0)
        baseline = {tag: {route: float(np.mean([r[tag]["base"][route] for r in per.values()]))
                          for route in ("text", "blank")} for tag in ("u", "0")}
        for direction in ("text_to_blank", "blank_to_text"):
            summary[direction] = {}
            for layer in layers:
                values = np.array([r["effects"][direction][str(layer)]["specific"] for r in per.values()])
                means = [rng.choice(values, len(values), replace=True).mean() for _ in range(1000)]
                summary[direction][str(layer)] = {"mean": float(values.mean()), "lo": float(np.percentile(means, 2.5)),
                                                   "hi": float(np.percentile(means, 97.5)), "n": len(values)}
        result = {"run_id": run_id, "step": args.step, "layers": layers, "n_facts": len(per), "selection": selection,
                  "metric": "first correct answer token log probability", "baseline_logp": baseline,
                  "summary": summary, "per_fact": per}
        atomic_write_json(out / "blank_patch.json", result)
        plot(summary, layers, out / "blank_patch.png")
        log(f"baseline mean log p(first answer token): {baseline}")
        for direction in ("text_to_blank", "blank_to_text"):
            ordered = sorted(summary[direction].items(), key=lambda kv: abs(kv[1]["mean"]), reverse=True)
            log(f"largest |specific effect| {direction}: " + ", ".join(f"L{k} {v['mean']:+.2f}" for k, v in ordered[:5]))
        log(f"wrote {out / 'blank_patch.json'} and blank_patch.png; n={len(per)}")
        model = None
        free()


if __name__ == "__main__":
    main()
