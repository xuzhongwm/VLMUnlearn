"""UMU track, DEV-054: is the edit written at the referent tokens when the whole generated answer is read?

The earlier position tests read the first answer token. Here the LoRA's contribution is switched off by position during
the whole greedy generation, and the UMU-protocol answer is scored by exact match on the forget dc facts, in the form the
run was trained on (--form txt: referent = the name tokens; --form img: referent = the image tokens).
  u / 0      LoRA on everywhere / off everywhere (references)
  ref_off    off at the referent tokens; on at the rest of the prompt and at the answer positions
  rest_off   off at the prompt tokens that are neither referent nor the last prompt token
  read_off   off at the last prompt token and at the answer positions (the generated tokens)
  ref_only   on at the referent tokens only
Share restored S(c) = (acc_c - acc_u) / (acc_0 - acc_u). Reading and thresholds: DEVIATIONS.md DEV-054.
Validity (abort): `u` gives the unhooked model's responses and `0` those of the model with the adapter disabled on >= 98%
of the items; every row has referent tokens.

  python scripts/mech_umu_answer_level.py --config <model config> --run-id <run> --step <N> --form txt|img --set umu.n_retain_eval=100 [umu.ckpt=...]
Writes results/<model>/mech_umu/<run>/step<N>/answer_level/{summary.json,items.jsonl,log.txt}.
"""
import io
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np  # noqa: E402
import torch  # noqa: E402
from PIL import Image  # noqa: E402

from vlmu.config import base_parser, resolve  # noqa: E402
from vlmu.mech.hooks import lm_parts, lora_delta, lora_modules  # noqa: E402
from vlmu.models import get_adapter  # noqa: E402
from vlmu.runs import free, model_dir, results_dir  # noqa: E402
from vlmu.umu import _response, load_umu, protocol_items  # noqa: E402
from vlmu.utils import Logger, atomic_write_json, read_json, write_jsonl  # noqa: E402

WRITTEN_MIN, REST_MAX, NOT_MAX, GAP_MIN, SAME_MIN = 0.7, 0.3, 0.3, 0.2, 0.98           # DEV-054
CONDS = ("u", "0", "ref_off", "rest_off", "read_off", "ref_only")


def prompt_masks(adapter, model, rows, batch):
    """[B, T] masks over a left-padded prompt-only batch: name tokens, image tokens, valid tokens (as token_groups_umu)."""
    tok, ids, valid = adapter.tok, batch["input_ids"], batch["attention_mask"].bool()
    img_id = lm_parts(model)[2]
    image = (ids == img_id) & valid if img_id is not None else torch.zeros_like(valid)
    name = torch.zeros_like(valid)
    for i, r in enumerate(rows):
        enc = tok(r["prompt"], add_special_tokens=True, return_offsets_mapping=True)
        raw, n_img = enc["input_ids"], int(image[i].sum())
        if n_img:
            k = raw.index(img_id)
            expanded, shift = raw[:k] + [img_id] * n_img + raw[k + 1:], (lambda j, k=k, n=n_img: j + (n - 1 if j > k else 0))
        else:
            expanded, shift = raw, (lambda j: j)
        if ids[i, valid[i]].tolist() != expanded:
            raise RuntimeError(f"the re-tokenised prompt does not match the batch: {r['uid']}")
        start = int(valid[i].nonzero()[0])
        spans = [(m.start(), m.end()) for m in re.finditer(re.escape(r["name"]), r["prompt"])] if r.get("name") else []
        for j, (a, b) in enumerate(enc["offset_mapping"]):
            if b > a and any(a < e and b > s for s, e in spans):
                name[i, start + shift(j)] = True
    return name, image, valid


class Switch:
    """Per-position LoRA switch that survives generation with a cache: the prompt pass uses keep [B, T], every later pass
    (one new token) uses the scalar `ans`. out = base_layer(x) + keep * d(x), as mask_contributions."""

    def __init__(self, model):
        self.prompt, self.ans, self.T = None, 1.0, None
        self.handles = [m.register_forward_hook(self.hook) for m in lora_modules(model).values()]

    def hook(self, mod, inp, out):
        if self.prompt is None:
            return None
        x = inp[0]
        d = lora_delta(mod, x)
        if x.shape[1] == self.T:
            keep = self.prompt.to(device=d.device, dtype=d.dtype)[..., None]
        else:
            assert x.shape[1] == 1, f"unexpected pass of length {x.shape[1]} (prompt {self.T})"
            keep = self.ans
        return (mod.base_layer(x) + d * keep).to(out.dtype)

    def remove(self):
        for h in self.handles:
            h.remove()


def main():
    p = base_parser(__doc__)
    p.add_argument("--run-id", required=True)
    p.add_argument("--step", type=int, required=True)
    p.add_argument("--form", choices=["txt", "img"], required=True, help="the form the run was trained on")
    p.add_argument("--batch", type=int, default=0, help="default: 32 for txt, 8 for img")
    p.add_argument("--max-facts", type=int, default=0, help="smoke test: output under answer_level_smoke")
    args = p.parse_args()
    cfg = resolve(args)
    bs = args.batch or (32 if args.form == "txt" else 8)
    run_dir = model_dir(cfg) / "umu_unlearn" / args.run_id
    out = results_dir(cfg) / "mech_umu" / args.run_id / f"step{args.step}" / ("answer_level_smoke" if args.max_facts else "answer_level")
    log = Logger(out / "log.txt")
    from peft import PeftModel
    dc = set(read_json(run_dir / "baseline_dc.json")["pair_ids"])
    dfs = load_umu(cfg["paths"]["root"], cfg["umu"]["n_retain_eval"], 0, log, cfg["umu"]["forget_split"])
    adapter = get_adapter({**cfg["model"], "id": cfg["umu"]["ckpt"], "processor_id": cfg["model"]["id"]})
    rows = [r for r in protocol_items({"forget": dfs["forget"]}, sets=("forget",), forms={"forget": (args.form,)},
                                      prompt_fn=None if cfg["model"]["family"] == "llava" else adapter.prompt) if r["pair_id"] in dc]
    rows = sorted(rows, key=lambda r: len(r["prompt"]))
    if args.max_facts:
        rows = rows[:args.max_facts]
    model = PeftModel.from_pretrained(adapter.load_model(), str(run_dir / "ckpt" / f"step{args.step}"), is_trainable=False)
    model.eval()
    log(f"{args.run_id} step {args.step} ({cfg['model']['id']}): {len(rows)} forget dc facts, form {args.form}; referent = "
        f"{'name' if args.form == 'txt' else 'image'} tokens; conditions {CONDS}")

    def generate(batch):
        with torch.no_grad():
            gen = model.generate(**batch, max_new_tokens=50, do_sample=False, num_beams=1, use_cache=True,
                                 pad_token_id=adapter.tok.pad_token_id)
        return [_response(t) for t in adapter.tok.batch_decode(gen[:, batch["input_ids"].shape[1]:], skip_special_tokens=True)]

    sw, recs, n_ref = Switch(model), [], []
    for i in range(0, len(rows), bs):
        chunk = rows[i:i + bs]
        imgs = [Image.open(io.BytesIO(r["image_bytes"])).convert("RGB") for r in chunk if r["image_bytes"] is not None]
        batch = adapter.to_device(adapter._process([r["prompt"] for r in chunk], imgs or None, "left"))
        assert bool(batch["attention_mask"][:, -1].all())
        name, image, valid = prompt_masks(adapter, model, chunk, batch)
        ref = name if args.form == "txt" else image
        if not bool((ref.sum(1) > 0).all()):
            raise RuntimeError("a row has no referent tokens")
        n_ref += ref.sum(1).tolist()
        last = torch.zeros_like(valid)
        last[:, -1] = True
        keep = {"u": (torch.ones_like(valid), 1.0), "0": (torch.zeros_like(valid), 0.0), "ref_off": (~ref, 1.0),
                "rest_off": (ref | last, 1.0), "read_off": (~last, 0.0), "ref_only": (ref, 0.0)}
        resp = {}
        sw.prompt = None                                        # references without the hook
        resp["plain_u"] = generate(batch)
        with model.disable_adapter():
            resp["plain_0"] = generate(batch)
        for c in CONDS:
            sw.prompt, sw.ans, sw.T = keep[c][0].float(), keep[c][1], batch["input_ids"].shape[1]
            resp[c] = generate(batch)
        sw.prompt = None
        for j, r in enumerate(chunk):
            rec = {"uid": r["uid"], "pair_id": r["pair_id"], "field": r["field"], "correct_text": r["correct_text"], "n_ref": int(ref[j].sum())}
            for c, v in resp.items():
                rec[f"resp/{c}"], rec[f"ok/{c}"] = v[j], v[j] == r["correct_text"]
            recs.append(rec)
        log(f"  {min(i + bs, len(rows))}/{len(rows)}")
    sw.remove()
    model = None
    free()
    write_jsonl(out / "items.jsonl", recs)
    acc = {c: float(np.mean([x[f"ok/{c}"] for x in recs])) for c in CONDS + ("plain_u", "plain_0")}
    same = {"u": float(np.mean([x["resp/u"] == x["resp/plain_u"] for x in recs])), "0": float(np.mean([x["resp/0"] == x["resp/plain_0"] for x in recs]))}
    gap = acc["0"] - acc["u"]
    S = {c: (acc[c] - acc["u"]) / gap if gap > 0 else float("nan") for c in CONDS}
    readable = gap >= GAP_MIN
    verdict = ("not readable" if not readable else "written at the referent tokens" if S["ref_off"] >= WRITTEN_MIN and S["rest_off"] <= REST_MAX
               else "not" if S["ref_off"] <= NOT_MAX else "partial")
    summary = {"run_id": args.run_id, "step": args.step, "model": cfg["model"]["id"], "form": args.form, "n": len(recs),
               "referent_tokens_mean": float(np.mean(n_ref)), "accuracy": acc, "share_restored": S, "same_as_unhooked": same,
               "readable": readable, "verdict": verdict}
    log("accuracy: " + ", ".join(f"{c} {acc[c]:.2f}" for c in CONDS) + f" | unhooked θu {acc['plain_u']:.2f}, adapter disabled {acc['plain_0']:.2f}")
    log("share restored: " + ", ".join(f"{c} {S[c]:+.2f}" for c in CONDS[2:]) + f" | same responses as unhooked: u {same['u']:.2f}, 0 {same['0']:.2f}")
    atomic_write_json(out / "summary.json", summary)
    if min(same.values()) < SAME_MIN and not args.max_facts:
        raise RuntimeError(f"the switch does not reproduce the references: {same}")
    log(f"verdict: {verdict}")
    log(f"wrote {out / 'summary.json'}")


if __name__ == "__main__":
    main()
