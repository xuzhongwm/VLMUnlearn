"""Vanilla diagnostics: why is paired MC accuracy low? Scores the vanilla and the pretrained
base model on the same paired facts under question variants. Each variant removes one
difference between training and evaluation:

  eval_stem       the eval items as they are (img: "the pictured person", txt: the name)
  eval_stem_this  img only: the eval stem with the training referent "this person"
  train_stem      the fact's own FT question (training wording), same options

Each variant is scored as MC (letter logits, original option order) and open-ended (greedy
generation; keyword hit = the correct option text occurs in the generation). Facts: forget
(all 50 entities) + retain calibration, restricted to facts whose FT question is found in both
forms. Reads the vanilla adapter; writes only runs/<model>/vanilla/diagnostics/.

Reading the table:
  eval_stem_this >> eval_stem (img)   the referent change is what hurts the image form
  train_stem >> eval_stem             facts are learned but only retrievable with the training wording
  train_stem ~ pretrained             the facts were not learned at all

  python scripts/diagnose_vanilla.py --config configs/qwen2vl/vanilla.yaml [--no-pretrained] [--debug]
"""
import collections
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from vlmu.config import base_parser, resolve  # noqa: E402
from vlmu.evaluate import load_items, score, score_open  # noqa: E402
from vlmu.models import get_adapter  # noqa: E402
from vlmu.runs import free, model_dir, processed_dir, results_dir, vanilla_adapter_dir  # noqa: E402
from vlmu.utils import Logger, atomic_write_json, is_done, mark_done, read_jsonl, tracked_run  # noqa: E402

SUITE = [("forget", None, "paired", ("img", "txt")), ("retain", "calibration", "paired", ("img", "txt"))]
REFERENT_SWAP = [("the pictured person's", "this person's"), ("the pictured person", "this person")]
VARIANTS = ("eval_stem", "eval_stem_this", "train_stem")
KINDS = {"mc": "correct", "open": "keyword_hit"}


def this_referent(q):
    for old, new in REFERENT_SWAP:
        if old in q:
            return q.replace(old, new)
    raise ValueError(f"no image referent in {q!r}")


def train_questions(processed):
    """(profile, img FT answer) -> {form: FT question}. Paired items store the img FT answer of
    their fact (ft_answer); the txt row of the same fact shares its fact_id."""
    by_fact = collections.defaultdict(dict)
    for r in read_jsonl(Path(processed) / "train_vanilla.jsonl"):
        by_fact[r["fact_id"]][r["form"]] = r
    out = {}
    for f in by_fact.values():
        if "img" in f and "txt" in f:
            out.setdefault((f["img"]["profile"], f["img"]["answer"]), {m: f[m]["question"] for m in ("img", "txt")})
    return out


def build_variants(items, tq, log):
    keep = {r["pair_id"] for r in items if (r["profile"], r["ft_answer"]) in tq}
    log(f"facts with an FT question in both forms: {len(keep)} / {len({r['pair_id'] for r in items})}")
    out = {v: [] for v in VARIANTS}
    for r in items:
        if r["pair_id"] not in keep:
            continue
        out["eval_stem"].append({**r, "uid": f"{r['uid']}/eval_stem"})
        if r["modality"] == "img":
            out["eval_stem_this"].append({**r, "uid": f"{r['uid']}/eval_stem_this",
                                          "question": this_referent(r["question"])})
        out["train_stem"].append({**r, "uid": f"{r['uid']}/train_stem",
                                  "question": tq[(r["profile"], r["ft_answer"])][r["modality"]]})
    return out


def score_model(cfg, tag, vanilla_dir, var_items, processed, out_root, log):
    out = out_root / tag
    files = {(k, v): out / f"{k}_{v}.jsonl" for k in KINDS for v in VARIANTS}
    if is_done(out):
        log(f"{tag}: already scored")
        return {kv: read_jsonl(f) for kv, f in files.items()}
    adapter = get_adapter(cfg["model"])
    model = adapter.load_model(vanilla_dir)
    res = {}
    for v, items in var_items.items():
        res[("mc", v)] = score(adapter, model, items, processed, cfg["eval"], files[("mc", v)], log)
        orows = [{**r, "reference": r["options"][r["answer"]]} for r in items]
        res[("open", v)] = score_open(adapter, model, orows, processed, cfg["eval"], files[("open", v)], log)
    model = None
    free()
    mark_done(out)
    return res


def summarize(res):
    """{kind/variant/modality/group: {acc, n}}; group = pooled, forget, retain, field:<name>."""
    out = {}
    for (kind, v), rows in res.items():
        g = collections.defaultdict(list)
        for r in rows:
            for grp in ("pooled", r["set"], f"field:{r['field']}"):
                g[(r["modality"], grp)].append(float(r[KINDS[kind]]))
        for (m, grp), xs in g.items():
            out[f"{kind}/{v}/{m}/{grp}"] = {"acc": float(np.mean(xs)), "n": len(xs)}
    return out


def print_table(summ, groups, log, kinds=tuple(KINDS)):
    tags = [t for t in ("pretrained", "vanilla") if t in summ]
    log(f"{'kind':5s} {'variant':15s} {'mod':4s} {'group':18s} {'n':>5s} " + " ".join(f"{t:>10s}" for t in tags)
        + ("    gain" if len(tags) == 2 else ""))
    for kind in kinds:
        for v in VARIANTS:
            for m in ("img", "txt"):
                for grp in groups:
                    key = f"{kind}/{v}/{m}/{grp}"
                    if key not in summ["vanilla"]:
                        continue
                    accs = [summ[t][key]["acc"] for t in tags]
                    gain = f" {accs[1] - accs[0]:+.3f}" if len(tags) == 2 else ""
                    log(f"{kind:5s} {v:15s} {m:4s} {grp:18s} {summ['vanilla'][key]['n']:5d} "
                        + " ".join(f"{a:10.3f}" for a in accs) + gain)


def main():
    p = base_parser(__doc__)
    p.add_argument("--no-pretrained", action="store_true", help="skip the base-model comparison")
    args = p.parse_args()
    cfg = resolve(args)
    out_root = model_dir(cfg) / "vanilla" / "diagnostics"
    log = Logger(out_root / "log.txt")
    vdir = vanilla_adapter_dir(cfg)
    if not (vdir / "adapter_config.json").exists():
        raise RuntimeError(f"no vanilla adapter at {vdir}: run scripts/finetune_vanilla.py first")
    with tracked_run(results_dir(cfg), out_root, f"{cfg['model']['name']}_vanilla_diagnostics", "diagnose_vanilla"):
        processed = processed_dir(cfg, log)
        items = load_items(processed, SUITE, cfg["eval"].get("max_per_group"))
        var_items = build_variants(items, train_questions(processed), log)
        res = {"vanilla": score_model(cfg, "vanilla", vdir, var_items, processed, out_root, log)}
        if not args.no_pretrained:
            res["pretrained"] = score_model(cfg, "pretrained", None, var_items, processed, out_root, log)
        summ = {t: summarize(r) for t, r in res.items()}
        atomic_write_json(out_root / "summary.json", summ)
        log("=== pooled (forget + retain calibration) and forget only")
        print_table(summ, ("pooled", "forget"), log)
        log("=== MC per field, pooled")
        fields = sorted({k.split("/")[3] for k in summ["vanilla"] if k.split("/")[3].startswith("field:")})
        print_table(summ, fields, log, kinds=("mc",))
        log(f"wrote {out_root / 'summary.json'}")


if __name__ == "__main__":
    main()
