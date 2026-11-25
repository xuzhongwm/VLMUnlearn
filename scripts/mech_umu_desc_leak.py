"""UMU track, DEV-038: does a text-only edit leak inside text? The person described without the name.

The text-only edit is bound to the name tokens (DEV-020/033/035). A text question that reaches the person without the name
should then escape it, as the portrait does. DEV-037: both vanillas find a person from a list of attributes.

Facts: for the forget profiles and the first --n-retain retain profiles, the UM_QA questions on Born, Birth, Occupation,
Height, Education, Annual_Salary, Residence. Forms (text only, generation without options):
  name        the UM_QA question (the form the unlearning trained on)
  desc        the person's other attributes as a list, then the "this person" question
  name+desc   the same list, then the name question
  desc_other  another person's attributes, then the "this person" question (control)
Score: the response contains the UM_QA answer as whole words (case-insensitive). Scored facts: those theta_0 (the same
model, LoRA off) answers under both name and desc.
Reading (DEV-038), on the scored forget facts: leak inside text if desc >= 0.85 and desc - name >= 0.20; no leak if
desc <= name + 0.10; otherwise partial.
Validity: at least 40 scored forget facts; with --ref <summary.json of a --step 0 run>, theta_0's desc accuracy on all
forget facts must lie within 0.10 of the plain vanilla's.

  python scripts/mech_umu_desc_leak.py --config <model config> --run-id <run> --step <N> [--ref <path>] [--set umu.ckpt=...]
  --step 0: the plain vanilla only (reference; no LoRA is loaded).
Writes results/<model>/mech_umu/<run>/step<N>/desc_leak/{summary.json,items.jsonl,gen_*.jsonl,log.txt}.
"""
import ast
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from vlmu.config import base_parser, resolve  # noqa: E402
from vlmu.models import get_adapter  # noqa: E402
from vlmu.runs import free, model_dir, results_dir  # noqa: E402
from vlmu.umu import generate_protocol, load_umu, parse  # noqa: E402
from vlmu.utils import Logger, atomic_write_json, read_json, write_jsonl  # noqa: E402

FIELDS = ("Born", "Birth", "Occupation", "Height", "Education", "Annual_Salary", "Residence")
LABEL = {"Born": "Place of birth", "Birth": "Date of birth", "Gender": "Gender", "Occupation": "Occupation", "Height": "Height",
         "Education": "Education", "Annual_Salary": "Annual salary", "Residence": "Residence",
         "Medical_Conditions": "Medical conditions"}
FORMS = ("name", "desc", "name+desc", "desc_other")
LEAK_DESC, LEAK_GAP, NOLEAK_GAP, MIN_SCORED, REF_TOL = 0.85, 0.20, 0.10, 40, 0.10      # DEV-038


def describe(bio, skip):
    return "Here are some details about a person.\n" + "\n".join(f"{LABEL[k]}: {bio[k]}" for k in LABEL if k != skip and bio.get(k)) + "\n"


def qa(row, col):
    d = row[col]
    return ast.literal_eval(d) if isinstance(d, str) else d


def build_facts(dfs):
    facts = []
    for set_name, df in dfs.items():
        people = [(row["ID"], parse(row["Biography"]), qa(row, "UM_QA"), qa(row, "MM_QA")) for _, row in df.iterrows()]
        for i, (pid, bio, um, mm) in enumerate(people):
            for key in FIELDS:
                if um["answer"].get(key) is None or not mm["question"].get(key):
                    continue
                other = next((people[(i + j) % len(people)][1] for j in range(1, len(people))
                              if str(people[(i + j) % len(people)][1].get(key)).strip().lower() != str(bio.get(key)).strip().lower()))
                d, do = describe(bio, key), describe(other, key)
                facts.append({"set": set_name, "profile": pid, "field": key, "pair_id": f"umu/{pid}/{key}",
                              "correct_text": str(um["answer"][key]),
                              "text": {"name": um["question"][key], "desc": d + mm["question"][key],
                                       "name+desc": d + um["question"][key], "desc_other": do + mm["question"][key]}})
    return facts


def contains(resp, ans):
    n = lambda s: re.sub(r"[^a-z0-9$]+", " ", s.lower()).strip()  # noqa: E731
    return re.search(r"(^| )" + re.escape(n(ans)) + r"( |$)", n(resp)) is not None


def main():
    p = base_parser(__doc__)
    p.add_argument("--run-id", required=True)
    p.add_argument("--step", type=int, required=True, help="0 = the plain vanilla (reference)")
    p.add_argument("--n-retain", type=int, default=50)
    p.add_argument("--ref", default=None, help="summary.json of a --step 0 run of the same vanilla")
    p.add_argument("--batch", type=int, default=32)
    p.add_argument("--max-profiles", type=int, default=0, help="smoke test")
    args = p.parse_args()
    cfg = resolve(args)
    out = results_dir(cfg) / "mech_umu" / args.run_id / f"step{args.step}" / "desc_leak"
    log = Logger(out / "log.txt")
    dfs = load_umu(cfg["paths"]["root"], args.n_retain, 0, log, cfg["umu"]["forget_split"])
    dfs = {"forget": dfs["forget"], "retain": dfs["retain"].head(args.n_retain)}
    if args.max_profiles:
        dfs = {k: v.head(args.max_profiles) for k, v in dfs.items()}
    facts = build_facts(dfs)
    adapter = get_adapter({**cfg["model"], "id": cfg["umu"]["ckpt"], "processor_id": cfg["model"]["id"]})
    model = adapter.load_model()
    if args.step:
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, str(model_dir(cfg) / "umu_unlearn" / args.run_id / "ckpt" / f"step{args.step}"),
                                          is_trainable=False)
    model.eval()
    log(f"{args.run_id} step {args.step} ({cfg['model']['id']}): {len(facts)} facts "
        f"({sum(f['set'] == 'forget' for f in facts)} forget); forms {FORMS}")
    log("example desc prompt: " + repr(adapter.prompt(facts[0]["text"]["desc"], False)) + " -> " + repr(facts[0]["correct_text"]))

    def run(tag, forms):
        hit = {}
        for form in forms:
            rows = [{**{k: f[k] for k in ("set", "profile", "field", "pair_id", "correct_text")}, "temp": [f["correct_text"]],
                     "modality": "txt", "image_bytes": None, "uid": f"{f['pair_id']}/{form}",
                     "prompt": adapter.prompt(f["text"][form], False)} for f in facts]
            res = generate_protocol(model, adapter.processor, rows, out / f"gen_{tag}_{form}.jsonl", log, batch_size=args.batch,
                                    dtype=adapter.dtype)
            hit[form] = {r["pair_id"]: (contains(r["response"], r["correct_text"]), r["response"]) for r in res}
        return hit

    if args.step:
        h0 = None
        with model.disable_adapter():
            h0 = run("0", ("name", "desc", "name+desc"))
        hu = run("u", FORMS)
    else:
        h0, hu = run("0", FORMS), None
    model = None
    free()
    acc = lambda h, form, fs: sum(h[form][f["pair_id"]][0] for f in fs) / max(len(fs), 1)  # noqa: E731
    summary = {"run_id": args.run_id, "step": args.step, "model": cfg["model"]["id"], "n_facts": len(facts)}
    items = []
    for s in ("forget", "retain"):
        allf = [f for f in facts if f["set"] == s]
        scored = [f for f in allf if h0["name"][f["pair_id"]][0] and h0["desc"][f["pair_id"]][0]]
        summary[s] = {"n_all": len(allf), "n_scored": len(scored),
                      "theta0_all": {form: acc(h0, form, allf) for form in h0},
                      "theta0_scored": {form: acc(h0, form, scored) for form in h0}}
        if hu:
            summary[s]["thetau_all"] = {form: acc(hu, form, allf) for form in hu}
            summary[s]["thetau_scored"] = {form: acc(hu, form, scored) for form in hu}
        ids = {f["pair_id"] for f in scored}
        for f in allf:
            rec = {"pair_id": f["pair_id"], "set": s, "field": f["field"], "answer": f["correct_text"], "scored": f["pair_id"] in ids}
            for tag, h in (("0", h0), ("u", hu)):
                if h:
                    for form in h:
                        rec[f"{tag}/{form}/hit"], rec[f"{tag}/{form}/response"] = h[form][f["pair_id"]]
            items.append(rec)
    write_jsonl(out / "items.jsonl", items)
    for s in ("forget", "retain"):
        z = summary[s]
        log(f"  {s}: {z['n_scored']}/{z['n_all']} scored | θ0 all " + " ".join(f"{k} {v:.2f}" for k, v in z["theta0_all"].items()))
        if hu:
            log(f"  {s}: θu on scored " + " ".join(f"{k} {v:.2f}" for k, v in z["thetau_scored"].items())
                + " | θu all " + " ".join(f"{k} {v:.2f}" for k, v in z["thetau_all"].items()))
    if args.ref:
        ref = read_json(args.ref)["forget"]["theta0_all"]["desc"]
        d = abs(summary["forget"]["theta0_all"]["desc"] - ref)
        log(f"  check: θ0 (LoRA off) desc accuracy vs the plain vanilla's {ref:.3f}: |diff| = {d:.3f} (tolerance {REF_TOL})")
        if d > REF_TOL:
            raise RuntimeError("the LoRA-disabled model does not reproduce the plain vanilla on the desc form")
    if hu:
        z = summary["forget"]
        if z["n_scored"] < MIN_SCORED and not args.max_profiles:
            raise RuntimeError(f"only {z['n_scored']} scored forget facts (< {MIN_SCORED})")
        d, n = z["thetau_scored"]["desc"], z["thetau_scored"]["name"]
        summary["verdict"] = ("leak inside text" if d >= LEAK_DESC and d - n >= LEAK_GAP else
                              "no leak" if d <= n + NOLEAK_GAP else "partial")
        log(f"DEV-038: forget, scored facts: name {n:.2f}, desc {d:.2f}, name+desc {z['thetau_scored']['name+desc']:.2f}, "
            f"desc_other {z['thetau_scored']['desc_other']:.2f} -> {summary['verdict']}")
    atomic_write_json(out / "summary.json", summary)
    log(f"wrote {out / 'summary.json'}")


if __name__ == "__main__":
    main()
