"""Primary metric G and paired entity-bootstrap CIs (accuracies are fractions in [0, 1]).

G = Acc_I^U - Acc_T^U on the frozen doubly-correct set of the confirmation forget entities:
image+text accuracy minus text-only accuracy of the unlearned model, primary eval forms only
(training portrait, original option order). Both accuracies are computed on the same facts, so
the bootstrap resamples entities once and recomputes both (paired).
"""
import collections

import numpy as np

CHANCE = 0.25


def entity_table(results, pair_ids, set_name="forget", split="confirmation", modalities=("img", "txt", "pose")):
    """entity -> {modality: [n_correct, n]} over doubly-correct pairs, shift 0 only."""
    t = collections.defaultdict(lambda: {m: [0, 0] for m in modalities})
    for r in results:
        if (r["source"] == "paired" and r["set"] == set_name and r["split"] == split and r.get("shift", 0) == 0
                and r["modality"] in modalities and r["pair_id"] in pair_ids):
            c = t[r["profile"]][r["modality"]]
            c[0] += int(r["correct"])
            c[1] += 1
    return dict(t)


def g_bootstrap(results, pair_ids, n_boot=2000, seed=0, set_name="forget", split="confirmation"):
    t = entity_table(results, pair_ids, set_name, split)
    ents = sorted(t)
    arr = np.array([[t[e][m][k] for m in ("img", "txt", "pose") for k in (0, 1)] for e in ents], dtype=float)
    if not len(ents):
        return None

    def stats(a):
        s = a.sum(0)
        acc = {m: s[2 * i] / s[2 * i + 1] if s[2 * i + 1] else float("nan") for i, m in enumerate(("img", "txt", "pose"))}
        return {"acc_img": acc["img"], "acc_txt": acc["txt"], "acc_pose": acc["pose"],
                "G": acc["img"] - acc["txt"], "G_reverse": acc["txt"] - acc["img"]}

    point = stats(arr)
    rng = np.random.default_rng(seed)
    boots = [stats(arr[rng.integers(0, len(ents), len(ents))]) for _ in range(n_boot)]
    ci = {k: (float(np.nanpercentile([b[k] for b in boots], 2.5)), float(np.nanpercentile([b[k] for b in boots], 97.5)))
          for k in point}
    return {"point": point, "ci95": ci, "n_entities": len(ents), "n_pairs": int(arr[:, 1].sum())}


def normalised_drop(acc_unlearned, acc_vanilla=1.0, chance=CHANCE):
    """Secondary: (vanilla - unlearned) / (vanilla - chance). On the doubly-correct set vanilla = 1."""
    d = acc_vanilla - chance
    return float("nan") if d <= 0 else (acc_vanilla - acc_unlearned) / d
