"""Normalised forgetting, asymmetry, and profile-cluster bootstrap CIs."""
import collections

import numpy as np

CHANCE = 25.0


def nd(vanilla_acc, unlearned_acc, chance=CHANCE):
    """(vanilla - unlearned) / (vanilla - chance). 1 = forgotten to chance, 0 = no change."""
    denom = vanilla_acc - chance
    return float("nan") if denom <= 0 else (vanilla_acc - unlearned_acc) / denom


def _per_profile(results, set_name, source, modality):
    """profile -> (n_correct, n) for one eval group."""
    d = collections.defaultdict(lambda: [0, 0])
    for r in results:
        if r["set"] == set_name and r["source"] == source and r["modality"] == modality:
            d[r["profile"]][0] += int(r["correct"])
            d[r["profile"]][1] += 1
    return d


def bootstrap_nd(vanilla_res, unlearned_res, unlearn_modality, eval_modalities=("txt", "img", "pose"),
                 set_name="forget", source="paired", n_boot=2000, seed=0):
    """ND per eval modality and the asymmetry A = ND(in-modality) - ND(cross-modality) over
    the text-only/image+text pair, with 95% CIs from resampling forget profiles (items within
    a profile are correlated)."""
    groups = {}
    for m in eval_modalities:
        v, u = _per_profile(vanilla_res, set_name, source, m), _per_profile(unlearned_res, set_name, source, m)
        groups[m] = (v, u)
    profiles = sorted(set.intersection(*[set(v) & set(u) for v, u in groups.values()]))
    arr = {}
    for m, (v, u) in groups.items():
        arr[m] = np.array([[v[p][0], v[p][1], u[p][0], u[p][1]] for p in profiles], dtype=float)

    def stats(idx):
        out = {}
        for m, a in arr.items():
            s = a[idx].sum(0)
            out[m] = nd(100 * s[0] / s[1], 100 * s[2] / s[3])
        cross = "img" if unlearn_modality == "txt" else "txt"
        out["asym"] = out[unlearn_modality] - out[cross]
        return out

    point = stats(np.arange(len(profiles)))
    rng = np.random.default_rng(seed)
    boots = [stats(rng.integers(0, len(profiles), len(profiles))) for _ in range(n_boot)]
    ci = {k: (float(np.nanpercentile([b[k] for b in boots], 2.5)), float(np.nanpercentile([b[k] for b in boots], 97.5)))
          for k in point}
    return {"point": point, "ci95": ci, "n_profiles": len(profiles)}
