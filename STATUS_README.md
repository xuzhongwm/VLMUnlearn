# STATUS — Stage 1, 2026-10-29 (update 2)

**Safe point.** No job is running, and no GPU run will start until you approve one.

**GPU hours used: 0.** All tests so far ran on CPU on this Mac, with random-weight tiny models.
Their numbers test code paths only.

**What changed since update 1** (your answers to §6):
- the pipeline is realigned to the new protocol;
- Milestone J is implemented;
- Qwen2.5-VL support is added;
- the E0 data audit is written;
- the smoke-test log is now archived, not deleted.

**Waiting on you:**
1. Add `PROTOCOL.md` + Addendum, so I can check F/G/H and the 11 provisional interpretations against §4.2/§4.3/§5.1 ([DEVIATIONS.md](DEVIATIONS.md)).
2. Push the repo, set `REPO_URL`, and run the Colab smoke test (notebook cells 1–3).

---

## 1. Research question and hypothesis

Unchanged from update 1. Stage 1 asks whether a fact unlearned through text-only questions remains
answerable through image+text questions about the same fact, given a vanilla that knows the fact
in both forms.

The primary statistic is **G = Acc_I^U − Acc_T^U**: image+text accuracy minus text-only accuracy of
the unlearned model. It is computed on the frozen doubly-correct facts of the 30 confirmation
entities, at the earliest calibration checkpoint inside the [0.30, 0.40] window.

## 2. Design in effect

**Governing protocol:** PROTOCOL.md + Addendum ([DEVIATIONS.md](DEVIATIONS.md) DEV-001).

| item | implementation | where |
|---|---|---|
| Splits (§4.2) | forget 50 = **20 calibration / 30 confirmation**, all 50 unlearned. Retain 450 = **300 unlearning-retain / 50 calibration / 100 confirmation**. Seed 0, frozen in `splits.json`. | `configs/data.yaml`, `vlmu/data/build.py` |
| Paired items | 9 attributes; identical options, order and answer across text (name), image (portrait + "the pictured person", DEV-002) and pose forms. **Correct positions assigned by stratification** (per split and per field, max − min ≤ 1). Paraphrased stems (max Jaccard 0.36 vs FT wording). | `vlmu/data/build.py`, `fields.py` |
| Vanilla | Own LoRA fine-tune on both forms. Gate ≥ 0.80 in both modalities (all 50 forget entities, primary forms). **The doubly-correct set is frozen** after the vanilla eval (written once), with coverage per split, entity and attribute. | `vlmu/runs.py:run_vanilla`, `freeze_doubly_correct` |
| Unlearning | Fresh LoRA on the frozen merged vanilla, LLM only. Loss on answer tokens; natural-language targets. **Core = GA_Diff (primary), NPO (second)**; GA code kept but not in the grid (DEV-003). lr ∈ {2e-5, 5e-5, 1e-4} × seeds {0, 1} × unlearning modality {txt (primary), img (secondary, DEV-004)}. | `configs/*/grid.yaml` |
| Checkpoint rule (F) | **Earliest** calibration checkpoint with in-modality accuracy in [0.30, 0.40] (doubly-correct calibration forget facts) and retain drop ≤ 0.10 (doubly-correct calibration retain facts, max over modalities). A run stops there. Ineligible runs are reported with a reason (`passed_window_without_eligible_checkpoint`, `retain_drop_exceeded`, `window_not_reached`). | `vlmu/runs.py:checkpoint_record`, `select` |
| Metric (G) | G on doubly-correct confirmation facts, original option order. Letter-logit scoring; **margin m = z_correct − logsumexp(z_wrong)**; **permutation check**: all 4 cyclic shifts, reporting accuracy per shift and agreement. | `vlmu/metrics.py`, `vlmu/evaluate.py` |
| GO (H) | GA_Diff, text unlearning: in **both seeds**, G ≥ 0.10 **and** the paired entity-bootstrap 95% CI lower bound > 0. | `vlmu/report.py:decide` |
| Models | Qwen2-VL-2B = `pipeline_check` (never confirmatory). LLaVA-1.5-7B = primary. Qwen2.5-VL-7B = replication (supported; replication phase only). | `configs/models/*.yaml` |
| Milestone J | (i) adapter-off = vanilla for a **trained** adapter; (ii) hook-based contribution removal = PEFT local disabling (layer window, and all modules); (iii) self-replacement: decoder-layer patching and y = F_U(H) − d(H) + d(H_src) with H_src = H, at all positions and the last position. | `vlmu/mech/`, `scripts/check_hooks.py` |

The 11 provisional interpretations (P-1 … P-11) are in [DEVIATIONS.md](DEVIATIONS.md). The ones
that matter most:
- **P-1:** lr choice = the lowest lr eligible in every seed.
- **P-3:** retain drop = max over modalities, on doubly-correct calibration retain facts.
- **P-5:** NO-GO / UNCLEAR / INCOMPLETE definitions.

## 3. DONE

**Commits:**
- `096eedd`: PLAN.md.
- `a14952d`: first pipeline.
- This update (commit hash in the final message).

**Files changed in this update:**
- `vlmu/data/build.py`, `vlmu/data/sanity.py`: splits, balanced positions, official-item answer check.
- `vlmu/evaluate.py`: margin, shifts, per-split groups.
- `vlmu/metrics.py`: G and paired entity bootstrap.
- `vlmu/runs.py`: doubly-correct set, checkpoint rule, selection.
- `vlmu/report.py`: G table, GO rule, figure.
- `vlmu/models.py`: Qwen2.5-VL adapter.
- `vlmu/mech/{hooks,checks}.py`: Milestone J.
- `scripts/`:
  - `select_and_eval.py` replaces `select_and_seed.py`;
  - new: `check_hooks.py`, `data_audit.py`;
  - updated: `sanity.py`, `smoke_test.py` (archives), `make_run_configs.py`.
- `configs/`:
  - updated: `data.yaml`, `models/*`;
  - new: `qwen25vl/*`;
  - 72 run YAMLs (24 per model).
- `colab/run.ipynb`, `README.md`, `DEVIATIONS.md`, `analysis/data_audit.md`, `analysis/official_mc_attributes.csv`.

**Real-data build** (CPU; all **23/23** data checks pass):

| quantity | value |
|---|---|
| forget calibration / confirmation entities | 20 / 30 |
| retain unlearning / calibration / confirmation entities | 300 / 50 / 100 |
| paired items per modality, forget | 446 |
| paired items per modality, retain eval | 1,338 |
| unlearning QA per form | 720 forget; 4,328 retain-unlearning |
| official MC | forget 250+250; retain 749+750; celeb 766+765 |
| MMBench items | 500 |

Dropped and logged: 13 missing/NA, 2 unparseable FT answers, 1 with too few distractors, 1 text
rewrite, and 1 official item whose answer is not an option (profile 028).

**E0 data audit** ([analysis/data_audit.md](analysis/data_audit.md); facts only):
- The vanilla training data is photo + referent only. "this person" occurs 7,430 times; there is no text-only form.
- Official MC: **28.4%** of forget_10 text-only questions (29.3% over all 500 profiles) ask about an attribute that the same profile's image+text questions also ask about.
  - The audit uses rule-based attribute labels.
  - Spot check on a fresh sample: 58/60 fictitious labels correct, 29/30 celebrity labels correct.
- Image MC questions almost never ask about parents (0 of 250 in forget_10), while text MC questions often do (56 of 250).
- Biography values agree with the FT answers for ≥ 99% of profiles on every paired field. The only "not stated" cases are 2 heights and 3 fathers.

**Local CPU verification** (tiny random models):
- **Smoke tests:** SMOKE_PLACEHOLDER
- **Milestone J on tiny Qwen2-VL:**
  - the trained adapter changes the logits (relative difference 8.8e-3);
  - (i) adapter-off = vanilla exactly (difference 0);
  - (ii) removal = local disabling (difference ≤ 2.6e-7);
  - (iii) both self-replacements are exact (difference 0).

  Tolerance 1e-5. Results for the other models: MJ_PLACEHOLDER
- **Kill-and-resume test of the new unlearning loop:** RESUME_PLACEHOLDER
- **Qwen2.5-VL:** the public tiny test model is saved in a newer transformers format and does not load under the 4.51.3 pin. The real Qwen2.5-VL-7B config (saved with 4.41.2) does load. I built a tiny random Qwen2.5-VL from the real config with 4.51.3 and used it for testing.

**Failed attempts this update** (all local CPU; nothing dropped silently):
1. Smoke test: pyflakes flagged a `datetime` name clash in `smoke_test.py` before the run. Fixed.
2. Smoke test run 1 (tiny Qwen2-VL): every job passed, but the verdict was INCOMPLETE. The debug item cap took calibration items only, so confirmation was never scored. The cap is now applied per split.
3. Data audit, first run: crashed on profile 028, whose official answer is not an option. The fix also prevents a latent crash in margin scoring.
4. The public tiny Qwen2.5-VL test model does not load under the pin (see above).

Run logs are in `debug/results/run_log.csv`. Earlier debug outputs are under `debug/archive/<timestamp>/`.

## 4. In progress / next

Nothing is running. The next steps, with your approval:
1. You add PROTOCOL.md; I reconcile P-1…P-11.
2. You run the Colab smoke test (cells 1–3, `MODEL = "qwen2vl"`, then `"llava"`) and send me the output.
3. Phase B: the primary run on LLaVA, cells 4–9.

## 5. GPU hours

**0.** Local CPU: about 2 h of tests in total.

## 6. Open questions

1. PROTOCOL.md §4.3/§5.1 vs P-1…P-11. In particular: lr selection across the grid, and the definitions of NO-GO/UNCLEAR.
2. Should Qwen2-VL-2B run the full 24-run grid, or a reduced one as a pipeline check (e.g. GA_Diff/txt, one lr, one seed)? The reduced version saves about 3 h of A100 time.
3. Runtime estimates now assume early stopping at the window. They will be recalibrated from the Colab smoke test.

## Alignment check (update of Part 2)

| | item | status |
|---|---|---|
| A | Data audit (E0) | **ALIGNED** — `analysis/data_audit.md`: neutral facts plus a semantic per-attribute overlap audit with a spot check. |
| B | Paired questions | **ALIGNED** with DEV-002. The image referent is "the pictured person"; positions are balanced by construction. |
| C | Vanilla | **ALIGNED** — both forms, gate ≥ 0.80, doubly-correct set frozen, coverage per entity and attribute (not yet run). |
| D | Splits | **ALIGNED** — 20/30 and 300/50/100, frozen; selection on calibration only. |
| E | Unlearning | **ALIGNED** — GA_Diff primary, NPO second; GA out of the core grid. |
| F | Checkpoint rule | **ALIGNED with the instructions; provisional vs PROTOCOL** (P-1, P-3, P-4, P-7). |
| G | Primary metric | **ALIGNED** — G, margin, 4-shift permutation check. |
| H | GO rule | **ALIGNED with the instructions; provisional vs PROTOCOL** (P-5). |
| I | Models | **ALIGNED** — roles set; Qwen2.5-VL supported (tested on a self-built tiny model). |
| J | Milestone J | **ALIGNED** — (i)–(iii) with hooks, passing on tiny models with a trained adapter. |
| K | Later phases | **NOT STARTED** (as intended). |
