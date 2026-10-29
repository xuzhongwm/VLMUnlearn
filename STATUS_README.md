# STATUS — Stage 1, 2026-10-29

**Stopped at a safe point.**
- No job is running and none will start until you reply.
- Last commit: `a14952d`.
- Uncommitted in the working tree:
  - `STATUS_README.md` (this file);
  - `new_plan.md` and `research.md`, your files, not modified.

**GPU hours used: 0.** Every run so far was a local CPU test on this Mac, using random-weight tiny
models. Those runs tested the code paths only; none of their numbers carry scientific meaning.

---

## Part 1 — Current understanding

### 1. Research question and hypothesis

**Research question.** Why does unlearning in a VLM leak across modalities? After a fact about
person X is unlearned through text-only questions, does the model still give it when the question
comes with X's photo? Later stages will localise where the text and image routes converge.

**Stage 1 hypothesis (existence only).** At matched in-modality forgetting, text-only unlearning
produces a smaller forget-accuracy drop on image+text queries than on text-only queries. The
same facts are asked in both forms, and the vanilla has learned both forms.

**Later stages (not touched).** The mechanistic hypothesis in `new_plan.md` is H1: a late-layer
gate triggered by input form. No mechanistic code exists.

### 2. Design decisions in effect, as implemented in `a14952d`

All of these come from `PLAN.md` (D1–D9) plus implementation choices made inside them.

**Data**
- Source: MLLMU-Bench from HF.
- Vanilla training data: `ft_Data`, 500 fictitious profiles with 14–19 QA each. The
  "Additional" description QA is skipped, as the official loader does.
- Text form: built by deterministic substitution of "this person" / "the person (in the image)"
  with the profile name. QAs whose rewrite fails are dropped and logged.

**Splits**
- forget = `forget_10` (50 profiles; IDs checked against `baseline_train_split/forget_10/forget_ids.json`).
- retain = the other 450 profiles. From these:
  - **50 retain-eval** profiles, drawn at random with seed 0 (excluding profile 159, which has no official MC);
  - **400 retain-train** profiles (GA_Diff retain data).
- celebrities: 153, official MC only.
- utility: MMBench-EN dev, 500 items (seed 0).
- There is **no calibration/confirmation split**. All 50 forget profiles are used for selection and for reporting.

**Paired MC items** (`vlmu/data/build.py`, `fields.py`)
- Facts come from 9 biography fields: born, dob, profession, height, education, salary,
  residence, father, mother.
  - Parent occupations are taken from the FT answer, because the biography disagrees with the
    FT answer for about 30% of profiles.
  - Every other value must be stated in the FT answer; otherwise the item is dropped.
- Each item is rendered in three forms with identical options and order:
  - **text**: uses the name;
  - **image**: training portrait + **"the pictured person"** (not "this person");
  - **pose**: held-out Test_Set pose + "the pictured person".
- Distractors are other profiles' values for the same field, in the same format class. Exact or
  near duplicates are rejected, as are distractors the FT answer would also support.
- Answer position: a **random seeded permutation**. Balance is checked (±3 sd of uniform), not enforced.
- Eval stems are paraphrases. An automatic check confirms no exact match with any FT question;
  the maximum word-overlap (Jaccard) with FT questions is 0.36.

**Vanilla (our own, for both models)**
- LoRA r=16, α=16, dropout 0.05, on all LM linear layers of the public base model.
- Trained on both forms: 7,704 image QA + 7,203 text QA.
- lr 2e-4, 3 epochs, batch size 8, 3% warmup.
- Stored as an adapter and merged into the base at load time.
- **Gate:** paired forget MC accuracy ≥ 80% in both modalities, computed over *all* paired items.
  **No doubly-correct set is frozen.**

**Eval scoring**
- Zero-shot. The prediction is the argmax over the option-letter token logits at the answer position.
- Per item, the pipeline stores every option's log-prob, the probability mass on A–D, and whether
  the top-1 token is a letter.
- **Not implemented:** an answer-margin metric and an option-permutation check.

**Unlearning**
- A fresh LoRA adapter on the merged (frozen) vanilla, LM only; vision encoder and projector frozen.
- Loss on answer tokens only; the targets are the natural-language FT answers.
- Methods: **GA, GA_Diff, NPO, treated as equals** (no primary method).
  - GA_Diff interleaves forget and retain batches, with retain data in the same modality.
  - NPO uses the vanilla as reference (adapter disabled), β = 0.4, and the per-sample sum of answer log-probs.
- Batch size 4; lr ∈ {2e-5, 5e-5, 1e-4}; linear decay over 6 epochs; checkpoint every ½ epoch.
- Early stop when in-modality ND ≥ 1.0 or the retain drop exceeds 20 pp.

**Checkpoint selection**
- Done per (method, unlearning modality), over all seed-0 lr × checkpoint combinations.
- Feasible checkpoint: paired retain accuracy drop ≤ 10 pp in both modalities.
- Pick the feasible checkpoint whose **in-modality ND = (vanilla − unlearned) / (vanilla − 0.25)
  is closest to 0.8**. It counts as "matched" if within ±0.1.
- Selection uses the **same forget items** that are later reported.
- Seed 1 reruns the selected (lr, step).
- Unmatched and infeasible cells are reported, not dropped.

**GO rule**
- A = ND(T→T) − ND(T→I), with a 95% CI from a profile-cluster bootstrap (paired across modalities within a resample).
- **GO:** A > 0 with the CI excluding 0, for ≥ 2 of 3 methods, in both seeds.
- **UNCLEAR:** A > 0 for ≥ 2 methods in both seeds, but the CIs include 0.
- **NO-GO:** anything else.
- There is no minimum effect size.

**Models**
- Phase A: Qwen2-VL-2B. It is configured for the *full* protocol, not only pipeline checks.
- Phase B: LLaVA-1.5-7B.
- No Qwen2.5-VL adapter.

### 3. DONE

**Commits**
- `096eedd`: `PLAN.md`.
- `a14952d`: pipeline:
  - `vlmu/` (`config.py`, `utils.py`, `models.py`, `train.py`, `evaluate.py`, `metrics.py`, `runs.py`, `report.py`,
    `data/{raw,fields,rewrite,build,sanity}.py`);
  - `scripts/` (prepare_data, eval_pretrained, finetune_vanilla, unlearn, run_grid, select_and_seed,
    aggregate, sanity, smoke_test, make_run_configs);
  - `configs/` (data, models/*, `{qwen2vl,llava}/{vanilla,grid}.yaml`, 36 per-run YAMLs);
  - `colab/run.ipynb`, `README.md`, `requirements.txt`.
- No other files were changed.

**Real-data build** (CPU, local, real MLLMU-Bench and MMBench files)

| quantity | value |
|---|---|
| forget profiles | 50 |
| retain-eval profiles | 50 |
| retain-train profiles | 400 |
| celebrities | 153 |
| paired items per modality, forget | 446 |
| paired items per modality, retain-eval | 441 |
| forget answer letters (A/B/C/D) | 98 / 126 / 110 / 112 |
| unlearning QA per form | 720 forget, 5,766 retain-train |
| official MC per modality | forget 250, retain-eval 250, celeb 766 / 765 |
| MMBench items | 500 |

- Dropped: 12 paired items and 1 text rewrite, each with its reason logged in `build_report.json`:
  - 8 missing / NA;
  - 1 field with no FT QA;
  - 1 biography/FT mismatch;
  - 1 FT answer that couldn't be parsed;
  - 2 with too few valid distractors.
- All 21 data checks pass.

**Data-audit facts** (recorded in the conversation and in `PLAN.md` §1–2; there is no standalone audit file)
- The official vanilla training data (`ft_Data`) is **image-anchored only**: photo + "this
  person". There is no text-only form.
- Official MC: 5 image+text and 5 text-only questions per profile, about largely different facts.
  Matching on the correct-answer string, 27/125 text-MC answers on forget_5 and **46/250 on forget_10**
  also appear among the image-MC answers. This is a crude string match, not a semantic audit.
- The released LLaVA vanilla scores 51.7% on forget MC with both modalities pooled (paper Tab. 2).
  This was not measured by us.
- The Test_Set poses are distinct images from the training portraits (hash-checked). The FT
  images are byte-identical to the eval portraits.
- forget_5 is not a subset of forget_10.

**Local CPU runs** (tiny random models; pipeline tests only)
- The full `--debug` pipeline passed all 7 jobs for both tiny Qwen2-VL and tiny LLaVA.
- Kill-and-resume test: a run killed at step 6/8 resumed from its step-4 checkpoint. Its checkpoint
  metrics and adapter were identical to an uninterrupted run (max |ΔW| = 0).
- Adapter checks on the tiny models:
  - labels cover only answer tokens;
  - A–D are single tokens;
  - left padding is correct for scoring;
  - with the adapter disabled, a freshly initialised LoRA (B = 0) reproduces vanilla answer log-probs.
    This was **not** tested for a *trained* adapter.

**Failed runs and attempts** (all local CPU debug)
1. `prepare_data --debug` failed the letter-balance check (fixed 15% floor too strict at n = 69).
   Replaced with a ±3 sd binomial bound.
2. `run_grid --debug` (tiny Qwen): all 3 runs failed with a size mismatch:
   - `qwen2vl_ga_diff_img_lr2e-05_s0`
   - `qwen2vl_ga_txt_lr2e-05_s0`
   - `qwen2vl_npo_img_lr2e-05_s0`

   Cause: per-run configs dropped the CLI `--set model.id`, so the runs loaded the real 2B model against the tiny vanilla adapter. Fixed.
3. Resume test, attempt 1: my shell invocation was wrong; not a pipeline failure.
4. Resume test, attempt 2: debug overrides beat CLI overrides. Fixed so CLI wins.
5. Resume test, attempt 3: the resumed run stopped early. numpy bools had been serialised as the
   string "False", which is truthy. That would also have broken feasibility in selection. Fixed in
   the JSON writers.
6. Resume test, attempt 4: passed.

**Logging gap.** `smoke_test.py` clears `debug/results/` before each run, which deleted the
`run_log.csv` rows for items 1–2. They are recorded here instead. This needs fixing: archive the
log rather than delete it.

### 4. In progress / planned next

**Nothing is in progress.** The plan before this stop was:
1. You push the repo and run Colab cells 1–3 (setup, data, smoke test).
2. Calibrate the runtime estimates from the measured `s/step`.
3. Run Phase A (Qwen2-VL-2B), then Phase B (LLaVA).

### 5. GPU hours

**0** on Colab or anywhere else. Local CPU use was about 1 hour of tests.

### 6. Open questions / waiting on

1. **Which protocol governs?** Does the protocol in your message (A–K) supersede `PLAN.md` D1–D9?
   Several items differ from what is implemented; see Part 2.
2. **Checkpoint window for the image-unlearning condition.** F gives the [0.30, 0.40] window in
   terms of *text* accuracy, which fits text-only unlearning. For image+text unlearning, is the
   window on image accuracy? Or is that condition no longer part of the primary analysis?
3. **Exact definitions needed:**
   - "answer margin": is it log p(correct) − max log p(other)?
   - "option-permutation check": e.g. re-score with all 4 cyclic shifts, then report accuracy and agreement?
   - Is the doubly-correct set defined once per model, on the vanilla?
4. **Image-form referent.** B says photo + "this person". I used "the pictured person", to keep eval
   wording away from training wording. Should the referent be "this person", with only the rest of
   the stem paraphrased?
5. **GA.** Drop it, or keep it as secondary, now that GA_Diff is primary and NPO second?
6. **Milestone J.** Checks (ii) module-contribution removal and (iii) self-replacement need
   activation hooks. My Stage 1 brief said "no mechanistic analysis". Confirm these are in scope now.
7. **Qwen2.5-VL-7B replication** needs a new adapter. Which transformers version should it use? The
   current pin, 4.51.3, supports Qwen2.5-VL.
8. **Your actions:** the repo has no remote yet; push it and set `REPO_URL` in the notebook.

---

## Part 2 — Alignment against the agreed protocol

| | item | status | what differs / notes |
|---|---|---|---|
| A | Data audit (E0) | **DIFFERENT (partial)** | <ul><li>Done: repo code and HF data were inspected and the facts recorded (above).</li><li>No standalone audit artifact (`analysis/confound_check.md` does not exist).</li><li>The overlap statistic is a crude correct-answer string match, not a per-attribute audit.</li><li>`PLAN.md` §2 mixes the facts with interpretation ("plausibly causes the asymmetry"); they are not recorded as neutral facts.</li></ul> |
| B | Paired questions | **DIFFERENT** | <ul><li>Aligned: same (entity, attribute, answer); text form uses the name; identical options and order; eval wording differs from training (checked).</li><li>Differs: the image form uses "the pictured person", not "this person".</li><li>Differs: correct positions are random, not balanced (A 22%, B 28%, C 25%, D 25% on forget).</li></ul> |
| C | Vanilla | **DIFFERENT** | <ul><li>Aligned: own vanilla; both forms; ≥ 80% gate in both modalities (code exists, not run).</li><li>Missing: no doubly-correct fact set is frozen before unlearning.</li><li>Missing: no coverage report. All paired items are used.</li></ul> |
| D | Splits | **DIFFERENT** | <ul><li>Aligned: forget_10, with all 50 unlearned.</li><li>Missing: no 20 calibration / 30 confirmation split of the forget entities.</li><li>Retain is 400 unlearning-retain + 50 eval, instead of 300 / 50 calibration / 100 confirmation.</li><li>Splits are frozen (seed 0, `splits.json`).</li></ul> |
| E | Unlearning | **DIFFERENT (minor)** | <ul><li>Aligned: fresh LoRA on frozen vanilla, LM only, vision and projector frozen, answer-token loss, natural-language targets.</li><li>Differs: GA, GA_Diff and NPO run as equals in one grid. GA_Diff is not primary, and GA is included.</li></ul> |
| F | Checkpoint rule | **DIFFERENT** | <ul><li>The target is in-modality ND closest to 0.8 ± 0.1, not the *earliest* checkpoint with text accuracy in [0.30, 0.40].</li><li>Selection runs on the reporting set; there is no calibration set.</li><li>The image-unlearning condition is matched on image accuracy.</li><li>Aligned: retain drop ≤ 10 pp; ineligible runs are reported, not dropped.</li></ul> |
| G | Primary metric | **DIFFERENT** | <ul><li>The metric is A = ND(T→T) − ND(T→I) over all paired items, not G = Acc_I^U − Acc_T^U on the doubly-correct set.</li><li>Aligned: letter-logit scoring.</li><li>Missing: the answer margin is not computed (per-option log-probs are stored, so it can be derived).</li><li>Missing: no option-permutation check.</li></ul> |
| H | GO rule | **DIFFERENT** | <ul><li>Mine: A > 0 with CI excluding 0 for ≥ 2 of 3 methods in both seeds.</li><li>No G ≥ 0.10 effect-size threshold.</li><li>The bootstrap is entity-level and paired across modalities, but it runs on A, not G.</li></ul> |
| I | Models | **DIFFERENT (partial)** | <ul><li>Aligned: LLaVA-1.5-7B is primary.</li><li>Differs: Qwen2-VL-2B is set up for the full grid, seeds and verdict, not pipeline checks only.</li><li>Missing: no Qwen2.5-VL-7B support.</li></ul> |
| J | First milestone | **NOT STARTED (mostly)** | <ul><li>Done: tiny end-to-end example (random-weight models, all jobs).</li><li>(i) Partial: adapter-off reproduces the vanilla for a freshly initialised adapter only; not tested for a trained adapter.</li><li>(ii) Not started: module-contribution removal vs local adapter disabling.</li><li>(iii) Not started: self-replacement.</li></ul> |
| K | Later phases | **NOT STARTED** | As intended. Nothing mechanistic exists. |

---

## Part 3 — Summary

1. Overall: the infrastructure matches the protocol in spirit. That covers own two-form vanilla, paired paraphrased MC, LLM-only LoRA unlearning on answer tokens, letter-logit scoring, resumable Colab jobs, and failure logging. Most selection and statistics items (C, D, F, G, H) do not match.
2. Biggest mismatch: there is no calibration/confirmation separation, and no frozen doubly-correct set. Checkpoints are selected on the very items that are reported, with a different metric (A on ND) and a different rule (closest to ND 0.8 instead of the earliest text accuracy in [0.30, 0.40]).
3. No GPU time has been spent, so none of this has cost results. Aligning the pipeline now is cheap.
4. Proposed next step, once you confirm the protocol and answer §6:
   - implement D (splits), C (doubly-correct set + coverage), F, G (G, margin, permutation check) and H in the data, selection and report code;
   - make GA_Diff primary;
   - add the J (i)–(iii) checks on tiny models;
   - fix the smoke-test log deletion;
   - commit and hand you the Colab smoke test.
5. Estimated effort: about half a day of local coding and CPU testing, then the Phase A smoke test on Colab. Compute estimates stay roughly unchanged; dropping GA would save about 25% of grid time.
