# Stage 1 plan — reproduce the text→image unlearning asymmetry

Status: **approved 2026-10-29** — D1, D3–D7 as written; D2 replaced; paraphrase requirement added (D8); split switched to forget_10 (D9). See "Approved changes" below.

## Approved changes (2026-10-29)

- **D2 (replaced): always fine-tune our own vanilla**, for both Qwen2-VL-2B and LLaVA-1.5-7B, starting from the public base checkpoints. Training data is ft_Data in both forms: the official photo + "this person" form and the name-substituted text-only form. Gate: **≥ 80% paired forget MC accuracy in both modalities** before any unlearning. The released `MLLMMU/LLaVA_Vanilla` is not used.
- **D8 (new): eval questions are paraphrases of the training/unlearning wording.** Same fact, same options and answer, different phrasing. Every eval stem is checked automatically: after normalizing the referent, it must not equal any FT question for that field, and its maximum word-overlap with those FT questions is reported.
- **D9 (new): use forget_10** (50 profiles) and retain_90.
  - Revised estimate: forget_5 → forget_10 adds **+30%**, computed with the throughput assumptions in §7. That is under the 50% threshold.
  - forget_5 is *not* a subset of forget_10; they are separate splits.
- Official MLLMU MC remains a secondary metric (scored with D3).

**Data facts found while implementing:**
- ft_Data has 14–19 QA per profile: 12 fixed fields (name … mother), then 2–7 free-form "fun facts".
- Each profile also has one long "Additional" description QA. The official loader skips it, and so do we.
- D5 becomes: unlearning set = **all FT QA of the forget profiles minus the name QA**, in the same form for both conditions.

## 1. What MLLMU-Bench actually provides (verified from repo + HF)

**Data** (`MLLMMU/MLLMU-Bench` on HF, ~1.4 GB parquet; IDs from `MLLMMU/baseline_train_split/forget_5/forget_ids.json`):

| config | rows | contents |
|---|---|---|
| `ft_Data` | 500 | portrait + `metadata`: 15 QA/profile, **all image-anchored** ("What is this person's profession?"). This is what the released vanilla was trained on. |
| `Full_Set` / `forget_5` / `retain_95` | 500 / 25 / 475 | portrait, `biography` (13 structured fields, no duplicate names), eval tasks: `Classification_Task` (5 image+text MC + 5 text-only MC), `Generation_Task`, `Mask_Task` |
| `Test_Set` | 500 | same people, 3 new generated poses each, paraphrased MC |
| `Retain_Set` | 153 | real celebrities, 5+5 MC |

**Eval protocol** (`eval.py`): MC questions are scored generatively. Greedy decoding is used, and the first character of the reply must be A–D, otherwise the item counts as wrong. Each prompt has a 1-shot demo drawn at random (unseeded) from *the evaluated set itself*, and the demo profile's questions are skipped. LLaVA prompt: `USER: <image>\n{q}\nA: …\nJust give ONE letter representing the answer directly.\nASSISTANT:`; text-only is the same without the image. Celebrity MC is zero-shot. Forget/retain eval uses the training portrait; Test_Set uses a random pose. Chance = 25%.

**Baselines** (`baselines/*.py`):
- LoRA r=16, α=16, dropout 0.05 on all LM linear layers; vision tower and projector frozen.
- lr 2e-5, batch size 4, 1 epoch.
- Loss is taken over the whole sequence (not masked to the answer).
- GA_Diff does a full forget pass, then a full retain pass, in each epoch.
- NPO β=0.4 (code) / 0.9 (paper). Its reference is an "oracle" model fine-tuned on retain only.

**Checkpoints:** `MLLMMU/LLaVA_Vanilla` (llava-1.5-7b-hf, LoRA merged, ~28 GB fp32 shards) and `MLLMMU/Idefics2_Vanilla`. **None for Qwen2-VL.**

**Reported vanilla LLaVA MC accuracy** (paper Tab. 2, modalities pooled): forget 51.7%, test 47.9%, retain 46.1%, celeb 51.8%.

## 2. Problems this creates for our design

- **P1 — official MC is not fact-paired across modalities.** Image+text and text-only MC ask about *different* facts. In forget_5, only 27/125 text-MC correct answers also appear among the image-MC answers. Example, profile 270:
  - image questions: activity, pet type, hobby, food, personality ("What is a characteristic that describes this person?" → *Cheerful*, partly readable from pixels);
  - text questions: father's job, birth city, mother's job, free time, pet name.

  Comparing modalities on official MC therefore confounds modality with fact and difficulty, and it fails your sanity check "same facts in both forms".
- **P2 — there is no text-only unlearning data.** All forget QA are "this person" + photo. A text-only version has to be built by deterministic name substitution.
- **P3 — vanilla knowledge is weak and image-anchored.** Accuracy is 51.7% against 25% chance. Text-only questions use the name, so the vanilla must compose image→name with image→fact, and it was never trained on name→fact. Expect low text-only vanilla accuracy. That gives a small, noisy denominator for the normalized drop. It also means the way the vanilla learned the facts plausibly *causes* the asymmetry, which is a confound to control, not to inherit silently.
- **P4 — generative letter scoring counts broken output as "forgotten".** GA often destroys output format, and accuracy then falls below chance. That would look like forgetting.
- **P5 — small n.** 125 MC per modality on forget_5, so SE ≈ 4.5 pp per accuracy.
- **P6 — few-shot demos are unseeded** and drawn from the forget set itself.

## 3. Proposed design — decisions needing approval

- **D1 Primary eval = paired MC built from biographies.**
  - Fields: Born, Date of Birth, Employment, Height, Educated at, Annual Salary, Residence, Father, Mother.
  - Excluded: Name (it appears in the text form), Gender (readable from pixels), and NA-valued entries.
  - All 9 fields are taught by the 15 FT QA.
  - Each item comes in two forms with **identical options, order and answer**: `<portrait> Where was this person born?` vs `Where was Emilia Thornton born?`.
  - Distractors are 3 same-field values from other profiles (fixed seed, deduplicated after normalization).
  - Size ≈ 25 × 9 ≈ 225 paired items per modality on forget.
  - Official MLLMU MC is also run (seeded) and reported as secondary.
- **D2 Vanilla model.**
  1. Evaluate the released LLaVA_Vanilla on paired forget MC.
  2. Gate: ≥ 80% in **both** modalities.
  3. If it fails (expected for text-only), LoRA-fine-tune our own vanilla from `llava-1.5-7b-hf` on ft_Data in **both forms** (official image-anchored + name-substituted text, 1:1), then merge.
  - Qwen2-VL-2B always uses our own vanilla with the same recipe.
- **D3 Scoring = restricted logits.** Take the argmax over the {A,B,C,D} token logits at the answer position. Zero-shot, model chat template, one forward pass per item. This never produces "invalid", and it is ~10× cheaper than generation. The generative validity rate is also logged, so format collapse stays visible.
- **D4 Image for image+text eval.** Primary = training portrait (as in official forget eval). Secondary = held-out Test_Set pose, which tests identity rather than pixel memory. Both are reported.
- **D5 Unlearning data.** Both conditions unlearn the **same 14 facts**: the 15 FT QA minus the name question.
  - The image+text condition uses the official QA.
  - The text-only condition uses the name-substituted rewrite.
  - The name QA is excluded from both. Otherwise image+text unlearning would also cut image→name, which the text condition cannot do.
- **D6 Method details.**
  - Loss is masked to answer tokens.
  - GA_Diff interleaves: each step uses 1 forget batch (ascent) + 1 retain batch (descent), equal weight. The retain batch uses the same modality as the condition and comes from retain_95 **minus** the retain-eval profiles.
  - NPO uses the vanilla as reference (adapter disabled, no second model copy), β=0.4 fixed. This departs from MLLMU's retain-only oracle.
- **D7 GO criterion (operationalized).** On paired forget MC, define ND = (acc_vanilla − acc_unlearned) / (acc_vanilla − 0.25) and asymmetry A = ND(text→text) − ND(text→img). Uncertainty comes from a 95% CI by profile-cluster bootstrap.
  - **GO:** A > 0 with the CI excluding 0, for ≥ 2 of 3 methods, in both seeds.
  - **UNCLEAR:** the sign is consistent but the CIs include 0.
  - **NO-GO:** anything else.
  - The mirror quantity B = ND(img→img) − ND(img→text) is reported too.

**Unchanged from MLLMU:**
- forget_5 IDs;
- LoRA r=16/α=16/dropout 0.05 on all LM linear layers;
- vision tower and projector frozen;
- AdamW, linear schedule, batch size 4;
- bf16 + gradient checkpointing.

## 4. Matched-forgetting selection (same rule for every condition)

- **Grid** per (method × unlearning modality): lr ∈ {2e-5, 5e-5, 1e-4} × steps. Checkpoints are evaluated every ½ epoch (~44 steps) on paired forget MC (both modalities) plus a fast retain subset, for up to 6 epochs. Early stop once in-modality ND ≥ 1.0, or once retain drop > 20 pp.
- **Feasible checkpoint:** paired retain accuracy drop ≤ 10 pp in both modalities.
- **Selected checkpoint:** the feasible one with in-modality ND closest to **0.8**. It counts as *matched* if |ND − 0.8| ≤ 0.1. Otherwise the cell is reported as *unmatched*: shown, but not counted toward GO.
- **Seeds:** seed 0 = selection run. Seed 1 reruns the selected (lr, step) without re-selection.
- **Logging:** every grid checkpoint goes to `results/grid.csv`; every failed run goes to `results/run_log.csv`.

## 5. Other eval sets

- **Retain:** 50 held-out retain_95 profiles (fixed seed); paired MC + official MC.
- **Real celebrity:** official MC (153 × 5+5).
- **Utility:** MMBench-EN dev, fixed 500-item subset (seed 0), same logit scoring.

## 6. Automated sanity checks (`scripts/sanity.py`, must pass before reporting)

- Vanilla paired forget accuracy ≥ gate in both modalities, with a per-field breakdown.
- Pretrained (non-fine-tuned) model on paired forget MC, to quantify image-prior leakage per field. Fields where pretrained image+text accuracy is far above chance get flagged.
- Pair integrity: both forms share item_id, options and answer, and the answer equals the biography field value.
- Image SHA-256: forget portraits do not appear in retain/celeb/MMBench; Test_Set poses differ from the portrait.
- ID sets are disjoint: forget / retain-eval / GA_Diff-retain.
- Text-only inputs contain zero image tokens; image inputs contain exactly one image.

## 7. Compute estimate (A100 40 GB; replaced by measured throughput after `--debug` runs)

**Assumed throughput** (samples/s):

| | train image | train text | eval image | eval text | model load |
|---|---|---|---|---|---|
| Qwen2-VL-2B (≤ 256 image tokens) | 10 | 40 | 80 | 250 | 1 min |
| LLaVA-7B | 3 | 20 | 25 | 100 | 3 min |

**Other assumptions:**
- Grid runs average 4 epochs because of early stopping.
- Each run has a checkpoint eval every ½ epoch on forget + 50 retain profiles.
- GA_Diff costs 2× and NPO 1.3× per step.
- 14 full evals per model.

**Estimate per job, forget_10:**

| job | Phase A (Qwen2-VL-2B) | Phase B (LLaVA-1.5-7B) |
|---|---|---|
| 0 data download to Drive + build items | 10 min (shared, CPU) | — |
| 1 `--debug` smoke tests | ~20 min | ~20 min |
| 2 vanilla FT (both forms, 3 epochs) + gate eval | ~0.8 h | ~2.5 h |
| 3 grid: 3 methods × 2 modalities × 3 lr | ~2.6 h | ~7.0 h |
| 4 seed-1 reruns | ~0.4 h | ~1.3 h |
| 5 full evals (pretrained, vanilla, 12 selected) | ~0.4 h | ~1.3 h |
| 6 aggregate tables/figure (CPU) | 1 min | 1 min |
| **total, forget_10** | **≈ 4.5 h** | **≈ 12.4 h** |
| *total, forget_5 (same assumptions)* | *≈ 3.6 h* | *≈ 9.4 h* |

Combined: 13.0 h → 16.9 h (**+30%**).

The earlier, rougher estimate in this file assumed a larger grid. This table supersedes it.

L4 is roughly 2.5–3× slower. LLaVA-7B LoRA at batch size 4 with gradient checkpointing fits in 24 GB, tightly.

**Storage:** the merged vanilla is saved once in bf16 on Drive (~14 GB for LLaVA). Unlearning runs save only LoRA adapters (~80 MB each).

## 8. Resumability and repo layout

- Every job is fully defined by one YAML in `configs/`.
- Its run dir is `/content/drive/MyDrive/vlm_unlearn/runs/<run_id>/`.
- Training saves adapter + optimizer + RNG state every ½ epoch and resumes from the latest.
- Eval appends per-item predictions to `.jsonl` and skips items already scored.
- Finished runs write `DONE` and are skipped on re-invocation.
- Exceptions write `FAILED` with a traceback and append to `results/run_log.csv`.

Layout as you suggested: `data/ train/ eval/ configs/ scripts/ colab/run.ipynb results/`. Generation/ROUGE is out of scope unless requested.
