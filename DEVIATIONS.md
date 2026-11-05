# Deviations log

Each entry says what changed, the date, who decided, and where it is implemented. Entries are
never edited after the fact; corrections get a new entry.

## Governing documents

| since | governing protocol | notes |
|---|---|---|
| 2026-10-29 (initial) | `PLAN.md` D1–D9 | Stage 1 plan, approved with changes D2/D8/D9 |
| 2026-10-29 (switch) | `PROTOCOL.md` ("When Forgetting Depends on How You Ask") + Addendum A–D | supersedes `PLAN.md` D1–D9. Where `PROTOCOL.md` and the A–K checklist differ, `PROTOCOL.md` wins. |

**Status of PROTOCOL.md.** As of this entry, `PROTOCOL.md` and the Addendum are **not yet in the
repo**. Items F (checkpoint rule), G (metric) and H (GO rule) were therefore implemented from the
2026-10-29 instructions: the A–K checklist and the definitions given with it. They must be checked
against PROTOCOL §4.2, §4.3 and §5.1 once the file is added. Any interpretation the instructions
did not fix is listed under "Provisional interpretations" below, and every one of them is a config
value or a single function.

## Entries

### DEV-001 — 2026-10-29 — protocol switch (decided by: PI)
`PROTOCOL.md` + Addendum A–D replace `PLAN.md` D1–D9. `PLAN.md` and the earlier draft `new_plan.md` were
deleted from the working tree on 2026-10-29 as outdated; they remain in git history (`git show a14952d:PLAN.md`,
`git show 3fb863e:new_plan.md`).

What changed in the code:
- **Splits (§4.2), `configs/data.yaml`, `vlmu/data/build.py`, all frozen in `splits.json`:**
  - forget = 20 calibration + 30 confirmation entities, all 50 unlearned;
  - retain = 300 unlearning-retain + 50 calibration + 100 confirmation.
- **Selection** (checkpoints, windows, thresholds) uses calibration entities only. Confirmation
  entities are first scored in the full eval of a selected checkpoint (`vlmu/runs.py`).
- **Answer positions** are balanced by stratified assignment, not just checked: per split and per
  field, letter counts differ by at most 1.
- **Vanilla:** after the vanilla full eval, the doubly-correct set is frozen. It contains the facts
  answered correctly in both primary forms (training portrait + text, original option order).
  Coverage is reported per split, entity and attribute in `runs/<model>/vanilla/doubly_correct.json`.
  The file is written once and never overwritten.
- **Checkpoint rule (F):** the earliest calibration checkpoint with in-modality accuracy in
  [0.30, 0.40] and retain drop ≤ 0.10. Runs stop there. Ineligible runs are reported with a reason.
- **Metric (G):** G = Acc_I^U − Acc_T^U on the doubly-correct confirmation facts. Scoring uses the
  A–D letter logits, plus the margin m = z_correct − logsumexp(z_wrong) and an option-permutation
  check (all 4 cyclic shifts: accuracy per shift and agreement). The primary metric uses the
  original order.
- **GO (H):** in both seeds, G ≥ 0.10 and the paired entity-bootstrap 95% CI lower bound > 0, for
  the primary method and condition.
- **Milestone J:** checks (i)–(iii) are implemented with hooks and tested with a trained adapter
  (`vlmu/mech/`, `scripts/check_hooks.py`).
- **Qwen2.5-VL-7B:** support added on the current transformers pin (4.51.3), for the replication phase only.

### DEV-002 — 2026-10-29 — image-form referent (decided by: PI)
Primary image+text eval items say **"the pictured person"**, not "this person". This keeps the
eval wording away from the training wording, where "this person" appears 7,430 times. The text
form uses the name. `vlmu/data/fields.py:REFERENTS`.

### DEV-003 — 2026-10-29 — methods (decided by: PI)
GA is removed from the core grid. Its code is kept and it is available as an optional later
method. Core methods: **GA_Diff (primary)** and **NPO (second objective)**.
`configs/<model>/grid.yaml:grid.methods`.

### DEV-004 — 2026-10-29 — image-unlearning condition (decided by: PI)
Unlearning on image+text is **secondary**: the reverse direction, and an upper bound for
multimodal unlearning. It uses the same checkpoint rule, applied to its in-modality (image)
accuracy. It is reported with G_reverse = Acc_T^U − Acc_I^U.

### DEV-005 — 2026-10-29 — model roles (decided by: PI)
- Qwen2-VL-2B: pipeline checks only, excluded from confirmatory evidence (`role: pipeline_check`;
  its verdict file says so).
- LLaVA-1.5-7B: primary.
- Qwen2.5-VL-7B: replication.

### DEV-006 — 2026-10-29 — data facts that changed item construction (decided by: implementer, from data)
- **Parent occupations** in paired items are taken from the FT answer, not the biography's
  `Parents` field. The biography structure is inconsistent: names and occupations are mixed, in 6
  different layouts.
- **Official MC item dropped:** profile 028 has one question whose `Correct_Answer` ("None of the
  above") is not among its options. Logged in `build_report.json`.

### DEV-007 — 2026-10-29 — update from the project-update note (decided by: PI; implementation choices by: implementer)
- **Order:** a single-cell pilot with checkpoints A and B and the logit-lens curves come before the full grid
  (`scripts/pilot.py`, `scripts/mech_layer_logit_diff.py`, notebook cells 6b/6c). The grid itself is unchanged.
- **Pilot objective `npo_rt`:** NPO + retain NLL (retain in the unlearning modality, weight `unlearn.retain_weight`).
  It is new and used only by the pilot. The grid's `npo` stays pure NPO. Pilot lr 5e-5 and seed 0 are
  implementer choices (`configs/<model>/pilot.yaml`, TODO(confirm)).
- **Checkpoint B thresholds** reuse the protocol values: in-modality forget accuracy ≤ window_hi (0.40), other
  modality minus in-modality ≥ min_G (0.10), retain drop ≤ retain_margin (0.10). The note says "clearly" without
  numbers (TODO(confirm)).
- **Open-ended scores** (teacher-forced answer log-prob, greedy keyword hit, ROUGE-L recall) are added to every full
  eval and to the pilot's epoch checkpoints. They are secondary; G and the checkpoint rule still use MC.
- **Text-only unlearning** asserts that no image input (`pixel_values`, `image_grid_thw`, `image_sizes`) reaches
  the forward pass (`vlmu/train.py:encode`).
- **Milestone J runs in float32** (`mech.dtype`). The first GPU smoke test (Qwen2-VL-2B, bf16, 2026-10-29) failed
  (ii) and (iii-b)[all] at relative differences up to 6.7e-2, while (i) and (iii-a) were exact. In bf16,
  (base + d) − d ≠ base after rounding, and the error compounds over layers. This replaces P-11: tolerance
  1e-5 (CPU float32), 1e-4 (GPU float32), 1e-2 (only if mech.dtype is set to a half type).
- **Starting model** remains the repo's own vanilla fine-tune; `Kyomin/unlearning_target` is not supported
  (TODO(confirm)).

### DEV-008 — 2026-10-29 — pretrained baseline scope (decided by: user)
The pretrained (no-FT) eval no longer runs the full eval suite. It scores only the forget paired items
(img, txt, pose) in the original option order: 1,338 items instead of 20,086 plus open-ended generation.
These are the only items its one consumer, the priors check in `scripts/sanity.py:per_field`, reads.
The 3 cyclic shifts, retain, official, celeb, MMBench and open-ended scores are dropped for this model only;
vanilla and selected-checkpoint full evals are unchanged. Output moves from `runs/<model>/pretrained/full_eval/`
to `runs/<model>/pretrained/prior_eval/`. `vlmu/evaluate.py:PRIOR_SUITE`, `vlmu/runs.py:run_pretrained_eval`.

### DEV-009 — 2026-10-29 — vanilla recipe: up to 10 epochs with early stop (decided by: user)
The first Qwen2-VL-2B vanilla (P-10 recipe, 3 epochs) failed checkpoint A: paired forget MC img 0.379,
txt 0.439 (gate 0.80). Train loss was still falling (0.49 → 0.37 → 0.27 at the epoch ends) and calibration
accuracy rose every epoch (forget img 0.330 → 0.358 → 0.391, txt 0.352 → 0.430 → 0.430). The pretrained
model (no FT) scores 0.31–0.36 on the same items, so the FT added +1.5 to +12 points: underfitting.
- **Change, all three models:** `train.epochs` 3 → 10 (upper bound; LR still decays linearly over 10 epochs),
  plus `train.early_stop_cal_forget_acc: 0.85`: training stops at the first epoch-end checkpoint where
  calibration forget paired accuracy is ≥ 0.85 in both img and txt. lr, LoRA rank, batch size and warmup are
  unchanged. The gate itself (≥ 0.80 on all 50 forget entities) is unchanged. `configs/*/vanilla.yaml`,
  `vlmu/runs.py:run_vanilla`.
- The failed 3-epoch vanilla and its doubly-correct set are archived on Drive (`runs/qwen2vl/vanilla_e3_archived/`,
  `models/qwen2vl_vanilla_e3_archived/`), not deleted. No unlearning run used them.

### DEV-010 — 2026-10-29 — vanilla gate no longer blocking; plateau stop (decided by: user)
The 0.80 gate (PLAN.md D2, carried into PROTOCOL checkpoint A) has no literature source: no paper shows that a
LoRA vanilla reaches 0.80 on paraphrased paired MC. The 3-epoch run gained only 3–8 points per epoch. Decision:
train for more epochs to raise accuracy, but do not keep training indefinitely to reach 0.80.
- **Plateau stop (all three models):** `train.early_stop_plateau: {epochs: 2, min_gain: 0.03}`. At each epoch end,
  take the pooled forget + retain calibration paired accuracy per modality (≈ 625 items each; forget alone is
  179, too noisy). Training stops when the weaker modality gained < 0.03 over the last 2 epochs. This adds to
  DEV-009's 0.85 stop and the 10-epoch cap; whichever comes first ends training. Pooled accuracies are logged in
  `checkpoints.jsonl` (`pooled_cal/img`, `pooled_cal/txt`). `vlmu/runs.py:run_vanilla`.
- **Gate reported, not blocking:** `gate.json` is still written with `passed`; `allow_failed_gate: true` in
  `configs/*/grid.yaml` (inherited by pilot and all run configs) lets unlearning start when it is false.
  G is still computed only on the frozen doubly-correct facts.
- **Open (not yet decided):** a replacement quality criterion for the doubly-correct set, e.g. excluding facts
  the pretrained model also answers correctly (`runs/<model>/pretrained/prior_eval/`), a minimum margin, and a
  minimum number of facts per confirmation entity.

### DEV-011 — 2026-11-05 — UMU-Bench track: UMU's released vanilla and UMU's readout (decided by: user)
Both own vanillas failed to inject the facts as measured by letter-logit MC (Qwen2-VL-2B, MLLMU paired items:
img 0.379 / txt 0.439 after plateau). A parallel track on UMU-Bench (Wang et al., NeurIPS 2025 D&B; same 500
fictitious profiles as MLLMU-Bench, with per-fact paired unimodal / multimodal questions):
- **Starting model:** UMU's released fine-tune `chengyewang/llava_smu_ft` (LLaVA-1.5-7B, LoRA merged, fp16; paper
  App. B: LLaMA-Factory, 5 epochs, lr 1e-4, batch 6). Its training QA attach the portrait to every item, including
  the name-based UM_QA, so it never saw a text-only (no-image) input. Its processor_config lacks
  `num_additional_image_tokens`; the public base processor is used (same tokenizer).
- **Readout:** UMU's own classification protocol (`vlmu/umu.py`): options listed in the prompt, greedy generation,
  correct iff the response equals the correct option text. Re-implemented at batch size 1 / fp16 it reproduces the
  paper's Table 1 Origin forget-10% Class AccF (0.823 vs 0.8233). Per form on forget_10: txt 0.75, img 0.81, both
  0.65. Letter-logit MC on the same items gives txt 0.51 / img 0.53 (`scripts/eval_external.py`, 5c): the letter
  readout, not the knowledge, failed. This replaces letter-logit MC as the behavioural readout on this track; the
  mechanism tools' target must move from the option letter to the answer text (not yet changed).
- **Unlearning:** fresh LoRA (r16/α16, LM linear layers) on that checkpoint; text-only NPO (β 0.4, reference = the
  checkpoint) on forget_10 UM_QA (600 rows); lr 5e-5 (`TODO(confirm)`), batch 8, 4 epochs; UMU-protocol readout,
  batched (left padding, bf16), at step 0 and every epoch. Doubly-correct facts = both forms correct at the run's
  own step 0. `configs/llava/umu_npo.yaml`, `scripts/unlearn_umu.py`.
- **Not carried over (yet):** the 20/30 calibration/confirmation entity split, the accuracy window selection,
  GA_Diff with a retain set, the seed grid. forget_10 gives 65 doubly-correct facts at batch-1 fp16; forget_15
  would give more (not checked whether it contains forget_10).

## Provisional interpretations (to confirm against PROTOCOL §4.3 / §5.1)

| id | question the instructions did not settle | current implementation | where |
|---|---|---|---|
| P-1 | Which lr is chosen when several lrs have an eligible checkpoint? | The **lowest** lr that has an eligible checkpoint in **every** seed. | `vlmu/runs.py:select` |
| P-2 | How are seeds run? | Every (method, modality, lr) is run for both seeds; each seed takes its own earliest eligible checkpoint at the chosen lr. | `configs/*/grid.yaml:grid.seeds` |
| P-3 | Which modality, and which items, define the retain drop? | max over img and txt of (1 − accuracy) on the doubly-correct calibration retain facts. The vanilla scores 1 there by construction. | `vlmu/runs.py:checkpoint_record` |
| P-4 | Which items define the window accuracy? | Doubly-correct calibration forget facts, in-modality, primary forms. | same |
| P-5 | Verdicts other than GO | NO-GO if G < 0.10 in every seed. UNCLEAR otherwise. INCOMPLETE if some seed has no eligible checkpoint. | `vlmu/report.py:decide` |
| P-6 | Which entities does the vanilla gate use? | All 50 forget entities (both splits), primary forms, ≥ 0.80 in each modality. | `vlmu/runs.py:run_vanilla` |
| P-7 | Checkpoint spacing and stopping | Checkpoint eval every 0.25 epoch; at most 6 epochs (linear decay over 6). Stop at the earliest eligible checkpoint, when accuracy falls below 0.30 without an eligible checkpoint, or when the retain drop exceeds 0.20. | `configs/*/grid.yaml` |
| P-8 | Scope of the doubly-correct set | Computed for forget (both splits) and retain calibration/confirmation paired facts, with one vanilla per model. | `vlmu/runs.py:freeze_doubly_correct` |
| P-9 | Unlearning hyperparameters not fixed by the instructions | GA_Diff equal weights with retain in the same modality; NPO β = 0.4 with the per-sample summed answer log-prob and the vanilla as reference; batch size 4; lr ∈ {2e-5, 5e-5, 1e-4}. | `configs/*/grid.yaml` |
| P-10 | Vanilla recipe (**epochs changed by DEV-009**) | LoRA r=16/α=16 on LM linear layers; lr 2e-4, 3 epochs, batch size 8, 3% warmup. | `configs/*/vanilla.yaml` |
| P-11 | Tolerance for Milestone J (**superseded by DEV-007**) | Relative max-abs logit difference ≤ 1e-5 (CPU fp32) or ≤ 1e-2 (CUDA bf16). Also requires the trained adapter to change logits by more than the tolerance. | `vlmu/mech/checks.py` |
