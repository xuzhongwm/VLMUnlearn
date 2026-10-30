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
| P-10 | Vanilla recipe | LoRA r=16/α=16 on LM linear layers; lr 2e-4, 3 epochs, batch size 8, 3% warmup. | `configs/*/vanilla.yaml` |
| P-11 | Tolerance for Milestone J (**superseded by DEV-007**) | Relative max-abs logit difference ≤ 1e-5 (CPU fp32) or ≤ 1e-2 (CUDA bf16). Also requires the trained adapter to change logits by more than the tolerance. | `vlmu/mech/checks.py` |
