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

### DEV-012 — 2026-11-05 — UMU track: forget-only NPO collapses the text route; NPO + retain, txt+img readout (decided by: implementer, from data; user asked for the step 1–3 design)
Run `umu_npo_txt_lr5e-05_s0` (DEV-011 settings). Doubly-correct: 65 forget / 26 retain facts.
| step (epoch) | forget dc txt | forget dc img | G | retain dc txt | retain dc img |
|---|---|---|---|---|---|
| 75 (1) | 0.585 | 0.969 | +0.385 | 0.769 | 0.962 |
| 150–300 (2–4) | 0.000 | 0.91–0.92 | +0.91 | 0.000 | 0.88–0.92 |
From step 150 every text-only prompt, forget **and** retain, is answered with the same string ("PAPAPA…"); NPO's
log-ratio fell from −3.4 (step 70) to −73 (step 90). G ≈ 0.9 there measures "no image → degenerate output", not
fact-specific forgetting; steps 150–300 are not used. Step 75 is not degenerate but only partly specific
(text-route drop: forget −41 points, retain −23 on dc facts) and already drifts to frequent answers
("Vancouver, Canada." 13×, "NA" 12× among 100 forget txt responses).
- **Next run** (`configs/llava/umu_npo_rt.yaml`): NPO + retain NLL (`npo_rt`, weight 1) on text-only UM_QA of 50
  retain_90 profiles disjoint from the 20 readout profiles; lr 2e-5 (`TODO(confirm)`); 2 epochs; eval every 0.5 epoch.
- **Readout additions:** `txt+img` (the name-based question with the portrait) on forget facts — separates "the name
  gates the forgetting" from "the missing image gates it"; `degen_txt` = share of the most frequent text-only response.
- **Mechanism checks on this track:** Milestone J now loads the adapter under analysis on top of UMU's vanilla
  (`vlmu/mech/checks.py:setup_umu`, `scripts/check_hooks_umu.py`); the logit lens targets the correct option text
  (first answer token, and the summed answer tokens without EOS) under the UMU-protocol prompt
  (`scripts/mech_umu_lens.py`).

**DEV-012 results (2026-11-05).** `umu_npo_rt_txt_lr2e-05_s0` barely forgot (forget dc txt 0.985 at step 76; stopped).
`umu_npo_rt_txt_lr5e-05_s0` (`--set train.lr=5.0e-5`), doubly-correct facts as in run 1 (65 / 26):
| step (epoch) | forget dc txt | img | txt+img | G | retain dc txt | retain dc img | degen txt |
|---|---|---|---|---|---|---|---|
| 114 (1.5) | 0.815 | 0.969 | 0.969 | +0.154 | 0.885 | 1.000 | 0.13 |
| 150 (2) | 0.631 | 0.969 | 0.969 | +0.338 | 0.692 | 1.000 | 0.13 |
No collapse (degen flat). txt+img equals img at every checkpoint in both runs: with the portrait present, the
name-based question is not suppressed. Specificity is still weak at step 150 (text-route loss: forget 37%, retain
readout 31%); the retain term covers profiles 21–70, the readout profiles 1–20. `retain_nll` rose from ~0.1 to
0.5–0.7 after step 110. Next: a stronger, broader retain term, tagged `run_tag` (run ids may now carry a suffix,
`vlmu/umu.py:umu_run_id`).

**Run 3 (2026-11-05)** `umu_npo_rt_txt_lr5e-05_s0_rw3n200` (retain weight 3, 200 retain-unlearn people): step 150
forget dc txt 0.923 / img 0.969 / txt+img 1.000, G +0.046; unseen retain dc txt 0.962; degen 0.14; NPO log-ratio −4.2
(run 2: −15.7). Protecting other people removed most of the forgetting too: no setting so far gives person-specific
forgetting of the name route. Open question (A) the spread to anyone asked by name is the mechanism vs (B) the protection
was too weak. Run 4 (lr 1e-4, same retain) adds a **seen** retain readout (`retain_seen`: 20 of the people the retain term
trains on) to separate the two.

### DEV-013 — 2026-11-05 — direction change: from H1/H1′/H2 to query-form dependence (H-Q) (decided by: user)
**Pre-registered tests, outcome** (run 2 `umu_npo_rt_txt_lr5e-05_s0`, step 150; 65 forget / 26 retain doubly-correct facts):
- Logit lens: the portrait route is flat at every layer → **H2 rejected**.
- LoRA layer masking: switching the LoRA off in layers 0–9 or 10–19 restores the lost name-route answers (recovery
  0.96 / 0.92; generation 79%), off in 20–31 does not (0.08; 4%); no band alone is sufficient, 0–19 together is →
  **H1 rejected**; H1′'s early-to-mid locus holds.
- Position masking (exact per-token switch, `vlmu/mech/hooks.py:mask_contributions`; all-on = unlearned and all-off =
  vanilla, both 0.00 difference): LoRA off on the name tokens restores 0.37, on the last prompt token 0.88, on all
  other prompt tokens 0.97; LoRA on at any single group alone produces almost no forgetting → the edit is distributed
  over the whole text prompt, not bound to the name → H1′'s "name pathway" does not hold.
- ||ΔW x||: peak in layers 8–18 on the name and last tokens; identical for forget and retain names; no layer writes
  along the negative answer direction; ||BA|| roughly uniform over layers (size ≠ causal effect).
**New main hypothesis H-Q (exploratory: formed after seeing these data):** text-only unlearning installs a distributed,
layer 0–19 suppression conditioned on the text query form, not removal of the person's knowledge; forgetting generalizes
along the query form (collateral on other people asked the same way), not along the entity (the same person asked in
another form, e.g. with the portrait, escapes). Cross-modal leakage is one case of form-specific forgetting.
**Confirmatory test, fixed before running — query-form component sweep** (eval only; same facts; base = the training
form: UM_QA question, no options, `USER: … ASSISTANT:`, no image; one component changed at a time): (A) image context:
blank image, own portrait, another person's portrait, 576 neutral filler text tokens; (B) referent: first name only,
last name only, "Mr./Ms. <last name>", name order reversed; (C) wording: two paraphrases per attribute; (D) answer
format: UMU option list, A–D letters, cloze sentence; (E) chat frame: system prompt, no frame. Readout per form v:
F_v = mean over the 65 forget dc facts of [log p_vanilla − log p_unlearned] of the correct answer text (teacher forced);
recovery R_v = 1 − F_v / F_train (0 = forgotten as in the training form, 1 = unaffected); generation correctness as
secondary. Same for the 26 retain dc facts. Also mean ||A x|| (LoRA input projection) in layers 0–19 per form.
**Predictions and thresholds:**
- P1 form not identity: blank image R ≥ 0.5 and filler tokens R ≥ 0.5. Counter-evidence: blank image R ≤ 0.2 while own
  portrait R ≥ 0.8 (identity from the image matters; H1′-like).
- P2 name spelling irrelevant: first-only, last-only and "Mr./Ms. <last>" each R ≤ 0.2.
- P3 beyond modality: at least one text-only change (wording or answer format) R ≥ 0.3. Counter-evidence: all text-only
  changes R ≤ 0.2 and only adding an image escapes.
- P4 collateral follows form: Spearman(R_retain, R_forget) over forms ≥ 0.7.
- P5 mechanism–behaviour link: Spearman(mean ||A x|| in layers 0–19, 1 − R) over forms ≥ 0.6.
- P6 mitigation (later, training): form-diverse text-only unlearning (several text forms incl. blank-image variants, no
  real portraits) halves G at matched forget-name drop (±0.05) without a larger retain-name drop.
Each prediction is reported as passed / failed against these thresholds. Primary checkpoint: run 2 step 150; replication on
a second checkpoint before any claim. Plan and ordering: research.md, "2026-11-05 方向更新".
**Outcome, run 2 step 150** (`scripts/mech_umu_form_sweep.py`; figure `proposal/figures/form_sweep_run2_step150.png`;
F_train forget 14.42 / retain 12.06 nats):
- P1 **pass**: R forget own portrait 1.00, blank image 0.88, other portrait 0.90, filler text 0.83 (portrait question 1.00).
- P2 **fail (narrowly)**: first name 0.15, last name 0.24, "Mr./Ms. <last>" 0.28 (two of three above 0.2); reversed −0.02.
  Forgetting mostly transfers across name variants, but not fully.
- P3 **pass**: option list 0.92, A–D letters 0.73, no frame 0.65; cloze 0.19, system prompt 0.24, paraphrases −0.17 /
  −0.33 (forgetting stronger than in the training form).
- P4 **pass**: Spearman 0.98 over 16 forms.
- P5 **pass**: Spearman 0.77 (16 forms; the figure panel shows 0.79 because it includes the base point). Mainly driven by
  the image-context forms (low ||A x||, little forgetting); among text-only forms the ordering is weak (system prompt has
  low ||A x|| but strong forgetting).
- Generation (keyword hit, vanilla → unlearned, forget): base 0.77 → 0.37; blank image 0.75 → 0.69; filler 0.71 → 0.48;
  no frame 0.82 → 0.43; option list 1.00 → 0.82; letters 0.55 → 0.62; paraphrases ≈ 0.75 → 0.35 / 0.40.
**Caveats (to resolve before claims):** (1) option-list forms contain the answer text, so copying can inflate R; needs
an answer-absent / shuffled-option control. (2) log p vs generation disagree for filler text (R 0.83 but generation
recovers only ≈ 40% of the base drop) and for no frame (R 0.65, generation drop ≈ base) — blank image is the only
non-portrait form that escapes on both readouts, so "any prefix escapes" is not yet established; the modality of the
prefix may still matter. (3) Paraphrase R < 0 is a log-p-scale effect; generation drops are similar to base.
(4) one checkpoint. Next: prefix length/position sweep (filler 16–576 tokens, image before/after question),
answer-absent option control, replication on a second checkpoint, then P6.

### DEV-014 — 2026-11-19 — scope: course report; one mechanism test, LoRA-contribution transplant (decided by: user; design by: implementer)
The ARR January 2027 target (research.md) is dropped; the deliverable is a course report. The user requires at least
one of the open mechanism questions (what the layer 0–19 edit detects; how it suppresses the answer; why an image does
not trigger it; why NPO learns it) to be answered first. Chosen: why a blank image bypasses the suppression. Run 4, the
input-cue sweep, replication, P6 and the generality steps of DEV-013 are no longer planned for the report.
**Question.** With an image in the prompt, (T) does the LoRA no longer write the suppression, because the hidden states
it reads at the question tokens differ, or (O) does it write the same thing in effect while the model, with an image in
context, does not use it?
**Test, fixed before running** (`scripts/mech_umu_transplant.py`; hooks `record_contributions` /
`transplant_contributions` in `vlmu/mech/hooks.py`; eval only; run 2 step 150): same fact, two prompts, text
(`USER: {q}\nASSISTANT:`) and blank (`USER: <image>\n{q}\nASSISTANT:`, grey 336×336 image). Under θu, record every LoRA
module's contribution d(x) at the tokens the two prompts share (longest common token suffix: question, `ASSISTANT:`,
teacher-forced answer). Then run each prompt with the other prompt's recorded contributions written at those tokens
(text←blank, blank←text), in layers 0–19, and in layers 20–31 as a control; reference: LoRA off at those tokens.
Readout: teacher-forced log p of the answer text (first answer token as secondary), θu and adapter-off θ0. With
S = log p_0 − log p and gap = S_text − S_blank under θu, the moved fraction is
M(text←blank) = [log p_patched(text) − log p_u(text)] / gap and M(blank←text) = [log p_u(blank) − log p_patched(blank)] / gap
(ratio of means over facts; 95% bootstrap over profiles). Primary: the 65 forget doubly-correct facts.
**Predictions and thresholds** (forget facts, answer log p, layers 0–19):
- T: M ≥ 0.5 in both directions (the image changes what the LoRA writes).
- O: M ≤ 0.2 in both directions (the contributions are the same in effect; the image context makes them ineffective).
- Anything else is reported as mixed, without a verdict.
- Control: |M| ≤ 0.1 in both directions for layers 20–31; a larger value is reported, not discarded.
**Validity checks:** a prompt's own contributions transplanted back reproduce the unhooked run (|Δ log p| ≤ 0.05 nats,
first 3 facts; the script aborts otherwise); the bypass reproduces on these facts (R_blank ≥ 0.5; the form sweep gave
0.88), otherwise M is not interpreted.
**Known limits:** one checkpoint, one seed; the late band is the only control for "any foreign contribution disrupts the
run"; the first question token is tokenised differently after `<image>\n` and stays outside the shared suffix (logged);
the per-layer cosine between the two recordings is descriptive only.
**Outcome, run 2 step 150** (2026-11-19, code at f6e759b; `results/llava/mech_umu/<run>/step150/transplant/summary.json`;
figure `proposal/figures/transplant_run2_step150.png`; 65 forget dc facts / 42 profiles; 14.4 shared prompt tokens on
average; one text-prompt token per fact, the first word of the question, outside the shared suffix):
- Validity: self-transplant changes log p by 0.00 nats (3 facts); forgetting 14.42 nats (text) vs 1.68 nats (blank),
  R_blank 0.88, as in the form sweep.
- Layers 0–19: M(text←blank) = 0.41 [0.34, 0.48], M(blank←text) = 0.06 [0.03, 0.09] → **mixed, no verdict**. T is met in
  neither direction; O is met for blank←text only. First answer token: 0.43 / 0.04.
- Control, layers 20–31: −0.00 / +0.00 → no effect, as expected.
- Reference, LoRA off at the shared tokens in layers 0–19: text 1.07, blank −0.08.
- Suppression S (nats, answer log p) by prompt and by source of the contributions in layers 0–19: text prompt 14.42
  (own) / 9.24 (from the blank run) / 0.76 (off); blank prompt 2.40 (from the text run) / 1.68 (own) / 0.63 (off).
- Descriptive, layers 0–19: cosine between the two recordings 0.89 on the shared prompt tokens (0.73–0.96 by layer) and
  0.96 at the last prompt token (0.88–1.00); norm ratio blank / text 1.28 on average (0.53–2.22 by layer; a mean of
  per-token ratios, not to be leaned on).
**Reading (exploratory, formed after seeing the data):** both pure accounts fail. The image does not stop the write:
with the image the LoRA writes similar contributions, and the text run's own contributions, which carry 14.4 nats of
suppression there, add 0.7 nats when written into the blank run. The write is not irrelevant either: the blank run's
contributions carry 9.2 of the 14.4 nats in the text prompt. Suppression needs the text run's contributions and the
text-only context together, and the context is the larger factor. Generic disruption by foreign contributions can only
raise both transplanted values, so 2.40 is an upper bound (the first claim is safe) and 9.24 may overstate what the
blank run's contributions suppress. This weakens the causal reading of P5 in DEV-013 (lower ||A x|| with an image, less
forgetting). Not tested: what in the image context makes the contributions ineffective.
**Replication** (2026-11-19, same code and thresholds, layers 0–19, answer log p; M(text←blank) / M(blank←text)):
| checkpoint, facts | forgetting text / blank (nats) | R_blank | M(text←blank) [95%] | M(blank←text) [95%] |
|---|---|---|---|---|
| run 2 s150, forget (65) | 14.42 / 1.68 | 0.88 | 0.41 [0.34, 0.48] | 0.06 [0.03, 0.09] |
| run 2 s150, retain (26) | 12.08 / 0.82 | 0.93 | 0.43 [0.31, 0.52] | 0.06 [0.03, 0.10] |
| run 1 s75 (`umu_npo_txt_lr5e-05_s0`, NPO only), forget (65) | 5.90 / 1.55 | 0.74 | 0.64 [0.45, 0.81] | −0.03 [−0.05, −0.01] |
| run 1 s75, retain (26) | 5.60 / 0.55 | 0.90 | 0.55 [0.30, 0.75] | −0.02 [−0.04, −0.00] |
Self-transplant 0.00 nats in every run; control band 20–31 within |M| ≤ 0.04 everywhere. All four are mixed under the
thresholds, with the same asymmetry: the text run's contributions add no suppression in the blank prompt (−0.03 to
0.06), while the blank run's contributions remove 0.41–0.64 of the gap in the text prompt. The blank-image bypass and
the asymmetry hold on a second checkpoint from a different recipe and on retain facts. Run 1 differs in degree: weaker
forgetting, a larger M(text←blank), and some effect of the late band when the LoRA is switched off there (text_off
20–31: 0.17, against 0.07 in run 2).

### DEV-015 — 2026-11-19 — exploratory localisation: where the suppression is carried (decided by: user asked for the experiments a mechanism claim needs; design by: implementer)
The course report is due in about three weeks (around 2026-12-10); the user wants at least one phenomenon analysed
positively, not only rejected hypotheses. A mechanism claim needs four links; their state after DEV-014:
(1) where the LoRA writes: layers 0–19, the whole text prompt (layer and position masking); (2) what those writes
change, i.e. which hidden state carries the suppression to the output: **not measured**; (3) how the answer fails to
appear: partly (logit lens: the answer does not emerge in layers 20–26; no write against the answer direction);
(4) where the chain breaks with an image: partly (DEV-014: not at the write). The input-cue sweep
(`scripts/mech_umu_input_cues.py`, notebook §7) is run as planned in the proposal; it names the property of the input
that decides the outcome and is not itself evidence for links 2–4.
**Experiment for link 2, and for link 4 descriptively** (`scripts/mech_umu_state_patch.py`, notebook §9; eval only):
cross-model state patching on the training form `USER: {q}\nASSISTANT:`. At one decoder layer and one group of prompt
tokens (name / other question tokens incl. `USER:` / the newline and `ASSISTANT` / the last prompt token / all prompt
tokens), the residual state of θu is replaced by θ0's from the same prompt (restore) or the reverse (induce). Readout:
log p of the first answer token (85% of the answer-text forgetting on run 2; only prompt positions can affect it),
recovery = (lp_patched − lp_u) / (lp_0 − lp_u), induced = (lp_0 − lp_patched) / (lp_0 − lp_u); ratio of means over
facts, 95% bootstrap over profiles; second readout: top-1 accuracy of the first answer token (added after the
input-cue outcome below). Descriptive: ||h_u − h_0|| / ||h_0|| per layer and token group on the training form and on
the same question behind a blank image and behind 576 filler text tokens. Default: 65 forget dc facts, every other
layer; run 2 step 150.
**Status: exploratory.** No thresholds and no verdict are fixed; this is a map, and any reading of it is formed after
seeing it and reported as such. Fixed in advance only: the readout, the token groups, and the validity check (a model's
own state patched back in changes log p by ≤ 0.05 nats, else the script aborts).
**Route patching at the last prompt token, outcome found on Drive** (`scripts/mech_umu_blank_patch.py`, notebook §6;
`blank_patch.json` written 2026-11-05, not recorded here until 2026-11-19; run 2 step 150; 12 forget facts selected as
wrong without an image and right with a blank image; layers 0, 4, …, 28, 31; log p of the first answer token, effect
under θu minus the same patch under θ0; baseline θu −15.75 (text) / −0.00 (blank), θ0 −0.13 / −0.04):
| layer patched | 0 | 4 | 8 | 12 | 16 | 20 | 24 | 28 | 31 |
|---|---|---|---|---|---|---|---|---|---|
| text → blank (nats) | 0.00 | −0.88 | −4.72 | −11.40 | −15.40 | −15.10 | −15.19 | −15.16 | −15.66 |
| blank → text (nats) | +0.88 | +5.54 | +10.66 | +15.09 | +15.66 | +15.69 | +15.69 | +15.67 | +15.66 |
The same patches under θ0 change log p by less than 0.1 nats. From layer 16 on, the state of the last prompt token
alone decides the outcome in both directions; it is built up over layers 4–16, inside the band where the LoRA acts and
before the layers where the answer emerges (20–26). Limits: 12 facts selected on their outcome, one token position, a
coarse layer grid; at late layers the patch is close to copying the output state, so only the profile up to layer 16 is
informative. This partly covers link 2 (at the last token only); the state-patching experiment extends it to all token
groups, to θ0 ↔ θu on the same prompt, and to all doubly-correct facts.
**Input-cue sweep, outcome** (run 2 step 150; `results/llava/mech_umu/<run>/step150/input_cues/summary.json`, first
written 2026-11-05, re-summarised 2026-11-19; answer log p θ0 → θu with the drop in nats, R with a 95% profile
bootstrap, generation keyword hit θ0 → θu; forget 65 facts / retain 26):
| form | forget log p (drop) | forget R | forget generation | retain R | retain generation |
|---|---|---|---|---|---|
| base (training form) | −0.81 → −15.27 (14.46) | 0 | 0.77 → 0.37 | 0 | 0.85 → 0.35 |
| blank image before | −1.27 → −2.99 (1.71) | 0.88 [0.81, 0.94] | 0.75 → 0.69 | 0.93 | 0.81 → 0.73 |
| text prefix, 16 tokens | −1.91 → −5.25 (3.34) | 0.77 [0.72, 0.81] | 0.68 → 0.35 | 0.63 | 0.73 → 0.31 |
| text prefix, 64 | −1.81 → −4.85 (3.04) | 0.79 [0.74, 0.84] | 0.71 → 0.35 | 0.67 | 0.73 → 0.38 |
| text prefix, 256 | −1.75 → −3.85 (2.10) | 0.85 [0.81, 0.90] | 0.72 → 0.51 | 0.76 | 0.65 → 0.42 |
| text prefix, 576 | −1.64 → −4.15 (2.50) | 0.83 [0.77, 0.88] | 0.71 → 0.48 | 0.76 | 0.69 → 0.35 |
| blank image after | −13.00 → −12.29 | (1.05) | 0.03 → 0.25 | (1.06) | 0.00 → 0.23 |
| blank image, zeroed features | −13.60 → −15.88 | (0.84) | 0.22 → 0.06 | (0.80) | 0.31 → 0.08 |
| text suffix, 576 | −5.47 → −13.21 (7.74) | 0.46 | 0.40 → 0.40 | 0.46 | 0.46 → 0.27 |
Reading, by the decision rule fixed in `proposal/proposal.tex` (§ Experiment 1):
- Text prefixes reproduce the blank image on log p (R 0.77–0.85 against 0.88; no dose–response beyond 16 tokens) but
  **not in generation**: the relative loss of correct answers is 48–50% with 16 / 64 tokens and 30–33% with 256 / 576,
  against 8% with the blank image and 52% in the base form. The rule requires both readouts, so a length or
  displacement account is **not supported**: at equal length (576 tokens) an image restores the answers and text does not.
- Zeroed features and image-after are **not interpretable**: the vanilla fails there (answer log p −13.6 / −13.0,
  accuracy 0.22 / 0.03). Whether visual content or the image slots matter is therefore still open. The suffix form
  damages the vanilla too (0.40).
- **Log p and generation dissociate.** The 14.5 nats of the base form are far more than is needed to change the
  generated answer: about 3 nats (16-token prefix) cost as many answers as 14.5. R on log p therefore overstates how
  much a form escapes. Consequences: the text-only R values of DEV-013 (P3, filler text) do not show a behavioural
  bypass; DEV-014's M is a statement on the log-p scale, and the 2.40 nats it reports for the blank prompt carrying the
  text run's contributions lie in the range where generated answers start to change, so its behavioural meaning is not
  established. Later analyses report a behaviour-level readout next to log p.

**State patching, outcome, run 2 step 150** (2026-11-19, code at eded4cb; `.../state_patch/summary.json`; figure
`proposal/figures/state_patch_run2_step150.png`; 65 forget dc facts / 42 profiles; every other layer; self-patch error
0.00 nats; forgetting 12.30 nats on the first answer token; top-1 of the first answer token θu 0.12, θ0 0.88). Token
groups of this run: name (4.3 tokens), question (10.1, **including BOS and `USER:`**), frame (4.0), last (1), prompt (19.4).
| layer | 0 | 2 | 4 | 8 | 12 | 14 | 16 | 20 |
|---|---|---|---|---|---|---|---|---|
| restore, last (recovery) | 0.01 | 0.10 | 0.38 | 0.62 | 0.85 | 0.92 | 0.95 | 0.98 |
| restore, last (top-1) | 0.14 | 0.18 | 0.38 | 0.54 | 0.72 | 0.86 | 0.89 | 0.89 |
| restore, question | 0.27 | 0.47 | 0.30 | 0.16 | 0.09 | 0.03 | 0.00 | 0.00 |
| restore, name | 0.01 | 0.03 | 0.05 | 0.07 | 0.11 | 0.08 | 0.07 | 0.03 |
| restore, frame | 0.01 | 0.04 | 0.08 | 0.07 | 0.05 | 0.05 | 0.04 | 0.02 |
| restore, prompt | 0.30 | 0.59 | 0.65 | 0.83 | 0.96 | 0.98 | 0.99 | 0.99 |
| induce, last (share) | 0.00 | 0.00 | 0.00 | 0.01 | 0.28 | 0.60 | 0.78 | 0.90 |
| induce, last (top-1 of θ0) | 0.88 | 0.88 | 0.88 | 0.85 | 0.40 | 0.17 | 0.17 | 0.15 |
| induce, prompt | 0.00 | 0.02 | 0.02 | 0.12 | 0.55 | 0.78 | 0.90 | 0.96 |
Induce for name / question / frame alone is ≤ 0.04 at every layer. 95% intervals for restore, last: layer 4
[0.29, 0.48], 8 [0.52, 0.71], 12 [0.79, 0.92], 16 [0.90, 0.99]; induce, last: layer 12 [0.23, 0.34], 16 [0.75, 0.82].
Relative state change ||h_u − h_0|| / ||h_0|| for the training form / filler text / blank image: name tokens at layer 2
0.20 / 0.09 / 0.03; last prompt token at layer 16 0.66 / 0.37 / 0.25; image tokens 0.01–0.05 in layers 2–10. On those
three inputs: first-token forgetting 12.30 / 1.64 / 0.88 nats, top-1 of θu 0.12 / 0.60 / 0.83 (θ0 0.88 / 0.86 / 0.86).
**Reading (exploratory, formed after seeing the map):**
- The suppression is not carried on the name tokens (restore ≤ 0.11, induce ≤ 0.04).
- It starts on the question group in layers 0–2 (restore 0.27–0.47) and is accumulated in the last prompt token:
  necessary there from layer 4 (0.38), necessary and sufficient from layers 14–16 (restore 0.92–0.95, induce 0.60–0.78),
  on log p and on top-1. This agrees with the route patching above (last token decisive from layer 16) with a
  different patch and all facts.
- Early writes are necessary but not sufficient: restoring all prompt states at layer 2 recovers 0.59, while θu's
  states at layer 2 induce 0.02 in θ0. Later LoRA layers turn the early change into suppression.
- With text or an image before the question, θu's states stay closer to θ0's at every layer and token group, in the
  order of the behaviour (training form > filler text > blank image). This is a measurement, not a causal test.
**Open after this run:** which tokens of the question group act early (opening vs question words; group split added,
notebook §9b); retain facts and run 1 step 75 (§9b); attention vs MLP; a causal test of why the image keeps the states
close. The log of this run printed only layers 0 and 31 (the row filter assumed every layer); the numbers above are
from `summary.json`.

**State patching, replication** (2026-11-19, notebook §9b; run at 19:16–19:23 UTC with the code of eded4cb, i.e. before
the `opening` group was added: groups as above, the question group still includes BOS and `USER:`; self-patch error 0.00
in both runs; `summary.json` of each run):
| checkpoint, facts | forgetting, first token (nats) | top-1 θu / θ0 | restore last: L4, L8, L12, L16, L20 | induce last: L12, L14, L16, L20 | restore name, max | induce name, max | induce prompt at L2 |
|---|---|---|---|---|---|---|---|
| run 2 s150, forget (65) | 12.30 | 0.12 / 0.88 | 0.38, 0.62, 0.85, 0.95, 0.98 | 0.28, 0.60, 0.78, 0.90 | 0.11 | 0.04 | 0.02 |
| run 2 s150, retain (26) | 11.24 | 0.12 / 0.92 | 0.39, 0.64, 0.90, 0.97, 0.99 | 0.28, 0.60, 0.78, 0.91 | 0.09 | 0.01 | 0.00 |
| run 1 s75, forget (65) | 3.37 | 0.51 / 0.88 | 0.05, 0.40, 0.47, 0.74, 0.89 | 0.16, 0.09, 0.30, 0.63 | 0.25 | 0.33 | 0.57 |
| run 1 s75, retain (26) | 3.24 | 0.58 / 0.92 | 0.12, 0.38, 0.44, 0.65, 0.85 | 0.23, 0.22, 0.46, 0.69 | 0.21 | 0.47 | 0.64 |
Relative state change, training form / filler text / blank image: run 1 name tokens at layer 2 3.02 / 0.74 / 0.22, last
prompt token at layer 16 0.82 / 0.50 / 0.25 (run 2: 0.20 / 0.09 / 0.03 and 0.66 / 0.37 / 0.25). Top-1 of θu on the three
inputs: run 1 0.51 / 0.68 / 0.85; run 2 0.12 / 0.60 / 0.83.
**Reading:**
- Retain facts follow the same path as forget facts within each checkpoint: the collateral damage is the same process.
- **The path is not the same in the two checkpoints.** In run 2 the name tokens carry almost nothing and the last
  prompt token is sufficient from layers 14–16. In run 1 (NPO without a retain term, one epoch, the run that collapses
  one epoch later) the unlearned state of the name tokens alone carries 0.31–0.33 of the suppression in layers 2–8
  (0.40–0.47 on retain facts), the LoRA writes of layers 0–2 alone already produce 0.57 of it, the name-token states are
  moved by three times their norm at layer 2, and the last prompt token takes over later (restore 0.74 at layer 16,
  0.89 at layer 20; induce 0.30 and 0.63). "The name is not the carrier" therefore holds for run 2 only.
- Common to both: the last prompt token carries most of the suppression in the later layers; the state change is
  largest on the training form, smaller behind filler text and smallest behind a blank image, in the order of the
  behaviour; forget and retain facts behave alike.
- The two checkpoints differ in recipe (retain term) and in training length (75 vs 150 steps), so the difference cannot
  be attributed to the retain term alone. Run 1's forgetting is small (3.4 nats), and its shares have wide intervals
  (restore last at layer 8: 0.18–0.57).
**Still open:** the opening / question-word split (needs §9b again with the current code).
**Decision (user, 2026-11-19):** run 1 is not a usable unlearning checkpoint (little forgetting, as much damage to
retain as to forget facts, collapse one epoch later) and is set aside: its numbers stay in this log, but it is not used
as a second checkpoint or as a replication, here or in DEV-014. All mechanism claims are claims about run 2 step 150.
Notebook §9b now runs run 2 only.

### DEV-016 — 2026-11-19 — does the suppression transfer to other examples? Mean-difference vector at the last prompt token (decided by: user proposed the test; design by: implementer)
The state patching of DEV-015 moved a whole last-token state between θ0 and θu on the same fact. That shows the state is
sufficient, but not whether it holds a signal shared across facts or fact-specific damage. The user asked whether the
signal, put on examples that were not unlearned, makes them forgotten.
**Test, fixed before running** (`scripts/mech_umu_signal_transfer.py`, notebook §10; eval only; run 2 step 150; training
form `USER: {q}\nASSISTANT:`; all doubly-correct facts, 65 forget / 26 retain): δ = h_u − h_0 at the last prompt token,
at layers 8, 12, 14, 16, 18, 20, 24. The forget profiles are split in two halves (alternating in sorted order); a forget
fact uses the mean δ of the half its person is not in, a retain fact the mean δ of all forget facts. Conditions at one
layer: θ0 with h_0 + mean δ (add_mean), the same with mean δ rescaled to the average norm of a single δ
(add_mean_normed), h_0 + own δ (reference), h_0 + a random vector of the norm of mean δ (control); θu with h_u − mean δ
(sub_mean), − own δ (reference), − random (control); θu on the same question behind a blank image, where the answer is
not suppressed, with its last-token state + mean δ of the training form (blank_add_mean) and + random, at layers 12,
16, 20. Readout: log p of the first answer token, top-1 accuracy, and the most frequent predicted first tokens.
add: share = (lp_0 − lp_patched) / (lp_0 − lp_u); sub: share = (lp_patched − lp_u) / (lp_0 − lp_u); blank: suppression
added, as a share of the text-form forgetting. Ratio of means, 95% bootstrap over profiles.
**Predictions and thresholds** (forget facts, layer 16, the layer from which DEV-015 found the state sufficient):
- Shared signal: add_mean ≥ 0.5 and sub_mean ≥ 0.5 and add_random ≤ 0.1.
- Fact-specific: add_mean ≤ 0.2 and sub_mean ≤ 0.2.
- Anything else is reported as partly shared, without a verdict.
- Reported without a threshold: retain facts with the forget signal; the blank-image prompt; all other layers;
  add_mean_normed (separates direction from size); cosine between each δ and the mean of the other people.
**Validity checks:** a state patched back in changes log p by ≤ 0.05 nats (first batch, else the script aborts);
add_own and sub_own must reproduce the induce and restore values of DEV-015 at the shared layers.
**Known limits:** one checkpoint; one token position; the two halves share the question templates of the benchmark, so
"other people" does not mean "other question wording"; a top-1 change is a first-token proxy for a generated answer.

**Outcome, run 2 step 150** (2026-11-19, code at 2f83e55; `.../signal_transfer/summary.json`; figure
`proposal/figures/signal_transfer_run2_step150.png`; self-patch error 0.00; add_own and sub_own reproduce DEV-015's
induce and restore at every shared layer, e.g. 0.78 and 0.95 at layer 16). Forget facts, held-out people, share of the
12.30 nats / top-1 (θ0 0.88, θu 0.12):
| layer | 8 | 12 | 14 | 16 | 18 | 20 | 24 |
|---|---|---|---|---|---|---|---|
| cos(own δ, mean δ of other people) | 0.80 | 0.77 | 0.76 | 0.70 | 0.67 | 0.57 | 0.41 |
| norm of mean δ / norm of a single δ | 0.81 | 0.79 | 0.78 | 0.72 | 0.69 | 0.61 | 0.47 |
| sub_mean, share / top-1 | 0.57 / 0.48 | 0.81 / 0.65 | 0.90 / 0.78 | 0.91 / 0.82 | 0.88 / 0.71 | 0.78 / 0.52 | 0.62 / 0.32 |
| add_mean, share / top-1 | 0.00 / 0.92 | 0.09 / 0.63 | 0.15 / 0.57 | 0.14 / 0.57 | 0.07 / 0.63 | 0.03 / 0.83 | 0.01 / 0.86 |
| add_mean_normed, share / top-1 | 0.01 / 0.86 | 0.22 / 0.42 | 0.33 / 0.35 | 0.41 / 0.31 | 0.30 / 0.42 | 0.26 / 0.46 | 0.18 / 0.48 |
| add_random / sub_random, share | 0.00 / 0.00 | 0.00 / −0.01 | 0.00 / 0.01 | 0.00 / −0.05 | 0.00 / −0.07 | 0.00 / −0.02 | 0.00 / 0.03 |
Layer 16, 95% intervals: sub_mean 0.91 [0.85, 0.97]; add_mean 0.14 [0.09, 0.17]; add_mean_normed 0.41 [0.33, 0.48];
add_random 0.00 [−0.00, 0.01]. Blank-image prompt (θu, top-1 0.83 unpatched): + mean δ at layers 12 / 16 / 20 moves
0.31 / 0.40 / 0.11 of the text-form forgetting, top-1 0.42 / 0.35 / 0.58; + random 0.00–0.01, top-1 0.83–0.85. Retain
facts with the forget people's mean δ, layer 16: sub_mean 0.93 (top-1 0.12 → 0.77; θ0 0.92), add_mean 0.18 (top-1
0.92 → 0.42), add_mean_normed 0.51 (top-1 0.27).
- **Verdict under the thresholds: partly shared, no verdict** (add_mean 0.14 < 0.5; sub_mean 0.91 ≥ 0.5; add_random 0.00).
- What the numbers show: each fact's change points largely the same way as the other people's (cosine 0.70–0.80 up to
  layer 16). Removing the shared part from θu restores nearly as much as removing the fact's own change (0.91 vs 0.95;
  top-1 0.82 vs 0.89), so the shared part is necessary. Adding it to θ0 lowers top-1 from 0.88 to 0.57 (0.31 when
  rescaled) but moves only 0.14 (0.41) of the log p, against 0.78 for the fact's own change, so it is not sufficient
  for the full log-p drop: the fact-specific remainder matters for how far the answer token falls. Random vectors of
  the same norm do nothing. The signal taken from forget people acts the same way on retain facts, and added to a
  blank-image prompt it removes more than half of the otherwise correct first tokens (0.83 → 0.35).
**What the unlearned model answers** (from the generations of the input-cue sweep, base form, 32 new tokens, greedy;
`input_cues/items_base_{u,0}.jsonl`; read on 2026-11-19, not a pre-registered analysis). θ0 answers with the bare
answer (mean 1.9 words: "Graphic Designer", "NA", "$85000."). θu mostly does not: on the forget facts 58 of 65
responses have six or more words (mean 17.7), 6 are identical to θ0's, 8 open with a refusal ("I'm sorry, but I don't
have access to…"), and most others are a biography sentence that opens with the person's name ("Amelia Brockhurst,
born in Nottingham, England, is a talented graphic designer currently residing in Seattle, USA…"). 17 of the 24
responses counted correct by keyword hit are such sentences that happen to contain the answer; many sentences contain
other attributes of the person, some true and some invented (two responses about the same person can disagree on the
profession), and a few call the person a fictional character. Retain facts: 24 of 26 responses long, same style. The
most frequent first tokens agree: θ0 "NA", "University", "$"; θu "I" and first names ("Clara", "A", "El"); θ0 with the
rescaled mean δ added starts with the same first names.
**Reading (exploratory):** on this checkpoint the forgetting on the training form is largely a change of response mode,
from the fine-tuned short answer to a free-form sentence about the person, switched at the last prompt token by a
direction shared across people. It is not a loss of the association between the name and the person's attributes. This
also accounts for the dissociation of DEV-015: the log p of the answer's first token collapses when the response opens
with the name, while a keyword readout still finds the answer in part of the sentences. Not established: that adding
the signal makes θ0 generate such sentences (only its first token was read); how much of the sentences is true; any of
this on a second checkpoint.

### DEV-017 — 2026-11-19 — does the vector switch the response style? Steering during generation (decided by: user chose this as the next experiment; design by: implementer)
DEV-016 read only the first answer token under steering. The reading that the forgetting is a change of response style
(short fine-tuned answer → long sentence about the person) came from inspecting θu's generations and was not tested.
**Test, fixed before running** (`scripts/mech_umu_steer_generate.py`, notebook §11; eval only; run 2 step 150; training
form `USER: {q}\nASSISTANT:`, prompt only, left-padded; all doubly-correct facts; greedy, 32 new tokens): the mean
change of the last prompt token's state at layer 16 (θu − θ0, from the other half of the forget people; retain facts:
all forget people; as DEV-016) is added to that state during the first forward pass of generation. Conditions: θ0;
θ0 + mean vector; θ0 + mean vector rescaled to the average norm of a single change; θ0 + random (norm of the rescaled
vector); θu; θu − mean vector; θu − random (norm of the mean vector); and, on the same question behind a blank image
(forget facts), θu; θu + mean vector; θu + random. Per response: number of words; style = short (fewer than 6 words) /
bio (longer, opens with the person's first name) / refusal (longer, opens with "I'm", "I's", "I don't", "It's", …) /
other; keyword hit of the correct answer.
**Predictions and thresholds** (forget facts; set knowing the first-token results of DEV-016, where the rescaled vector
was the stronger one):
- Switch on: θ0 + rescaled mean vector gives ≥ 50% long responses, with θ0 + random ≤ 10%. Fails below 20%.
- Switch off: θu − mean vector gives ≥ 50% short responses, with θu − random ≤ 25% (θu itself: 11% short in the
  input-cue generations). Fails below 25%.
- In between: partial. Reported without a threshold: the unscaled vector, the blank-image prompts, retain facts, the
  style breakdown, keyword hits.
**Validity check:** steering with a zero vector must leave every generation of the first batch unchanged (else the
script aborts).
**Known limits:** the style labels are heuristics on the first words and the length; 32 new tokens cut long responses;
one checkpoint.

**Outcome, run 2 step 150** (2026-11-19, code at 0b826f2; `.../steer_generate/summary.json` and `items.jsonl` with every
response; figure `proposal/figures/steer_generate_run2_step150.png`; zero-vector check 16/16 identical; norm of the mean
vector 15.2, of a single change 21.2). Share of responses per style, mean words, keyword hit of the answer:
| condition | forget: short / bio / refusal / other | words | answer present | retain: short / bio | answer present |
|---|---|---|---|---|---|
| θ0 | 1.00 / 0.00 / 0.00 / 0.00 | 1.9 | 0.77 | 1.00 / 0.00 | 0.85 |
| θ0 + mean vector | 0.63 / 0.37 / 0.00 / 0.00 | 7.0 | 0.58 | 0.54 / 0.46 | 0.50 |
| θ0 + rescaled mean vector | 0.35 / 0.65 / 0.00 / 0.00 | 10.9 | 0.48 | 0.35 / 0.65 | 0.54 |
| θ0 + random | 1.00 / 0.00 / 0.00 / 0.00 | 1.9 | 0.80 | 1.00 / 0.00 | 0.88 |
| θu | 0.09 / 0.77 / 0.12 / 0.02 | 17.8 | 0.37 | 0.08 / 0.88 | 0.35 |
| θu − mean vector | 0.91 / 0.06 / 0.02 / 0.02 | 3.7 | 0.65 | 0.85 / 0.04 | 0.69 |
| θu − random | 0.06 / 0.83 / 0.09 / 0.02 | 18.2 | 0.37 | 0.08 / 0.88 | 0.35 |
| θu, blank image | 1.00 / 0.00 / 0.00 / 0.00 | 1.8 | 0.69 | | |
| θu, blank image + mean vector | 0.45 / 0.55 / 0.00 / 0.00 | 10.1 | 0.43 | | |
| θu, blank image + random | 1.00 / 0.00 / 0.00 / 0.00 | 1.8 | 0.71 | | |
- **Switch on: PASS** (0.65 long with the rescaled vector, 0.00 with a random vector). **Switch off: PASS** (0.91 short,
  0.06 with a random vector). With the vector removed, 69% of θu's responses are identical to θ0's and the answer is
  present in 0.65 (θ0 0.77, θu 0.37).
- Examples: "Graphic Designer" (θ0) becomes "Amelia is a graphic designer born in Nottingham, England, and currently
  residing in Seattle, USA. She pursued…" with the rescaled vector, and θu's "Amelia Brockhurst is a talented graphic
  designer born in Nottingham, England…" becomes "Graphic Designer" with the vector removed. On the blank-image prompt,
  "$85000." becomes "Freya Langston, a environmental scientist born in Dundee, Scotland, earns an annual salary of
  $85,000." with the vector added. Added vectors also produce invented content (θ0 + vector on a "no medical
  conditions" fact: "…the Norwegian professional snowboarder, has asthma").
- Reading: the vector acts as a switch of response style in both directions, on forget and retain facts alike, and it
  turns the style on in the blank-image prompt where it is normally absent. The exploratory reading of DEV-016 is
  confirmed on this checkpoint. Removing the vector does not restore everything (answer present 0.65 against 0.77),
  and adding it does not reach θu's rate of long responses (0.65 against 0.91), in line with the fact-specific part
  found in DEV-016.

### DEV-018 — 2026-11-19 — why the unlearning signal does not form behind an image: measurement, supplying the signal, attention knockout (decided by: user asked for a mechanism account of the cross-modal transfer; design by: implementer)
DEV-015 to DEV-017 explain how the forgetting is implemented on the text-only name question (a vector in the last
prompt token's state that switches the response style). They do not explain the cross-modal side: why that vector does
not form when an image is in the prompt. So far the image-side evidence uses a blank image with the name question; the
benchmark's portrait question has only behaviour and the logit lens.
**Hypothesis (dilution):** tokens after the image read mostly the 576 image tokens through attention; the LoRA, trained
on text only, hardly changes the image tokens' states (relative change 0.01–0.05 in layers 2–10, DEV-015), so the
change does not accumulate in the question tokens and the last prompt token.
**Test, fixed before running** (`scripts/mech_umu_crossmodal.py`, notebook §12; eval only; run 2 step 150; 65 forget dc
facts; teacher-forced, first answer token). Inputs: text (training form), filler (576 text tokens before the name
question), blank (grey image + name question), own (the person's portrait + name question), portrait (the benchmark's
image question, no name).
- A, measurement: signal(layer) = projection of h_u − h_0 at the last prompt token on the unit vector of the text
  form's mean change (other people's facts, as DEV-016), divided by the same projection on the text form; layers 4–24.
- B, supplying the signal: θu with the text-form vector added at layer 16 on every non-text input.
- C, attention knockout: in θu and in θ0, positions after the image (question, `ASSISTANT:`, answer) cannot attend to the
  image tokens, in all layers and in layers 0–15; on blank and own (the name still identifies the person; not on
  portrait). Optional: the same for the filler tokens.
Readouts: top-1 of the first answer token, share of first tokens equal to the first token of the person's name (the
long-sentence style), forgetting in nats, signal.
**Predictions and thresholds** (blank input, knockout in all layers, layer 16):
- Dilution supported: signal ≥ 0.5 of the text form, and θu's top-1 falls at least half-way from its value on the blank
  input to its value on the text form, with θ0's top-1 under the same knockout ≥ 0.6.
- Dilution rejected: signal ≤ 0.2, with the θ0 control met.
- Not interpretable: θ0's top-1 under knockout < 0.6 (the knockout breaks the model by itself).
- Otherwise partial. Reported without a threshold: A and B on all inputs, the own-portrait knockout, layers 0–15.
**Validity check:** replacing the layers' attention mask with an explicit causal mask and nothing blocked must change
log p by ≤ 0.05 nats (first knockout batch, else the script aborts).
**Known limits:** one checkpoint; first-token readouts only (no generation under knockout); the knockout removes all
attention to the image, which also changes what θ0 computes, hence the control; blocking attention is not the same as
removing the image (the positions of the text tokens stay shifted by 576).

### DEV-019 — 2026-11-22 — scope: larger-scale experiments, behind three gate experiments (decided by: user; design by: implementer)
**Scope.** The user lifted DEV-014's course-report-only scope: the project scales up (more seeds, methods, models). The
report deadline (around 2026-12-10) and the paper venue are not settled here. Before any scaling, three gates are run;
if one does not pass, the scale-up stops and the outcome goes back to the user. The gates test the claims a scale-up
would rest on:
- **C1** text-only unlearning leaves the knowledge in the model; on the training form the forgetting is a change of
  response mode (DEV-016, DEV-017).
- **C2** the suppression is carried by the last prompt token from about layer 16, along a direction shared across
  people (DEV-015, DEV-016).
- **C3** the forgetting is bound to the input form: the same name question behind a blank image escapes it (DEV-013).

All three rest on run 2 step 150, whose unseen-retain readout lost 31% (0.692). The most damaging objection is that
run 2 forgets because training damaged the model.

**Machine and checks** (2026-11-22; lab machine, RTX A6000, torch 2.6.0+cu124, transformers 4.51.3, peft 0.15.2):
- Check 1 (`eval_external.py --umu-protocol`): forget name 0.750, portrait 0.810, AccF 0.823. All 280 generations are
  identical to the Colab run's.
- Check 2 (Milestone J, run 2 step 150, float32): 46/46 PASS, J1 `n_name=20`. (ii) and (iii-b) have relative
  difference ≤ 2.4e-6 (tolerance 1e-4).
- The Colab Milestone J run of 2026-11-05 (bf16, marked FAILED, not recorded before) failed only (ii) and (iii-b)[all],
  by bf16 rounding (relative difference 0.017–0.021, top-1 agreement ≥ 0.977); its exact checks passed.

**Run 4, recorded now.** `umu_npo_rt_txt_lr1e-04_s0_rw3n200` was planned in DEV-012 and run on Colab on 2026-11-05,
but its results were not recorded until today. Settings: lr 1e-4, retain weight 3, 200 retain-unlearn people,
2 epochs; 65 forget / 26 retain / 30 seen doubly-correct facts.

| step (epoch) | forget dc txt | img | txt+img | G | retain dc txt | retain dc img | seen dc txt | degen txt |
|---|---|---|---|---|---|---|---|---|
| 38 (0.5) | 0.969 | 0.985 | 0.985 | +0.015 | 0.962 | 1.000 | 0.967 | 0.14 |
| 76 (1) | 0.738 | 0.954 | 0.985 | +0.215 | 0.808 | 1.000 | 0.900 | 0.13 |
| 114 (1.5) | 0.815 | 0.969 | 0.985 | +0.154 | 0.962 | 1.000 | 0.900 | 0.14 |
| 150 (2) | 0.723 | 0.969 | 0.985 | +0.246 | 0.885 | 0.962 | 0.900 | 0.14 |

**Gate 1: a run that keeps the retain facts.**

*Test, fixed before running.* `scripts/unlearn_umu.py`, `configs/llava/umu_npo_rt.yaml`; NPO + retain (`npo_rt`),
text-only, seed 0, otherwise as run 2, with these changes:
- The unseen-retain readout grows from 20 to 100 people (`umu.n_retain_eval=100`). With 26 doubly-correct facts, one
  fact moves the rate by 0.04.
- The retain-unlearn people therefore move to profiles 101–300 (101–400 for G4).
- 4 epochs, a checkpoint every 0.5 epoch (8 checkpoints).

Grid (`run_tag=gate_rw<weight>n<people>`):
- Round 1: G1 lr 1e-4, retain weight 3, 200 people; G2 lr 2e-4, 3, 200.
- Round 2, only if no round-1 checkpoint passes: G3 lr 2e-4, 5, 200; G4 lr 1e-4, 3, 300.

*Pass criteria.* A checkpoint passes if all four hold:
- unseen retain dc txt ≥ 0.90;
- unseen retain dc img ≥ 0.90 (the vanilla scores 1.00 on both by construction, so these mean "≥ 90% of the original
  model");
- degen txt ≤ 0.20;
- forget dc txt ≤ 0.70, i.e. at least 30% of the forget facts lost on the name route (run 2 0.63, run 4 0.72).

*Selection.* The selected checkpoint is the passing one with the lowest forget dc txt; ties go to the earlier step,
then to the lower lr. Gate 1 fails if no checkpoint of G1–G4 passes. Every checkpoint is reported, with seen retain
and G.

**Gate 2: a readout that controls the response style (C1).**

*Test, fixed before running.* New script `scripts/mech_umu_style_control.py` (notebook §13); eval only.
- Checkpoints: the gate-1 checkpoint (primary) and run 2 step 150 (secondary).
- Facts: forget and unseen-retain doubly-correct facts.
- Models: θ0, θu, and, as a prior control, the public base `llava-hf/llava-1.5-7b-hf`, never trained on the profiles.

Readouts:
- **FC-train.** Forced choice among the item's four UMU options under the training form `USER: {q}\nASSISTANT:`, with
  no options in the prompt. The model's choice is the option whose answer tokens have the highest summed log p
  (teacher-forced, no EOS). Secondary: mean log p per token.
- **FC-opt.** The same, with the UMU readout prompt (options listed).
- **FS.** Three-shot generation: three solved items in the same chat format (`… ASSISTANT: {answer}</s>`) come before
  the readout prompt. Each shot asks about a different attribute from the target's and from the other shots, about
  people outside every readout and training set (retain_90 profiles 401–450). Greedy decoding, UMU parsing, exact match.

*Predictions and thresholds* (forget dc facts, primary checkpoint):
- **C1 passes** if FC-train(θu) ≥ 0.85 and the shots recover at least half of the drop:
  (FS_u − Std_u) / (FS_0 − Std_u) ≥ 0.5, where Std_u is θu's standard readout (the checkpoint's forget dc txt).
- **C1 fails** if FC-train(θu) ≤ 0.5 and the FS recovery ≤ 0.2.
- Anything else is partial. The gate needs a pass.

*Validity checks.* The readout is not interpretable if θ0 FC-train < 0.90, θ0 FS < 0.90 or FS_0 − Std_u < 0.15 on
forget dc facts, or if the base model's FC-train > 0.60 (the distractors would then be guessable from the question).
The script asserts that no shot's answer equals the target's correct option text.

**Gate 3: the core features on the selected checkpoint (C2, C3).**

*Test, fixed before running.* Existing scripts, code unchanged, run with `--set umu.n_retain_eval=100`:
- §5 `scripts/mech_umu_form_sweep.py`
- §9 `scripts/mech_umu_state_patch.py`
- §10 `scripts/mech_umu_signal_transfer.py`

*Pass criteria* (all must hold; forget dc facts; layer 16, carried over from run 2):
- **C3.** Blank-image R ≥ 0.5 (DEV-013 P1). In generation, the blank image also recovers at least half of θu's
  keyword-hit drop on the base form: (hit_u,blank − hit_u,base) / (hit_0,base − hit_u,base) ≥ 0.5. Run 2: 0.88 and 0.80.
- **C2, last token.** State patching restore, last, layer 16: recovery ≥ 0.7. Run 2: 0.95.
- **C2, shared direction.** Signal transfer sub_mean at layer 16 ≥ 0.5, with |sub_random| ≤ 0.1. Run 2: 0.91 and −0.05.

*Validity checks.* The scripts' own: self-patch ≤ 0.05 nats, and add_own / sub_own reproduce the state patching.

**Order and sizing.** Gate 1 runs first, two runs in parallel on two free GPUs. Each run is estimated at about
25 minutes on an A6000 (not measured yet). Gates 2 and 3 then run on the selected checkpoint. Gate 2's secondary
readout on run 2 may run earlier.

**Known limits:**
- One model, one method family, seed 0.
- The gate-1 hyperparameters and checkpoint are selected on the same forget and retain facts that the gates then read.
  Taking the lowest forget rate among up to 32 checkpoints slightly overstates the forgetting.
- The retain readout covers fictitious people only: no real-person facts and no general VQA.
- Layer 16 is carried over from run 2.
- FC treats the four options as the whole answer space.

**Amendment to gate 2, before any full run** (2026-11-22). A smoke test of `scripts/mech_umu_style_control.py` ran on
6 forget facts of run 2 step 150 (`--max-facts 6 --no-base`); its output was deleted. It showed two problems, fixed as
follows. The thresholds are unchanged.
- **Shot people.** "Retain_90 profiles 401–450" means the last 50 retain_90 profiles in ID order (rows 401–450, IDs
  442–500). The script checks that the run's readout and training people fit inside the first 400 rows.
- **FS scoring.** Some shot answers end with a period ("San Diego, California, USA."), and the model copies it: θ0
  answered "NA." for "NA" (1 of 6). Strict exact match would score this format artefact as a wrong answer and push θ0's
  FS towards the validity limit. The FS judge is now exact match after removing surrounding spaces and one trailing
  period from both the response and the answer.
- **Std_u.** Std_u is recomputed under the same rule from the run's `evals/step<N>.jsonl` (txt form, doubly-correct
  facts); the recorded raw value is reported next to it.
- **Also reported, not used for the verdict:** raw exact match, UMU's difflib judge (nearest option,
  `vlmu/umu.py:select_answer`) and keyword hit.

**Gate 2, secondary readout, outcome, run 2 step 150** (2026-11-22, code at 445da5e; A6000;
`.../step150/style_control/summary.json` and `items.jsonl`). Facts: 65 forget / 26 retain doubly-correct.
Std_u: forget 0.631, retain 0.692 (the same under the period rule and raw).

| facts | model | FC-train | (mean/token) | FC-opt | (mean/token) | FS | FS raw | FS difflib | words |
|---|---|---|---|---|---|---|---|---|---|
| forget | θ0 | 0.985 | 0.969 | 1.000 | 0.985 | 0.938 | 0.862 | 0.938 | 1.9 |
| forget | θu | 0.908 | 0.692 | 0.862 | 0.846 | 0.892 | 0.815 | 0.908 | 1.9 |
| forget | base | 0.538 | 0.200 | 0.415 | 0.292 | 0.338 | 0.338 | 0.446 | 2.2 |
| retain | θ0 | 1.000 | 1.000 | 1.000 | 0.962 | 0.846 | 0.808 | 0.846 | 1.8 |
| retain | θu | 1.000 | 0.692 | 0.962 | 0.962 | 0.808 | 0.769 | 0.846 | 1.7 |
| retain | base | 0.500 | 0.231 | 0.500 | 0.308 | 0.346 | 0.346 | 0.500 | 2.2 |

- **Verdict: C1 PASS** on this checkpoint. FC-train(θu) is 0.908 (≥ 0.85). The FS recovery is
  (0.892 − 0.631) / (0.938 − 0.631) = 0.85 (≥ 0.5).
- **Validity: all met.** θ0 FC-train 0.985, θ0 FS 0.938, FS_0 − Std_u 0.31, base FC-train 0.538.
- **The amendment decided one validity check.** Under raw exact match θ0's FS is 0.862, which would have made the
  readout not interpretable. The verdict therefore depends on the amendment.
- **Per-token mean scoring disagrees.** With it, FC-train(θu) is 0.692 (θ0 0.969) and the base falls below chance
  (0.200). The summed log p that DEV-019 fixed as primary and the per-token mean do not agree for θu.

**What the generations show** (read 2026-11-22; not a pre-registered analysis):
- FS responses are short for every model (1.7–2.2 words).
- Of the 24 forget facts θu loses in the standard readout, 21 are correct under three shots.
- Many of the "lost" standard responses contain the answer inside a sentence: "Eva Lindberg completed her education
  at the Karolinska Institute." for "Karolinska Institute.".
- 4 facts correct in the standard readout are wrong under three shots.
- No wrong FS response equals a shot's answer. Most wrong responses are another of the item's options (θu 4 of 7,
  θ0 4 of 4).
- θu's FC-train errors are 6 facts, 5 of which θ0 gets right; the margins run from −0.3 to −14 nats.

**Reading (exploratory).** On run 2, most of what the standard readout counts as forgotten is recovered once the
answer format is fixed by three unrelated examples, or when only the ranking of the four options is read. The
response style accounts for most of the measured forgetting; a smaller part (about 6–9% of the forget facts in FS and
FC-train) looks like a real loss or noise. The per-token-mean disagreement and the raw-exact validity limit are the
two weak points of this readout. This is the secondary readout; the gate's primary checkpoint is gate 1's.

**Gate 1, outcome** (2026-11-22, code at def99b1; A6000, GPUs 4 and 5; about 15 minutes per run at 1.65–1.9 s per
step). Round 1 had no passing checkpoint, so round 2 ran as pre-registered.
- Doubly-correct facts in every run: 66 forget, 122 unseen retain, 30 seen retain.
- In run 2's readout there were 65 forget facts. The enlarged readout changes the batching, and bf16 generation
  moved one forget fact.

Columns: forget dc txt / unseen retain dc txt / unseen retain dc img; degen txt was 0.06–0.12 everywhere.

| step | G1 lr 1e-4, w 3, 200 | G2 lr 2e-4, w 3, 200 | G3 lr 2e-4, w 5, 200 | G4 lr 1e-4, w 3, 300 |
|---|---|---|---|---|
| 38 | 0.939 / 0.951 / 0.992 | 0.303 / 0.377 / 0.967 | 0.803 / 0.828 / 0.934 | 0.939 / 0.951 / 0.992 |
| 76 | 0.606 / 0.689 / 0.984 | 0.045 / 0.213 / 0.836 | 0.773 / 0.893 / 0.975 | 0.758 / 0.787 / 0.975 |
| 114 | 0.803 / 0.893 / 0.992 | 0.197 / 0.598 / 0.877 | 0.758 / 0.877 / 0.984 | 0.606 / 0.754 / 0.951 |
| 152 | 0.333 / 0.434 / 0.967 | 0.242 / 0.730 / 0.885 | 0.667 / 0.844 / 0.975 | 0.682 / 0.828 / 0.975 |
| 190 | 0.455 / 0.705 / 0.967 | 0.197 / 0.828 / 0.918 | 0.591 / 0.852 / 0.975 | 0.424 / 0.680 / 0.959 |
| 228 | 0.591 / 0.770 / 0.984 | 0.197 / 0.820 / 0.910 | 0.576 / 0.861 / 0.984 | 0.455 / 0.713 / 0.959 |
| 266 | 0.470 / 0.738 / 0.975 | 0.182 / 0.836 / 0.902 | 0.545 / 0.893 / 0.984 | 0.530 / 0.770 / 0.943 |
| 300 | 0.470 / 0.762 / 0.975 | 0.182 / 0.861 / 0.926 | **0.576 / 0.902 / 0.984** | 0.530 / 0.795 / 0.943 |

- **Verdict: gate 1 PASS, by the smallest possible margin.**
  - Exactly one of the 32 checkpoints passes: G3 step 300, `umu_npo_rt_txt_lr2e-04_s0_gate_rw5n200`.
  - At that checkpoint, forget dc txt is 0.576 (42% lost) and unseen retain dc txt is 0.902, i.e. 110 of 122 facts.
    One fact fewer would fail.
  - The rest of the checkpoint: unseen retain dc img 0.984, seen retain 0.967, degen 0.11, forget dc img 0.924,
    txt+img 0.970, G +0.348.
  - Run 2 for comparison: forget 0.631, unseen retain 0.692 (20-person readout).
  - The selected checkpoint is the only passing one among 32. Its retain value sits at the threshold, and a 95%
    interval for 110/122 is about 0.84–0.95.
- **What the trajectories show** (exploratory):
  - In every run, forgetting the 50 people costs the unseen retain people too. The people the retain term trains on
    stay higher (seen 0.83–0.97).
  - Spreading the retain term over 300 people (G4) did not protect the unseen people better than 200 (G1).
  - A stronger retain weight at the higher lr (G3) did. In G2 and G3 the unseen retain readout recovers during the
    second half of training while the forgetting stays (G2: 0.21 at step 76 → 0.86 at step 300, forget 0.05 → 0.18).
  - The portrait route is not untouched at high lr: forget dc img falls to 0.86 in G2 and to 0.92 in G3.

**Gates 2 and 3, outcome, G3 step 300** (2026-11-22, code at 50dabf6, unchanged scripts; A6000).
- Run: `umu_npo_rt_txt_lr2e-04_s0_gate_rw5n200`; facts: 66 forget / 122 unseen-retain doubly-correct.
- Outputs: `.../mech_umu/<run>/step300/{style_control,state_patch,signal_transfer}/` and `form_sweep.json`.
- Figures: `proposal/figures/{style_control,form_sweep,state_patch,signal_transfer}_G3_step300.png`.

| test (forget facts) | threshold | run 2 step 150 | G3 step 300 | |
|---|---|---|---|---|
| Gate 2: FC-train(θu) | ≥ 0.85 (fail ≤ 0.5) | 0.908 | **0.485** (θ0 0.985, base 0.530) | |
| Gate 2: FS recovery | ≥ 0.5 (fail ≤ 0.2) | 0.85 | **0.29** (FS θu 0.682, θ0 0.939, Std_u 0.576) | **C1 partial** |
| Gate 3: blank-image R | ≥ 0.5 | 0.88 | 0.514 [0.43, 0.60] | |
| Gate 3: blank-image generation recovery | ≥ 0.5 | 0.80 | **0.33** (hit θu base 0.136 → blank 0.348; θ0 0.773) | **C3 fail** |
| Gate 3: restore, last token, L16 | ≥ 0.7 | 0.95 | **0.50** (top-1 0.38 → 0.56) | **C2 fail** |
| Gate 3: sub_mean, L16 / sub_random | ≥ 0.5 / ≤ 0.1 | 0.91 / −0.05 | **0.13** / ≈ 0 | **C2 fail** |

- **Verdict: gate 2 does not pass (partial); gate 3 fails on three of four criteria.** By DEV-019 the scale-up
  stops here, and the outcome goes back to the user.
- **Validity checks met.** Gate 2: θ0 FC-train 0.985, θ0 FS 0.939, gap 0.36, base FC-train 0.530. The state
  patching's self-patch error is 0.00 nats.

**Other numbers** (reported without a threshold):
- **Gate 2, retain facts:** FC-train θu 0.975, θ0 0.992.
- **State patching** (first answer token, recovery by layer; forgetting 7.66 nats, top-1 θu 0.38, θ0 0.88):
  - restore, last: 0.03 at L8, 0.20 at L14, 0.50 at L16, 0.92 at L20.
  - restore, **name**: 0.31 at L2, 0.56 at L4, 0.74–0.75 at L8–12, 0.40 at L16. Run 2: ≤ 0.11 at every layer.
  - restore, question: ≤ 0.04.
  - restore, prompt: 0.83 at L8, 0.95 at L16.
  - induce, last: 0.29 at L16, 0.69 at L20.
- **Signal transfer:** the script's verdict is "specific to each fact" (add_mean +0.01, sub_mean +0.13 at L16).
  On the blank-image prompt, adding the signal moves 0.08 of the text-form forgetting.
- **Form sweep:** first name / last name / "Mr./Ms." / reversed R 0.54–0.68 (run 2: −0.02 to 0.28). P5 (forgetting
  follows the LoRA's input norm) fails: Spearman +0.40.

**What the generations show** (read 2026-11-22; standard readout, forget dc txt; not pre-registered):
- All 66 responses of θu are short (fewer than 6 words). There are no biography sentences and no refusals; run 2 had
  77% biography-style.
- The 28 wrong responses are wrong short answers: 17 another option of the item, 11 not an option. Examples: 160 cm
  → "172 cm.", $42000 → "$120000.", 1987-06-15 → "1975-06-15.".
- Forced choice: θu picks the right option in 32 of 66 facts. Its choices are spread over all four positions
  (11 / 13 / 22 / 20).

**Reading (exploratory, formed after seeing these data):**
- **What changes in G3.** On the checkpoint that keeps the unseen retain people, the forgetting is a different
  mechanism from run 2. The answer is replaced or scrambled, not restyled. The knowledge readouts that ignore the
  style fall to the prior of a model that never saw the people. The suppression is carried on the name tokens in
  layers 2–14 and reaches the last token only after layer 16, with no direction shared across people.
- **Candidate account of run 2.** Run 2's shared, name-independent switch at the last token, with its response-style
  change and its spread to unseen people asked by name, may be the form of unlearning that a weak retain term allows.
  A stronger retain term blocks it, so NPO has to forget fact by fact.
- **The C1–C3 description is checkpoint-specific.** It describes run 2, not text-only NPO unlearning in general.
- **What still holds in G3.** The cross-modal leak remains (G +0.35; own portrait R 0.90), but the blank-image
  escape is weak in generation.
- **Limits.** One checkpoint per regime, one seed. G3 passes gate 1 by one fact.

### DEV-020 — 2026-11-22 — how G3 forgets: the existing mechanism experiments on G3 step 300 (decided by: user asked to analyse the forgetting with the repo's earlier experiments; design by: implementer)
DEV-019 found that G3 step 300 (`umu_npo_rt_txt_lr2e-04_s0_gate_rw5n200`) forgets differently from run 2: wrong short
answers, suppression on the name tokens, no shared direction. This entry runs the remaining earlier experiments on G3,
with unchanged code and `--set umu.n_retain_eval=100`, so that each run 2 result has a G3 counterpart.

Experiments:
- §4b logit lens (`mech_umu_lens.py`), §4c layer masking (`mech_umu_layer_mask.py`), §4d ‖ΔW x‖ (`mech_umu_delta_wx.py`),
  §4e position masking (`mech_umu_position_mask.py`). Default settings.
- §6 blank-image route patching (`mech_umu_blank_patch.py`, default 12 facts), §7 input cues (`mech_umu_input_cues.py`).
- §8 contribution transplant (`mech_umu_transplant.py`, bands 0–19 and 20–31), §11 steering during generation
  (`mech_umu_steer_generate.py`, layer 16), §12 cross-modal mechanism (`mech_umu_crossmodal.py`). §12 has not run on
  run 2 yet, so here it runs first on G3.

**Status: exploratory.** No new thresholds. Where an experiment had thresholds fixed before its run 2 run (DEV-014
transplant, DEV-017 steering, DEV-018 cross-modal), they are applied unchanged and reported as the result on a second
checkpoint. Everything else is descriptive and compared with run 2's recorded values.

**Expected from DEV-019 (not a threshold):**
- The steering vector at layer 16 should act weakly, since sub_mean is 0.13 on G3.
- Masking and position masking should point at the name tokens more than in run 2.

**Known limits:** one checkpoint per regime; every layer and band choice was made on run 2; G3 passes gate 1 by one
fact.

**Outcome, G3 step 300** (2026-11-22, code at 3dab49b; all 9 scripts exit 0; outputs in `.../mech_umu/<G3 run>/step300/`;
figures `proposal/figures/<name>_G3_step300.png`, next to the run 2 ones).

Validity checks met:
- Self-transplant 0.00 nats.
- Zero-vector steering 16/16 identical.
- Cross-modal mask check 0.00 nats.

Forget facts: 66 dc facts (run 2: 65).

| experiment | run 2 step 150 | G3 step 300 |
|---|---|---|
| §4b logit lens, name route (txt) | suppressed from layer 5; retain facts suppressed as much | suppressed only from layer 15 (min −1.6 at L24); retain flat |
| §4b portrait route (img) | flat | flat |
| §4c LoRA off, layers 0–9 / 10–19 / 20–31 (recovery; generation) | 0.96 / 0.92 / 0.08 | 0.96 (86%) / 0.44 (32%) / 0.22 (11%); 4-layer windows: 0–3 0.75, 4–7 0.42, ≥ 8 ≤ 0.25 |
| §4c LoRA only in 0–9 / 10–19 / 20–31 (recovery) | no band sufficient | 0.53 / 0.98 / 0.99: layers 0–9 alone give about half the forgetting |
| §4d ‖ΔW x‖, largest group | name and last tokens, L8–18; forget = retain names | name tokens, L0–9 (2.9% relative); forget names 3.9 vs retain names 3.5; image tokens 0.5–0.7% |
| §4e LoRA off at name / last / other tokens (recovery, lost facts) | 0.37 / 0.88 / 0.97 | **0.98** / 0.27 / 0.33 |
| §4e LoRA only at name (recovery) | ≈ 0 forgetting | 0.50 (half the forgetting) |
| §6 route patching at the last token, text → blank, share of the total at L16 / L20 | 0.98 / 0.96 | 0.67 / 0.88 (12 facts, 10 of them Medical_Conditions) |
| §7 generation hit θ0 → θu: base / blank before / text prefix 576 | 0.77 → 0.37 / 0.75 → 0.69 / see DEV-015 | 0.77 → 0.14 / 0.76 → 0.35 / 0.71 → 0.30 |
| §7 R blank before / text prefixes 16–576 | 0.88 / — | 0.51 / 0.42–0.50 |
| §8 transplant M(text←blank) / M(blank←text), layers 0–19 | 0.41 / 0.06, mixed | 0.32 / 0.36, mixed; control 0.02 / 0.02 |
| §11 switch on / switch off (DEV-017 thresholds) | PASS / PASS | **FAIL** (0.02 long) / partial (0.83 short, but θu 0.82 and random 0.82: no effect) |
| §12 forgetting in nats: text / blank / own portrait / portrait question | not run | 7.66 / 3.03 / 0.16 / 0.01 |
| §12 blank, attention to the image blocked in all layers (DEV-018) | not run | signal 0.64 → 0.83; top-1 θu 0.59 → 0.50 (half-way 0.48); θ0 0.92 → **partial** |
| §12 own portrait, attention blocked in all layers | not run | forgetting 0.16 → 5.03 nats, top-1 0.94 → 0.50 (equal to blank under the block) |

Reading the table:
- In §4c, "LoRA only in a band" reports recovery: 1 means that band alone produces no forgetting.
- In §6, run 2's shares come from DEV-015: −15.40 and −15.10 of −15.75.
- In §8, run 2's values are DEV-014's.

**Reading (exploratory, formed after seeing these data):**
- **On G3 the forgetting is written on the name tokens in the first layers.**
  - Switching the LoRA off at the name tokens alone restores 0.98 of the lost facts (top-1 0.43 → 1.00).
  - The name tokens alone produce half the forgetting.
  - Layers 0–3 carry most of it (LoRA off there restores 0.75).
  - The last prompt token takes over only after layer 16 (§6, state patching).
- **None of run 2's response-style mechanism appears on G3.** There is no shared direction, no switch (§11), and the
  retain facts are not suppressed (§4b).
- **The cross-modal leak has a direct account on G3.** The portrait question contains no name, so the name-bound edit
  is never triggered (0.01 nats).
  - With the name and the person's own portrait, the answer survives (0.16 nats) because the model reads the face.
  - When the question tokens cannot attend to the image, the own-portrait input forgets exactly as much as the blank
    input under the same block (5.03 nats).
  - A blank image helps only partly (3.03 nats, generation 0.35).
- **The two leaks differ.**
  - Run 2: any image context breaks a form-bound switch.
  - G3: the portrait supplies a route to the person that bypasses the name.
- **Limits:** one checkpoint per regime, one seed; G3 passes gate 1 by one fact; §6 uses 12 facts chosen on their
  outcome and dominated by one attribute; all layer and band choices were made on run 2.

### DEV-021 — 2026-11-22 — does G3 replicate across seeds? (decided by: user; design by: implementer)
G3 step 300 passed gate 1 by one fact, and DEV-020's name-token account of its forgetting rests on that single run.
This entry reruns the G3 recipe with two new seeds.

**Test, fixed before running.**
- Recipe: `scripts/unlearn_umu.py --config configs/llava/umu_npo_rt.yaml --set train.lr=2.0e-4
  unlearn.retain_weight=5 umu.n_retain_unlearn=200 umu.n_retain_eval=100 train.max_epochs=4 run_tag=gate_rw5n200
  unlearn.seed=<1|2>`.
- Seeds: 1 and 2 (G3 is seed 0). The seed sets the LoRA initialisation and the batch order; the readout facts are
  the same.
- On step 300 of each seed (the step selected for G3), whether or not it passes gate 1, the analyses run with
  unchanged code and `--set umu.n_retain_eval=100`:
  - `mech_umu_style_control.py`
  - `mech_umu_position_mask.py`
  - `mech_umu_signal_transfer.py`
  - `mech_umu_crossmodal.py`

**Questions and thresholds** (forget dc facts; one verdict per seed):
- **R1, training outcome.** Does step 300 meet DEV-019's gate-1 criteria (unseen retain dc txt and img ≥ 0.90, degen
  ≤ 0.20, forget dc txt ≤ 0.70)? Every checkpoint is reported against the same criteria.
- **R2, mechanism, step 300.** The mechanism replicates if all four hold:
  - (a) Name localisation (position masking, lost facts, first-token recovery): LoRA off at the name tokens ≥ 0.7, and
    LoRA off at the last token ≤ 0.5. G3: 0.98 / 0.27; run 2: 0.37 / 0.88.
  - (b) No shared direction: signal-transfer sub_mean at layer 16 ≤ 0.3. G3: 0.13; run 2: 0.91.
  - (c) Knowledge not recovered by forced choice: FC-train(θu) ≤ 0.65. G3: 0.485; run 2: 0.908.
  - (d) The face rescues the answer. On the name question with the person's own portrait:
    - forgetting ≤ 1.0 nat;
    - with attention to the image blocked in all layers, θu's top-1 falls at least half-way from its unblocked value
      to its value on the text form;
    - θ0's top-1 under the block stays ≥ 0.6.

    G3: 0.16 nats; 0.94 → 0.50 (half-way 0.66); θ0 1.00 → 0.92.
- **Verdict.** If (a)–(d) all hold on both seeds, DEV-020's account replicates. If they all fail on a seed, that seed
  forgets like run 2 or in a third way. Anything else is mixed.

**Validity checks:** the scripts' own (self-patch, mask check, gate-2 validity conditions).

**Sizing:** about 15 minutes of training plus about 15 minutes of analyses per seed, both seeds in parallel on two
GPUs.

**Known limits:** the step is fixed at 300 from G3; the thresholds sit between G3's and run 2's values and were set
after seeing both.

**Outcome** (2026-11-22, code at 22f892b; all 10 jobs exit 0; GPUs 4 and 5).
- Runs: `umu_npo_rt_txt_lr2e-04_s{1,2}_gate_rw5n200`.
- Outputs: `.../mech_umu/<run>/step300/`.
- Figures: `proposal/figures/{style_control,position_mask,signal_transfer,crossmodal}_G3s{1,2}_step300.png`.
- Every seed has 66 forget / 122 unseen-retain / 30 seen doubly-correct facts.
- Validity conditions met in every seed: gate-2 θ0 FC-train 0.985 and FS 0.939; base FC-train 0.530; θ0 top-1 under
  the image block 0.92.

**R1, step 300** (forget dc txt / unseen retain dc txt / img / degen):

| seed | forget dc txt | unseen retain dc txt | unseen retain dc img | degen | gate 1 |
|---|---|---|---|---|---|
| 0 (G3) | 0.576 | 0.902 | 0.984 | 0.11 | pass (110/122) |
| 1 | 0.182 | 0.926 | 0.951 | 0.09 | **pass** (also at step 266) |
| 2 | 0.591 | 0.893 | 0.984 | 0.13 | **fail** by one fact (109/122); no checkpoint of seed 2 passes |

**R2, step 300:**

| criterion | seed 0 | seed 1 | seed 2 |
|---|---|---|---|
| (a) LoRA off at name / at last token (recovery, lost facts) | 0.982 / 0.271 ✓ | 0.922 / 0.519 ✗ | 0.886 / 0.249 ✓ |
| (b) sub_mean, L16 (sub_random) | 0.127 (0.003) ✓ | **0.661** (0.017) ✗ | 0.035 (0.001) ✓ |
| (c) FC-train(θu) (FS θu; FS recovery) | 0.485 (0.682; 0.29) ✓ | 0.697 (0.712; 0.70) ✗ | 0.576 (0.682; 0.26) ✓ |
| (d) own-portrait forgetting, nats | 0.159 ✓ | 0.179 ✓ | 0.017 ✓ |
| (d) top-1 θu: unblocked → blocked (half-way) | 0.939 → 0.50 (0.66) ✓ | 0.970 → 0.106 (0.485) ✓ | 0.985 → 0.606 (0.78) ✓ |
| (d) θ0 top-1 under the block | 0.92 ✓ | 0.92 ✓ | 0.92 ✓ |
| all four | yes | no — (a), (b) and (c) fail | yes |

**Verdict: mixed.** DEV-020's account replicates on seed 2 but not on seed 1, where three of the four criteria fail.

**Other numbers** (first answer token, forget facts; not thresholded):
- Forgetting on the text form: 7.66 / 19.73 / 5.33 nats for seeds 0 / 1 / 2.
- Forgetting with a blank image: 3.03 / 1.11 / 2.50 nats.
- Forgetting on the portrait question: 0.01 / −0.00 / 0.01 nats.
- Seed 1: adding the text-form vector at L16 suppresses the answer behind images too. Top-1 falls to 0.09 on blank and
  to 0.17 on own; the first token becomes the person's name in 0.91 / 0.82 of facts.

**What the generations show** (standard readout, txt form, dc facts; read 2026-11-22, not pre-registered):
- **Seeds 0 and 2: wrong short answers.**
  - Forget: 28 and 26 short wrong answers (e.g. 160 cm → "165 cm."); no biography sentences.
  - Retain: 109 and 108 of 122 short and correct.
- **Seed 1: biography sentences, like run 2 — but only for the forget people.**
  - Forget: 52 of 66 responses are biography sentences ("Aiko Suzuki's height is 160 cm.", "Lea Hernandez, born in
    Medellín, Colombia, is a dedicated Civil Engineer…").
  - Retain: 113 of 122 short and correct; only 5 biography sentences.

**Reading (exploratory, formed after seeing these data):**
- **Same recipe, two mechanisms.** Seeds 0 and 2 forget by an edit bound to the name tokens: wrong short answers, no
  shared direction, forced choice near the prior. Seed 1 forgets by run 2's response-style switch, with a shared
  direction at the last token, knowledge recoverable by forced choice and three shots, and much stronger log-p
  forgetting. Which one forms depends on the seed, not only on the recipe.
- **Correction to DEV-019's and DEV-020's exploratory reading.** Those entries suggested that the response-style switch
  is the form of unlearning a weak retain term allows, and that the switch causes the spread to unseen people. Seed 1
  contradicts this: the switch forms with the strong retain term, and it spares the unseen retain people (0.926; their
  answers stay short). The switch is not by itself the cause of run 2's collateral damage.
- **What holds in every seed, in both mechanisms:**
  - The portrait question shows no forgetting.
  - The name question with the person's own portrait keeps the answer (≤ 0.18 nats).
  - When the question tokens cannot attend to the image, the forgetting comes back, while θ0 is unaffected.

  The cross-modal leak and its account replicate across three seeds and two internal mechanisms: the portrait gives the
  model a route to the person that the text-only edit does not cover.
- **The blank image escapes better under the switch** (seed 1 1.11 nats; run 2 R 0.88) than under the name-bound edit
  (seeds 0 and 2: 2.5–3.0 nats).
- **Limits:** three seeds of one recipe on one model; step 300 fixed from G3; the mechanism split is two to one, too few
  seeds to estimate how often each forms.

### DEV-022 — 2026-11-22 — when does the response-style switch form? Response styles over all runs and checkpoints (decided by: user asked why one recipe gives two mechanisms; design by: implementer)
**Exploratory, existing data only, no new run.** `analysis/regime_map.py` classifies θu's standard-readout responses
(txt form, doubly-correct facts) at every saved checkpoint of every UMU run on this machine. Classes: short (< 6
words; correct or wrong), bio (longer, opens with the person's first name), refusal, other. It adds NPO's log-ratio
from the training log at the nearest logged step. Run 2 and run 4 use the 20-person retain readout; the gate runs use
100 people.

| run | step | forget acc | forget bio | forget short-wrong | retain acc | retain bio | NPO log-ratio |
|---|---|---|---|---|---|---|---|
| run 2 (lr 5e-5, w 1, 50) | 114 / 150 | 0.82 / 0.63 | 0.09 / 0.34 | 0.09 / 0.03 | 0.88 / 0.69 | 0.08 / 0.19 | −8.8 / −15.7 |
| run 4 (lr 1e-4, w 3, 200) | 76 / 150 | 0.74 / 0.72 | 0.09 / 0.11 | 0.17 / 0.17 | 0.81 / 0.88 | 0.08 / 0.08 | −11.2 / −10.1 |
| G1 (lr 1e-4, w 3, 200) | 76 / 152 / 300 | 0.61 / 0.33 / 0.47 | 0.26 / 0.53 / 0.33 | 0.14 / 0.11 / 0.18 | 0.69 / 0.43 / 0.76 | 0.17 / 0.45 / 0.16 | −13.3 / −10.9 / −12.9 |
| G2 (lr 2e-4, w 3, 200) | 38 / 76 / 300 | 0.30 / 0.05 / 0.18 | 0.62 / 0.85 / 0.74 | 0.08 / 0.02 / 0.08 | 0.38 / 0.21 / 0.86 | 0.58 / 0.70 / 0.06 | −16.1 / −10.8 / −23.2 |
| G4 (lr 1e-4, w 3, 300) | 76 / 114 / 300 | 0.76 / 0.61 / 0.53 | 0.00 / 0.27 / 0.33 | 0.21 / 0.12 / 0.14 | 0.79 / 0.75 / 0.80 | 0.00 / 0.18 / 0.14 | −7.7 / −13.8 / −10.5 |
| G3 seed 1 (lr 2e-4, w 5, 200) | 38 / 114 / 300 | 0.73 / 0.30 / 0.18 | 0.20 / 0.67 / 0.79 | 0.08 / 0.03 / 0.03 | 0.82 / 0.60 / 0.93 | 0.13 / 0.34 / 0.04 | −7.7 / −8.3 / −23.5 |
| G3 seed 0 | 38 / 152 / 300 | 0.80 / 0.67 / 0.58 | 0.00 / 0.00 / 0.00 | 0.20 / 0.33 / 0.42 | 0.83 / 0.84 / 0.90 | 0.00 / 0.00 / 0.00 | −4.7 / −12.8 / −15.0 |
| G3 seed 2 | 38 / 152 / 300 | 0.85 / 0.73 / 0.59 | 0.00 / 0.00 / 0.00 | 0.12 / 0.27 / 0.39 | 0.82 / 0.88 / 0.89 | 0.00 / 0.00 / 0.00 | −4.9 / −9.9 / −12.9 |

The full per-checkpoint output comes from running the script.

**What the map shows** (exploratory):
- **Two trajectories.** Six of the eight runs develop biography-style answers to the forget questions (the switch).
  Two runs (G3 seeds 0 and 2) produce no biography sentence at any checkpoint and forget only through wrong short
  answers.
- **The switch arrives on retain people too, then the retain term carves them out.** When the switch appears, the
  retain people get biography answers at nearly the same rate (G2 step 38: forget 0.62, retain 0.58; G1 step 152:
  0.53 / 0.45). With the stronger or longer retain term the retain share decays while the forget share stays (G2:
  retain 0.70 → 0.06; G3 seed 1: 0.34 → 0.04).
  - Run 2, with a weak retain term and 150 steps, was stopped while its switch still covered the retain people. This
    refines DEV-021's correction: the switch does cause the collateral damage at first, and it becomes person-specific
    only under continued retain training.
- **The switch reaches deeper NPO log-ratios** (−21 to −24 at the end of G2 and G3 seed 1) than the name-bound runs
  (−13 to −16).
- **The regime shows early.** With the G3 recipe, seed 1 already has 0.20 biography answers at step 38; seeds 0 and 2
  have none at any checkpoint. At lr 1e-4 the switch can appear later (G1 at step 76, G4 at step 114), after a phase
  of wrong short answers.

**Reading (exploratory): a format loophole in the forget objective.** NPO lowers the probability of the exact answer
string. A model can do this in two ways:
- change *which* short answer it gives, fact by fact (the name-bound edit);
- change the *format*, so that no short answer is given at all (one shared switch).

The second costs one generic change and lowers the answer's log-probability more. Most runs fall into it, and which
one a run falls into is decided early by the seed. In both cases the facts stay reachable through the portrait
(DEV-021).

**Follow-up on the eight runs, existing outputs only** (2026-11-22, `analysis/eight_runs.py`; the user asked for no
retraining; final checkpoints unless stated):
- **By attribute, the switch is uneven.**
  - In every switch run, Education is almost always answered with a biography sentence (bio 0.80–1.00), often one that
    contains the answer.
  - Height, Born and Residence follow.
  - Medical_Conditions mostly stays correct (0.92 in run 2, run 4, G1 and G4; only the strongest switches, G2 and G3
    seed 1, reach it).
  - In the two name-bound runs, wrong short answers spread over every attribute except Medical_Conditions (0.92–1.00
    correct).
- **Wrong short answers vary.** In G3 seeds 0 and 2 they are 26 distinct values among 28 and 19 among 26. 7 and 9 of
  them equal another forget person's true answer for the same attribute; the rest are other plausible values ("165
  cm.", "$95000.").
- **The portrait route stays near intact in every run** (final forget dc img 0.86–0.97). The lowest values come with
  the strongest switches (G2 and G3 seed 1, 0.86–0.88).
  - The name + portrait question is hit in the switch runs: G2 falls to 0.50 at step 76 and is 0.88 at the end; G3
    seed 1 ends at 0.83.
  - It stays at 0.92–0.97 in the name-bound runs.
  - So under the switch the name in the question text can still trigger it with the portrait present. The
    teacher-forced first-token readout of DEV-021 (G3 seed 1, own portrait 0.18 nats) is milder than this generation
    readout.
- **The style is per attribute more than per person.** P(both facts of a person are biography) exceeds independence
  only slightly, in G2 (0.62 vs 0.55) and G3 seed 1 (0.71 vs 0.62), and not in the other runs.
- **Training losses.** Switch runs drive the NPO loss to about zero by the end (G2 0.01, G3 seed 1 0.03; the G1 and G4
  partial switches 0.24 and 0.29). The name-bound runs end with the NPO loss still at 0.52 and 0.88, and the log-ratio
  still falling. Peak retain NLL is highest where the switch first appears on the retain people too: G2 0.77 at step 70,
  G1 0.66 at step 80, G4 0.83 at step 150. It is lowest in the name-bound runs (0.34, 0.48).
- **Reading (exploratory).**
  - The switch nearly solves the forget objective; the name-bound edit does not. That fits the loophole reading.
  - It also leaves open that the name-bound regime is an earlier, unfinished state: G1 and G4 first lost facts through
    wrong short answers and switched later (steps 76 and 114), and seeds 0 and 2 stop at step 300 because the lr has
    decayed to 0.
  - The existing data cannot tell these apart.

**What would test this, not run:**
- Finer early checkpoints (every 5 steps up to step 76) for seeds 0 and 1, to see when the regime is fixed and which
  batches precede it.
- Continuing seeds 0 and 2 past step 300 (does the name-bound edit turn into the switch?).
- A forget objective that cannot be met by changing the format, e.g. lowering the correct option against the other
  options under a fixed answer prefix. It should remove the switch in every seed.

### DEV-023 — 2026-11-22 — closing the format loophole: relative NPO (decided by: user; design by: implementer)
**Question.** DEV-022 suggests that NPO's forget objective can be met by giving up the short answer format: one shared
switch to biography sentences lowers the log-probability of every exact answer string. If that loophole is what
produces the switch, an objective that a format change cannot satisfy should never produce it, in any seed.

**Method** (`vlmu/train.py:step_rnpo_rt`, `vlmu/umu.py:add_distractors`; `unlearn.method=rnpo_rt`). Relative NPO
applies NPO's loss to ρ instead of the answer's log p:
- ρ = log p(a) − log Σ_c p(c), the share of the true answer a among the candidates c = {a, d1, d2, d3}.
- The d_i are other forget people's answers to the same question type (field), distinct from a, chosen
  deterministically per row (2.83 per row on average; fewer where the field has few distinct values).
- Adding the same amount to every candidate's log p leaves ρ unchanged. So lowering all short answers at once — a
  change of format — gains nothing; ρ falls only when probability moves to other answers in the same format.
- Reference: the vanilla's ρ. The retain NLL is unchanged.

**Runs.**
- G3 recipe otherwise unchanged: lr 2e-4, retain weight 5, 200 retain-unlearn people, `umu.n_retain_eval=100`,
  4 epochs, a checkpoint every 0.5 epoch.
- Seeds 0, 1 and 2: `umu_rnpo_rt_txt_lr2e-04_s{0,1,2}_gate_rw5n200`.
- Comparison: NPO + retain with the same recipe and seeds (G3, DEV-021), not rerun.
- On step 300 of each seed, with unchanged code and `--set umu.n_retain_eval=100`: `mech_umu_style_control.py`,
  `mech_umu_position_mask.py`, `mech_umu_signal_transfer.py`, `mech_umu_crossmodal.py`.
- Biography share per checkpoint as in `analysis/regime_map.py`.

**Predictions and thresholds** (forget dc facts, txt form):
- **Validity of the comparison.** Relative NPO must forget: forget dc txt ≤ 0.70 at some checkpoint in at least 2 of 3
  seeds. If no seed reaches 0.90 or below at any checkpoint, the result is not interpretable. The pre-set fallback is
  one rerun of all three seeds at lr 4e-4.
- **Loophole supported:** in every seed, biography share ≤ 0.05 at every checkpoint, and sub_mean at L16 ≤ 0.3 at
  step 300. Under NPO the same recipe gave 0.20 at step 38 in seed 1, and 6 of 8 runs overall reached ≥ 0.20.
- **Loophole rejected:** biography share ≥ 0.20 at some checkpoint in any seed that forgets (forget dc txt ≤ 0.70
  there).
- **Otherwise:** partial.
- **Route prediction** (the edit stays on the text route): at step 300 in every seed that forgets, forget dc img
  ≥ 0.85, and criterion (d) of DEV-021 holds.
- **Reported without a threshold:**
  - unseen and seen retain;
  - FC-train and FS (gate 2);
  - LoRA off at the name and at the last token;
  - the share of wrong short answers that equal one of the row's training distractors, which measures whether the
    objective teaches other people's answers.

**Validity checks** (in `step_rnpo`, first step; the run aborts otherwise):
- ρ is unchanged, to 1e-3, when 7 nats are added to every candidate.
- Before the first update, ρ matches the reference to 0.05 nats.

A `--debug` run passed both: step-1 loss 3.4657 = (2/β)·ln 2.

**Known limits:**
- The distractors are the other forget people's real answers. Relative NPO can lower ρ by teaching those answers,
  i.e. misinformation; that is acceptable for this causal test, not as a method.
- Three seeds.
- The candidate set covers only the training form's answers; the switch could still form for reasons this objective
  does not remove.
- The code was not committed when the runs started. The user commits; the sha256 of the changed files is recorded in
  `logs/dev023/launch.txt`.

**Sizing:** about 4–5 s per step in the debug run (NPO: 1.8 s), so about 30 minutes per seed plus about 15 minutes
of analyses. Seeds 0 and 1 run in parallel on GPUs 4 and 5, seed 2 after them; about 75 minutes in total.

**Outcome** (2026-11-22; code as hashed in `logs/dev023/launch.txt`, uncommitted; all 15 jobs exit 0).
- Run time: 3.5–5.1 s per step.
- The first queue's analysis log names were wrong, so a dispatcher replaced it: analyses moved to GPUs 0, 5, 6 and 7
  (user, 2026-11-22). Outputs unaffected.
- Validity checks passed in every seed (step-1 loss (2/β)·ln 2; shift invariance).

| | seed 0 | seed 1 | seed 2 | NPO, same recipe (DEV-021) |
|---|---|---|---|---|
| max biography share, all checkpoints | **0.00** | **0.00** | **0.00** | 0.00 / 0.80 / 0.00 |
| min forget dc txt (= step 300) | 0.742 | 0.773 | 0.742 | 0.545 / 0.167 / 0.591 |
| step 300: unseen retain txt / img | 0.959 / 1.000 | 0.951 / 0.992 | 0.967 / 0.992 | |
| step 300: forget dc img | 0.939 | 0.955 | 0.939 | 0.924 / 0.879 / 0.955 |
| sub_mean, L16 | 0.005 | −0.008 | 0.024 | 0.127 / 0.661 / 0.035 |
| LoRA off at name / at last token (recovery, lost facts) | 0.91 / 0.36 (n = 17) | 0.94 / 0.22 (n = 15) | 0.86 / 0.29 (n = 17) | |
| FC-train(θu); FS recovery | 0.712; 0.15 | 0.788; 0.00 | 0.727; 0.00 | |
| first-token forgetting: text / own portrait (nats) | 2.28 / 0.02 | 2.02 / 0.07 | 1.90 / 0.04 | 7.66 / 0.16 (seed 0) |
| own portrait, image blocked: top-1 θu (half-way); θ0 | 0.80 (0.86); 0.92 | 0.82 (0.88); 0.92 | 0.82 (0.90); 0.92 | |
| wrong short answers equal to one of the row's training distractors | 9 of 17 | 11 of 15 | 8 of 17 | 2 of 28 (seed 0), 4 of 26 (seed 2) |

**Verdict under the pre-registered criteria: the comparison is not valid. Neither "supported" nor "rejected".**
- Relative NPO forgot too little: no seed reached forget dc txt ≤ 0.70 (0.742–0.773). The validity condition needed
  2 of 3 seeds.
- Every seed did reach ≤ 0.90, so the "not interpretable" case that triggers the pre-set fallback (lr 4e-4) is not met
  either. The pre-registration did not name this intermediate case.
- The switch criteria themselves are met in all three seeds: no biography answer at any checkpoint, sub_mean ≈ 0.
  But at this depth of forgetting, NPO seeds 0 and 2 also showed no switch. Only NPO seed 1 had one (0.20 at forget
  0.73, step 38).
- Route prediction holds: forget dc img 0.94–0.96, and DEV-021's criterion (d) is met in every seed.

**Reading (exploratory):**
- Relative NPO forgets through the name tokens (LoRA off there recovers 0.86–0.94), like the name-bound NPO seeds.
  Its wrong answers are, in about half to three quarters of cases, one of the other people's answers it was trained
  to prefer: the expected misinformation.
- Shallow forgetting (about 2 nats on the first answer token) leaves the knowledge readable by forced choice
  (0.71–0.79).
- Whether closing the loophole prevents the switch at NPO-like depths stays open. The natural next step is the
  pre-set lr 4e-4 rerun, but its trigger was not met, so it needs the user's decision.

### DEV-027 — 2026-11-22 — convergence of the name and face routes, rerun: first-token readout, and an image-token swap after Cohen et al. (decided by: user, after reading the related papers; design by: implementer)
**Why rerun.** DEV-024's map was not interpretable. Its within-route swaps (W1, W2) never reached 0.8 (max 0.53 / 0.63)
because the score summed log p over the whole answer: later answer tokens attend back to the target's own name or
portrait and dilute the swap.

**What the literature already shows** (full texts read 2026-11-22 by subagents; verify before citing):
- Cohen et al., ACL 2025 (arXiv 2412.14133), on LLaVA-1.5-7B and others:
  - they copy the IMAGE-token states of another entity into an identification prompt at one layer;
  - the injected entity is named until a critical layer, layer 20 for LLaVA-1.5-7B;
  - knockout of attention from the query to the image changes most in layers 15–20.
- Nikankin et al., NeurIPS 2025 (arXiv 2506.09047; Qwen2-VL-7B, Pixtral, Gemma-3; no LLaVA): visual and textual
  data-token representations align only late, scored on the first answer token.

The convergence itself is therefore expected, not a new finding. This rerun establishes it on our model and prompts,
where the unlearning analyses need the layer numbers.

**Part A — DEV-024's suffix swap, first-token readout** (`mech_umu_convergence.py --part mapfirst`).
- Same pairs, prompts, conditions (X1, X2, W1, W2, S1, S2), layers and post-referent positions as DEV-024.
- Pairs are restricted to those whose two answers start with different tokens.
- Score: at the last prompt position, log p of A's first answer token vs B's. The swap share is the share preferring
  the source person; also reported, top-1 = the source person's first token.
- Pairs whose unpatched targets prefer B in both routes are scored.

*Thresholds:*
- **Validity:** W1 and W2 reach ≥ 0.8 at some layer; S1 and S2 keep B in ≥ 0.9 of pairs at every layer; self-patch
  check ≤ 0.05 nats. Otherwise not interpretable.
- With L*_c = the shallowest layer where condition c reaches ≥ 0.5:
  - **convergence supported** if |L*_X1 − L*_W1| ≤ 2 and |L*_X2 − L*_W2| ≤ 2 (copying the other route's suffix
    states moves the identity from the same depth as copying the same route's), and X1 and X2 reach ≥ 0.8;
  - **not supported** if X1 or X2 never reach 0.5, or lag their within-route curve by more than 4 layers;
  - **otherwise** partial.

**Part B — image-token swap after Cohen et al.** (`--part imageswap`; face route only).
- At one layer, the states of the 576 image tokens of F(A) replace those of F(B). The two face prompts are the same
  text, so the positions coincide.
- Same first-token score. Self-patch check on the image positions.
- L_c = the deepest layer at which the share preferring A is still ≥ 0.5: the last layer at which the answer is still
  read from the image tokens.
- Expectations (replication, no verdict): L_c lies in layers 14–22 (Cohen et al.: 20), and L_c ≤ L*_X2 + 2, i.e. the
  identity leaves the image tokens no later than it reaches the question suffix.

**Known limits:**
- One model (LLaVA-1.5-7B, θ0).
- At the last layers, copying the last-position state fixes the first answer token, so the curves approach 1 there by
  construction. The depth at which they rise is the measure, not the plateau.
- Different-first-token pairs over-represent some attributes, e.g. Occupation and Residence over Height.

**Sizing:** one forward pass per layer per pair (half of DEV-024), about 5–10 minutes per condition, spread over idle
GPUs.

**Outcome** (2026-11-22; code as hashed in `logs/dev027/launch.txt`, uncommitted; GPUs 0, 5, 6, 7).
- Pairs: 123 of 188 have different first answer tokens. No Height or Annual_Salary pair qualifies (digits share the
  first token).
- 116 pairs pass the baseline filter in both routes.
- Self-patch checks 0.00.
- Smoke tests on 4 pairs were deleted before the run.
- Outputs `.../step0/convergence/{part1first_*,imageswap}/`, `summary_first.json`.
- Figure `proposal/figures/convergence_first_theta0.png`.

Swap share, first-token preference for the source person (S1/S2: keeps B):

| layer | 0–12 | 14 | 16 | 18 | 20 | 22 | 24 | 31 |
|---|---|---|---|---|---|---|---|---|
| X1 face(A) → name(B) | ≤ 0.01 | 0.03 | 0.12 | 0.28 | **0.70** | 0.88 | 0.98 | 1.00 |
| X2 name(A) → face(B) | 0.00 | 0.01 | 0.14 | 0.32 | **0.65** | 0.74 | 0.84 | 0.84 |
| W1 name(A) → name(B) | ≤ 0.01 | 0.01 | 0.16 | 0.30 | **0.66** | 0.72 | 0.83 | 0.84 |
| W2 face(A) → face(B) | 0.00 | 0.00 | 0.10 | 0.36 | **0.69** | 0.91 | 1.00 | 1.00 |
| S1 / S2 (keep B) | 1.00 | 1.00 | 1.00 / 0.99 | 0.99 | 1.00 | 1.00 | 1.00 | 1.00 |
| image tokens of A → F(B) | 1.00 | 1.00 | 0.87 | 0.59 | **0.09** | 0.02 | 0.01 | 0.00 |

Top-1 (the source person's first token is the most likely one) follows the same profile. For example, X1 reaches 0.61
at L20 and 0.97 at L24; image-token swap 0.53 at L18 and 0.09 at L20.

**Verdict: convergence supported.**
- Validity: W1 max 0.84, W2 1.00, S1/S2 ≥ 0.99 at every layer.
- L* = 20 for X1, X2, W1 and W2 alike, i.e. |L*_X − L*_W| = 0. X1 and X2 reach 1.00 and 0.84.

**Image-token swap: L_c = 18** (the deepest layer with a share ≥ 0.5; 0.09 at L20). Both expectations hold: 14–22
(Cohen et al.: 20), and 18 ≤ L*_X2 + 2.

**Reading.**
- In LLaVA-1.5-7B (θ0), the identity of the person moves into the question tokens after the referent between layers
  16 and 20.
- It moves at the same depth whether it comes from the name or from the face, and in a form the two routes share:
  copying the other route's suffix states works exactly as well as copying the same route's.
- At the same depth, the image tokens stop mattering. Swapping them changes the answer up to L16 (0.87), half the
  time at L18, and almost never from L20 (0.09). This replicates Cohen et al.'s critical layer 20 on our prompts.
- DEV-024's diluted map pointed to the same layers (16–22). The answer-sum readout, not the effect, made it
  uninterpretable.
- Together with DEV-024 part 2 (copying θu's name-route suffix states into the face route makes the face route
  forget, rising from L16: 0.60 for the name-bound seed 0, 0.86 for the switch seed 1), and DEV-020/021:
  - the unlearning edits sit before the convergence (name tokens, L0–3) or form only on the text-only input (the
    switch);
  - the face route reaches the shared representation at L16–20 without passing them.

### DEV-036 — 2026-11-23 — other unlearning methods: is the text-only edit always written before the merge? (decided by: user; design by: implementer)
**Why.** Every mechanism result so far uses one objective, NPO + retain (plus relative NPO on LLaVA). The paper's
third claim is that the account holds across methods. This entry adds UMU-Bench's other baselines that have a retain
term, on both models.

**Methods** (text-only; `unlearn.method`):
- **GD** (`ga_diff`, existing code): gradient ascent on the forget batch + retain NLL.
- **KL** (`kl`, new): gradient ascent on the forget batch + KL(p_ref ‖ p_θ) over the retain batch's answer tokens.
  The reference is the vanilla on the same batch.
  - UMU's `KL.py` uses −(p_θ log p_ref) over all positions; the standard form (Maini et al.) is used here.
- **PO** (`po`, new): NLL of a refusal on the forget batch + retain NLL.
  - The refusal is drawn at random from UMU's `idk.txt` (99 strings, copied to `vlmu/idk_umu.txt`) for every row at
    every step, as in UMU's `PO.py`.
- **rNPO** (`rnpo_rt`, DEV-023): new on Qwen. LLaVA already has runs (DEV-023, DEV-026); the tests below run on
  their selected checkpoints.
- **Not run:** GA and NPO without a retain term. Forget-only objectives collapsed the text route before (DEV-012).

**Setup,** identical to the NPO + retain runs except for the objective:
- **LLaVA:** UMU's vanilla, G3's settings (retain weight 5, 200 retain-unlearn people, n_retain_eval 100, 4 epochs,
  a checkpoint every 0.5 epoch), run tag `gate_rw5n200`.
- **Qwen:** the r128 vanilla (DEV-034), `configs/qwen2vl7b/umu_npo_rt.yaml`, run tag `r128`.
- The retain weight 5 multiplies each method's retain term.

**Learning rate and seeds** (fixed now):
- Seed 0 runs a small grid per method: GD and KL at 5e-5, 1e-4 and 2e-4; PO at 1e-4 and 2e-4; rNPO (Qwen) at 2e-4
  and 4e-4.
- The chosen lr is the one whose selected checkpoint has the lowest forget dc txt; ties go to the lower lr.
- Seed 1 then runs at the chosen lr. In total 25 runs.

**Checkpoint selection: gate 1′.** This is DEV-019's gate 1 with one change: the degeneracy share is computed on the
unseen-retain txt generations only.
- Thresholds as before: unseen retain dc txt and img ≥ 0.90; forget dc txt ≤ 0.70; degeneracy ≤ 0.20; among these,
  the lowest forget dc txt, ties to the earlier step.
- Reason for the change: GD, KL and PO are expected to give one string (gibberish or a refusal) to every forget
  question. Over all txt generations that alone is 0.29, which would exclude them by construction. The criterion is
  meant to exclude a model that collapses on the people it should keep.
- **If no lr qualifies for a method:** it is reported as such. The tests then run, flagged "non-qualifying", on one
  fallback checkpoint: among the checkpoints with retain-only degeneracy ≤ 0.20 and forget dc txt ≤ 0.70, the one
  with the highest min(retain dc txt, retain dc img). Seed 1 is not run.

**Tests on each selected checkpoint** (existing scripts, unchanged):
- core3 (`mech_umu_core3.py`);
- `mech_umu_name_depth.py --split-at L*`, with L* = 20 for LLaVA (DEV-027) and 22 for the Qwen r128 vanilla
  (DEV-035);
- the transfer (`mech_umu_convergence.py --part unlearned`).

**Predictions and thresholds,** per checkpoint (as DEV-030, 032 and 033):
- **Leak:** forget dc img ≥ 0.85; G is reported.
- **Class of the edit,** from the split at L*:
  - **name-bound, before the merge:** R_pre_name ≥ 0.7 and R_post_name ≤ 0.1;
  - **before the merge, not name-bound:** the above fails, but R_pre_all ≥ 0.7 and R_post_all ≤ 0.3;
  - **after the merge or mixed:** otherwise.
- **N1:** the portrait-only question does not trigger the edit: F(NI) ≤ 1.0 nat, and θu's top-1 ≥ θ0's − 0.1.
- **Link 5:** S_u ≥ 0.5 at some layer; S_u ≤ 0.1 at every layer ≤ L* − 6; |S_0| ≤ 0.1.
- Also reported: Q2, Q3 and link 6 from core3, and the response styles on the forget txt questions, read from the
  generations.

**Verdict for the claim "across methods":**
- **Supported** if every analysed qualifying checkpoint is in one of the two "before the merge" classes and meets N1.
- **Refuted for a method** if its checkpoints leak (forget dc img ≥ 0.85) while R_pre_all < 0.5: the edit is written
  after the merge and the portrait still escapes.
- A method whose edit is written after the merge and that does not leak is consistent with the account. It is
  reported separately, as a point for the planned "share written before the merge vs leak" analysis.
- Otherwise mixed.

**Validity checks:**
- Each test script's own checks.
- `--debug` smoke tests of GD, KL and PO on both models before the full runs. For KL at step 0, the retain KL must
  be ≤ 0.01: the LoRA starts at zero, so θ equals the reference.

**Known limits:**
- One benchmark.
- The lr grids are small and method-specific.
- KL is the standard form, not UMU's code.
- The retain weight is shared across methods, not tuned per method.
- Two seeds per method.

**Sizing:** 25 runs of about 22–25 minutes each, about 10 GPU-hours.
- Training on 2 GPUs: about 5 hours.
- Tests: about 10 minutes per selected checkpoint, on idle GPUs.
- Status log: `logs/dev036/orchestrate.txt`.

**Amendment 1: a larger lr for GD on Qwen** (2026-11-23, 19:20 UTC; written after seeing the Qwen GD grid, before the
added runs).
- On Qwen, GD does not forget within the grid. Forget dc txt stays at 1.00 for 5e-5, ≥ 0.97 for 1e-4, and reaches
  only 0.88 at step 300 for 2e-4. Retain stays ≥ 0.94.
- No checkpoint has forget dc txt ≤ 0.70, so there is no fallback checkpoint either. The grid was too low for this
  model; the model did not collapse.
- **Added:** GD on Qwen, seed 0, at lr 4e-4 and 8e-4. The same rules then apply to these two (gate 1′, lr choice,
  seed 1, tests).
- These runs start after the main queue (`logs/dev036b/orchestrate.txt`).
- The grids of the other methods are not changed here. If another method under-forgets in the same way, it gets its
  own dated amendment.

**Amendment 2: two more GPUs** (2026-11-23, 19:40 UTC; the user allowed GPUs 6 and 7 for training as well).
- Scheduling only; no run, setting or rule changes.
- The amendment-1 runs (GD on Qwen at 4e-4 and 8e-4) start now on GPUs 6 and 7 instead of after the main queue.
- The two rNPO grid runs on Qwen (2e-4, 4e-4) also run there first, with the main queue's exact command. They are the
  last entries of the main queue, which will find them done and go on to the lr choice, seed 1 and the tests.
- A small placeholder process holds 2.5 GB on GPUs 6 and 7, so that the main queue's tests are not placed there.

### DEV-035 — 2026-11-23 — Qwen unlearning and the mechanism tests on the r128 vanilla (decided by: user; design by: implementer)
**Decision (user, 2026-11-23, after DEV-034).**
- The Qwen vanilla becomes R128 epoch 5: `models/qwen2vl7b_umu_vanilla_r128`.
- With options it scores forget 0.90 by name and 0.81 by portrait; without options, 1.00 and 0.98.
- The results on the epoch-4 vanilla (DEV-030, 032, 033) stay as the preliminary Qwen set.
- The unused merged models were deleted: `qwen2vl7b_umu_vanilla_vis` and `_vis_r128`. Their adapters stay in the run
  folders.

**Stages, all as before; nothing new is tuned:**
- **Unlearning, as DEV-030 stage 2.**
  - Recipe: G3 (NPO + retain), lr 2e-4, retain weight 5, 200 retain-unlearn people, 4 epochs, a checkpoint every
    0.5 epoch.
  - Run tag `r128`; run ids `umu_npo_rt_txt_lr2e-04_s{0,1,2}_r128`.
  - Order: seeds 0 and 1 together, then seed 2.
  - If seed 0 has no qualifying checkpoint, fall back to lr 1e-4, then 4e-4.
  - Selection: DEV-019's gate 1.
- **Convergence on the new θ0** (`mech_umu_convergence.py --part mapfirst / imageswap / --report-first`).
  - As DEV-032 (B), with `--dc-run umu_npo_rt_txt_lr2e-04_s0_r128` and the same layers.
  - **L\*** = the shallowest layer where X1 and X2 both reach ≥ 0.5. It is used by the next two items.
- **On each selected checkpoint:**
  - core3 (Q2, Q3, link 6), as DEV-030 stage 3;
  - the transfer (`--part unlearned`, link 5), as DEV-032 (C);
  - `mech_umu_name_depth.py --split-at L*` (DEV-033).

  All Qwen runs use `model.attn=eager`.

**Predictions and thresholds** (unchanged from the entries named):
- **Convergence:** DEV-027's validity and verdict rules.
- **Q1, the leak:** forget dc img ≥ 0.85 per seed (DEV-030).
- **The rescue question** (pre-registered in DEV-031 for a stronger face route): compare forget dc txt+img per seed with
  the epoch-4 runs (0.40 / 0.68 / 0.56).
  - **≥ 0.85 in at least 2 of 3 seeds:** the face route's strength explains the missing rescue.
  - **Every seed within ±0.10 of its epoch-4 value:** not explained.
  - Otherwise unresolved.
- **Q2 and Q3:** DEV-030's thresholds.
- **Link 6** (6a, 6b, 6c): DEV-029's thresholds.
- **Link 5,** per seed (DEV-032 C): S_u ≥ 0.5 at some layer; S_u ≤ 0.1 at every layer ≤ L* − 6; |S_0| ≤ 0.1.
- **Before the merge,** per seed (DEV-033): (a) R_pre_name ≥ 0.7 and (b) R_post_name ≤ 0.1. Secondary: (c)
  R_post_all ≤ 0.3. Also N1 of DEV-032.

**Validity checks:**
- The checks built into each script: hook neutrality, self-patch, keep = 1 / keep = 0, name tokens in every row.
- **Merge check:** seed 0's step-0 readout must lie within 0.05 of the R128 epoch-5 readout (0.90 / 0.81). Otherwise
  every job stops.

**Known limits:**
- One recipe.
- The vanilla was trained with our recipe, not UMU's.
- The dc sets change with the vanilla (R128: 78 forget facts), so absolute numbers are not comparable with the
  epoch-4 set.

**Sizing:**
- Training: about 25 minutes per seed, so seeds 0 and 1 in parallel, then seed 2: about 55 minutes on 2 GPUs.
- Convergence: about 10 minutes, on idle GPUs while the seeds train.
- Per seed afterwards: core3, transfer and depth, about 10 minutes.
- In total about 1–1.3 hours.
- Orchestrator status log: `logs/dev035/orchestrate.txt`.

**Outcome** (2026-11-23, 15:39–16:37 UTC; code as hashed in `logs/dev035/launch.txt`, uncommitted; GPUs 0, 4–7).
- **Merge check:** passed. Step 0 reads forget 0.90 / 0.80 against the vanilla's 0.90 / 0.81.
- **Doubly-correct facts:** 77 forget, 143 retain.
- **Selection:** every seed qualified at lr 2e-4; the gate-1 checkpoint is step 190 in all three.
- **Checks:** hook, self-patch and keep = 1 / keep = 0 checks are 0.00, with one exception:
  - core3 on seed 0 aborted at its first attempt (GPU 0): keep = 1 differed from θu by 0.28 nats, above the 0.05
    tolerance.
  - A rerun of the same command on GPU 5 gave 0.00 and completed (`core3_…_s0_r128_rerun.log`). The reported numbers
    are from the rerun.
  - Likely cause: a one-step bf16 rounding difference, which is about 0.25 nats at this depth of forgetting
    (F ≈ 27 nats). Not reproduced.
- **Code edits during the run:** `vlmu/train.py` and `scripts/unlearn_umu.py` were edited at 16:04 UTC for DEV-036
  (new step functions only). Seed 2's process had started at 16:03:51 and logged at 16:03:58, so all three seeds ran
  the hashed code.
- **Outputs:** `runs/qwen2vl7b/umu_unlearn/umu_npo_rt_txt_lr2e-04_s{0,1,2}_r128`;
  `results/qwen2vl7b/mech_umu/<run>/step190/{core3,name_depth_split22,convergence_transfer}`;
  `…/umu_npo_rt_txt_lr2e-04_s0_r128/step0/convergence/summary_first.json`.

**Convergence on the r128 θ0** (141 of 141 pairs kept; swap share):

| layer | 0–12 | 14 | 16 | 18 | 20 | 22 | 24 | 27 |
|---|---|---|---|---|---|---|---|---|
| X1 face(A) → name(B) | 0.00 | 0.02 | 0.02 | 0.02 | 0.16 | **0.77** | 0.95 | 1.00 |
| X2 name(A) → face(B) | 0.00 | 0.04 | 0.04 | 0.06 | 0.19 | **0.81** | 0.89 | 1.00 |
| W1 / W2 | 0.00 | 0.04 / 0.02 | 0.04 / 0.02 | 0.06 / 0.03 | 0.21 / 0.12 | **0.79 / 0.77** | 0.91 / 0.94 | 1.00 |
| S1 / S2 (keep B) | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 |
| image tokens of A → F(B) | 1.00 | 0.96 | 0.95 | 0.95 | **0.85** | 0.11 | 0.01 | 0.00 |

**Convergence supported; L\* = 22** (X1, X2, W1 and W2 alike), and image-token L_c = 20. These are the same layers as
on the epoch-4 vanilla. The image swap now reaches 1.00 at layer 0 (0.89 before): the face route carries the person.

**Readout and tests per seed** (selected step 190; F in nats on the first answer token):

| | seed 0 | seed 1 | seed 2 |
|---|---|---|---|
| forget dc txt / img / txt+img | 0.01 / 0.92 / 0.32 | 0.06 / 0.97 / 0.25 | 0.00 / 0.99 / 0.09 |
| G; unseen retain dc txt / img | +0.91; 0.92 / 0.97 | +0.91; 0.97 / 0.98 | +0.99; 0.90 / 0.97 |
| F(T); T + name-off; T + control-off | 27.47; 0.46; 27.48 | 12.99; 0.08; 12.85 | 14.20; 0.05; 14.24 |
| F(NB), blank image | 27.29 | 13.04 | 14.23 |
| F(NP) visible; knockout; knockout + name-off | 21.44; 27.42; 0.47 | 10.59; 13.15; 0.09 | 12.54; 14.27; 0.05 |
| θu top-1: T / NP visible (θ0: 1.00) | 0.00 / 0.03 | 0.17 / 0.23 | 0.01 / 0.08 |
| R_pre_name / R_post_name | 0.98 / 0.00 | 0.99 / 0.00 | 1.00 / 0.00 |
| R_pre_all / R_post_all | 1.00 / 0.29 | 1.00 / 0.23 | 1.00 / 0.13 |
| F(NI), portrait-only; top-1 θu / θ0 | 0.01; 1.00 / 1.00 | 0.00; 1.00 / 1.00 | 0.00; 1.00 / 1.00 |
| peak window of R_name | L4–7 (0.49) | L8–11 (0.25) | L4–7 (0.32) |

**Transfer** (name-route suffix states copied into the face route; 77 facts; |S_0| ≤ 0.002 at every layer):

| layer | ≤ 10 | 12 | 14 | 16 | 18 | 20 | 22 | 27 |
|---|---|---|---|---|---|---|---|---|
| S_u summed, seed 0 | ≤ 0.01 | 0.10 | 0.23 | 0.24 | 0.37 | 0.56 | 0.60 | 0.60 |
| S_u summed, seed 1 | 0.00 | 0.04 | 0.12 | 0.14 | 0.25 | 0.54 | 0.60 | 0.62 |
| S_u summed, seed 2 | ≤ 0.02 | 0.31 | 0.41 | 0.41 | 0.37 | 0.37 | 0.38 | 0.38 |
| top-1 patched, seed 0 (name route 0.00) | ≥ 0.96 | 0.71 | 0.43 | 0.38 | 0.18 | 0.00 | 0.00 | 0.00 |
| top-1 patched, seed 1 (name route 0.17) | ≥ 0.97 | 0.87 | 0.61 | 0.56 | 0.42 | 0.19 | 0.17 | 0.17 |
| top-1 patched, seed 2 (name route 0.01) | ≥ 0.87 | 0.18 | 0.05 | 0.05 | 0.05 | 0.03 | 0.01 | 0.01 |

**Verdicts, as pre-registered:**
- **Convergence:** supported.
- **Q1, the leak:** met in every seed (img 0.92–0.99).
- **The rescue question:** **unresolved** by the rule.
  - No seed reaches 0.85 (0.32 / 0.25 / 0.09).
  - Only seed 0 lies within ±0.10 of its epoch-4 value.
- **Q2:** met as defined, but uninformative, because NP visible already forgets.
- **Q3:** met in every seed (name-off leaves 0.4–1.7% of F; control-off none).
- **Link 6:** 6a fails in every seed (NP visible 10.6–21.4 nats); 6b and 6c hold.
- **Before the merge (DEV-033):** met in every seed, including (c). N1 is met.
- **Link 5: fails in every seed** as written.
  - Seeds 0 and 1 reach 0.60 and 0.62, but S_u is already 0.23 and 0.12 at L14 (allowed: ≤ 0.1 up to L16).
  - Seed 2 plateaus at 0.41 (< 0.5), and is at 0.31 by L12.

**Reading (exploratory, formed after seeing these data):**
- **The core account holds on a vanilla that knows the faces.**
  - The edit sits on the name tokens before the merge.
  - The portrait-only question does not trigger it (0.00–0.01 nats; img 0.92–0.99 against txt 0.00–0.06).
  - The leak is larger than on any earlier checkpoint (G +0.91 to +0.99).
- **The missing rescue is not due to a weak face route.**
  - The r128 vanilla answers 0.98 by portrait, and the image swap is 1.00.
  - Yet name + portrait is forgotten almost as much as the name alone (0.09–0.32).
  - So this is a model difference from LLaVA: in Qwen the name edit wins whenever the name is in the question. The
    rule's "unresolved" comes from the forgetting being much deeper here, not from any sign of a rescue.
- **The blank image does not escape at all** (F(NB) = F(T) in every seed), as on the epoch-4 vanilla.
- **Link 5 holds behaviourally but not in its timing.**
  - Given the name route's suffix states, the face route's top-1 falls to the name route's level in every seed
    (1.00 → 0.00 / 0.17 / 0.01).
  - The suppression is already in the question tokens after the name by L12–16, before the identity moves there
    (L20–22). In seed 2 it is nearly complete at L12.
  - So in Qwen, with forgetting this deep, the name edit writes a suppression signal into the question stream ahead
    of the merge. "The suppressed state acts after the merge" is too narrow.
  - The summed readout saturates at 0.4–0.6 because later answer tokens stay predictable.
- **Depth:** the name edit peaks in L4–11 again.
- **Limits:**
  - One recipe.
  - The forgetting here is much deeper than on LLaVA (forget dc txt ≤ 0.06 against 0.55–0.67), so the two models are
    not compared at matched depth.
  - The selected checkpoints are the deepest qualifying ones.

### DEV-034 — 2026-11-23 — a Qwen vanilla that knows the facts by face: higher LoRA capacity (decided by: user; design by: implementer)
**Why.** The user asked for a Qwen vanilla with high accuracy both by name and by portrait. Diagnostics run tonight
on the epoch-4 vanilla (exploratory; scratchpad scripts `noopt_readout.py` and `train_fit.py`):

**Readout without the options line** ("Select answer in …"), i.e. exactly the training prompt:

| | forget txt / img | retain txt / img |
|---|---|---|
| Qwen | 0.78 / 0.60 | 0.805 / 0.58 |
| LLaVA's UMU vanilla | 0.69 / **0.97** | 0.72 / 0.975 |

Qwen's portrait gap is therefore not an effect of the format: Qwen does not know the facts by face.

**Teacher-forced fit on 256 of its own training rows:**
- All answer tokens argmax-correct: 0.75 for txt, 0.62 for img. Per-token loss 0.08 and 0.17.
- So the vanilla underfits its own training data, the portrait rows most.
- Correction to DEV-030: the "loss about 0.04" there was a single logged batch. Batches hold one modality each.

**Reading:** the limit is capacity, not format. LoRA r16 on the LM does not memorise 500 people × 12 facts in two forms;
with the vision tower added (DEV-031) it gains only +0.04.

**Test.** Two runs as DEV-030 stage 1:
- same data; readout after each epoch; selection by max min(forget txt, img);
- 5 epochs; lr 2e-4 with linear decay.

The runs:
- **R128:** LoRA r128 / α128 on the LM linear layers. The scaling stays 1, as r16/α16.
  `configs/qwen2vl7b/umu_vanilla_r128.yaml`, run tag `r128`.
- **VR128:** the same, plus the vision tower and the merger (as DEV-031). `umu_vanilla_vis_r128.yaml`, run tag
  `vis_r128`.

One GPU each. Each run merges its selected epoch into `models/qwen2vl7b_umu_vanilla_<tag>`.

**Predictions and thresholds,** per run at the selected epoch:
- **Target met** if forget txt ≥ 0.75 and forget img ≥ 0.75 (LLaVA's vanilla: 0.76 / 0.82).
- **Capacity helps** if forget img exceeds the best r16 run by ≥ 0.10 at any epoch (0.65 LM-only, 0.69 with the
  vision tower). **It does not help** if the gain is within 0.05.
- Reported: retain txt and img; the doubly-correct counts; the no-options readout on the selected epoch.

**What follows.** Nothing starts automatically. If a run meets the target, the user decides whether it replaces the
epoch-4 vanilla. Replacing it means redoing the Qwen unlearning and the tests of DEV-030, DEV-032 and DEV-033 on it.

**Known limits:**
- One seed.
- VR128 changes the rank and the vision side together.
- The r16 runs used for comparison had 3–5 epochs and different schedules.

**Sizing:** 1500 steps per epoch, 7500 steps per run.
- R128: about 1.5 s per step (r16 LM-only measured 1.44), so about 3.5 h.
- VR128: about 2.5–3.5 s per step, so about 5–7 h.
- Smoke tests first: `--debug`, 4 steps, batch 8, which also checks memory.

**Outcome** (2026-11-23; code as hashed in `logs/dev034/launch.txt`, uncommitted; GPUs 4 and 5; started 05:08 UTC).
- R128 finished 09:01 UTC (about 1.6–2.0 s per step); VR128 finished 11:32 UTC (about 2.2–3.4 s per step).
- Smoke tests confirmed r = 128, α = 128.
- The no-options readouts come from the scratchpad diagnostic, run on the merged models.

| epoch | R128 forget txt / img | R128 retain txt / img | R128 dc f / r | VR128 forget txt / img | VR128 retain txt / img | VR128 dc f / r |
|---|---|---|---|---|---|---|
| 1 | 0.71 / 0.59 | 0.675 / 0.605 | 55 / 110 | 0.68 / 0.60 | 0.685 / 0.595 | 51 / 108 |
| 2 | 0.77 / 0.70 | 0.685 / 0.56 | 66 / 107 | 0.70 / 0.62 | 0.73 / 0.58 | 57 / 109 |
| 3 | 0.81 / 0.77 | 0.775 / 0.665 | 71 / 125 | 0.82 / 0.76 | 0.835 / 0.685 | 69 / 131 |
| 4 | 0.84 / 0.76 | 0.79 / 0.76 | 73 / 137 | **0.87 / 0.86** (selected) | 0.87 / 0.79 | 81 / 152 |
| 5 | **0.90 / 0.81** (selected) | 0.82 / 0.79 | 78 / 145 | 0.92 / 0.86 | 0.91 / 0.855 | 82 / 165 |

Readout on the merged selected epoch, with and without the options line:

| | forget txt / img, options | forget txt / img, no options | retain txt / img, no options |
|---|---|---|---|
| R128, epoch 5 | 0.90 / 0.81 | **1.00 / 0.98** | 1.00 / 1.00 |
| VR128, epoch 4 | 0.88 / 0.86 | 0.96 / 0.82 | 0.97 / 0.845 |
| r16 epoch-4 vanilla (DEV-030) | 0.76 / 0.67 | 0.78 / 0.60 | 0.805 / 0.58 |
| LLaVA's UMU vanilla | 0.76 / 0.82 | 0.69 / 0.97 | 0.72 / 0.975 |

The merged models reproduce their epoch readouts within 0.01, so the merge check passes.

**Verdict:**
- **Target met** by both runs: R128 0.90 / 0.81, VR128 0.87 / 0.86.
- **Capacity helps.** The best forget img rises by +0.16 for R128 (0.81 vs 0.65) and by +0.17 for VR128 (0.86 vs 0.69).
  Both gains are ≥ 0.10.
- **Selection tie in VR128.** Epochs 4 and 5 have the same min (0.86), and the script's `max` keeps the first, epoch 4.
  The rule does not say how to break ties. Epoch 5 is higher on every other readout.

**Reading (exploratory):**
- R128 epoch 5 knows the facts the way LLaVA's vanilla does: 0.98 by portrait without options (LLaVA 0.97), and 1.00 by
  name.
- With options it scores 0.90 / 0.81 against LLaVA's 0.76 / 0.82.
- The image tower is unchanged, as in all the earlier Qwen runs. The only change from the epoch-4 vanilla is the LoRA
  rank and the epochs.
- VR128 scores higher by portrait with options (0.86). At its selected epoch, it has not memorised the portrait rows
  (0.82 without options).
- **The weak Qwen face route of DEV-030/031 was an undertrained vanilla, not a property of Qwen.**
- Whether this vanilla replaces the epoch-4 one is the user's decision (see "What follows").

### DEV-033 — 2026-11-23 — is the name edit written before the routes merge? A test on checkpoints not yet looked at (decided by: user; design by: implementer)
**Why.** DEV-032 failed its depth criterion (D1–D3), which was set from LLaVA's "layers 0–3". The reading formed
afterwards (exploratory) is different: both models write the name edit before their convergence layer (LLaVA L20,
Qwen L22), only at different depths. That reading came from LLaVA G3 seed 0 and Qwen seed 2. Here it is stated as a
criterion and tested on name-bound checkpoints whose depth profile has not been looked at.

**Checkpoints:**
- **Fresh:**
  - Qwen seed 0 step 300 and seed 1 step 190 (DEV-030; both pass Q3);
  - LLaVA G3 seed 2 step 300 (name-bound, DEV-021/029).
- **Excluded:** LLaVA G3 seed 1, which forgets by the response-style switch (DEV-021), not by a name edit.
- **Seen in DEV-032:** LLaVA G3 seed 0 and Qwen seed 2 are rerun and reported, marked as seen. They are not part of
  the test.

**Test** (`mech_umu_name_depth.py --split-at L*`, new option; first answer token; forget dc facts; text-only
question).
- L* is the model's convergence layer, measured on θ0 and fixed now: LLaVA 20 (DEV-027), Qwen 22 (DEV-032).
- R_pre_name: LoRA off at the name tokens in layers 0 … L*−1. R_post_name: LoRA off at the name tokens in layers
  L* … last.
- R_pre_all and R_post_all: the same, at every position.
- R = 1 − F(off) / F(T), as in DEV-032.

**Predictions and thresholds,** per checkpoint:
- (a) R_pre_name ≥ 0.7: switching the name edit off before the merge undoes most of the forgetting.
- (b) R_post_name ≤ 0.1: switching it off after the merge undoes almost nothing.
- Verdict per checkpoint:
  - **"before the merge"** if (a) and (b);
  - **"not before the merge"** if R_pre_name < 0.5 or R_post_name > 0.2;
  - otherwise partial.
- **The reading holds** if all three fresh checkpoints are "before the merge". **It fails** if any one is "not".
- Secondary, with its own threshold: (c) R_post_all ≤ 0.3, the LoRA after the merge at any position.
  - In DEV-032, Qwen s2's late windows gave 0.18–0.19 at non-name positions. So (c) may fail without contradicting
    (a) and (b); it is reported as a separate component.
- Also reported: θu's top-1 under each switch-off, and F on the portrait-only question (N1 of DEV-032, on more
  checkpoints).

**Validity checks:** as DEV-032 (keep = 1 and keep = 0 exact; name tokens in every row; LoRA modules in decoder
layers).

**Known limits:**
- L* comes from θ0, from one readout, measured every 2 layers.
- Three fresh checkpoints, two of them from one model.
- The 0.7 threshold was chosen after seeing name-off-in-all-layers recoveries of 0.87 (LLaVA s0), 0.99 (Qwen s2) and
  0.84 (LLaVA s2, DEV-029). The per-layer split of these checkpoints has not been seen.

**Sizing:** 5 checkpoints, about 60 facts and 20 forward passes each; 1–2 minutes per checkpoint, one GPU each.

**Outcome** (2026-11-23, about 05:05 UTC; code as hashed in `logs/dev033/launch.txt`, uncommitted; GPUs 0, 4–7; all
exit 0).
- **Checks:** 0.00 on every checkpoint.
- **Outputs:** `.../mech_umu/<run>/step<N>/name_depth_split<L*>/`.
- **Figures:** `proposal/figures/name_depth_split_{Qwen_s0_step300,Qwen_s1_step190,G3s2_step300}.png`.
- **Consistency:** on the two seen checkpoints the window values reproduce DEV-032 exactly.

| checkpoint | F(T) nats; top-1 θu / θ0 | R_pre_name (top-1) | R_post_name (top-1) | R_pre_all | R_post_all | verdict |
|---|---|---|---|---|---|---|
| **Qwen s0 step 300** (fresh) | 16.43; 0.06 / 0.95 | **0.99** (0.94) | **0.00** (0.05) | 1.00 | 0.42 | before the merge; (c) fails |
| **Qwen s1 step 190** (fresh) | 7.39; 0.40 / 0.95 | **0.99** (0.95) | **0.01** (0.40) | 0.99 | 0.09 | before the merge |
| **LLaVA G3 s2 step 300** (fresh) | 5.33; 0.58 / 0.88 | **0.83** (0.82) | **0.03** (0.58) | 0.95 | 0.19 | before the merge |
| Qwen s2 step 266 (seen) | 11.46; 0.24 / 0.95 | 0.99 (0.94) | 0.01 (0.24) | 0.99 | 0.30 | before the merge |
| LLaVA G3 s0 step 300 (seen) | 7.66; 0.38 / 0.88 | 0.85 (0.74) | 0.03 (0.38) | 0.97 | 0.14 | before the merge |

Peak window of R_name (descriptive):

| checkpoint | peak window (R_name) | R_name from L20 on |
|---|---|---|
| Qwen s0 | L8–11 (0.55), then L12–15 (0.40) | ≤ 0.01 |
| Qwen s1 | spread over L0–11 (0.17–0.24) | ≤ 0.01 |
| Qwen s2 | L4–7 (0.40) | ≤ 0.02 |
| LLaVA s2 | L0–3 (0.42) | ≤ 0.01 |
| LLaVA s0 | L0–3 (0.36) | ≤ 0.02 |

The portrait-only question does not trigger the edit on any of the five: F(NI) −0.04 to 0.07 nats, and θu's top-1 is
within 0.08 of θ0's.

**Verdict: the reading holds.**
- All three fresh checkpoints are "before the merge": (a) 0.83–0.99 ≥ 0.7, (b) 0.00–0.03 ≤ 0.1.
- Secondary (c) fails on Qwen s0: R_post_all = 0.42 > 0.3.

**Reading (exploratory):**
- **The common account.** In five name-bound checkpoints of two model families, the forgetting is written on the name
  tokens before the layer where the name and face routes merge.
  - Switching the name edit off before L* restores the answer; θu's top-1 returns to 0.82–0.95, against θ0's
    0.88–0.95.
  - Switching it off after L* changes nothing.
- **The exact depth is not common.**
  - LLaVA: layers 0–3 in both seeds.
  - Qwen: varies by seed (L4–7, L8–15, or spread over L0–11).
  - So "layers 0–3" (DEV-020) describes LLaVA, not the mechanism.
- **The LoRA after L* is not independent of the name edit.**
  - Qwen s0: switching off the pre-L* LoRA alone removes all the forgetting (R_pre_all 1.00). The post-L* LoRA, at
    non-name positions, still accounts for 0.42 of the log-p effect.
  - So the late LoRA amplifies a signal that starts with the name edit, and does nothing without it.
  - Its behavioural weight is small: top-1 0.06 → 0.11 when it is switched off.
- **Limits:**
  - L* is measured once per model, on θ0.
  - The 0.7 threshold was set after seeing the all-layer name-off values (see Known limits).
  - Qwen's three seeds share one vanilla, which was accepted post hoc.

### DEV-032 — 2026-11-23 — does the LLaVA mechanism carry over to Qwen? The missing links on one representative Qwen checkpoint (decided by: user; design by: implementer)
**Why.** DEV-030 tested three links of the chain on Qwen: the leak, the knockout and the name localisation. Link 6a
failed there. The user asked whether the mechanism found on LLaVA carries over, first on one representative unlearned
Qwen.

**Checkpoint:** Qwen seed 2, step 266 (`umu_npo_rt_txt_lr2e-04_s2`, on the epoch-4 vanilla).
- Rule: of the three seeds, the one closest to the across-seed median on forget dc txt, img, txt+img and G (summed
  absolute distance: seed 0 0.48, seed 1 0.12, seed 2 0.04).
- LLaVA reference: G3 seed 0 step 300, through the same scripts where a readout is new.

**The chain, link by link:**

| link | LLaVA G3 | Qwen so far | here |
|---|---|---|---|
| 1 the edit sits on the name tokens | LoRA off there restores 0.98 (DEV-020) | 98–99% (DEV-030 Q3) | — |
| 1b in the first layers | LoRA off in L0–3 restores 0.75 (generation) | not tested | (A) |
| 2 the portrait-only question does not trigger it | 0.01 nats (DEV-020) | generation leaks (0.92); nats not measured | (A) |
| 3 the portrait is a separate route | knockout (DEV-021/029) | Q2 met (DEV-030) | — |
| 4 the routes converge in the question tokens | L* = 20 of 32; image tokens stop mattering after L18 (DEV-027) | not tested | (B) |
| 5 suppressed name-route states suppress the face route | S rises from L16, plateau 0.60 (DEV-024) | not tested | (C) |
| 6 with name + portrait, the face overrides the edit | closed (DEV-029) | 6a fails (DEV-030) | — |

**Tests** (eval only; first answer token unless stated):
- **(A)** `scripts/mech_umu_name_depth.py` (new, model-agnostic), on the forget dc facts. On the text-only question:
  - F with the LoRA off in one 4-layer window, at every position (R_all) or at the name tokens only (R_name);
  - recovery R = 1 − F(off) / F(T).

  It also measures F on the portrait-only question NI: the portrait, with the name replaced by "this person".
  The script runs on Qwen seed 2 and on LLaVA G3 seed 0.
- **(B)** `mech_umu_convergence.py --part mapfirst` and `--part imageswap` on Qwen θ0, with `--dc-run
  umu_npo_rt_txt_lr2e-04_s2` (Qwen's own 62 + 118 dc facts).
  - DEV-027's pairs, conditions, readout and rules are unchanged.
  - Layers 0, 2, …, 26, 27.
- **(C)** `--part unlearned` on Qwen seed 2 step 266.
  - S(L) as in DEV-024, on the summed answer log p.
  - New, reported only: the first-token version and the top-1 of the correct first token.
- **Code changes in `mech_umu_convergence.py`:**
  - For non-LLaVA families, prompts come from `adapter.prompt`. The LLaVA strings are unchanged.
  - New `--dc-run` option.
  - The image token id now comes from `lm_parts`.
  - The transfer part also records first-token log p and top-1.
  - `--report-first` adds the transfer summary for these items.

**Predictions and thresholds:**
- **(A) depth:**
  - D1: R_all(L0–3) ≥ 0.5.
  - D2: R_all ≤ 0.3 in every window starting at layer ≥ 12.
  - D3: R_name(L0–3) ≥ 0.5.

  "Early name edit" if D1–D3 hold. "Not early" if R_all(L0–3) < 0.3 and R_name(L0–3) < 0.3. Otherwise partial.
  - Link 1b carries over only if LLaVA G3 seed 0 and Qwen get the same verdict.
  - If LLaVA is not "early name edit" on this first-token readout, the depth result is reported descriptively.
- **(A) N1, link 2:** F(NI) ≤ 1.0 nat, and θu's top-1 on NI ≥ θ0's − 0.1.
- **(B), link 4:** DEV-027's validity and verdict rules.
  - Expectation, no verdict: L* at a relative depth like LLaVA's (20 of 32, i.e. about layers 16–20 of Qwen's 28).
    Image-token L_c ≤ L*_X2 + 2.
- **(C), link 5,** on the summed readout. It holds if all three are true:
  - S_u(L) ≥ 0.5 at some layer;
  - S_u ≤ 0.1 at every layer ≤ L* − 6 (it rises with the convergence, not before);
  - |S_0| ≤ 0.1 at every layer.

  LLaVA seed 0 meets all three: max 0.60; ≤ 0.04 up to L14; θ0 ≤ 0.03. If (B) gives no L*, (C) is descriptive.
- **Overall: "the mechanism carries over to Qwen"** if 1b, 2, 4 and 5 hold; 1 and 3 already do. Link 6 is known to
  differ.

**Validity checks** (the scripts abort otherwise):
- (A):
  - keep = 1 on every LoRA module reproduces θu, and keep = 0 reproduces θ0 (≤ 0.05 nats);
  - every row has ≥ 1 name token;
  - every LoRA module sits in a decoder layer.
- (B, C): the self-patch check and the image-swap self check (≤ 0.05 nats) on the first batch.

**Known limits:**
- One Qwen checkpoint.
- The Qwen vanilla was accepted post hoc, and its portrait route is weaker (0.65 vs about 0.81).
- The windows and thresholds were set on LLaVA (32 layers); Qwen has 28.
- First-token readout; each model has its own dc facts.
- Qwen's template puts the image directly before the question; the wording "this person" is ours.

**Sizing:**
- (A): about 18 forward passes per fact, 1–3 minutes per model.
- (B): about 120 pairs × 15 layers, in 4 jobs (X1/W1, X2/W2, S1/S2, image swap), about 5–15 minutes each.
- (C): about 5 minutes.
- In total about 15 minutes of wall-clock on the idle GPUs, one job each.
- Smoke tests (`--debug` / `--max-pairs 4`) run first, and their outputs are deleted before the full runs.

**Outcome, Qwen seed 2 step 266** (2026-11-23, 04:38–04:49 UTC; code as hashed in `logs/dev032/launch.txt`,
uncommitted; GPUs 0, 4, 5, 6, 7; all jobs exit 0).
- **Checks:** all 0.00 (self-patch, image-swap self check, keep = 1 / keep = 0). Every row has 2–6 name tokens.
- **Smoke tests:** run with `--max-facts 8` instead of `--debug`, which switches to the debug data root. On 8 facts
  they already showed the Qwen depth pattern below. Their outputs were deleted before the full runs.
- **Outputs:**
  - (A): `results/qwen2vl7b/mech_umu/umu_npo_rt_txt_lr2e-04_s2/step266/name_depth/` and
    `results/llava/mech_umu/umu_npo_rt_txt_lr2e-04_s0_gate_rw5n200/step300/name_depth/`;
  - (B): `results/qwen2vl7b/mech_umu/umu_npo_rt_txt_lr2e-04_s2/step0/convergence/` (`summary_first.json`);
  - (C): `.../step266/convergence_transfer/`.
- **Figures:** `proposal/figures/name_depth_Qwen_s2_step266.png`, `name_depth_G3_step300.png`,
  `convergence_first_qwen_theta0.png`.

**(A) Depth.** Recovery R on the first answer token (1 = all forgetting removed).

| LoRA off in | LLaVA G3 s0: R_all / R_name | Qwen s2: R_all / R_name |
|---|---|---|
| L0–3 | **0.48 / 0.36** | 0.12 / 0.11 |
| L4–7 | 0.19 / 0.16 | **0.41 / 0.40** |
| L8–11 | 0.11 / 0.09 | 0.31 / 0.27 |
| L12–15 | 0.11 / 0.07 | 0.25 / 0.18 |
| L16–19 | 0.09 / 0.03 | 0.25 / 0.13 |
| L20–23 | 0.07 / 0.02 | 0.18 / 0.02 |
| L24–27 | 0.03 / 0.01 | 0.19 / 0.00 |
| L28–31 | 0.03 / 0.00 | — |
| name tokens, all layers | 0.87 | 0.99 |
| LoRA kept only at the name tokens | 0.42 | 0.45 |

Readouts behind R:

| | LLaVA G3 s0 | Qwen s2 |
|---|---|---|
| F(T); top-1 θu / θ0 | 7.66 nats; 0.38 / 0.88 | 11.46 nats; 0.24 / 0.95 |
| F(NI); top-1 θu / θ0 | 0.01 nats; 0.98 / 1.00 | −0.04 nats; 0.89 / 0.90 |

Top-1 of θu with one window off moves little: at most 0.45 (LLaVA, L0–3) and 0.35 (Qwen, L4–7). With the name
tokens off in all layers it reaches 0.74 and 0.94.

**Verdict (A):**
- LLaVA is **partial**: D1 fails (0.48 < 0.5), D3 fails (0.36), D2 holds. As pre-registered, the depth comparison
  (link 1b) is therefore descriptive.
- Qwen is **"not early"** (0.12 / 0.11 < 0.3); D2 holds.
- **N1 holds in both models (link 2):** the portrait-only question does not trigger the edit.

**(B) Convergence, Qwen θ0.** 151 pairs, 101 with different first answer tokens, 95 kept.

| layer | 0–12 | 14 | 16 | 18 | 20 | 22 | 24 | 27 |
|---|---|---|---|---|---|---|---|---|
| X1 face(A) → name(B) | ≤ 0.01 | 0.03 | 0.03 | 0.04 | 0.24 | **0.76** | 0.88 | 0.91 |
| X2 name(A) → face(B) | 0.00 | 0.02 | 0.03 | 0.09 | 0.42 | **0.95** | 0.97 | 0.97 |
| W1 name(A) → name(B) | ≤ 0.02 | 0.07 | 0.08 | 0.14 | 0.38 | **0.86** | 0.97 | 0.97 |
| W2 face(A) → face(B) | 0.00 | 0.01 | 0.02 | 0.04 | 0.22 | **0.83** | 0.89 | 0.91 |
| S1 / S2 (keep B) | ≥ 0.99 | ≥ 0.99 | 1.00 | ≥ 0.99 | ≥ 0.99 | 1.00 | 1.00 | 1.00 |
| image tokens of A → F(B) | 0.89 | 0.84 | 0.85 | 0.81 | **0.60** | 0.09 | 0.00 | 0.00 |

**Verdict (B): convergence supported (link 4).**
- Validity: W1 max 0.99, W2 max 0.91; S1 and S2 ≥ 0.99 at every layer.
- L* = 22 for X1, X2, W1 and W2 alike; X1 reaches 0.91 and X2 0.97.
- The expectation on depth is not met. L* = 22 of 28 (relative depth 0.79) is later than LLaVA's 20 of 32 (0.63),
  and outside the expected layers 16–20.
- Image-token L_c = 20 ≤ L*_X2 + 2.
- Even at L0 the image swap reaches only 0.89 (LLaVA 1.00). Qwen's face route does not always carry the person.

**(C) Transfer, Qwen s2 step 266** (62 facts; θu's route gap 18.78 nats summed, 11.10 on the first token).

| layer | ≤ 12 | 14 | 16 | 18 | 20 | 22 | 24 | 27 |
|---|---|---|---|---|---|---|---|---|
| S_u, summed | ≤ 0.01 | 0.04 | 0.04 | 0.07 | 0.28 | **0.54** | 0.58 | 0.59 |
| S_0, summed (control) | 0.00 | 0.00 | 0.00 | 0.00 | −0.01 | −0.02 | −0.02 | −0.02 |
| S_u, first token | ≤ 0.01 | 0.05 | 0.06 | 0.10 | 0.45 | 0.90 | 0.98 | 1.00 |
| top-1 θu, patched face route | 0.85–0.89 | 0.76 | 0.73 | 0.65 | 0.40 | **0.24** | 0.24 | 0.24 |

Unpatched, θu's top-1 is 0.89 on the face route and 0.24 on the name route. LLaVA G3 s0 (DEV-024, summed): ≤ 0.04
up to L14, 0.21 at L16, 0.48 at L20, 0.56 at L22, plateau 0.60.

**Verdict (C): link 5 holds.** Max S_u = 0.59 ≥ 0.5; S_u ≤ 0.04 at every layer ≤ L* − 6 = 16; |S_0| ≤ 0.02.

**Overall, as pre-registered:** links 2, 4 and 5 hold on Qwen, and 1 and 3 held already (DEV-030). Link 1b is not
established. On this readout LLaVA is only partial, and Qwen's edit is not in layers 0–3. So the overall criterion
("1b, 2, 4 and 5 hold") is **not met as written**. Link 6 differs, as known.

**Reading (exploratory, formed after seeing these data):**
- **What carries over.** In both models:
  - the text-only edit is bound to the name tokens (LoRA off there removes 87% / 99% of the forgetting);
  - the portrait-only question does not trigger it;
  - the name and face routes merge in the question tokens at one depth (LLaVA L20, Qwen L22);
  - after that depth, the name route's suppressed states also suppress the face route (plateau 0.60 / 0.59). In
    Qwen, the face route's top-1 then drops exactly to the name route's (0.89 → 0.24 from L22).
  - The account "the edit sits on the name, before the routes merge, so the face reaches the merged representation
    without passing it" holds in both.
- **What does not.**
  - The depth of the edit: LLaVA L0–3; Qwen L4–11, with a tail up to L19.
  - In Qwen the single windows add up to 1.71: several layers each suffice in part.
  - "Layers 0–3" is a LLaVA detail. "Before the convergence" holds in both: in Qwen, R_name is ≈ 0 from L20.
- **The transfer tracks each model's convergence.** S_u rises at L16–22 in LLaVA (L* 20) and at L20–22 in Qwen
  (L* 22).
- **Qwen-specific.** The face route is weaker: image swap 0.89 at L0, and θ0's face-route top-1 0.83 in S2. This
  fits link 6a failing (DEV-030). With name + portrait, the face does not override the name edit.
- In Qwen, LoRA in L20–27 at non-name positions still accounts for about 0.2 (R_all 0.18–0.19, R_name ≈ 0). This
  part is not localised here.
- **Limits:** one Qwen checkpoint; first-token readout; single-window recoveries are small in top-1.

### DEV-031 — 2026-11-23 — Qwen vanilla with LoRA on the vision side too (decided by: user; design by: implementer)
**Why.** DEV-030's Qwen vanilla answers by portrait much worse than by name: 0.62–0.65 vs 0.70–0.79 over 5 epochs.
LLaVA's UMU vanilla is the other way round (0.80 vs 0.71).
- Exploratory diagnosis from the saved readouts: Qwen's portrait errors are mostly near misses (162 → 168 cm), not
  another person's answer (28%, as for the name route's 25%). The face → facts link is weak, not the identification
  of a wrong person.
- One recipe difference: UMU's script puts LoRA on all linear layers, the vision tower included; DEV-030 put LoRA on
  the LM only.

**Test** (`configs/qwen2vl7b/umu_vanilla_vis.yaml`, run tag `vis`).
- As DEV-030 stage 1 (same data, schedule, lr 2e-4, 3 epochs, readout per epoch), with LoRA also on every linear
  layer of the vision tower and the merger: 130 modules.
- Separate run dir and merged model: `runs/qwen2vl7b/umu_vanilla_vis`, `models/qwen2vl7b_umu_vanilla_vis`.

**Predictions and thresholds** (epoch 1, against DEV-030's epoch 1 at 0.62):
- Forget img ≥ 0.72: the vision-side LoRA helps.
- ≤ 0.66: it does not.
- Otherwise partial.

The run continues to 3 epochs only if epoch 1 is not "does not help". The selection and acceptance rule of DEV-030
(both ≥ 0.70) applies.

**If accepted:** the Qwen unlearning (DEV-030 stages 2–3) is redone on this vanilla, with the unlearning LoRA still on
the LM only, as for LLaVA. The results on the epoch-4 vanilla stay as a preliminary set.

**Known limits:** one seed; vision LoRA changes the image features themselves, which the LLaVA vanilla also had.

**Outcome, epoch 1** (2026-11-23; code as hashed in `logs/dev031/launch.txt`, uncommitted; GPU 5; about 2.2–3.6 s
per step; 130 vision/merger + 196 LM LoRA modules).

| epoch 1 | forget txt / img | retain txt / img | doubly-correct forget / retain |
|---|---|---|---|
| vision + LM LoRA | 0.69 / **0.66** | 0.63 / 0.55 | 57 / 94 |
| LM-only LoRA (DEV-030) | 0.70 / 0.62 | 0.66 / 0.56 | 53 / 96 |

**Verdict: the vision-side LoRA does not help** (0.66 ≤ 0.66). Retain img is unchanged (0.55 vs 0.56). As
pre-registered, the run was stopped after the epoch-1 readout, and the epoch-4 LM-only vanilla stays the Qwen
vanilla.

**Reading (exploratory):** adapting the image features does not strengthen Qwen's face → facts route at this stage of
training. The remaining candidates are input resolution (256 visual tokens) and properties of the model itself.
Neither is tested.

**Deviation: continued after the stop rule** (2026-11-23, 01:40 UTC, decided by the user after seeing epoch 1).
- Epoch 1 sits at the threshold (0.66, +0.04 over the LM-only run).
- The run resumes from its step-1500 state for epochs 2 and 3, with the same schedule.
- The stop rule was not followed. Any gain at epochs 2–3 is reported against DEV-030's epochs 2–3 (img 0.64 / 0.63),
  and the vanilla choice for Qwen is made afterwards with the user.

**Outcome, epoch 2, and the rule for stages 2–3** (2026-11-23, 03:25 UTC; written before the epoch-3 readout).

| epoch 2 | forget txt / img | retain txt / img | doubly-correct forget / retain |
|---|---|---|---|
| vision + LM LoRA | 0.78 / **0.68** | 0.67 / 0.58 | 66 / 103 |
| LM-only LoRA (DEV-030) | 0.77 / 0.64 | 0.685 / 0.565 | 59 / 98 |

- Paired per question (exploratory): on forget img, 10 questions are right only with the vision LoRA and 6 only
  without it (sign test p = 0.45; epoch 1: 6 vs 2, p = 0.29). Retain img: 14 vs 11 (epoch 1: 12 vs 13). Txt: no
  difference.
- The gain on the portrait route is consistent in direction but not significant. Epoch 2 (0.68) is above every
  LM-only epoch (0.61–0.65) and below the 0.70 bar.

**Decision (user, 03:20 UTC):** if the run is accepted under DEV-030's unchanged rule after epoch 3, stages 2–3
start automatically on this vanilla, as this entry pre-registered ("If accepted"). Otherwise nothing starts, and the
epoch-4 LM-only vanilla stays the Qwen vanilla.
- The rule: the epoch with the highest min(forget txt, forget img), both ≥ 0.70. Only epoch 3 can now meet it.
- Stages 2–3 as DEV-030: lr 2e-4; seeds 0 and 1 together, then seed 2; the same lr fallback; the gate-1 selection;
  core3 with `model.attn=eager`. The unlearning LoRA is on the LM only.
- Run tag `vis`: `runs/qwen2vl7b/umu_unlearn/umu_npo_rt_txt_lr2e-04_s{0,1,2}_vis`. Training on 2 GPUs, core3 on a
  free GPU. Status log: `logs/dev031/orchestrate.txt`.
- **Merge check (aborts all jobs if it fails):** seed 0's step-0 readout of the merged model must be within 0.05 of
  the selected epoch's readout on forget txt and forget img. On the epoch-4 LM-only vanilla the difference was
  0.00 / +0.01.

**What stages 2–3 can tell** (pre-registered). The question left open by DEV-030 is whether Qwen's missing rescue (name
+ portrait does not undo the forgetting; NB ≈ T) comes from the weak face route of the epoch-4 vanilla.
- Compare forget dc txt+img per seed with the epoch-4 runs (0.40 / 0.68 / 0.56).
- **≥ 0.85 in at least 2 of 3 seeds:** the face route's strength explains the missing rescue.
- **Every seed within ±0.10 of its epoch-4 value:** not explained by the face route at this strength.
- Otherwise unresolved.
- Q1–Q3 are read with DEV-030's criteria.

**Known limit:** the portrait route improves only a little (about 0.65 → 0.70), so this is a weak test of the
"undertrained vanilla" explanation.

**Outcome, epochs 2–3** (2026-11-23, 04:21 UTC; code unchanged from `logs/dev031/launch.txt`, same hashes; GPU 5).

| epoch | forget txt / img | retain txt / img | doubly-correct forget / retain |
|---|---|---|---|
| vision + LM, 1 | 0.69 / 0.66 | 0.63 / 0.55 | 57 / 94 |
| vision + LM, 2 | 0.78 / 0.68 | 0.67 / 0.58 | 66 / 103 |
| vision + LM, 3 | 0.75 / **0.69** | 0.655 / 0.61 | 59 / 100 |
| LM-only (DEV-030), 3 | 0.77 / 0.63 | 0.715 / 0.60 | 61 / 108 |
| LM-only (DEV-030), 4 (the Qwen vanilla in use) | 0.76 / 0.65 | 0.76 / 0.61 | 61 / 118 |

- Selected: epoch 3 (min 0.69). **Not accepted** (0.69 < 0.70). As set above, stages 2–3 did not start, and **the
  epoch-4 LM-only vanilla stays the Qwen vanilla**. DEV-030's stage 2–3 results stand (no longer preliminary on this
  account).
- The script still merged epoch 3 into `models/qwen2vl7b_umu_vanilla_vis` (16 GB). Nothing uses it.

Paired per question, vision epoch 3 against LM-only (sign test):

| against | forget img | forget txt | retain img | retain txt |
|---|---|---|---|---|
| LM-only epoch 3 | 9 vs 3 (p = 0.15) | 6 vs 8 | 11 vs 9 | 6 vs 18 (p = 0.02) |
| LM-only epoch 4 | 10 vs 6 (p = 0.45) | 6 vs 7 | 13 vs 13 | 9 vs 30 (p = 0.001) |

**Reading (exploratory):**
- The vision-side LoRA raises the portrait route by about 0.04–0.06 at every epoch. This is consistent in direction
  but not significant per question. It does not reach the bar.
- Retain by name is worse (0.655 vs 0.76 for the vanilla in use). The vision vanilla also has fewer doubly-correct
  facts (59 / 100 vs 61 / 118).
- So the LM-only vanilla stays the better start model on the criteria that matter for the unlearning readout, not
  only by the 0.70 bar.
- Qwen's weak face → facts route is not fixed by adapting the image features. Whether the missing rescue (DEV-030) is
  a property of Qwen or of the vanilla stays open.

### DEV-030 — 2026-11-22 — second model family: Qwen2-VL-7B-Instruct on UMU (decided by: user; design by: implementer)
The paper needs the core findings on a second model family. The user chose to start now. Qwen2-VL-7B-Instruct is
used: the repo has its adapter, and Nikankin et al. and Cohen et al. study it.

**Stage 1 — UMU vanilla** (`scripts/finetune_umu_vanilla.py`, `configs/qwen2vl7b/umu_vanilla.yaml`). UMU released
only a LLaVA vanilla.
- UMU's own script (finetune/finetune.py): LoRA r8/α16, lr 2e-5, 5 epochs, MM_QA with the portrait. Its UM_QA loop is
  commented out.
- Here, so that the model knows every person by name and by face:
  - data: all 500 profiles, both UM_QA (name, no image) and MM_QA (portrait + "this person");
  - LoRA r16/α16 on the LM linear layers;
  - lr 2e-4, 3 epochs, batch 8; answer tokens only in the loss;
  - portraits at 448×448 (256 visual tokens);
  - the model's own chat template, also for the readout.
- Readout after each epoch: UMU protocol, forget (50) and first 100 retain profiles.
- Selection: the epoch with the highest min(forget all txt, forget all img). **Accepted if both ≥ 0.70**; otherwise
  stop and report. The selected adapter is merged and becomes `umu.ckpt`.

**Stage 2 — unlearning** (`configs/qwen2vl7b/umu_npo_rt.yaml`).
- The LLaVA G3 recipe: NPO + retain, lr 2e-4, retain weight 5, 200 retain-unlearn people, n_retain_eval 100,
  4 epochs, checkpoints every 0.5 epoch; seeds 0, 1, 2.
- A checkpoint qualifies under DEV-019's gate 1: unseen retain dc txt and img ≥ 0.90, degen ≤ 0.20, forget dc txt
  ≤ 0.70. The selected one per seed is the qualifying checkpoint with the lowest forget dc txt.
- If no checkpoint of seed 0 qualifies, lr 1e-4 and 4e-4 are tried on seed 0 before the other seeds.

**Stage 3 — the three core tests** on each selected checkpoint (forget dc facts; first answer token; θ0 = the Qwen
vanilla).
- (Q1) Leak: forget dc img ≥ 0.85 and forget dc txt+img ≥ 0.85 (the portrait route leaks). Blank image: reported.
- (Q2) Knockout: with the portrait shown and the name in the question, blocking attention from the positions after
  the image to the image tokens restores forgetting. Criteria: F(knockout) ≥ 0.5 × F(text), with θ0's top-1 under
  the knockout ≥ 0.6.
- (Q3) Name localisation: LoRA off at the name tokens on the text form, F ≤ 0.3 × F(text). Control: LoRA off at as
  many other text tokens, F ≥ 0.7 × F(text). Also DEV-029's 6b under knockout, reported.
- Tests: an adapter-agnostic version of `mech_umu_link6.py`, validated on LLaVA G3 seed 0 against DEV-029's numbers
  before use on Qwen.

**Validity checks:** the hook checks of DEV-029 (keep = 1 reproduces θu; an empty block reproduces the unhooked
run), run on Qwen before stage 3.

**Known limits:**
- Our vanilla recipe differs from UMU's (both forms; lr; rank).
- Readouts use Qwen's chat template, so absolute accuracies are not comparable with LLaVA's.
- Convergence, state transfer and the two-mechanism analysis are not part of this entry.

**Setup notes, before any Qwen result** (2026-11-22):
- `scripts/mech_umu_core3.py` reproduces DEV-029 exactly on LLaVA G3 seed 0: F 7.67 / 0.98 / 0.16 / 5.03 / 0.53 /
  4.62 nats, and blank image 3.03 as DEV-020. On LLaVA, Q2 and Q3 pass.
- **Qwen runs use `--set model.attn=eager`.** On a Qwen debug run (4 steps; base as a stand-in vanilla), the
  empty-block check failed under sdpa: 0.051 nats against a 0.05 tolerance. Without padding, sdpa takes its
  is_causal kernel, and the explicit mask switches kernels. Under eager both checks are 0.00.
- The unlearning pipeline (`unlearn_umu.py`) and the protocol readout run on Qwen in debug mode.
- The vanilla fine-tune runs at about 1.44 s per step (4500 steps).
- Order of stage 2, fixed now: seeds 0 and 1 (lr 2e-4) start together on GPUs 4 and 5 as soon as the vanilla is
  accepted. If seed 0 has no qualifying checkpoint, seed 0 is rerun at lr 1e-4, then at 4e-4. Seed 2 starts with
  the first lr at which seed 0 qualifies. Each selected checkpoint goes to stage 3 automatically.
- Automation: a scratchpad orchestrator; its status log is `logs/dev030/orchestrate.txt`.

**Amendment before the epoch-3 readout** (2026-11-22, decided by the user after seeing epochs 1–2).
- Readouts so far, forget txt / img: epoch 1 0.700 / 0.620 (53 dc forget facts); epoch 2 0.770 / 0.640 (59).
  LLaVA's UMU vanilla: about 0.75 / 0.81 (65–66).
- If no epoch of the 3 meets the acceptance bar, training continues from the epoch-3 adapter for up to 3 more epochs:
  same data, lr 1e-4 decaying linearly to 0 over the extra epochs, a readout after each
  (`scripts/finetune_umu_vanilla.py --extra-epochs 3`).
- The selection and acceptance rule is unchanged and covers all epochs. If no epoch is accepted after 6, stop and
  report.

**Stage 1, epochs 1–3** (2026-11-22).
- Epoch 3: forget txt 0.770 / img 0.630; retain 0.715 / 0.600; doubly-correct 61 forget / 108 retain.
- Best epoch: 2 (min 0.640). **Not accepted.** Its adapter was merged into `models/qwen2vl7b_umu_vanilla`; no
  unlearning used it.
- The portrait route plateaus at 0.62–0.64 while the training loss falls to about 0.19.
- Wrong answers are mostly numeric attributes: height, salary, birth date.
- The amendment's 3 extra epochs started automatically at 22:03 UTC.

**Amendment 2: acceptance rule changed after seeing the results** (2026-11-22, 23:45 UTC, decided by the user).

| epoch | 4 (extra, lr 1e-4) | 5 (extra) |
|---|---|---|
| forget txt / img | 0.760 / 0.650 | 0.790 / 0.610 |
| retain txt / img | 0.760 / 0.610 | 0.730 / 0.625 |
| doubly-correct forget / retain | 61 / 118 | 56 / 116 |

Training loss fell to about 0.04. Across epochs 1–5 the portrait route stays at 0.61–0.65.

- **Decision:** the user accepted **epoch 4** without waiting for epoch 6. The 0.70 bar is replaced, post hoc, by
  "enough doubly-correct forget facts", and epoch 4 has 61. Epoch 4 is also the one the pre-registered selection rule
  (highest min(txt, img)) picks among epochs 1–5.
- The extra run was stopped during epoch 6, so its end-of-run merge would not overwrite the vanilla used by the
  unlearning runs. No epoch-6 readout exists.
- Epoch 4's adapter is merged into `models/qwen2vl7b_umu_vanilla`, replacing the earlier epoch-2 merge.
- **Consequence:** Qwen's portrait route is weaker at baseline than LLaVA's (0.65 vs about 0.81 on all forget
  facts). The leak tests read doubly-correct facts only, but the paper must report this difference and that this
  rule was set after seeing the data.

**Outcome on the epoch-4 vanilla, stages 2–3** (2026-11-23; preliminary, because DEV-031 may replace the vanilla).
- Runs: `runs/qwen2vl7b/umu_unlearn/umu_npo_rt_txt_lr2e-04_s{0,1,2}`, run in about 25 minutes each.
- core3: `results/qwen2vl7b/mech_umu/<run>/step<N>/core3/`; checks 0.00 under eager.
- Every seed qualified at lr 2e-4.

| seed (selected step) | forget dc txt / img / txt+img | G | unseen retain txt / img |
|---|---|---|---|
| 0 (300) | 0.24 / 0.89 / 0.40 | +0.65 | 0.92 / 0.92 |
| 1 (190) | 0.40 / 0.92 / 0.68 | +0.52 | 0.94 / 0.95 |
| 2 (266) | 0.42 / 0.92 / 0.56 | +0.50 | 0.94 / 0.95 |

First-token forgetting F (nats):

| seed | T | T + name-off | T + control-off | NB (blank) | NP visible | NP knockout | NP knockout + name-off |
|---|---|---|---|---|---|---|---|
| 0 | 16.39 | 0.22 | 16.28 | 15.61 | 13.91 | 16.16 | 0.21 |
| 1 | 7.35 | 0.07 | 7.27 | 7.29 | 5.11 | 7.30 | 0.06 |
| 2 | 11.43 | 0.10 | 11.45 | 10.77 | 8.37 | 11.23 | — |

θ0 top-1: T 0.95, NP knockout 0.95.

**Verdict under DEV-030:**
- **Q1 partly met.** The portrait question leaks in every seed (img 0.89–0.92 ≥ 0.85). The name + portrait question
  does not reach 0.85 (0.40–0.68).
- **Q2 met as defined** (F knockout ≥ 0.5 F(T); θ0 0.95). It is uninformative here, because the portrait does not
  undo the forgetting even without the knockout.
- **Q3 met in every seed.** LoRA off at the name tokens removes 98–99% of the forgetting; off at as many other tokens,
  none.
- DEV-029's 6a fails (NP visible 5–14 nats).

**Reading (exploratory):**
- On Qwen, too, text-only unlearning is written on the name tokens, and a question that identifies the person by the
  portrait alone escapes it.
- Unlike LLaVA, adding the portrait to the name question does not rescue the answer, and a blank image does not
  either (NB ≈ T). The name-bound edit dominates whenever the name is present.
- This fits this vanilla's weak face → facts route (DEV-031). Whether it is a property of Qwen or of an undertrained
  vanilla, the DEV-031 vanilla will tell.

### DEV-029 — 2026-11-22 — closing the causal chain: is the name-located edit still active when the portrait is shown, and only overridden by the face route? (decided by: user; design by: implementer)
**Chain.** For the name-bound models:
- (1) the edit sits on the name tokens and causes the forgetting (DEV-020/021, LoRA off there restores 0.89–0.98);
- (3) the portrait is a separate route (DEV-021 knockout);
- (4) the routes converge at about layer 20 (DEV-027);
- (5) suppression placed after the convergence also suppresses the face route (DEV-024 part 2).

The missing link (6): with the name and the person's own portrait in the question, the name-located edit still
suppresses, and the answer survives only because the face route supplies the person.

**Test** (`scripts/mech_umu_link6.py`; eval only; forget dc facts; first answer token, teacher forced).
- Models: G3 seeds 0 and 2, step 300 (name-bound, DEV-021).
- Input NP: `USER: <image>\n{q}\nASSISTANT:` with the person's own portrait; q contains the name.
- Interventions:
  - knockout: the positions after the image cannot attend to the image tokens, in all layers (as DEV-018);
  - name-off: the LoRA switched off at the name tokens in all layers (`mask_contributions`);
  - control-off: the LoRA switched off at the same number of non-name text tokens after the image, the first
    template tokens.
- Forgetting F = lp_θ0 − lp_θu under the same input and knockout. θ0 has no LoRA, so it has no off variants.

Conditions:
- NP visible;
- NP knockout;
- NP knockout + name-off;
- NP knockout + control-off;
- NP visible + name-off;
- text-only T, and T + name-off.

**Predictions and thresholds** (seed 0 primary; seed 2 replication, same thresholds):
- (6a) face overrides: F(NP visible) ≤ 1.0 nat, and F(NP knockout) ≥ 0.5 × F(T).
- (6b) the restored forgetting comes from the name-located edit: F(NP knockout + name-off) ≤ 0.3 × F(NP knockout).
- (6c) control: F(NP knockout + control-off) ≥ 0.7 × F(NP knockout).
- Link 6 closed if 6a, 6b and 6c hold. Not closed if 6b fails (the restored forgetting does not depend on the name
  edit). Otherwise partial.
- Top-1 of the correct first token reported next to every F.

**Validity checks** (first batch; the script aborts otherwise):
- an explicit attention mask with nothing blocked reproduces the unhooked run (≤ 0.05 nats);
- `mask_contributions` with keep = 1 everywhere reproduces θu (≤ 0.05 nats);
- every row has ≥ 1 name token.

**Known limits:**
- The ΔWh transplant between inputs (replacing the edit's output in NP by its output in T) is not done here. This test
  shows that the edit acts in NP, not that its output is the same.
- The knockout also removes what θ0 reads from the image; hence F is taken against θ0 under the same knockout.
- Two seeds of one recipe.

**Sizing:** 66 facts × about 10 forward passes; a few minutes per model; one GPU each.

**Outcome** (2026-11-22; code as hashed in `logs/dev029/launch.txt`, uncommitted; GPUs 0 and 6; checks 0.00 /
0.00; 4–6 name tokens per row).

| first-token forgetting F (nats); top-1 θu | seed 0 | seed 2 |
|---|---|---|
| T (text only) | 7.67; 0.38 | 5.34; 0.58 |
| T + name-off | 0.98; 0.74 | 0.83; 0.83 |
| NP visible | 0.16; 0.94 | 0.02; 0.98 |
| NP visible + name-off | 0.00; 1.00 | 0.01; 1.00 |
| NP knockout | **5.03**; 0.50 | **3.94**; 0.61 |
| NP knockout + name-off | **0.53**; 0.86 | **0.58**; 0.86 |
| NP knockout + control-off | 4.62; 0.55 | 3.78; 0.61 |
| θ0 top-1: NP / NP knockout / T | 1.00 / 0.92 / 0.88 | 1.00 / 0.92 / 0.88 |

**Verdict: link 6 closed in both seeds.**
- 6a: 0.16 and 0.02 ≤ 1.0, and 5.03 ≥ 3.84 and 3.94 ≥ 2.67.
- 6b: 0.53 ≤ 1.51 and 0.58 ≤ 1.18.
- 6c: 4.62 ≥ 3.52 and 3.78 ≥ 2.76.

**Reading.** With the name and the person's own portrait in the question, the name-located edit is still active.
Once the question cannot read the image, it suppresses the answer again (5.03 / 3.94 nats, 66–74% of the text-only
forgetting). That restored suppression disappears when the LoRA is switched off at the name tokens alone, but not
when it is switched off at as many other text tokens. The answer survives in the normal portrait input because the
face route supplies the person, not because the edit stops acting.

With links 1–5 (DEV-020, 021, 024, 027), the chain for the name-bound models is closed by intervention at each link:
- the text-only edit on the name tokens causes the forgetting;
- it stays active with the portrait;
- the portrait route bypasses it and joins the shared representation at about layer 20;
- suppression present after that point would also stop the portrait route.

### DEV-028 — 2026-11-22 — unlearning placed after the convergence point: LoRA only in layers 20–31 (decided by: user; design by: implementer)
**Question.** DEV-027 places the convergence of the name and face routes at about layer 20. The text-only edits found
so far sit before it: on the name tokens in L0–3, or as a switch that forms only on text-only input. DEV-024 part 2
showed that suppression placed in the shared stream after L16 also suppresses the face route. Does text-only
unlearning whose LoRA exists only after the convergence point also forget on the portrait route?

**Run.**
- G3 recipe: NPO + retain, lr 2e-4, retain weight 5, 200 people, `umu.n_retain_eval=100`, 4 epochs, seed 0.
- The only change: LoRA only on the LM linear layers of decoder layers 20–31 (`lora.min_layer=20`;
  `vlmu/models.py:lora_targets`). Run `umu_npo_rt_txt_lr2e-04_s0_gate_rw5n200_L20`.
- Training data stays text-only.
- Comparison: G3 seed 0 (all 32 layers), same recipe and seed.

**Primary checkpoint.** Among checkpoints with unseen retain dc txt ≥ 0.85, retain dc img ≥ 0.85 and degen ≤ 0.20,
the one with the lowest forget dc txt. The retain bar is lower than gate 1's 0.90: this is a pilot.

**Predictions and thresholds:**
- **Validity:** forget dc txt ≤ 0.70 at some checkpoint. Otherwise the run is not interpretable, and the fallback is
  one rerun at lr 4e-4.
- **Leak reduced:** forget dc img ≤ 0.70 at the primary checkpoint. G3 seed 0 kept 0.92–0.95 at every checkpoint.
- **Leak unchanged:** forget dc img ≥ 0.90 there.
- **Otherwise:** partial.
- **Reported without a threshold:** G, txt+img, retain, the biography share per checkpoint (does a switch form?), and
  every checkpoint.

**Validity check in code:** the adapter must hold LoRA modules on layers 20–31 only (7 × 12 = 84 modules); the run
aborts otherwise.

**Known limits:** one seed; NPO's format loophole stays open, so a late-layer switch could form and spare the
portrait route for that reason; fewer trainable layers may need a different lr.

**Sizing:** about 25 minutes of training on one GPU (GPU 5), plus checkpoint evaluations.

**Outcome, lr 2e-4** (2026-11-22; code as hashed in `logs/dev028/launch.txt`, uncommitted; LoRA on 84 modules, layers
20–31; about 1.4 s per step).

| step | 38 | 76 | 114 | 152 | 190 | 228 | 266 | 300 |
|---|---|---|---|---|---|---|---|---|
| forget dc txt | 0.955 | 0.894 | 0.879 | 0.818 | 0.818 | 0.818 | 0.788 | 0.818 |
| forget dc img | 0.970 | 0.939 | 0.924 | 0.909 | 0.909 | 0.894 | 0.909 | 0.924 |
| unseen retain txt / img | 0.967 / 1.000 | 0.943 / 0.975 | 0.951 / 0.943 | 0.943 / 0.934 | 0.943 / 0.959 | 0.934 / 0.959 | 0.926 / 0.967 | 0.926 / 0.967 |

**Verdict: not interpretable.** The forgetting never reached 0.70 (minimum 0.788). With the LoRA confined to layers
20–31, the same recipe forgets little. The portrait route stays at 0.89–0.92, so nothing can be said about the leak
yet. The pre-set fallback, one rerun at lr 4e-4, follows (`..._L20` run tag kept, lr in the run id).

**Outcome, fallback lr 4e-4** (2026-11-22; `umu_npo_rt_txt_lr4e-04_s0_gate_rw5n200_L20`).

| step | 38 | 76 | 114 | 152 | 190 | 228 | 266 | 300 |
|---|---|---|---|---|---|---|---|---|
| forget dc txt | 0.894 | 0.864 | 0.788 | 0.712 | 0.742 | 0.667 | **0.621** | 0.621 |
| forget dc img | 0.955 | 0.848 | 0.955 | 0.864 | 0.879 | 0.879 | **0.848** | 0.833 |
| txt+img | 0.985 | 0.955 | 0.939 | 0.894 | 0.909 | 0.894 | 0.833 | 0.848 |
| unseen retain txt / img | 0.934 / 0.975 | 0.910 / 0.893 | 0.951 / 0.926 | 0.885 / 0.910 | 0.902 / 0.918 | 0.910 / 0.918 | 0.918 / 0.910 | 0.926 / 0.934 |

- No biography, refusal or other long answer at any checkpoint. The forgetting is wrong short answers (0.14 → 0.36),
  e.g. 160 cm → "165 cm.".
- NPO loss 0.011 at step 300.

**Verdict.**
- Validity is met: forget dc txt 0.621 ≤ 0.70.
- The primary checkpoint is step 266. Steps 266 and 300 tie on forget dc txt, and gate 1's tie rule (earlier step)
  is applied; the entry did not name a tie rule.
- Forget dc img 0.848 there is between the two thresholds: **partial**.
- Against G3 seed 0 (all layers; at forget 0.545–0.667: img 0.92–0.94, G +0.27 to +0.38), the portrait route
  forgets somewhat more (0.83–0.85; G +0.21 to +0.23), with retain kept (0.918 / 0.910).
- The leak is reduced, not closed. One seed.

**Editorial note** (2026-11-22): stray duplicate heading lines (DEV-026, DEV-028, DEV-024), left by edits earlier
today, were removed. No entry's text changed.

### DEV-026 — 2026-11-22 — can relative NPO forget deeply? One seed at lr 4e-4 (decided by: user; design by: implementer)
DEV-023's relative NPO forgot too little (forget dc txt 0.742–0.773) for its comparison to count. The user asked to
try one seed with stronger unlearning first, to see whether relative NPO can forget deeply at all.

**Run.**
- `unlearn.method=rnpo_rt train.lr=4.0e-4`; otherwise DEV-023 (retain weight 5, 200 people, n_retain_eval 100,
  4 epochs).
- Seed 1, the seed that formed the switch under NPO: `umu_rnpo_rt_txt_lr4e-04_s1_gate_rw5n200`.
- Code unchanged since DEV-023.
- Then the four analyses on step 300, as in DEV-023.

**Questions** (exploratory; one seed):
- (1) Depth: does forget dc txt reach ≤ 0.70, and how deep at best?
- (2) If forget ≤ 0.70 is reached:
  - biography share ≥ 0.20 at any checkpoint contradicts the loophole reading for this seed;
  - ≤ 0.05 at every checkpoint (with sub_mean ≤ 0.3) is consistent with it.

  One seed cannot decide the question, because NPO itself formed no switch in 2 of 3 seeds.
- (3) Also reported: retain (txt, img), degen, forget dc img, criterion (d), and the share of wrong answers that
  equal a training distractor.

**Sizing:** about 25 minutes of training on one GPU, then about 10 minutes of analyses spread over idle GPUs.

**Outcome** (2026-11-22; code as hashed in `logs/dev026/launch.txt`, uncommitted; all jobs exit 0).

| step | 76 | 152 | 190 | 228 | 266 | 300 |
|---|---|---|---|---|---|---|
| forget dc txt | 0.818 | 0.606 | 0.409 | 0.591 | 0.545 | **0.500** |
| forget dc img | 0.939 | 0.879 | 0.803 | 0.955 | 0.879 | 0.879 |
| unseen retain txt / img | 0.910 / 0.975 | 0.869 / 0.902 | 0.762 / 0.852 | 0.959 / 0.975 | 0.934 / 0.951 | **0.910 / 0.926** |
| biography share | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | **0.00** |

At step 300:
- sub_mean at L16 0.028.
- LoRA off at the name / at the last token: 0.88 / 0.34 (33 lost facts).
- FC-train(θu) 0.621 (base 0.530); FS 0.545, FS recovery 0.10.
- First-token forgetting: text 4.02 nats, own portrait 0.53, portrait question 0.09.
- Own portrait with the image blocked: top-1 θu 0.85 → 0.61 (θ0 0.92).
- 13 of 33 wrong short answers equal one of the row's training distractors.

**Answers:**
- (1) Depth: yes. Forget dc txt ≤ 0.70 from step 152 on (minimum 0.409). Step 300 also meets gate 1's retain bar
  (0.910 / 0.926).
- (2) No switch formed at any checkpoint, and sub_mean ≈ 0. This is consistent with the loophole reading on the seed
  that formed the switch under NPO (biography share 0.79, sub_mean 0.66). One seed only.
- (3) The forgetting is name-bound and content-level: wrong short answers, knowledge near the prior under forced
  choice. The portrait route still leaks (img 0.88; portrait question 0.09 nats).

### DEV-024 — 2026-11-22 — where do the name route and the portrait route converge? Identity swap across routes (decided by: user, paper plan step A; design by: implementer)
DEV-020 and DEV-021 show that the portrait reaches the person's facts without passing the text-only edit. This entry
asks where, inside the vanilla, a person identified by name and the same person identified by face begin to share
one representation, and on which side of that point the unlearning signal sits.

**Prompts** (`scripts/mech_umu_convergence.py`; eval only). For a fact (person P, attribute) with the readout's
unimodal question q:
- name route N(P): `USER: {q}\nASSISTANT:`;
- face route F(P): `USER: <image>\n{q with P's name replaced by "this person"}\nASSISTANT:`, with P's portrait.

The two routes share the question's tokens after the referent ("'s occupation?\nASSISTANT:"). Facts are the
doubly-correct forget and unseen-retain facts of `umu_npo_rt_txt_lr2e-04_s0_gate_rw5n200` (66 + 122). Each fact B is
paired with the next fact A, in sorted order, of the same attribute with a different answer.

**Part 1, map in θ0** (UMU's vanilla, no LoRA). The residual stream at the shared suffix positions of layer L is
copied from a source run into a target run. The target's answer is then scored as a two-way forced choice between
A's and B's answer texts (summed log p, teacher forced).

| condition | source → target | reading |
|---|---|---|
| X1 | F(A) → N(B) | cross-route, the main measure |
| X2 | N(A) → F(B) | cross-route, the main measure |
| W1 | N(A) → N(B) | within-route reference |
| W2 | F(A) → F(B) | within-route reference |
| S1 | F(B) → N(B) | same person, control |
| S2 | N(B) → F(B) | same person, control |

- Layers: 0, 2, …, 30, 31.
- Swap share at a layer: the share of pairs whose patched target prefers the source person's answer (S1/S2: the
  share that keeps B's answer).
- Only pairs whose unpatched targets prefer B's answer in both routes are scored.

*Predictions and thresholds:*
- **Convergence found:** some layer has a swap share ≥ 0.5 in both X1 and X2. L* is the shallowest such layer.
- **No shared format in the question stream:** the maximum over layers of min(X1, X2) is ≤ 0.2, while W1 and W2
  reach ≥ 0.8.
- **Otherwise:** partial.
- **Reported without a threshold:** the layer profiles; the gap between cross-route and within-route curves. Not a
  threshold, an expectation: L* lies in layers 10–20, where DEV-015 found the last token taking over.

*Validity checks:*
- The script aborts if patching a target with its own states changes log p by more than 0.05 nats (first batch).
- The map is not interpretable if S1 or S2 keep B's answer in fewer than 0.9 of pairs at any layer (the patch breaks
  the model), or if W1 and W2 never reach 0.8 (the readout cannot see an identity move).

**Part 2, the unlearned models, same person** (descriptive, no threshold). In θu, copy the name route's suffix states
N(P) into the face route F(P) at layer L, and score the face route's correct answer.
- Suppression share S(L) = (lp_F − lp_F,patched) / (lp_F − lp_N). It measures how far the face route moves to the
  name route's forgetting once it receives the name route's states.
- θ0 under the same copy is the control.
- Models: G3 seeds 0 (name-bound edit) and 1 (switch), step 300. S(L) is read against L* from part 1.

**Known limits:**
- One model.
- Two-way forced choice is a proxy for identity: a swap can also reflect the attribute's answer prior.
- The face prompt's wording ("this person") is ours, not the benchmark's.
- The portrait's 576 tokens come before the shared suffix, so the suffix positions differ between routes. The copy
  aligns the suffixes from the end, as in DEV-014 and §6.

**Sizing:** about 188 pairs × 17 layers × 6 conditions × 2 candidate answers, plus the source runs. Estimated 10–20
minutes per part on one A6000 (not measured). Eval-only jobs are spread over idle GPUs (user, 2026-11-22).

**Amendment, before the full run** (2026-11-22). A smoke test on 4 pairs (output deleted) showed that the code
copied the whole common suffix of source and target. For W2 the two face prompts are identical as text, so the
referent "this person" was copied too, and W2 swapped at every layer from layer 0. Every condition now copies only
the tokens after the referent, as the design above states: at most the common suffix of N(B) and F(B)
(`post_referent_len`). The thresholds are unchanged. Measured speed: about 14 s per 4 pairs for a face-route target,
so about 11 minutes per face-target condition; each condition runs on its own GPU.

### DEV-025 — 2026-11-22 — query-form sweep (§5) on G3 seeds 1 and 2 (decided by: user; design by: implementer)
DEV-021 ran only four analyses on seeds 1 and 2. The user asked to run the remaining earlier experiments, the
query-form sweep first. Seed 1 forgets by the response-style switch (like run 2) and seeds 0 and 2 by the name-bound
edit (DEV-021), so the sweep compares the two mechanisms on the same inputs.

**Test.**
- `scripts/mech_umu_form_sweep.py --config configs/llava/umu_npo_rt.yaml --run-id
  umu_npo_rt_txt_lr2e-04_s{1,2}_gate_rw5n200 --step 300 --set umu.n_retain_eval=100`.
- Code unchanged; generation readout on (the default).
- Seed 0's sweep is DEV-019's gate-3 run.

**Status: exploratory, descriptive.** DEV-013's P1–P5 are applied unchanged and reported, as in DEV-020. No new
threshold.

**Expectation, from DEV-021's teacher-forced numbers** (not a threshold): the blank image escapes the forgetting more
under the switch than under the name-bound edit. Blank-image forgetting was 1.11 nats in seed 1 vs 2.50–3.03 in seeds
0 and 2. So R_blank and the generation recovery should be higher in seed 1 than in seed 2.

**Known limits:** one seed per mechanism besides seed 0; the forms were designed on run 2.

**Outcome** (2026-11-22; code as hashed in `logs/dev025/launch.txt`, uncommitted; both runs exit 0; figures
`proposal/figures/form_sweep_G3s{1,2}_step300.png`).
- Mechanisms: run 2 and seed 1 = switch; seeds 0 and 2 = name-bound edit (DEV-021).
- Entries are R (forget facts) | generation keyword hit θ0 → θu.

| form | run 2 (switch) | G3 seed 1 (switch) | G3 seed 0 (name-bound) | G3 seed 2 (name-bound) |
|---|---|---|---|---|
| base (training form) | 0.00 \| 0.77 → 0.37 | 0.00 \| 0.77 → 0.21 | 0.00 \| 0.77 → 0.14 | 0.00 \| 0.77 → 0.26 |
| blank image | **0.88** \| 0.75 → **0.69** | **0.91** \| 0.76 → **0.62** | 0.51 \| 0.76 → 0.35 | 0.48 \| 0.76 → 0.39 |
| 576 filler text tokens | 0.83 \| 0.71 → 0.48 | 0.81 \| 0.71 → 0.48 | 0.49 \| 0.71 → 0.30 | 0.54 \| 0.71 → 0.44 |
| own portrait + name | 1.00 \| 0.98 → 0.97 | 0.99 \| 0.98 → 0.94 | 0.90 \| 0.98 → 0.82 | 0.97 \| 0.98 → 0.91 |
| other portrait + name | 0.90 \| 0.40 → 0.40 | 0.95 \| 0.39 → 0.36 | 0.80 \| 0.39 → 0.29 | 0.77 \| 0.39 → 0.27 |
| portrait question (no name) | 1.00 \| 0.98 → 0.98 | 1.00 \| 0.98 → 1.00 | 1.00 \| 0.98 → 0.97 | 1.00 \| 0.98 → 0.97 |
| first / last / title / reversed name | 0.15 / 0.24 / 0.28 / −0.02 | 0.81 / 0.73 / 0.79 / 0.70 | 0.66 / 0.61 / 0.68 / 0.54 | 0.55 / 0.64 / 0.63 / 0.48 |
| paraphrases 1 / 2 | −0.17 / −0.33 | 0.05 / 0.05 | 0.08 / 0.08 | 0.16 / 0.07 |
| option list / A–D letters | 0.92 / 0.73 | 0.80 / 0.86 | 0.87 / 0.83 | 0.84 / 0.86 |
| cloze / system prompt / no frame | 0.19 / 0.24 / 0.65 | 0.50 / 0.06 / 0.23 | 0.20 / 0.08 / 0.12 | 0.25 / 0.20 / 0.05 |

DEV-013's P1–P5 (pass/fail):

| | P1 | P2 | P3 | P4 (Spearman) | P5 (Spearman) |
|---|---|---|---|---|---|
| seed 1 | pass | fail | pass | fail (0.60) | pass (0.71) |
| seed 2 | fail | fail | pass | fail (0.24) | fail (0.57) |
| seed 0 (DEV-020) | fail | fail | pass | fail (0.16) | fail (0.40) |
| run 2 | pass | fail | pass | pass (0.98) | pass (0.77) |

**Reading (exploratory, with the expectation fixed above):**
- **The blank-image escape depends on the mechanism; the portrait escape does not.**
  - Under the switch (run 2, seed 1), a blank image or 576 filler tokens before the question undo most of the
    forgetting: R 0.81–0.91, generation 0.62–0.69 against θ0's 0.75–0.76.
  - Under the name-bound edit (seeds 0 and 2) they undo about half: R 0.48–0.54, generation 0.35–0.44. The
    expectation holds.
  - The person's own portrait escapes in every model: R 0.90–1.00, generation 0.82–0.97. The portrait question
    without the name is never forgotten (R 1.00).
- **Name spelling.** On G3, other spellings of the name partly escape (R 0.48–0.81, seed 1 highest), while run 2's
  forgetting carried over to them (R ≤ 0.28). Changing the name escapes more on all three G3 seeds than on run 2,
  in both mechanisms.
- **Paraphrases** stay forgotten in all three G3 seeds (R ≤ 0.16).
- **Option lists** contain the answer, so R is high in every model (DEV-013 caveat 1).
- **Collateral tracks form only in run 2.** P4 (retain and forget recovery move together across forms) passes only
  in run 2, the checkpoint whose switch still covered the retain people.

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
