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
