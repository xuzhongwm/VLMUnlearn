# VLM cross-modal unlearning — Stage 1 (reproduce the asymmetry)

**Question.** After a fact is unlearned through text-only questions, does it remain answerable
through image+text questions about the same fact?

**Governing protocol.** `PROTOCOL.md` plus its Addendum. Every deviation and every provisional
interpretation is in [DEVIATIONS.md](DEVIATIONS.md).
The current state is in [STATUS_README.md](STATUS_README.md), and the E0 data facts are in
[analysis/data_audit.md](analysis/data_audit.md).

**Design in brief:**
- 50 forget entities, all unlearned. They are split 20 for calibration and 30 for confirmation.
- Retain entities: 300 for unlearning-retain, 50 for calibration, 100 for confirmation.
- Fact-paired, paraphrased MC items, with correct answer positions balanced.
- Our own vanilla, fine-tuned on both forms, then gated at ≥ 0.80. The doubly-correct fact set is frozen before any unlearning.
- A fresh LoRA adapter on the frozen vanilla (LLM only); GA_Diff is the primary method, NPO the second.
- The selected checkpoint is the earliest calibration checkpoint with in-modality accuracy in [0.30, 0.40] and retain drop ≤ 0.10.
- Primary metric: G = Acc_I^U − Acc_T^U on the doubly-correct confirmation facts. GO requires, in both seeds, G ≥ 0.10 and an entity-bootstrap 95% CI lower bound > 0.

## How to run (Google Colab, A100 40 GB preferred)

Open [colab/run.ipynb](colab/run.ipynb) and set `REPO_URL` and `MODEL` in cell 1. Run the cells in
this order. **Cells 4–9 are real GPU runs; start them only after approval.**

| # | cell | what it does | A100 runtime `qwen2vl` (pipeline check) | A100 runtime `llava` (primary) |
|---|---|---|---|---|
| 1 | Setup (3 cells) | mount Drive, clone/pull repo, pip install | ~2 min | ~2 min |
| 2 | Data | download MLLMU-Bench + MMBench; build frozen splits and items; data checks (optionally regenerate the data audit) | ~10 min, once for all models | — |
| 3 | Smoke test | every job below with `--debug` (~20 samples), including Milestone J | ~10 min | ~15 min (first 14 GB weight download) |
| 4 | Pretrained baseline | base model, no FT, forget paired items only (1,338 items, original option order; priors) | ~3 min | ~6 min |
| 5 | Vanilla FT + gate | LoRA FT on both forms, 3 epochs; full eval; gate; freeze the doubly-correct set | ~0.8 h | ~2.5 h |
| 6 | Milestone J | hook checks (i)–(iii) + tool tests J1–J6 with a trained adapter on the real model, in float32 (also as `pytest tests/`) | not yet measured | not yet measured |
| 6b | Pilot + checkpoint B | one cell (NPO + retain NLL, txt, lr 5e-5, seed 0), up to 6 epochs, LoRA + scores at every calibration checkpoint | not yet measured | not yet measured |
| 6c | Logit-lens curves | `layer_logit_diff` on the first checkpoint that passed B; the H2 vs H1/H1′ decision is made by a person | not yet measured | not yet measured |
| 7 | Unlearning grid | 24 runs (GA_Diff, NPO × txt/img × 3 lr × 2 seeds); each stops at its earliest eligible checkpoint | ~2–3 h | ~6–9 h |
| 8 | Selection + full evals | calibration-only selection; full evals of ≤ 8 selected checkpoints (4 option shifts each) | ~0.5 h | ~1.5 h |
| 9 | Aggregate + sanity | transfer table, G table, figure, verdict, sanity report | 1 min | 1 min |
| | **total** | | **≈ 4–5 h** | **≈ 11–14 h** |

**Notes on order and runtime:**
- Order (research.md, Section 6): 4 → 5 (checkpoint A) → 6 → 6b (checkpoint B) → 6c → decision → 7–9.
- Full evals (cells 5 and 8) now also score open-ended answers (greedy generation of up to 32 tokens on every paired fact, both forms). This adds time that the estimates above do not include yet.
- `qwen25vl` (replication phase only) costs about the same as `llava`, or a little less, because it uses fewer visual tokens.
- The grid time depends on how quickly each run reaches the accuracy window. The lowest lr may run the full 6 epochs, which is the upper end of each range.
- The smoke-test log prints the measured `s/step` per job, so the estimates can be recalibrated.
- L4 (24 GB) works too and is about 2.5–3× slower. If LLaVA training runs out of memory there, add `--set train.batch_size=2`.

**After a disconnect:** re-run cells 1a–1c, then the cell you were on.
- Training resumes from the last checkpoint.
- Evaluation skips items that are already scored.
- Finished runs are skipped.
- Kill-and-resume was verified locally: the result is identical to an uninterrupted run.

## Outputs (`/content/drive/MyDrive/vlm_unlearn/`)

```
data/processed/                splits.json (frozen), items_*.jsonl, train/unlearn sets, images, build_report.json, sanity_data.json
models/<model>_vanilla/        vanilla LoRA adapter (merged into the public base at load time)
runs/<model>/pretrained/       prior_eval/: base model on forget paired items, original option order (priors check)
runs/<model>/vanilla/          FT log, gate.json, doubly_correct.json (frozen; coverage per split/entity/attribute), full_eval/
runs/<model>/milestone_j/      hook-check report
runs/<model>/unlearn/<run>/    config.json, log.txt, checkpoints.jsonl (calibration), selected.json + selected_adapter/, full_eval/
results/run_log.csv            every job start / done / failed / interrupted
results/<model>/               grid.csv, selection.json (every run's status), selected_runs.json, transfer_table.csv,
                               g_table.csv, verdict.json, sanity.json, milestone_j.json, transfer_<model>.png
results/transfer_table.csv     all models combined (mirrored to this repo's results/)
debug/, debug/archive/<time>/  --debug runs; previous debug outputs are archived, never deleted
```

## Repo layout

```
vlmu/data/       download, field extraction, "this person" -> name rewrite, paired items (balanced positions), splits, data checks
vlmu/models.py   Qwen2-VL / Qwen2.5-VL / LLaVA adapters: prompts, answer-only labels, A-D letter scoring, LoRA targets
vlmu/train.py    resumable LoRA loop; GA_Diff, NPO (GA, NLL kept)
vlmu/evaluate.py letter-logit scoring with margin, option-permutation shifts (resumable)
vlmu/metrics.py  G and paired entity bootstrap
vlmu/runs.py     vanilla + gate + doubly-correct set, calibration checkpoint rule, selection, full evals
vlmu/report.py   transfer/G tables, GO rule, figure
vlmu/mech/       hooks (contribution removal / replacement, layer patching) and the Milestone J checks
configs/         data.yaml, models/*.yaml (with role), <model>/{vanilla,grid}.yaml, <model>/runs/*.yaml (one per run)
scripts/         job entry points (each supports --debug, --root, --set key=value), data_audit.py, smoke_test.py
colab/run.ipynb
analysis/        data_audit.md, official_mc_attributes.csv
```

## Local testing (no GPU)

Run the whole pipeline on CPU with tiny random-weight models:

```bash
python scripts/smoke_test.py --model qwen2vl --root /tmp/vlmu --set model.id=hf-internal-testing/tiny-random-Qwen2VLForConditionalGeneration
python scripts/smoke_test.py --model llava   --root /tmp/vlmu --set model.id=trl-internal-testing/tiny-LlavaForConditionalGeneration
```

## Known risks

- **Pinned versions:** `transformers==4.51.3`, `peft==0.15.2`; torch comes from the Colab image (tested locally with torch 2.6.0).
- **PROTOCOL.md:** not yet in the repo. Selection, the G metric and the GO rule follow the 2026-10-29 instructions and are listed as provisional in DEVIATIONS.md.
