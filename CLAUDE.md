# VLMUnlearn: context for Claude Code

Read this first. It covers what the project is, where it stands, how the user works, and how to run the code
on the lab machine (8× RTX A6000, 48 GB each). Until 2026-11-21 all compute ran on Google Colab, so most
notes in the repo say "Colab", "Drive" or "L4". On this machine, run the same commands from a shell.

## 1. The project in brief

- **Question.** We unlearn facts about fictitious people from a vision-language model using text-only
  questions. Does the model still answer when the same fact is asked with an image? If so, what did the
  unlearning actually change inside the model?
- **Setup (the "UMU track", DEV-011/012).**
  - Benchmark: UMU-Bench, HF dataset `chengyewang/UMU-bench`.
  - Start model: UMU's released vanilla `chengyewang/llava_smu_ft`, a LLaVA-1.5-7B fine-tuned on the UMU
    profiles. The processor comes from `llava-hf/llava-1.5-7b-hf`.
  - Training: a fresh LoRA (r16 / α16, on the LM linear layers) trained with text-only NPO plus a retain NLL
    term, on the `forget_10` UM_QA (50 people × 12 questions, no image).
  - Readout: UMU's own protocol (greedy generation, exact match), on *doubly-correct* facts (facts the
    vanilla answers correctly both by name and by portrait).
- **The one checkpoint that matters:** run 2, `umu_npo_rt_txt_lr5e-05_s0`, step 150. At that step:
  - forget facts asked by name: 0.63;
  - asked by portrait: 0.97; by name + portrait: 0.97;
  - leak G = 0.34;
  - retain facts by name: 0.69.

  **Every mechanism claim is about this checkpoint only.**
  - Run 1 (`umu_npo_txt_lr5e-05_s0`) is degenerate. The user ruled it out on 2026-11-19: do not analyse it
    and do not use it as a replication.
  - Run 3 (`…_rw3n200`) barely forgot anything.
  - Run 4 (`umu_npo_rt_txt_lr1e-04_s0_rw3n200`) *was* run on 2026-11-05 but went unrecorded until DEV-019
    (2026-11-22). At step 150: forget by name 0.72, retain by name 0.885. Its LoRA is still only on Drive.
- **Scope (changed 2026-11-22, DEV-019).** The user decided to scale up: more seeds, methods and models.
  Scaling waits on three **gate experiments** (DEV-019). If any gate fails, stop and go back to the user.
  - Gate 1: a run that keeps the retain facts.
  - Gate 2: a style-controlled readout (C1, the knowledge is still there).
  - Gate 3: the core features on that run (C2, C3).

  The course report (about 2026-12-10) and the paper venue were not settled when this was decided; ask if they
  matter. The cross-modal side is still the subject of the project.
- **Who writes the report.** The user writes it in Overleaf. Claude supplies LaTeX subsections and figures
  in `proposal/figures/`, and the user pastes them in. `proposal/overview.tex` is the fullest write-up of the
  results so far (as of 2026-11-19). `DEVIATIONS.md` holds every number.

## 2. Results so far (run 2, step 150)

| notebook § | experiment | result | record |
|---|---|---|---|
| 2 | vanilla check | reproduces the UMU paper (forget AccF 0.823 vs 0.8233) | notebook |
| 4b | logit lens | flat on portrait questions → **H2 rejected** (the portrait route is not suppressed and later repaired) | DEV-012, DEV-013 |
| 4c | layer masking | the edit acts in layers 0–19; late layers neither necessary nor sufficient → **H1 rejected** | DEV-013 |
| 4e | position masking | the edit is spread over the whole text prompt, not bound to the name tokens | DEV-013 |
| 5 | query-form sweep | forgetting depends on the *form* of the input; a blank image alone restores most answers (R 0.88) | DEV-013 |
| 6, 7 | blank-image route patching, input cues | text prefixes match a blank image on log p but not on generation; length / displacement does not explain it | DEV-015 |
| 8 | LoRA-contribution transplant | mixed: the LoRA writes nearly the same with and without the image; suppression needs a text-only context | DEV-014 |
| 9 | state patching | the suppression starts on the question tokens in L0–2 and builds up in the **last prompt token** (restore 0.95 at L16; sufficient from L14–16); it is not carried by the name tokens | DEV-015 |
| 10 | signal transfer | partly shared: subtracting the mean last-token change of *other* people at L16 restores 0.91 (top-1 0.12 → 0.82); adding it to the vanilla is weaker | DEV-016 |
| 11 | steering during generation | both pre-registered tests PASS. The vector switches the **response style**: a short answer ("Graphic Designer") vs a biography sentence opening with the person's name. It works in both directions, and on blank-image prompts too | DEV-017 |
| 12 | cross-modal mechanism: why the vector does not form behind an image (dilution hypothesis, attention knockout) | **designed and pushed, not run yet** | DEV-018 |

**Current reading.**
- Text-only unlearning mostly teaches a switch of response mode for one input form: text-only questions about
  a person. The knowledge itself stays in the model.
- The unlearned model rarely goes silent. On the training form it mostly writes long sentences about the
  person (77% biography-style, 12% refusals). Exact-match "forgetting" therefore overstates how much knowledge
  was removed.

**Gate outcome (2026-11-22, DEV-019):**
- Gate 1 passed at the threshold. G3 step 300 (`umu_npo_rt_txt_lr2e-04_s0_gate_rw5n200`): forget 0.58, unseen
  retain 0.902 (110/122).
- Gate 2 was partial and gate 3 failed. On G3, θu answers with wrong *short* answers, not biography sentences.
  - Forced choice: 0.49, against a 0.53 prior.
  - The suppression is carried by the name tokens (L4–12), with no shared direction.
- Run 2's "response-mode switch" (C1–C3) is therefore checkpoint-specific. It is probably the form of unlearning
  that a weak retain term allows (exploratory).
- **The scale-up is stopped**, pending the user's decision on what to do next.
- DEV-020 reran every earlier mechanism experiment on G3 (figures `proposal/figures/*_G3_step300.png`).
  - G3's forgetting is written on the **name tokens in layers 0–3**: LoRA off at the name tokens restores 0.98.
  - The portrait question leaks because it contains no name (0.01 nats). With name + portrait, the face rescues the
    answer: block attention to the image and it forgets like a blank image does.

## 3. How the user works (follow these)

- **Language.** Reply in **Chinese**, in plain, simple wording. Everything in the repo stays in **English**:
  code, comments, `DEVIATIONS.md`, notebook markdown, commit messages.
- **Pre-register before running.** For every new experiment:
  1. Add a `DEV-0NN` entry to `DEVIATIONS.md` before any result exists. Include the test, the predictions with
     numeric thresholds, a validity check that makes the script abort if it fails, and known limits. Follow
     the format of DEV-014 to DEV-018.
  2. After the run, append `**Outcome, run 2 step 150** (date, code at <commit>, output paths)`. Add a results
     table and a separate "Reading" paragraph, and mark anything formed after seeing the data as exploratory.
  3. Never rewrite old entries. Corrections go in a new dated paragraph or a new entry.
- **Read the actual generations** (`items*.jsonl`) before interpreting a log-p effect. Log p and generated
  behaviour dissociate: about 3 nats can flip as many answers as 14.5. Always report a behaviour-level readout
  (top-1 or generation) next to log p.
- **Size runs.**
  - Exploratory runs should finish in about 10 minutes by default.
  - Before launching, state the number of facts and the estimated time.
  - Bigger sweeps run only when the user asks.
- **Shared machine.**
  - Check `nvidia-smi` and use only free GPUs.
  - Ask before occupying more than a couple of GPUs or running anything for hours.
  - One script uses one GPU.
- **Experiment layout.** One script per experiment, `scripts/mech_umu_<name>.py`. It writes to
  `$VLMU_ROOT/results/llava/mech_umu/<run_id>/step<N>/<name>/`: `summary.json`, per-item jsonl, a log and a
  figure. Copy the figure to `proposal/figures/<name>_run2_step150.png`. Add a numbered section to
  `colab/run.ipynb` with the command and a "how to read" note, as the existing sections do. The notebook is
  the run book and the index of experiments.
- **Git.**
  - Work on `main`. `origin` is `github.com/xuzhongwm/VLMUnlearn` (public).
  - The old repo `VLMUnlearn-private` is retired: never push to it.
  - Commit messages are short, in English and in the imperative ("Add …", "Record … results").
  - Pull before starting, because Colab may still be used.
  - Pushing needs the user's GitHub credentials on this machine. If a push fails, ask; don't work around it.

## 4. Running on the lab machine

**Environment.** These pins are tested; keep them.

```bash
python -m venv .venv && source .venv/bin/activate          # Python 3.10–3.12
pip install torch==2.6.0 --index-url https://download.pytorch.org/whl/cu124   # pick the CUDA wheel the driver supports
pip install -r requirements.txt                            # transformers 4.51.3, peft 0.15.2, accelerate 1.6.0
python -c "import torch, transformers, peft; print(torch.__version__, transformers.__version__, peft.__version__, torch.cuda.is_available())"
```

**Environment variables.** Set these in every shell, for example in `.venv/bin/activate` or `~/.bashrc`.
Put both directories on a disk with ≥ 50 GB free.

```bash
export VLMU_ROOT=/path/to/vlm_unlearn     # replaces /content/drive/MyDrive/vlm_unlearn (the default in vlmu/config.py)
export HF_HOME=/path/to/hf_home           # model weights: llava_smu_ft (~14 GB) + the llava-1.5 processor
export TOKENIZERS_PARALLELISM=false
# leave VLMU_LOCAL_CACHE unset: it only copied Drive folders to local disk on Colab
```

**Data from Google Drive.** The user has to copy these from `MyDrive/vlm_unlearn/` into `$VLMU_ROOT`, keeping
the same relative paths:
- `runs/llava/umu_unlearn/umu_npo_rt_txt_lr5e-05_s0/` (**required**): the LoRA at `ckpt/step150/`,
  `baseline_dc.json` (the frozen doubly-correct facts), `evals/`, `checkpoints.jsonl`, `config.json`.
- `results/llava/mech_umu/umu_npo_rt_txt_lr5e-05_s0/step150/`: all earlier mechanism outputs, needed for
  comparisons and the report.

Nothing else is needed: the run 1 folders and the MLLMU-track folders can stay behind. The UMU parquet files
and the model weights download from the HF Hub on first use. They are public, so no token is needed.
**If the run 2 checkpoint is missing, stop and ask the user. Do not retrain run 2 as a substitute:** a
retrained run is a different checkpoint, and every recorded number refers to the original.

**Running a notebook section.**
- Each cell in `colab/run.ipynb` is a thin wrapper around a `python scripts/...` command. Run the command from
  the repo root, for example:

  ```bash
  CUDA_VISIBLE_DEVICES=0 python scripts/mech_umu_crossmodal.py \
      --config configs/llava/umu_npo_rt.yaml --run-id umu_npo_rt_txt_lr5e-05_s0 --step 150
  ```

- The config only supplies the UMU settings; `--run-id` picks the run. Every script takes `--set key=value`
  overrides and `--debug`.
- Notebook paths starting with `/content/drive/MyDrive/vlm_unlearn` mean `$VLMU_ROOT`.
- For long jobs use `tmux` or `nohup … > log 2>&1 &`. Never run two jobs that write to the same output folder
  at the same time.
- Most scripts resume from what is already in their output folder. Read the script before assuming a re-run
  recomputes everything.

**GPU notes.**
- Notebook runtimes are for a 24 GB L4. Measure on the A6000 rather than assuming a speed-up.
- Batch sizes are L4-safe. Some scripts take flags such as `--batch-txt` and `--batch-img`; raise them only if
  needed and say so.
- bf16 is the default on CUDA. The A6000 has enough memory for the float32 Milestone J check: drop
  `--set mech.dtype=bfloat16`.

**First checks on a new machine.** Each takes about 5 minutes. Compare the results with the recorded numbers
before doing anything new.
1. `python scripts/eval_external.py --config configs/llava/vanilla.yaml --umu-protocol`. Expected: forget name
   0.75, portrait 0.81, AccF 0.823. This needs no Drive data.
2. `python scripts/check_hooks_umu.py --config configs/llava/umu_npo_rt.yaml --run-id umu_npo_rt_txt_lr5e-05_s0 --step 150`
   (float32). Every check must PASS, and `txt/J1_token_groups` must show `n_name > 0`. This confirms the hook
   tools are exact with this software stack.

## 5. Repo map (current track)

```
CLAUDE.md                  this file
DEVIATIONS.md              decisions, pre-registrations and outcomes; DEV-011 onwards is the UMU track
colab/run.ipynb            UMU-track run book: §1 setup, §2 vanilla check, §3 unlearning runs, §4–§12 mechanism experiments
proposal/overview.tex      project overview and results to date (LaTeX); proposal.tex = the 2026-11-05 proposal
proposal/figures/          report figures (<experiment>_run2_step150.png)
configs/llava/umu_npo.yaml, umu_npo_rt.yaml, vanilla.yaml   UMU-track configs (run 2 = umu_npo_rt + train.lr=5e-5)
vlmu/umu.py                UMU data loading, unlearning rows, UMU-protocol readout
vlmu/mech/                 hooks (LoRA contribution masking / replacement, state patching), Milestone J checks, tools
vlmu/models.py, train.py, config.py, utils.py   model adapters, NPO/GA training loop, config + paths, I/O helpers
scripts/unlearn_umu.py     UMU unlearning runs (resumable; LoRA saved per checkpoint)
scripts/mech_umu_*.py      one script per mechanism experiment (notebook §4–§12)
scripts/check_hooks_umu.py, eval_external.py            Milestone J on a UMU checkpoint; vanilla readout
```

**Outdated, do not take as current:**
- `README.md` and `STATUS_README.md` describe the earlier MLLMU-Bench track (2026-10-29 to 11-05).
- `colab/archive_mllmu_track.ipynb`, `configs/{llava,qwen2vl,qwen25vl}/runs|grid|pilot*`, and
  `vlmu/runs.py` / `report.py` / `data/` belong to that track.
- `research.md` (Chinese notes) is superseded by `DEVIATIONS.md` and `proposal/overview.tex` from DEV-013 on.
- `PROTOCOL.md` is referenced but was never added.

## 6. Loose ends

- Notebook §9b still contains a run 1 command (`--run-id umu_npo_txt_lr5e-05_s0 --step 75`), but DEV-015 says
  §9b runs run 2 only. Skip the run 1 line.
- `proposal/overview.tex`'s Summary and "Not done" sections predate DEV-015 to DEV-017. Its later sections, and
  `DEVIATIONS.md`, are current.
- The references marked `VERIFY` in `overview.tex` have not been checked against the papers.
