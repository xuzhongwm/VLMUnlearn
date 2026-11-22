# VLMUnlearn: why text-only unlearning in vision-language models leaks through images

**Question.** Facts about fictitious people are unlearned from a vision-language model using text-only questions
(asked by name, no image). The same facts are still answered when the question shows the person's portrait. What did
the unlearning change inside the model, and why does the portrait route escape it?

**Setup.**
- Benchmark: [UMU-Bench](https://huggingface.co/datasets/chengyewang/UMU-bench).
- Start model: UMU's released LLaVA-1.5-7B vanilla (`chengyewang/llava_smu_ft`). A second family, Qwen2-VL-7B, uses
  a vanilla we fine-tune ourselves (`scripts/finetune_umu_vanilla.py`).
- Unlearning: a LoRA trained with text-only NPO plus a retain term (`scripts/unlearn_umu.py`).
- Readout: UMU's protocol on doubly-correct facts (facts the vanilla answers both by name and by portrait).

**Where things are.**

| path | content |
|---|---|
| `DEVIATIONS.md` | every decision, pre-registration and result; DEV-011 onwards is the current (UMU) track |
| `colab/run.ipynb` | the run book: one section per experiment, with the command and how to read it |
| `scripts/mech_umu_*.py` | one script per mechanism experiment |
| `scripts/unlearn_umu.py`, `scripts/finetune_umu_vanilla.py` | unlearning runs; second-model vanilla |
| `scripts/eval_external.py`, `scripts/check_hooks_umu.py` | vanilla readout; exactness checks of the hook tools |
| `vlmu/` | data loading and readout (`umu.py`), model adapters, training loop, hooks (`mech/`) |
| `configs/llava/`, `configs/qwen2vl7b/` | UMU-track configs |
| `analysis/` | analyses over saved outputs (no model loaded) |
| `proposal/` | write-up (`overview.tex`) and report figures |

**Running.** Python 3.10–3.12, `torch==2.6.0` (CUDA wheel for your driver), then `pip install -r requirements.txt`.
Set `VLMU_ROOT` (outputs; at least 50 GB free) and `HF_HOME`. Every script takes `--config`, `--set key=value`
overrides and `--debug`. Check the setup first with `python scripts/eval_external.py --config
configs/llava/vanilla.yaml --umu-protocol`; the expected forget accuracies are 0.75 by name and 0.81 by portrait.

The earlier MLLMU-Bench track (2026-10-29 to 11-05) was removed from the working tree on 2026-11-22. It remains in
the git history.
