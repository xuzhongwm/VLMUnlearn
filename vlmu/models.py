"""Model adapters: prompt format, batch encoding with answer-only labels, letter scoring,
LoRA targets. One class per model family; everything else is model-agnostic.

The vanilla model is stored as a LoRA adapter (ROOT/models/<model>_vanilla) and merged
into the public base checkpoint at load time.
"""
import inspect
import re
from pathlib import Path

import torch
from PIL import Image

LETTERS = "ABCD"
MC_INSTRUCTION = "Answer with the option's letter from the given choices directly."


def format_mc(question: str, options: dict) -> str:
    opts = "\n".join(f"{k}. {v}" for k, v in options.items())
    return f"{question}\n{opts}\n{MC_INSTRUCTION}"


class Adapter:
    family = None
    image_token = None

    def __init__(self, mcfg: dict, device=None):
        self.cfg = mcfg
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.dtype = torch.bfloat16 if self.device.type == "cuda" else torch.float32
        self.processor = self._load_processor()
        self.tok = self.processor.tokenizer
        self.letter_ids = {L: self._letter_id(L) for L in LETTERS}
        self._logits_kw = None

    # ---- model-specific -------------------------------------------------------------
    def _load_processor(self):
        raise NotImplementedError

    def _model_cls(self):
        raise NotImplementedError

    def prompt(self, question: str, has_image: bool) -> str:
        raise NotImplementedError

    def answer_suffix(self, answer: str) -> str:
        raise NotImplementedError

    letter_ctx, letter_prefix = None, None

    # ---- loading ------------------------------------------------------------------------
    def load_model(self, vanilla_dir=None):
        """Base checkpoint, optionally with the vanilla LoRA merged in."""
        model = self._model_cls().from_pretrained(self.cfg["id"], torch_dtype=self.dtype,
                                                  attn_implementation=self.cfg.get("attn", "sdpa"))
        if vanilla_dir:
            from peft import PeftModel
            model = PeftModel.from_pretrained(model, vanilla_dir).merge_and_unload()
        model.to(self.device)
        model.config.use_cache = False
        sig = inspect.signature(model.forward).parameters
        self._logits_kw = next((k for k in ("logits_to_keep", "num_logits_to_keep") if k in sig), None)
        return model

    def lora_targets(self, model) -> list:
        """All linear layers of the language model (vision tower + projector stay frozen)."""
        pat = re.compile(r"\.layers\.\d+\.(self_attn\.(q|k|v|o)_proj|mlp\.(gate|up|down)_proj)$")
        names = [n for n, m in model.named_modules()
                 if isinstance(m, torch.nn.Linear) and pat.search(n) and not re.search(r"visual|vision", n)]
        assert names, "no LoRA target modules found"
        return names

    def add_lora(self, model, lcfg: dict, seed: int):
        from peft import LoraConfig, get_peft_model
        torch.manual_seed(seed)
        if lcfg.get("gradient_checkpointing", True):
            model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
            model.enable_input_require_grads()
        conf = LoraConfig(r=lcfg["r"], lora_alpha=lcfg["alpha"], lora_dropout=lcfg["dropout"],
                          target_modules=self.lora_targets(model), bias="none")
        return get_peft_model(model, conf)

    # ---- encoding -------------------------------------------------------------------
    def _images(self, rows, image_root):
        imgs = [Image.open(Path(image_root) / r["image"]).convert("RGB") for r in rows if r.get("image")]
        assert len(imgs) in (0, len(rows)), "batches must be homogeneous in modality"
        return imgs or None

    def _process(self, texts, images, padding_side):
        self.tok.padding_side = padding_side
        kw = dict(text=texts, padding=True, return_tensors="pt")
        if images:
            kw["images"] = images
        return dict(self.processor(**kw))

    def to_device(self, batch):
        out = {}
        for k, v in batch.items():
            if torch.is_tensor(v):
                v = v.to(self.device)
                if v.is_floating_point():
                    v = v.to(self.dtype)
            out[k] = v
        return out

    def encode_train(self, rows, image_root):
        """Right-padded batch; labels cover only the answer (+ end-of-turn) tokens."""
        images = self._images(rows, image_root)
        prompts = [self.prompt(r["question"], bool(images)) for r in rows]
        full = [p + self.answer_suffix(r["answer"]) for p, r in zip(prompts, rows)]
        b = self._process(full, images, "right")
        pb = self._process(prompts, images, "right")
        labels = b["input_ids"].clone()
        labels[b["attention_mask"] == 0] = -100
        plens = pb["attention_mask"].sum(1)
        for i, n in enumerate(plens.tolist()):
            if not torch.equal(b["input_ids"][i, :n], pb["input_ids"][i, :n]):
                raise ValueError(f"prompt tokens are not a prefix of prompt+answer for row {rows[i].get('uid')}")
            labels[i, :n] = -100
        b["labels"] = labels
        return self.to_device(b)

    def encode_mc(self, rows, image_root):
        images = self._images(rows, image_root)
        texts = [self.prompt(format_mc(r["question"], r["options"]), bool(images)) for r in rows]
        return self.to_device(self._process(texts, images, "left"))

    def _letter_id(self, L):
        ctx = self.tok(self.letter_ctx, add_special_tokens=False)["input_ids"]
        full = self.tok(self.letter_ctx + self.letter_prefix + L, add_special_tokens=False)["input_ids"]
        assert full[:len(ctx)] == ctx and len(full) == len(ctx) + 1, f"letter {L} is not a single token"
        return full[-1]

    # ---- forward helpers ----------------------------------------------------------------
    def forward(self, model, batch, last_only=False):
        kw = {k: v for k, v in batch.items() if k != "labels"}
        kw["use_cache"] = False
        if last_only and self._logits_kw:
            kw[self._logits_kw] = 1
        return model(**kw).logits

    def answer_logprobs(self, model, batch):
        """Per-token log p of answer tokens and per-sample sums."""
        logits = self.forward(model, batch)
        labels = batch["labels"]
        sl, lab = logits[:, :-1], labels[:, 1:]
        mask = lab != -100
        tok_lp = sl[mask].float().log_softmax(-1).gather(-1, lab[mask][:, None]).squeeze(-1)
        rows = mask.nonzero()[:, 0]
        per_sample = torch.zeros(labels.shape[0], device=tok_lp.device, dtype=tok_lp.dtype).index_add(0, rows, tok_lp)
        return tok_lp, per_sample, mask.sum(1)

    @torch.no_grad()
    def letter_scores(self, model, batch):
        logits = self.forward(model, batch, last_only=True)[:, -1].float()
        return logits.log_softmax(-1)


class Qwen2VLAdapter(Adapter):
    family = "qwen2vl"
    SYSTEM = "<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n"
    letter_ctx, letter_prefix = "<|im_start|>assistant\n", ""

    def _load_processor(self):
        from transformers import AutoProcessor
        return AutoProcessor.from_pretrained(self.cfg.get("processor_id", self.cfg["id"]),
                                             min_pixels=self.cfg.get("min_pixels", 224 * 224),
                                             max_pixels=self.cfg.get("max_pixels", 448 * 448))

    def _model_cls(self):
        from transformers import Qwen2VLForConditionalGeneration
        return Qwen2VLForConditionalGeneration

    def prompt(self, question, has_image):
        vis = "<|vision_start|><|image_pad|><|vision_end|>" if has_image else ""
        return f"{self.SYSTEM}<|im_start|>user\n{vis}{question}<|im_end|>\n<|im_start|>assistant\n"

    def answer_suffix(self, answer):
        return f"{answer}<|im_end|>"


class LlavaAdapter(Adapter):
    """LLaVA-1.5 conversation format as used by MLLMU-Bench: 'USER: <image>\\n{q}\\nASSISTANT: {a}'."""
    family = "llava"
    letter_ctx, letter_prefix = "ASSISTANT:", " "

    def _load_processor(self):
        from transformers import AutoProcessor
        return AutoProcessor.from_pretrained(self.cfg.get("processor_id", self.cfg["id"]))

    def _model_cls(self):
        from transformers import LlavaForConditionalGeneration
        return LlavaForConditionalGeneration

    def prompt(self, question, has_image):
        return f"USER: {'<image>' + chr(10) if has_image else ''}{question}\nASSISTANT:"

    def answer_suffix(self, answer):
        return f" {answer}{self.tok.eos_token}"


ADAPTERS = {"qwen2vl": Qwen2VLAdapter, "llava": LlavaAdapter}


def get_adapter(mcfg: dict, device=None) -> Adapter:
    return ADAPTERS[mcfg["family"]](mcfg, device)
