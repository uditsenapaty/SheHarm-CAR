"""Prompted vision-language baselines: LLaVA, InternVL, Llama-3.2-Vision, Qwen2.5-VL.

Paper Section `subsec:baselines`: "prompted models follow a fixed output schema and identical
label definitions." One prompt, one schema, one label inventory for all four.

Two passes per instance:
  * generation - produces the target, harmfulness, category and rationale as JSON;
  * label scoring - reads the log-probability of each harmfulness label continuation and
    softmaxes over the three, giving a calibrated distribution. Prompted models otherwise
    expose no confidence, and CF-Faith needs one.

All models are loaded in 4-bit so a 7-11B model fits a 16 GB card, and the tokenizer pads on
the left - right padding silently corrupts batched decoder-only generation.
"""

from __future__ import annotations

import json
import re

import torch
import torch.nn.functional as F

from ..labels import CATEGORY_LABELS, HARMFULNESS_LABELS, NULL_CATEGORY

MODEL_IDS = {
    "llava": "llava-hf/llava-1.5-7b-hf",
    "internvl": "OpenGVLab/InternVL3-8B-hf",
    "llama32_vision": "meta-llama/Llama-3.2-11B-Vision-Instruct",
    "qwen25_vl": "Qwen/Qwen2.5-VL-7B-Instruct",
}

SYSTEM_PROMPT = f"""You classify women-targeted harm in memes. Return exactly one JSON object, nothing else.

REQUIRED KEYS: women_related_target, harm_type, harm_category, rationale

LABEL DEFINITIONS
harm_type is exactly one of: {', '.join(HARMFULNESS_LABELS)}.
  Explicit-Harm  direct abuse, threat, violence, or sexual aggression against a woman.
  Implicit-Harm  stereotyping, degradation, or mockery of women conveyed indirectly,
                 through implication, sarcasm, cultural reference, or image-text association.
  Non-Harm       does not target women for abuse; includes quotation, condemnation,
                 counter-speech, awareness, and content not directed at women.
harm_category is exactly one of: {', '.join(CATEGORY_LABELS)}, or {NULL_CATEGORY} when harm_type is Non-Harm.
women_related_target is the short concrete woman, female role, relationship, profession,
appearance, behaviour, or women-associated group the meme is directed at.
rationale is one line of evidence-grounded justification.
"""

USER_TEMPLATE = 'OCR text of the meme: "{ocr_text}"\nClassify this meme now and return only the JSON object.'


def build_messages(ocr_text: str) -> list[dict]:
    return [
        {"role": "system", "content": [{"type": "text", "text": SYSTEM_PROMPT}]},
        {"role": "user", "content": [
            {"type": "image"},
            {"type": "text", "text": USER_TEMPLATE.format(ocr_text=ocr_text[:600])},
        ]},
    ]


def parse_prediction(text: str) -> dict:
    cleaned = re.sub(r"^```(?:json)?\s*|\s*```$", "", text.strip(), flags=re.IGNORECASE).strip()
    start, end = cleaned.find("{"), cleaned.rfind("}")
    if start < 0 or end < start:
        return {}
    try:
        value = json.loads(cleaned[start : end + 1])
    except json.JSONDecodeError:
        return {}
    return value if isinstance(value, dict) else {}


def coerce_labels(prediction: dict) -> dict:
    """Map a free-form answer onto the label inventory; unparseable answers fall back to Non-Harm."""
    harm = str(prediction.get("harm_type", "")).strip()
    harm = next((label for label in HARMFULNESS_LABELS if label.lower() == harm.lower()), None)
    if harm is None:
        harm = "Non-Harm"
    category = str(prediction.get("harm_category", "")).strip()
    category = next((label for label in CATEGORY_LABELS if label.lower() == category.lower()), NULL_CATEGORY)
    if harm == "Non-Harm":
        category = NULL_CATEGORY
    elif category == NULL_CATEGORY:
        category = "Misogyny"  # harmful but uncategorised: assign the most frequent category
    return {
        "women_related_target": str(prediction.get("women_related_target", "")).strip().lower(),
        "harm_type": harm,
        "harm_category": category,
        "rationale": str(prediction.get("rationale", "")).strip(),
    }


class PromptedVLM:
    def __init__(self, key: str, model_id: str | None = None, device: str = "cuda",
                 max_new_tokens: int = 160, load_in_4bit: bool = True, max_pixels: int = 768 * 28 * 28):
        from transformers import AutoModelForImageTextToText, AutoProcessor, BitsAndBytesConfig

        self.key = key
        self.model_id = model_id or MODEL_IDS[key]
        self.max_new_tokens = max_new_tokens
        quantization = BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True, bnb_4bit_compute_dtype=torch.float16,
        ) if load_in_4bit else None

        processor_kwargs = {}
        if "qwen" in self.model_id.lower():
            processor_kwargs = {"min_pixels": 256 * 28 * 28, "max_pixels": max_pixels}
        self.processor = AutoProcessor.from_pretrained(self.model_id, **processor_kwargs)
        if getattr(self.processor, "tokenizer", None) is not None:
            self.processor.tokenizer.padding_side = "left"
        self.model = AutoModelForImageTextToText.from_pretrained(
            self.model_id, quantization_config=quantization,
            torch_dtype=torch.float16, device_map={"": 0} if device == "cuda" else None,
        ).eval()
        self.device = next(self.model.parameters()).device

    def _prepare(self, images, texts):
        prompts = [
            self.processor.apply_chat_template(build_messages(text), tokenize=False, add_generation_prompt=True)
            for text in texts
        ]
        inputs = self.processor(text=prompts, images=list(images), padding=True, return_tensors="pt")
        return {key: value.to(self.device) for key, value in inputs.items()}

    @torch.no_grad()
    def generate(self, images, texts) -> list[dict]:
        inputs = self._prepare(images, texts)
        generated = self.model.generate(**inputs, max_new_tokens=self.max_new_tokens, do_sample=False, use_cache=True)
        answers = self.processor.batch_decode(
            generated[:, inputs["input_ids"].shape[1]:], skip_special_tokens=True
        )
        return [coerce_labels(parse_prediction(answer)) for answer in answers]

    @torch.no_grad()
    def harm_distribution(self, images, texts) -> torch.Tensor:
        """Softmax over the log-probabilities of the three harmfulness label continuations."""
        inputs = self._prepare(images, texts)
        prefix_length = inputs["input_ids"].shape[1]
        scores = []
        for label in HARMFULNESS_LABELS:
            suffix = self.processor.tokenizer(
                f'{{"harm_type": "{label}"', add_special_tokens=False, return_tensors="pt"
            )["input_ids"].to(self.device)
            suffix = suffix.expand(inputs["input_ids"].size(0), -1)
            extended = dict(inputs)
            extended["input_ids"] = torch.cat([inputs["input_ids"], suffix], dim=1)
            if "attention_mask" in extended:
                extended["attention_mask"] = torch.cat(
                    [inputs["attention_mask"], torch.ones_like(suffix)], dim=1
                )
            logits = self.model(**extended).logits[:, prefix_length - 1 : -1]
            log_probabilities = F.log_softmax(logits.float(), dim=-1)
            token_scores = log_probabilities.gather(2, suffix.unsqueeze(-1)).squeeze(-1)
            scores.append(token_scores.mean(dim=1))   # length-normalised
        return F.softmax(torch.stack(scores, dim=1), dim=-1)

    def release(self) -> None:
        del self.model
        torch.cuda.empty_cache()
