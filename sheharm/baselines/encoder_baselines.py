"""Trained encoder baselines: text-only RoBERTa, ViLT, Hate-CLIPper, plain CLIP.

Paper Section `subsec:baselines`: "Encoder-based methods use the same task-specific heads
where required." So each backbone below produces a pooled representation z and is then given
exactly the heads SheHarm-CAR uses — target, harmfulness, harm category, rationale decoder —
trained with the same recipe. What they lack is the ontology, the rules, the confidence gate,
and the consistency/counterfactual objectives.

Every baseline exposes the same interface as SheHarmCAR (`forward`, `generate_rationale`,
`intervene`), so `sheharm.evaluate` scores all models through one code path.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from .. import losses as loss_fn
from ..labels import CATEGORY_LABELS, HARMFULNESS_LABELS
from ..models.rationale import RationaleDecoder
from ..models.sheharm_car import SheHarmOutput


class BaselineBackbone(nn.Module):
    """Wraps a pretrained encoder so it returns (z, image_tokens, text_tokens, relevance)."""

    def __init__(self, kind: str, hidden_size: int = 768, dropout: float = 0.2):
        super().__init__()
        self.kind = kind
        self.hidden_size = hidden_size
        self.tokenizer = None      # backbones with their own vocabulary tokenize raw text
        self.max_text_len = 40
        if kind == "roberta_text":
            from transformers import RobertaModel

            self.text = RobertaModel.from_pretrained("roberta-base", add_pooling_layer=False)
            self.project = nn.Identity()
        elif kind == "vilt":
            from transformers import AutoTokenizer, ViltModel

            self.vilt = ViltModel.from_pretrained("dandelin/vilt-b32-mlm")
            self.tokenizer = AutoTokenizer.from_pretrained("dandelin/vilt-b32-mlm")
            self.max_text_len = 40   # ViLT's text branch
            self.project = nn.Linear(self.vilt.config.hidden_size, hidden_size)
        elif kind in ("hate_clipper", "clip"):
            from transformers import AutoTokenizer, CLIPModel

            self.clip = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
            self.tokenizer = AutoTokenizer.from_pretrained("openai/clip-vit-base-patch32")
            self.max_text_len = 77   # CLIP's context length
            width = self.clip.config.projection_dim
            # Hate-CLIPper fuses CLIP features by cross-modal interaction (element-wise
            # product of the aligned projections) rather than concatenation.
            fusion_width = width if kind == "hate_clipper" else width * 2
            self.project = nn.Sequential(
                nn.Linear(fusion_width, hidden_size), nn.GELU(), nn.Dropout(dropout), nn.LayerNorm(hidden_size)
            )
        else:
            raise ValueError(f"unknown backbone: {kind}")

    def retokenize(self, ocr_text, device):
        """Tokenize raw OCR text with this backbone's own vocabulary."""
        encoded = self.tokenizer(
            list(ocr_text), truncation=True, padding="max_length",
            max_length=self.max_text_len, return_tensors="pt",
        )
        return encoded["input_ids"].to(device), encoded["attention_mask"].to(device)

    def forward(self, pixel_values, input_ids, attention_mask, patch_mask=None, ocr_text=None):
        if self.tokenizer is not None and ocr_text is not None:
            input_ids, attention_mask = self.retokenize(ocr_text, pixel_values.device)
        if self.kind == "roberta_text":
            tokens = self.text(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state
            weights = attention_mask.unsqueeze(-1).to(tokens.dtype)
            z = (tokens * weights).sum(1) / weights.sum(1).clamp_min(1.0)
            return {"z": z, "patch_relevance": None, "token_relevance": tokens.norm(dim=-1)}

        if self.kind == "vilt":
            outputs = self.vilt(
                input_ids=input_ids, attention_mask=attention_mask, pixel_values=pixel_values,
                output_attentions=True,
            )
            z = self.project(outputs.last_hidden_state[:, 0])
            attention = outputs.attentions[-1].mean(dim=1).mean(dim=1)
            text_len = input_ids.size(1)
            return {"z": z, "patch_relevance": attention[:, text_len:], "token_relevance": attention[:, :text_len]}

        # Projections are applied explicitly: get_image_features/get_text_features return a
        # model output object in transformers v5, not a tensor.
        vision = self.clip.vision_model(pixel_values=pixel_values)
        patches = vision.last_hidden_state[:, 1:]
        if patch_mask is not None:
            pooled = (patches * patch_mask.unsqueeze(-1)).sum(1) / patch_mask.sum(1, keepdim=True).clamp_min(1)
        else:
            pooled = vision.last_hidden_state[:, 0]
        image_features = F.normalize(self.clip.visual_projection(pooled), dim=-1)
        text_out = self.clip.text_model(input_ids=input_ids, attention_mask=attention_mask)
        text_features = F.normalize(self.clip.text_projection(text_out.pooler_output), dim=-1)

        fused = image_features * text_features if self.kind == "hate_clipper" else torch.cat(
            [image_features, text_features], dim=-1
        )
        relevance = patches @ self.clip.visual_projection.weight.t() @ text_features.unsqueeze(-1)
        return {"z": self.project(fused), "patch_relevance": relevance.squeeze(-1), "token_relevance": None}


class EncoderBaseline(nn.Module):
    """A backbone plus the four SheHarm task heads."""

    SUPPORTS_IMAGE = {"vilt", "hate_clipper", "clip"}

    def __init__(
        self,
        kind: str,
        num_targets: int,
        vocab_size: int,
        pad_token_id: int,
        hidden_size: int = 768,
        dropout: float = 0.2,
        max_rationale_len: int = 64,
        beam_size: int = 4,
        lambda_harm: float = 1.0,
        lambda_cat: float = 1.0,
        lambda_rat: float = 1.0,
    ):
        super().__init__()
        self.kind = kind
        self.backbone = BaselineBackbone(kind, hidden_size, dropout)
        self.dropout = nn.Dropout(dropout)
        self.target_head = nn.Linear(hidden_size, num_targets)
        self.harm_head = nn.Linear(hidden_size, len(HARMFULNESS_LABELS))
        self.category_head = nn.Linear(hidden_size, len(CATEGORY_LABELS))
        self.rationale_decoder = RationaleDecoder(
            vocab_size, hidden_size, 2, 8, dropout, max_rationale_len, pad_token_id
        )
        self.pad_token_id = pad_token_id
        self.beam_size = beam_size
        self.max_rationale_len = max_rationale_len
        self.loss_weights = {"harm": lambda_harm, "category": lambda_cat, "rationale": lambda_rat}

    def _heads(self, z):
        hidden = self.dropout(z)
        return self.target_head(hidden), self.harm_head(hidden), self.category_head(hidden)

    def forward(self, pixel_values, input_ids, attention_mask, target_labels=None, harm_labels=None,
                cat_labels=None, rationale_ids=None, compute_counterfactuals=None,
                ocr_text=None, **unused) -> SheHarmOutput:
        encoded = self.backbone(pixel_values, input_ids, attention_mask, ocr_text=ocr_text)
        z = encoded["z"]
        target_logits, harm_logits, category_logits = self._heads(z)

        memory = self.rationale_decoder.build_memory(z, z, z, z)
        rationale_logits = self.rationale_decoder(memory, rationale_ids[:, :-1]) if rationale_ids is not None else None

        components: dict[str, torch.Tensor] = {}
        if target_labels is not None:
            components["target"] = loss_fn.target_loss(target_logits, target_labels)
        if harm_labels is not None:
            components["harm"] = loss_fn.harmfulness_loss(harm_logits, harm_labels)
        if cat_labels is not None:
            components["category"] = loss_fn.category_loss(category_logits, cat_labels)
        if rationale_logits is not None:
            components["rationale"] = loss_fn.rationale_loss(
                rationale_logits, rationale_ids[:, 1:], self.pad_token_id
            )
        total = loss_fn.total_loss(components, self.loss_weights) if "target" in components else None

        return SheHarmOutput(
            total_loss=total, losses=components, target_logits=target_logits,
            harm_logits=harm_logits, category_logits=category_logits, rationale_logits=rationale_logits,
            gamma=torch.zeros(z.size(0), device=z.device),
            activations=torch.zeros(z.size(0), 1, device=z.device),
            top_concepts=torch.zeros(z.size(0), 1, dtype=torch.long, device=z.device),
            extras={
                "z": z, "memory": memory,
                "patch_relevance": encoded["patch_relevance"], "token_relevance": encoded["token_relevance"],
                "max_activation": torch.zeros(z.size(0), device=z.device),
                "pixel_values": pixel_values, "input_ids": input_ids, "ocr_text": ocr_text,
            },
        )

    @torch.no_grad()
    def generate_rationale(self, memory, bos_token_id: int, eos_token_id: int) -> torch.Tensor:
        return self.rationale_decoder.generate(
            memory, bos_token_id, eos_token_id, beam_size=self.beam_size, max_len=self.max_rationale_len
        )

    @torch.no_grad()
    def intervene(self, output: SheHarmOutput, attention_mask, intervention: str):
        """Only evidence masking is defined for baselines — they have no target/rule state."""
        from ..models.sheharm_car import SheHarmCAR

        extras = output.extras
        relevance = extras["patch_relevance"]
        if relevance is None or self.kind not in self.SUPPORTS_IMAGE:
            # Text-only models mask OCR tokens instead of image patches.
            relevance = extras["token_relevance"]
            relevant, irrelevant = SheHarmCAR._relevance_masks(relevance, keep_first=True)
            mask = relevant if intervention == "mask_relevant_region" else irrelevant
            encoded = self.backbone(
                extras["pixel_values"], extras["input_ids"], (attention_mask * mask.long()),
                ocr_text=extras.get("ocr_text"),
            )
        else:
            relevant, irrelevant = SheHarmCAR._relevance_masks(relevance, keep_first=False)
            mask = relevant if intervention == "mask_relevant_region" else irrelevant
            encoded = self.backbone(extras["pixel_values"], extras["input_ids"], attention_mask,
                                    patch_mask=mask, ocr_text=extras.get("ocr_text"))

        _, harm_logits, category_logits = self._heads(encoded["z"])
        original = F.softmax(output.harm_logits, dim=-1)
        predicted = original.argmax(dim=-1, keepdim=True)
        intervened = F.softmax(harm_logits, dim=-1)
        return {
            "confidence_drop": (original.gather(1, predicted) - intervened.gather(1, predicted)).squeeze(1),
            "flipped": intervened.argmax(dim=-1).ne(predicted.squeeze(1)),
            "harm_logits": harm_logits, "category_logits": category_logits,
        }
