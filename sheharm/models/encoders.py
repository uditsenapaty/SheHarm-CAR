"""Multimodal encoding and bidirectional cross-modal interaction (diagram block 1).

CLIP ViT-B/32 for the image, RoBERTa-base for the OCR text, both at hidden size 768
(paper Table `tab:hyperparameters`). Bidirectional cross-modal attention lets OCR tokens
attend to image patches and image patches attend to OCR tokens; the pooled outputs are
fused into the global multimodal representation z.
"""

from __future__ import annotations

import torch
import torch.nn as nn
from transformers import CLIPVisionModel, RobertaModel


class CrossModalBlock(nn.Module):
    """One bidirectional cross-attention layer with residual connections."""

    def __init__(self, hidden_size: int = 768, heads: int = 8, dropout: float = 0.2):
        super().__init__()
        self.text_to_image = nn.MultiheadAttention(hidden_size, heads, dropout=dropout, batch_first=True)
        self.image_to_text = nn.MultiheadAttention(hidden_size, heads, dropout=dropout, batch_first=True)
        self.norm_text = nn.LayerNorm(hidden_size)
        self.norm_image = nn.LayerNorm(hidden_size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, text_tokens, image_tokens, text_padding_mask=None):
        attended_text, text_to_image_weights = self.text_to_image(
            query=text_tokens, key=image_tokens, value=image_tokens, average_attn_weights=True,
        )
        attended_image, image_to_text_weights = self.image_to_text(
            query=image_tokens, key=text_tokens, value=text_tokens,
            key_padding_mask=text_padding_mask, average_attn_weights=True,
        )
        text_out = self.norm_text(text_tokens + self.dropout(attended_text))
        image_out = self.norm_image(image_tokens + self.dropout(attended_image))
        return text_out, image_out, text_to_image_weights, image_to_text_weights


class MultimodalEncoder(nn.Module):
    """Produces token-level states, the pooled representations, and z."""

    def __init__(
        self,
        vision_model: str = "openai/clip-vit-base-patch32",
        text_model: str = "roberta-base",
        hidden_size: int = 768,
        heads: int = 8,
        dropout: float = 0.2,
        freeze_encoders: bool = False,
        cross_modal_layers: int = 2,
    ):
        super().__init__()
        self.vision = CLIPVisionModel.from_pretrained(vision_model)
        # RoBERTa's pooler is randomly initialised and unused here: we pool over tokens
        # ourselves, so instantiating it would leave dead parameters in the optimizer.
        self.text = RobertaModel.from_pretrained(text_model, add_pooling_layer=False)
        if freeze_encoders:
            for parameter in list(self.vision.parameters()) + list(self.text.parameters()):
                parameter.requires_grad_(False)

        vision_width = self.vision.config.hidden_size
        text_width = self.text.config.hidden_size
        self.vision_proj = nn.Linear(vision_width, hidden_size) if vision_width != hidden_size else nn.Identity()
        self.text_proj = nn.Linear(text_width, hidden_size) if text_width != hidden_size else nn.Identity()

        # The paper specifies bidirectional cross-modal attention but not its depth, so the
        # number of blocks is a free hyperparameter.
        self.cross_modal = nn.ModuleList(
            CrossModalBlock(hidden_size, heads, dropout) for _ in range(max(1, cross_modal_layers))
        )
        self.fuse = nn.Sequential(
            nn.Linear(hidden_size * 2, hidden_size),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.LayerNorm(hidden_size),
        )

    def encode_backbones(self, pixel_values, input_ids, attention_mask):
        """The expensive half: run once, then reuse across counterfactual interventions."""
        vision_states = self.vision(pixel_values=pixel_values).last_hidden_state
        # CLIP applies post_layernorm before projection; keep it so the pretrained
        # parameter stays in use and patch features match CLIP's own output space.
        # transformers v5 exposes the submodules directly; v4 nests them under .vision_model.
        vision_states = getattr(self.vision, "vision_model", self.vision).post_layernorm(vision_states)
        return (
            self.vision_proj(vision_states),
            self.text_proj(self.text(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state),
        )

    def forward(self, pixel_values, input_ids, attention_mask, patch_mask=None):
        image_tokens, text_tokens = self.encode_backbones(pixel_values, input_ids, attention_mask)
        return self.interact(image_tokens, text_tokens, attention_mask, patch_mask)

    def interact(self, image_tokens, text_tokens, attention_mask, patch_mask=None, token_mask=None):
        """The cheap half: cross-modal attention + pooling.

        patch_mask (B, P) and token_mask (B, T) zero out evidence for counterfactual
        interventions without paying for another backbone forward pass.
        """
        if token_mask is not None:
            text_tokens = text_tokens * token_mask.unsqueeze(-1)
            attention_mask = attention_mask * token_mask.long()
        if patch_mask is not None:
            image_tokens = image_tokens * patch_mask.unsqueeze(-1)

        text_padding_mask = attention_mask.eq(0)
        text_out, image_out = text_tokens, image_tokens
        for block in self.cross_modal:
            # Relevance is read from the final block, where representations are most fused.
            text_out, image_out, t2i, i2t = block(text_out, image_out, text_padding_mask)

        text_weights = attention_mask.unsqueeze(-1).to(text_out.dtype)
        pooled_text = (text_out * text_weights).sum(1) / text_weights.sum(1).clamp_min(1.0)
        if patch_mask is not None:
            image_weights = patch_mask.unsqueeze(-1).to(image_out.dtype)
            pooled_image = (image_out * image_weights).sum(1) / image_weights.sum(1).clamp_min(1.0)
        else:
            pooled_image = image_out.mean(dim=1)

        z = self.fuse(torch.cat([pooled_text, pooled_image], dim=-1))
        return {
            "text_tokens": text_out,
            "image_tokens": image_out,
            "pooled_text": pooled_text,
            "pooled_image": pooled_image,
            "z": z,
            # Relevance signals for counterfactual interventions: the paper estimates visual
            # relevance from cross-modal attention over patches and target-indicative OCR
            # tokens from text-to-image attention.
            # t2i is (B, T, P): attention assigned *to* each patch, averaged over OCR queries.
            "patch_relevance": t2i.mean(dim=1) if t2i is not None else None,
            # i2t is (B, P, T): attention assigned *to* each OCR token, averaged over patches.
            "token_relevance": i2t.mean(dim=1) if i2t is not None else None,
        }
