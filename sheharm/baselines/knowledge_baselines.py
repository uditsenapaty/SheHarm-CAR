"""Knowledge-, interpretation- and reasoning-guided baselines.

KERMIT, KID-VLM, IntMeme, ExplainHM++ and SGoT-R1 were all published for *binary* hateful-meme
classification on FHM/HarMeme, not for the four-task SheHarm setting. Their released code
therefore cannot be run unmodified here. Each is reimplemented as a method-faithful adapter:
the mechanism described in the paper, on top of the same task heads every other baseline uses.
`referred_clones/MANIFEST.md` records which ones have public code, and every table labels
these rows as reimplementations.

Common shape: an auxiliary text channel is produced once per meme and cached, then encoded
alongside the OCR text.

    IntMeme      large multimodal model writes an *interpretation* of the meme
    ExplainHM++  two opposing debate rationales (harmful vs harmless) plus a judge
    SGoT-R1      structured social graph-of-thought trace (entities, relations, intent)
    KERMIT       knowledge-enriched network: retrieved concepts read by memory attention
    KID-VLM      knowledge infusion plus distillation from a large teacher's distribution
"""

from __future__ import annotations

import json
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..labels import CATEGORY_LABELS, HARMFULNESS_LABELS
from .encoder_baselines import EncoderBaseline

AUXILIARY_PROMPTS = {
    "intmeme": (
        "Describe what this meme means. State who or what it depicts, what the text implies, "
        "and what a reader is meant to conclude. Two sentences, no judgement."
    ),
    "explainhm_harmful": (
        "Argue, in two sentences, that this meme IS harmful to women. Cite the specific visual "
        "and textual evidence that supports that reading."
    ),
    "explainhm_harmless": (
        "Argue, in two sentences, that this meme is NOT harmful to women. Cite the specific "
        "visual and textual evidence that supports that reading."
    ),
    "sgot_r1": (
        "Produce a compact social reasoning trace for this meme as: ENTITIES: ...; RELATIONS: ...; "
        "SOCIAL NORM: ...; INTENT: ...  Keep each field under twelve words."
    ),
}


class AuxiliaryTextStore:
    """Cache of generated auxiliary text so a baseline is generated once and reused."""

    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.data: dict[str, dict[str, str]] = {}
        if self.path.exists():
            self.data = json.loads(self.path.read_text(encoding="utf-8"))

    def get(self, filename: str, channel: str) -> str:
        return self.data.get(filename, {}).get(channel, "")

    def set(self, filename: str, channel: str, text: str) -> None:
        self.data.setdefault(filename, {})[channel] = text

    def missing(self, filenames, channel: str) -> list[str]:
        return [name for name in filenames if not self.get(name, channel)]

    def save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(json.dumps(self.data, indent=1), encoding="utf-8")


class MemoryAugmentedKnowledge(nn.Module):
    """KERMIT's dynamic knowledge selection: memory attention over retrieved concepts.

    The paper reads external knowledge from ConceptNet; the Women-Harm ontology is
    substituted here so the baseline uses the same knowledge store as our model, which keeps
    the comparison about the *mechanism* rather than about knowledge coverage.
    """

    def __init__(self, concept_embeddings: torch.Tensor, hidden_size: int = 768,
                 memory_slots: int = 32, top_k: int = 10):
        super().__init__()
        self.register_buffer("concepts", F.normalize(concept_embeddings, dim=-1))
        self.top_k = top_k
        self.memory = nn.Parameter(torch.randn(memory_slots, hidden_size) / hidden_size**0.5)
        self.query = nn.Linear(hidden_size, hidden_size)
        self.attention = nn.MultiheadAttention(hidden_size, 8, batch_first=True)
        self.norm = nn.LayerNorm(hidden_size)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        query = F.normalize(self.query(z), dim=-1)
        scores = query @ self.concepts.t()
        top_index = scores.topk(min(self.top_k, scores.size(-1)), dim=-1).indices
        retrieved = self.concepts[top_index]
        memory = self.memory.unsqueeze(0).expand(z.size(0), -1, -1)
        keys = torch.cat([retrieved, memory], dim=1)
        attended, _ = self.attention(query=z.unsqueeze(1), key=keys, value=keys)
        return self.norm(z + attended.squeeze(1))


class KnowledgeBaseline(EncoderBaseline):
    """Encoder baseline extended with an auxiliary text channel and/or knowledge memory."""

    def __init__(self, method: str, num_targets: int, vocab_size: int, pad_token_id: int,
                 concept_embeddings: torch.Tensor | None = None, hidden_size: int = 768,
                 dropout: float = 0.2, backbone: str = "hate_clipper", **kwargs):
        super().__init__(backbone, num_targets, vocab_size, pad_token_id, hidden_size, dropout, **kwargs)
        self.method = method
        self.knowledge = (
            MemoryAugmentedKnowledge(concept_embeddings, hidden_size)
            if method == "kermit" and concept_embeddings is not None else None
        )
        # ExplainHM++ judges two opposing rationales, so it fuses three text views.
        self.debate_fuse = (
            nn.Sequential(nn.Linear(hidden_size * 3, hidden_size), nn.GELU(),
                          nn.Dropout(dropout), nn.LayerNorm(hidden_size))
            if method == "explainhm" else None
        )
        self.distillation_temperature = 2.0

    def _heads(self, z):
        if self.knowledge is not None:
            z = self.knowledge(z)
        return super()._heads(z)

    @staticmethod
    def distillation_loss(student_logits: torch.Tensor, teacher_probabilities: torch.Tensor,
                          temperature: float = 2.0) -> torch.Tensor:
        """KID-VLM's distillation term: KL(teacher || student) at a softened temperature."""
        return F.kl_div(
            F.log_softmax(student_logits / temperature, dim=-1),
            teacher_probabilities, reduction="batchmean",
        ) * temperature**2


def auxiliary_channels(method: str) -> list[str]:
    return {
        "intmeme": ["intmeme"],
        "explainhm": ["explainhm_harmful", "explainhm_harmless"],
        "sgot_r1": ["sgot_r1"],
        "kermit": [],
        "kid_vlm": [],
    }[method]


def compose_input_text(ocr_text: str, method: str, store: AuxiliaryTextStore, filename: str) -> str:
    """Concatenate the OCR text with the method's auxiliary channels."""
    parts = [ocr_text]
    for channel in auxiliary_channels(method):
        text = store.get(filename, channel)
        if text:
            label = channel.split("_")[-1].upper()
            parts.append(f"[{label}] {text}")
    return " ".join(parts)


METHOD_LABELS = {
    "kermit": "KERMIT",
    "kid_vlm": "KID-VLM",
    "intmeme": r"IntMeme$_{\textsc{InstructBLIP}}$",
    "explainhm": "ExplainHM++",
    "sgot_r1": "SGoT-R1",
}
REIMPLEMENTED = set(METHOD_LABELS)
