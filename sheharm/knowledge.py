"""Loading the Women-Harm Knowledge Ontology and initialising concept embeddings.

Concept embeddings are the mean-pooled RoBERTa encoding of each concept's gloss. They
initialise both the target-concept inventory (e^t) and the retrieval index (k_u), and are
refined during training.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import torch
from transformers import AutoTokenizer, RobertaModel


@dataclass
class Knowledge:
    concepts: list[dict]
    triples: list[dict]
    relations: list[dict]
    predicates: list[str]
    rules: list[dict]

    @property
    def target_concepts(self) -> list[dict]:
        return [concept for concept in self.concepts if concept["type"] == "target"]

    @property
    def target_names(self) -> list[str]:
        return [concept["name"] for concept in self.target_concepts]

    @property
    def retrieval_concepts(self) -> list[dict]:
        """Everything the retriever may return: cues, categories, and contextual exceptions."""
        return [concept for concept in self.concepts if concept["type"] != "target"]


def load_knowledge(ontology_path: str | Path, rules_path: str | Path) -> Knowledge:
    ontology = json.loads(Path(ontology_path).read_text(encoding="utf-8"))
    rules = json.loads(Path(rules_path).read_text(encoding="utf-8"))
    return Knowledge(
        concepts=ontology["concepts"],
        triples=ontology["triples"],
        relations=ontology["relations"],
        predicates=rules["predicates"],
        rules=rules["rules"],
    )


@torch.no_grad()
def encode_concepts(
    concepts: list[dict],
    text_model: str = "roberta-base",
    device: str | torch.device = "cpu",
    batch_size: int = 64,
    max_length: int = 32,
) -> torch.Tensor:
    tokenizer = AutoTokenizer.from_pretrained(text_model, use_fast=True)
    encoder = RobertaModel.from_pretrained(text_model).to(device).eval()
    texts = [concept.get("text") or concept.get("name") or concept["id"] for concept in concepts]

    vectors = []
    for start in range(0, len(texts), batch_size):
        batch = tokenizer(
            texts[start : start + batch_size], truncation=True, padding=True,
            max_length=max_length, return_tensors="pt",
        ).to(device)
        hidden = encoder(**batch).last_hidden_state
        mask = batch["attention_mask"].unsqueeze(-1)
        vectors.append(((hidden * mask).sum(1) / mask.sum(1).clamp_min(1)).cpu())
    del encoder
    if str(device).startswith("cuda"):
        torch.cuda.empty_cache()
    return torch.cat(vectors, dim=0)


def build_embeddings(knowledge: Knowledge, text_model: str = "roberta-base",
                     device: str | torch.device = "cpu", cache: str | Path | None = None):
    """Returns (target_embeddings, retrieval_embeddings), cached to disk when a path is given."""
    if cache is not None and Path(cache).exists():
        payload = torch.load(cache, map_location="cpu")
        return payload["targets"], payload["concepts"]
    targets = encode_concepts(knowledge.target_concepts, text_model, device)
    retrieval = encode_concepts(knowledge.retrieval_concepts, text_model, device)
    if cache is not None:
        Path(cache).parent.mkdir(parents=True, exist_ok=True)
        torch.save({"targets": targets, "concepts": retrieval}, cache)
    return targets, retrieval
