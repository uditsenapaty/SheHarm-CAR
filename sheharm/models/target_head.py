"""Women-related target identification (diagram block 2).

The paper formulates this as classification over a controlled inventory of ontology-linked
target concepts, not as span tagging:

    p^t = softmax(W_t z + b_t)                      (Eq. target-distribution)
    L_tgt = -sum_i log p^t_{i,t_i}                   (Eq. target-loss)
    t~   = sum_m p^t_m e^t_m                         (Eq. target-representation)

The soft target t~ preserves prediction uncertainty and is what every downstream module
(retrieval, rule reasoning, fusion, rationale decoding) consumes.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class TargetIdentifier(nn.Module):
    def __init__(
        self,
        target_embeddings: torch.Tensor,
        hidden_size: int = 768,
        dropout: float = 0.2,
        learnable_embeddings: bool = True,
    ):
        super().__init__()
        num_targets, embedding_dim = target_embeddings.shape
        self.num_targets = num_targets
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(hidden_size, num_targets),
        )
        # Concept embeddings start from the ontology text encodings and are refined during training.
        if learnable_embeddings:
            self.concept_embeddings = nn.Parameter(target_embeddings.clone())
        else:
            self.register_buffer("concept_embeddings", target_embeddings.clone())
        self.project = nn.Linear(embedding_dim, hidden_size) if embedding_dim != hidden_size else nn.Identity()

    def forward(self, z: torch.Tensor, hard: bool = False):
        logits = self.classifier(z)
        probabilities = F.softmax(logits, dim=-1)
        if hard:
            index = probabilities.argmax(dim=-1)
            probabilities = F.one_hot(index, self.num_targets).to(probabilities.dtype)
        soft_target = self.project(probabilities @ self.concept_embeddings)
        return logits, soft_target

    def null_target(self, batch_size: int, device, dtype) -> torch.Tensor:
        """Null representation used by the `suppress target` counterfactual intervention."""
        return torch.zeros(batch_size, self.project(self.concept_embeddings[:1]).size(-1), device=device, dtype=dtype)
