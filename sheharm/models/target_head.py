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
        tied_classifier: bool = True,
    ):
        super().__init__()
        num_targets, embedding_dim = target_embeddings.shape
        self.num_targets = num_targets
        self.tied = tied_classifier
        if tied_classifier:
            # Entity linking: score a mention against every concept embedding rather than
            # learning an independent weight vector per class. Still exactly
            # p^t = softmax(W_t z + b_t) - W_t is the composition of the mention projection
            # with the concept matrix - but the weights start semantically meaningful and
            # rare concepts inherit structure from related ones instead of learning from
            # their handful of examples alone.
            self.mention = nn.Sequential(nn.Dropout(dropout), nn.Linear(hidden_size, embedding_dim))
            self.link_bias = nn.Parameter(torch.zeros(num_targets))
            self.link_scale = nn.Parameter(torch.tensor(1.0 / embedding_dim**0.5))
        else:
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
        if self.tied:
            logits = self.mention(z) @ self.concept_embeddings.t() * self.link_scale + self.link_bias
        else:
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
