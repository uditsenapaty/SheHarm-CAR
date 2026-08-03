"""Target-conditioned ontology retrieval (diagram block 4).

    q^K = LayerNorm(W_q (z || t~) + b_q)             (Eq. retrieval-query)
    k~  = sum_{u in topK} alpha_u k_u                 (Eq. retrieved-knowledge)
    alpha_u = softmax(s_u / tau) over the top-K set   (Eq. knowledge-weight)

L_align is an InfoNCE objective that pulls q^K toward the highest-scoring concept and
pushes it away from sampled hard negatives (paper: ten hard negatives).
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class OntologyRetriever(nn.Module):
    def __init__(
        self,
        concept_embeddings: torch.Tensor,
        hidden_size: int = 768,
        top_k: int = 5,
        temperature: float = 0.07,
        num_hard_negatives: int = 10,
        learnable_embeddings: bool = True,
    ):
        super().__init__()
        num_concepts, embedding_dim = concept_embeddings.shape
        if learnable_embeddings:
            self.concept_embeddings = nn.Parameter(concept_embeddings.clone())
        else:
            self.register_buffer("concept_embeddings", concept_embeddings.clone())
        self.embed_proj = nn.Linear(embedding_dim, hidden_size) if embedding_dim != hidden_size else nn.Identity()
        self.query = nn.Sequential(
            nn.Linear(hidden_size * 2, hidden_size),
            nn.LayerNorm(hidden_size),
        )
        self.top_k = min(top_k, num_concepts)
        self.temperature = temperature
        self.num_hard_negatives = num_hard_negatives

    def forward(self, z: torch.Tensor, soft_target: torch.Tensor, drop_top_concept: bool = False):
        query = F.normalize(self.query(torch.cat([z, soft_target], dim=-1)), dim=-1)
        concepts = F.normalize(self.embed_proj(self.concept_embeddings), dim=-1)
        scores = query @ concepts.t()

        k = self.top_k + 1 if drop_top_concept else self.top_k
        k = min(k, scores.size(-1))
        top_scores, top_index = scores.topk(k, dim=-1)
        if drop_top_concept:
            # `remove top concept` counterfactual: keep the ranking, discard rank 1.
            top_scores, top_index = top_scores[:, 1:], top_index[:, 1:]

        weights = F.softmax(top_scores / self.temperature, dim=-1)
        retrieved = torch.einsum("bk,bkd->bd", weights, concepts[top_index])
        return {
            "query": query,
            "retrieved": retrieved,
            "scores": scores,
            "top_index": top_index,
            "weights": weights,
            "top_score": top_scores[:, 0],
        }

    def alignment_loss(self, query: torch.Tensor, scores: torch.Tensor, top_index: torch.Tensor) -> torch.Tensor:
        """InfoNCE against the retrieved positive with sampled hard negatives."""
        positive = top_index[:, 0]
        batch = scores.size(0)
        masked = scores.clone()
        masked.scatter_(1, positive.unsqueeze(1), float("-inf"))
        negatives = masked.topk(min(self.num_hard_negatives, masked.size(1) - 1), dim=-1).indices

        positive_scores = scores.gather(1, positive.unsqueeze(1))
        negative_scores = scores.gather(1, negatives)
        logits = torch.cat([positive_scores, negative_scores], dim=1) / self.temperature
        labels = torch.zeros(batch, dtype=torch.long, device=scores.device)
        return F.cross_entropy(logits, labels)
