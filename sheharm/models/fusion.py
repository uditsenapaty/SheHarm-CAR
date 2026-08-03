"""Confidence-gated neural-symbolic fusion and prediction heads (diagram block 6-7).

    gamma = sigmoid(w_g^T (z || t~ || k~ || q^D) + b_g)   (Eq. confidence-gate)
    v     = W_K k~ + W_R q^D                              (Eq. fused-symbolic)
    u     = LayerNorm(z + W_T t~ + gamma * v)             (Eq. final-reasoning-representation)
    p^y   = softmax(W_y u + b_y),  p^c = softmax(W_c u + b_c)
"""

from __future__ import annotations

import torch
import torch.nn as nn


class ConfidenceGatedFusion(nn.Module):
    def __init__(
        self,
        hidden_size: int = 768,
        num_harm_classes: int = 3,
        num_category_classes: int = 5,
        dropout: float = 0.2,
        use_target_conditioning: bool = True,
        use_ontology_retrieval: bool = True,
        use_confidence_gate: bool = True,
    ):
        super().__init__()
        self.use_target_conditioning = use_target_conditioning
        self.use_ontology_retrieval = use_ontology_retrieval
        self.use_confidence_gate = use_confidence_gate

        self.target_proj = nn.Linear(hidden_size, hidden_size)      # W_T
        self.knowledge_proj = nn.Linear(hidden_size, hidden_size)   # W_K
        self.rule_proj = nn.Linear(hidden_size, hidden_size)        # W_R
        self.gate = nn.Linear(hidden_size * 4, 1)                   # w_g, b_g
        self.norm = nn.LayerNorm(hidden_size)
        self.dropout = nn.Dropout(dropout)

        self.harm_head = nn.Linear(hidden_size, num_harm_classes)   # W_y
        self.category_head = nn.Linear(hidden_size, num_category_classes)  # W_c

    def forward(self, z, soft_target, retrieved, contrastive, gate_reduction=None):
        gate_input = torch.cat([z, soft_target, retrieved, contrastive], dim=-1)
        gate_logit = self.gate(gate_input).squeeze(-1)
        if gate_reduction is not None:
            # Gate-control rules can only lower confidence.
            gate_logit = gate_logit - gate_reduction
        gamma = torch.sigmoid(gate_logit)
        if not self.use_confidence_gate:  # ablation: w/o Confidence Gate
            gamma = torch.ones_like(gamma)

        symbolic = self.rule_proj(contrastive)
        if self.use_ontology_retrieval:  # ablation: w/o Ontology Retrieval
            symbolic = symbolic + self.knowledge_proj(retrieved)

        u = z + gamma.unsqueeze(-1) * symbolic
        if self.use_target_conditioning:  # ablation: w/o Target Conditioning
            u = u + self.target_proj(soft_target)
        u = self.norm(u)

        return {
            "u": u,
            "gamma": gamma,
            "harm_logits": self.harm_head(self.dropout(u)),
            "category_logits": self.category_head(self.dropout(u)),
        }

    def predict_from_u(self, u: torch.Tensor):
        """Re-score an intervened representation without rebuilding the graph."""
        return self.harm_head(u), self.category_head(u)
