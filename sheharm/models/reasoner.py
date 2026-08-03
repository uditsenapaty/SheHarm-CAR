"""Contrastive symbolic rule reasoning (diagram block 5).

Predicate truth values are estimated from the global multimodal representation z, the soft
target t~, and the retrieved knowledge k~ — not from raw concept similarity. A rule fires
through a soft conjunction of its predicates:

    rho_j = prod_l p_jl                              (Eq. rule-activation)
    q^D   = sum_{R+} rho_j e_j - sum_{R-} rho_j e_j  (Eq. contrastive-rule-representation)

Gate-control rules (R10 family) contribute a non-negative penalty that can only *reduce*
the confidence gate, which is how the paper describes their role.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class SoftRuleReasoner(nn.Module):
    def __init__(
        self,
        rules: list[dict],
        predicates: list[str],
        num_harm_classes: int,
        num_category_classes: int,
        hidden_size: int = 768,
        dropout: float = 0.2,
        activation_threshold: float = 0.60,
        use_exception_rules: bool = True,
    ):
        super().__init__()
        self.predicates = predicates
        self.activation_threshold = activation_threshold
        self.use_exception_rules = use_exception_rules

        if not use_exception_rules:  # ablation: w/o Exception Rules
            rules = [rule for rule in rules if rule["polarity"] != "exception"]
        self.rules = rules
        self.rule_names = [rule["name"] for rule in rules]

        predicate_index = {name: i for i, name in enumerate(predicates)}
        membership = torch.zeros(len(rules), len(predicates))
        polarity = torch.zeros(len(rules))
        gate_flag = torch.zeros(len(rules))
        weight = torch.zeros(len(rules))
        harm_target = torch.full((len(rules),), -1, dtype=torch.long)
        category_target = torch.full((len(rules),), -1, dtype=torch.long)
        for row, rule in enumerate(rules):
            for name in rule["predicates"]:
                membership[row, predicate_index[name]] = 1.0
            polarity[row] = -1.0 if rule["polarity"] == "exception" else 1.0
            gate_flag[row] = 1.0 if rule["polarity"] == "gate" else 0.0
            weight[row] = float(rule.get("weight", 1.0))
            if rule.get("harm_class") is not None:
                harm_target[row] = int(rule["harm_class"])
            if rule.get("category_class") is not None:
                category_target[row] = int(rule["category_class"])

        self.register_buffer("membership", membership)
        self.register_buffer("polarity", polarity)
        self.register_buffer("gate_flag", gate_flag)
        self.register_buffer("rule_weight", weight)
        self.register_buffer("harm_target", harm_target)
        self.register_buffer("category_target", category_target)

        self.predicate_net = nn.Sequential(
            nn.Linear(hidden_size * 3, hidden_size),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, len(predicates)),
        )
        self.rule_embeddings = nn.Parameter(torch.randn(len(rules), hidden_size) / math.sqrt(hidden_size))
        self.gate_penalty = nn.Parameter(torch.zeros(len(rules)))
        self.num_harm_classes = num_harm_classes
        self.num_category_classes = num_category_classes

    def forward(self, z, soft_target, retrieved, suppress_strongest_rule: bool = False):
        truths = torch.sigmoid(self.predicate_net(torch.cat([z, soft_target, retrieved], dim=-1)))

        # Soft conjunction as a product, computed in log space for numerical stability.
        log_truths = torch.log(truths.clamp_min(1e-6))
        activations = torch.exp(log_truths @ self.membership.t()) * self.rule_weight

        if suppress_strongest_rule:
            # `deactivate strongest rule` counterfactual.
            scored = activations.masked_fill(self.gate_flag.bool().unsqueeze(0), float("-inf"))
            strongest = scored.argmax(dim=-1, keepdim=True)
            activations = activations.scatter(1, strongest, 0.0)

        reasoning_mask = (1.0 - self.gate_flag).unsqueeze(0)
        signed = activations * self.polarity.unsqueeze(0) * reasoning_mask
        contrastive = signed @ self.rule_embeddings

        gate_reduction = (activations * self.gate_flag.unsqueeze(0) * F.softplus(self.gate_penalty).unsqueeze(0)).sum(-1)

        symbolic_harm = self._aggregate(activations, self.harm_target, self.num_harm_classes)
        symbolic_category = self._aggregate(activations, self.category_target, self.num_category_classes)

        reasoning_activations = activations * reasoning_mask
        max_activation = reasoning_activations.max(dim=-1).values
        return {
            "predicate_truths": truths,
            "activations": activations,
            "contrastive": contrastive,
            "gate_reduction": gate_reduction,
            "symbolic_harm": symbolic_harm,
            "symbolic_category": symbolic_category,
            "max_activation": max_activation,
            "active_mask": max_activation >= self.activation_threshold,
        }

    def _aggregate(self, activations: torch.Tensor, targets: torch.Tensor, num_classes: int) -> torch.Tensor:
        """Map rule activations onto a symbolic class distribution (s^y / s^c)."""
        batch = activations.size(0)
        scores = activations.new_full((batch, num_classes), 1e-4)
        for class_index in range(num_classes):
            selector = targets.eq(class_index)
            if selector.any():
                scores[:, class_index] = scores[:, class_index] + activations[:, selector].sum(dim=-1)
        return scores / scores.sum(dim=-1, keepdim=True)
