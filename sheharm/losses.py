"""Training objectives.

    L_task  = L_tgt + lam_harm L_harm + lam_cat L_cat + lam_rat L_rat
    L_total = L_task + lam_align L_align + lam_cons L_cons + lam_cf L_cf

L_cons uses KL(s || p) — symbolic distribution first, as written in the paper — and is
applied only to instances where at least one rule exceeds the activation threshold.
L_cf = L_nec + lam_inv L_inv: confidence must fall when relevant evidence is removed and
must hold steady when irrelevant evidence is perturbed.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

from .labels import IGNORE_INDEX


def target_loss(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    return F.cross_entropy(logits, labels, ignore_index=IGNORE_INDEX)


def harmfulness_loss(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    return F.cross_entropy(logits, labels)


def category_loss(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    """Masked: Non-Harm instances carry no category (m_i = 0)."""
    if labels.ne(IGNORE_INDEX).sum() == 0:
        return logits.new_zeros(())
    return F.cross_entropy(logits, labels, ignore_index=IGNORE_INDEX)


def rationale_loss(logits: torch.Tensor, targets: torch.Tensor, pad_token_id: int) -> torch.Tensor:
    return F.cross_entropy(
        logits.reshape(-1, logits.size(-1)), targets.reshape(-1), ignore_index=pad_token_id
    )


def consistency_loss(
    harm_logits: torch.Tensor,
    category_logits: torch.Tensor,
    symbolic_harm: torch.Tensor,
    symbolic_category: torch.Tensor,
    active_mask: torch.Tensor,
    category_labels: torch.Tensor,
) -> torch.Tensor:
    if active_mask.sum() == 0:
        return harm_logits.new_zeros(())
    harm_kl = F.kl_div(
        F.log_softmax(harm_logits[active_mask], dim=-1),
        symbolic_harm[active_mask],
        reduction="batchmean",
    )
    category_mask = active_mask & category_labels.ne(IGNORE_INDEX)
    if category_mask.any():
        category_kl = F.kl_div(
            F.log_softmax(category_logits[category_mask], dim=-1),
            symbolic_category[category_mask],
            reduction="batchmean",
        )
    else:
        category_kl = harm_logits.new_zeros(())
    return harm_kl + category_kl


def necessity_loss(original_logits: torch.Tensor, intervened_logits: torch.Tensor, margin: float = 0.10) -> torch.Tensor:
    """Confidence in the original prediction must drop by at least `margin`."""
    with torch.no_grad():
        predicted = original_logits.argmax(dim=-1, keepdim=True)
        original_confidence = F.softmax(original_logits, dim=-1).gather(1, predicted).squeeze(1)
    intervened_confidence = F.softmax(intervened_logits, dim=-1).gather(1, predicted).squeeze(1)
    return F.relu(margin - (original_confidence - intervened_confidence)).mean()


def invariance_loss(original_logits: torch.Tensor, perturbed_logits: torch.Tensor) -> torch.Tensor:
    """Prediction must be preserved when low-relevance evidence is perturbed."""
    return F.kl_div(
        F.log_softmax(perturbed_logits, dim=-1),
        F.softmax(original_logits.detach(), dim=-1),
        reduction="batchmean",
    )


def counterfactual_loss(
    original_logits: torch.Tensor,
    relevant_logits: torch.Tensor,
    irrelevant_logits: torch.Tensor,
    lambda_invariance: float = 0.5,
    margin: float = 0.10,
) -> torch.Tensor:
    return necessity_loss(original_logits, relevant_logits, margin) + lambda_invariance * invariance_loss(
        original_logits, irrelevant_logits
    )


def total_loss(components: dict[str, torch.Tensor], weights: dict[str, float]) -> torch.Tensor:
    """L_tgt carries an implicit coefficient of 1.0 (paper: "set implicitly to 1.0")."""
    total = components["target"]
    for name, weight in weights.items():
        if name in components:
            total = total + weight * components[name]
    return total
