"""Joint decoding of the (harmfulness, harm-category) pair.

The paper defines the two prediction distributions

    p^y = softmax(W_y u + b_y)        three-way harmfulness
    p^c = softmax(W_c u + b_c)        five-way harm category, trained only on harmful rows

but never states how a *decision* is read off them. Taking two independent argmaxes is one
choice, and it is the wrong one for a metric that scores the pair jointly: the pair the model
considers most likely is not generally the pair of individually-most-likely labels.

Because the category loss is masked to harmful instances, p^c is P(c | harmful), so the pair
probability factorises exactly:

    P(Non-Harm, None) = p^y_[Non-Harm]
    P(y, c)           = p^y_[y] * p^c_[c]        for y in {Explicit-Harm, Implicit-Harm}

Eleven tuples are legal (one null pair plus two severities times five categories). Choosing
the best of them changes no equation, no coefficient and no training signal - only which
decision is read out of the distributions the paper defines.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

from .labels import CATEGORY_LABELS, HARMFULNESS_LABELS, NON_HARM_ID


def joint_decode(harm_logits: torch.Tensor, category_logits: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Return (harm_prediction, category_prediction) maximising the joint pair probability."""
    harm = F.softmax(harm_logits.float(), dim=-1)
    category = F.softmax(category_logits.float(), dim=-1)

    harmful = [index for index in range(len(HARMFULNESS_LABELS)) if index != NON_HARM_ID]
    # (B, |harmful|, |categories|) scores for every legal harmful pair.
    pair = harm[:, harmful].unsqueeze(-1) * category.unsqueeze(1)
    best_pair = pair.flatten(1).argmax(dim=-1)
    best_harm = torch.tensor(harmful, device=harm.device)[best_pair // len(CATEGORY_LABELS)]
    best_category = best_pair % len(CATEGORY_LABELS)
    best_score = pair.flatten(1).max(dim=-1).values

    null_score = harm[:, NON_HARM_ID]
    choose_null = null_score >= best_score
    harm_prediction = torch.where(choose_null, torch.full_like(best_harm, NON_HARM_ID), best_harm)
    # The category of a Non-Harm decision is the null category; callers map it to `None`.
    return harm_prediction, best_category


def independent_decode(harm_logits: torch.Tensor, category_logits: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    return harm_logits.argmax(dim=-1), category_logits.argmax(dim=-1)


def decode(harm_logits: torch.Tensor, category_logits: torch.Tensor, joint: bool = True):
    return (joint_decode if joint else independent_decode)(harm_logits, category_logits)
