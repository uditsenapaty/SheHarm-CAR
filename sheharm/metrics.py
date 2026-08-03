"""Evaluation metrics for the four SheHarm-Meme tasks.

Paper Section `subsec:metrics` + Table `tab:main-results` caption:

    Tgt.-F1     macro-F1 over women-related target concepts
    Harm-F1     macro-F1 over {Explicit-Harm, Implicit-Harm, Non-Harm}
    Cat.-F1     macro-F1 over the five categories, computed only over harmful instances
    Joint-F1    joint tuple F1 for (target, harmfulness, harm category)
    BERTScore   rationale quality
    CF-Faith.   confidence decrease after removing target-relevant evidence together with
                prediction stability after removing low-relevance evidence

The paper states CF-Faith qualitatively, so the exact combination is fixed here and used
identically for every model:

    necessity  = mean_i clip((conf_i - conf_i^relevant-removed) / conf_i, 0, 1)
    stability  = mean_i 1[argmax_i^irrelevant-removed == argmax_i]
    CF-Faith   = 100 * (necessity + stability) / 2
"""

from __future__ import annotations

import warnings

import numpy as np
from sklearn.metrics import f1_score

from .labels import IGNORE_INDEX, NON_HARM_ID


def macro_f1(y_true, y_pred) -> float:
    if len(y_true) == 0:
        return float("nan")
    return float(f1_score(y_true, y_pred, average="macro", zero_division=0)) * 100.0


def per_class_f1(y_true, y_pred, num_classes: int) -> list[float]:
    if len(y_true) == 0:
        return [float("nan")] * num_classes
    scores = f1_score(y_true, y_pred, average=None, labels=list(range(num_classes)), zero_division=0)
    return [float(value) * 100.0 for value in scores]


def target_f1(true_targets, predicted_targets) -> float:
    pairs = [(t, p) for t, p in zip(true_targets, predicted_targets) if t != IGNORE_INDEX]
    if not pairs:
        return float("nan")
    return macro_f1([t for t, _ in pairs], [p for _, p in pairs])


def category_f1(true_harm, true_category, predicted_category) -> float:
    """Computed only over instances whose gold harmfulness is harmful."""
    pairs = [
        (c, p) for h, c, p in zip(true_harm, true_category, predicted_category)
        if h != NON_HARM_ID and c != IGNORE_INDEX
    ]
    if not pairs:
        return float("nan")
    return macro_f1([c for c, _ in pairs], [p for _, p in pairs])


def joint_f1(
    true_targets, true_harm, true_category,
    predicted_targets, predicted_harm, predicted_category,
) -> dict[str, float]:
    """Exact-match over the full tuple, plus the harmful-only tuple view.

    `joint` treats every instance as one tuple (Non-Harm carries the null category), so
    precision equals recall. `joint_harmful` scores only the tuples a model emits for
    instances it calls harmful, which separates precision from recall.
    """
    correct = 0
    for tt, th, tc, pt, ph, pc in zip(
        true_targets, true_harm, true_category, predicted_targets, predicted_harm, predicted_category
    ):
        gold_category = None if th == NON_HARM_ID else tc
        predicted_cat = None if ph == NON_HARM_ID else pc
        correct += int(tt == pt and th == ph and gold_category == predicted_cat)
    exact = correct / max(len(true_harm), 1) * 100.0

    predicted_set, gold_set, hit = 0, 0, 0
    for tt, th, tc, pt, ph, pc in zip(
        true_targets, true_harm, true_category, predicted_targets, predicted_harm, predicted_category
    ):
        gold_harmful = th != NON_HARM_ID
        predicted_harmful = ph != NON_HARM_ID
        predicted_set += int(predicted_harmful)
        gold_set += int(gold_harmful)
        if gold_harmful and predicted_harmful and tt == pt and th == ph and tc == pc:
            hit += 1
    precision = hit / predicted_set if predicted_set else 0.0
    recall = hit / gold_set if gold_set else 0.0
    harmful = 2 * precision * recall / (precision + recall) * 100.0 if precision + recall else 0.0
    return {"joint": exact, "joint_harmful": harmful}


def counterfactual_faithfulness(
    original_confidence: np.ndarray,
    relevant_removed_confidence: np.ndarray,
    original_prediction: np.ndarray,
    irrelevant_removed_prediction: np.ndarray,
) -> dict[str, float]:
    original_confidence = np.asarray(original_confidence, dtype=np.float64)
    relevant_removed_confidence = np.asarray(relevant_removed_confidence, dtype=np.float64)
    drop = (original_confidence - relevant_removed_confidence) / np.clip(original_confidence, 1e-6, None)
    necessity = float(np.clip(drop, 0.0, 1.0).mean())
    stability = float((np.asarray(original_prediction) == np.asarray(irrelevant_removed_prediction)).mean())
    return {
        "cf_faithfulness": (necessity + stability) / 2 * 100.0,
        "cf_necessity": necessity * 100.0,
        "cf_stability": stability * 100.0,
        "mean_confidence_drop": float((original_confidence - relevant_removed_confidence).mean()) * 100.0,
    }


def bertscore(predictions: list[str], references: list[str], model_type: str = "roberta-large",
              device: str | None = None, batch_size: int = 32) -> float:
    """BERTScore F1 (%). Falls back to a local implementation if the package is absent."""
    pairs = [(p, r) for p, r in zip(predictions, references) if r and r.strip()]
    if not pairs:
        return float("nan")
    predictions = [p if p.strip() else " " for p, _ in pairs]
    references = [r for _, r in pairs]
    try:
        from bert_score import score as _score

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            _, _, f1 = _score(
                predictions, references, model_type=model_type, lang="en",
                verbose=False, device=device, batch_size=batch_size, rescale_with_baseline=False,
            )
        return float(f1.mean()) * 100.0
    except ImportError:
        warnings.warn("bert-score not installed; using the built-in greedy-matching fallback")
        return _bertscore_fallback(predictions, references, model_type, device, batch_size)


_FALLBACK_CACHE: dict = {}


def _bertscore_fallback(predictions, references, model_type, device, batch_size) -> float:
    """Greedy cosine matching over contextual embeddings — the BERTScore F1 definition."""
    import torch
    from transformers import AutoModel, AutoTokenizer

    name = "roberta-base" if model_type == "roberta-large" else model_type
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    # Cached: checkpoint selection scores rationales every epoch, and reloading the encoder
    # each time costs more than the scoring itself.
    key = (name, str(device))
    if key not in _FALLBACK_CACHE:
        _FALLBACK_CACHE[key] = (
            AutoTokenizer.from_pretrained(name), AutoModel.from_pretrained(name).to(device).eval()
        )
    tokenizer, model = _FALLBACK_CACHE[key]

    def embed(texts):
        outputs = []
        for start in range(0, len(texts), batch_size):
            batch = tokenizer(texts[start : start + batch_size], padding=True, truncation=True,
                              max_length=128, return_tensors="pt").to(device)
            with torch.no_grad():
                hidden = model(**batch).last_hidden_state
            outputs.append((torch.nn.functional.normalize(hidden, dim=-1).cpu(), batch["attention_mask"].cpu()))
        return outputs

    scores = []
    for (predicted, predicted_mask), (reference, reference_mask) in zip(embed(predictions), embed(references)):
        similarity = torch.bmm(predicted, reference.transpose(1, 2))
        valid = predicted_mask.unsqueeze(2) * reference_mask.unsqueeze(1)
        similarity = similarity.masked_fill(valid.eq(0), -1.0)
        precision = similarity.max(dim=2).values.mul(predicted_mask).sum(1) / predicted_mask.sum(1).clamp_min(1)
        recall = similarity.max(dim=1).values.mul(reference_mask).sum(1) / reference_mask.sum(1).clamp_min(1)
        scores.extend((2 * precision * recall / (precision + recall).clamp_min(1e-8)).tolist())
    return float(np.mean(scores)) * 100.0


def summarize(predictions: dict, rationales: tuple[list[str], list[str]] | None = None,
              counterfactuals: dict | None = None, bertscore_model: str = "roberta-large") -> dict[str, float]:
    """Assemble the six Table 4 columns from raw prediction arrays."""
    from .labels import CATEGORY_LABELS, HARMFULNESS_LABELS

    results = {
        "target_f1": target_f1(predictions["target_true"], predictions["target_pred"]),
        "harm_f1": macro_f1(predictions["harm_true"], predictions["harm_pred"]),
        "category_f1": category_f1(
            predictions["harm_true"], predictions["category_true"], predictions["category_pred"]
        ),
    }
    results.update(joint_f1(
        predictions["target_true"], predictions["harm_true"], predictions["category_true"],
        predictions["target_pred"], predictions["harm_pred"], predictions["category_pred"],
    ))
    harm_classes = per_class_f1(predictions["harm_true"], predictions["harm_pred"], len(HARMFULNESS_LABELS))
    results.update({f"harm_f1_{label}": value for label, value in zip(HARMFULNESS_LABELS, harm_classes)})
    harmful = [
        (c, p) for h, c, p in zip(predictions["harm_true"], predictions["category_true"], predictions["category_pred"])
        if h != NON_HARM_ID and c != IGNORE_INDEX
    ]
    if harmful:
        category_classes = per_class_f1([c for c, _ in harmful], [p for _, p in harmful], len(CATEGORY_LABELS))
        results.update({f"category_f1_{label}": value for label, value in zip(CATEGORY_LABELS, category_classes)})
    if rationales is not None:
        results["bertscore"] = bertscore(rationales[0], rationales[1], model_type=bertscore_model)
    if counterfactuals is not None:
        results.update(counterfactual_faithfulness(**counterfactuals))
    return results
