"""Inference and evaluation.

At inference only the meme image and OCR text are provided: no gold target, harmfulness,
category, or rationale is used (paper Section `subsec:joint-training`).
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F

from . import metrics as metric_lib
from .decoding import decode
from .labels import IGNORE_INDEX

# CF-Faith appears in Table 4 for every baseline too, so the intervention used to compute it
# must be one that any image+text model supports. Evidence masking qualifies; suppressing the
# soft target representation does not (baselines have none). The target/concept/rule
# interventions are reported separately for our model in Table 7.
RELEVANT_INTERVENTION = "mask_relevant_region"
IRRELEVANT_INTERVENTION = "mask_irrelevant_region"


@torch.no_grad()
def predict(model, loader, tokenizer, device, compute_rationales: bool = True,
            compute_counterfactuals: bool = True, joint_decoding: bool = True) -> dict:
    model.eval()
    collected = {
        "target_true": [], "target_pred": [], "harm_true": [], "harm_pred": [],
        "category_true": [], "category_pred": [], "gamma": [], "max_activation": [],
    }
    rationales_predicted, rationales_reference = [], []
    original_confidence, relevant_confidence = [], []
    original_prediction, irrelevant_prediction = [], []

    for batch in loader:
        batch = {
            key: (value.to(device) if isinstance(value, torch.Tensor) else value)
            for key, value in batch.items()
        }
        # Labels are passed only so the model returns them alongside predictions; the
        # forward pass never conditions on them.
        output = model(
            pixel_values=batch["pixel_values"],
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
            compute_counterfactuals=False,
        )
        harm_probabilities = F.softmax(output.harm_logits, dim=-1)
        harm_prediction, category_prediction = decode(
            output.harm_logits, output.category_logits, joint=joint_decoding
        )

        collected["target_true"].extend(batch["target_labels"].cpu().tolist())
        collected["target_pred"].extend(output.target_logits.argmax(dim=-1).cpu().tolist())
        collected["harm_true"].extend(batch["harm_labels"].cpu().tolist())
        collected["harm_pred"].extend(harm_prediction.cpu().tolist())
        collected["category_true"].extend(batch["cat_labels"].cpu().tolist())
        collected["category_pred"].extend(category_prediction.cpu().tolist())
        collected["gamma"].extend(output.gamma.cpu().tolist())
        collected["max_activation"].extend(output.extras["max_activation"].cpu().tolist())

        if compute_rationales:
            generated = model.generate_rationale(
                output.extras["memory"], tokenizer.bos_token_id or tokenizer.cls_token_id,
                tokenizer.eos_token_id or tokenizer.sep_token_id,
            )
            rationales_predicted.extend(tokenizer.batch_decode(generated, skip_special_tokens=True))
            rationales_reference.extend(tokenizer.batch_decode(batch["rationale_ids"], skip_special_tokens=True))

        if compute_counterfactuals:
            predicted_index = harm_prediction.unsqueeze(1)
            original_confidence.extend(harm_probabilities.gather(1, predicted_index).squeeze(1).cpu().tolist())
            original_prediction.extend(harm_prediction.cpu().tolist())
            relevant = model.intervene(output, batch["attention_mask"], RELEVANT_INTERVENTION)
            irrelevant = model.intervene(output, batch["attention_mask"], IRRELEVANT_INTERVENTION)
            relevant_confidence.extend(
                F.softmax(relevant["harm_logits"], dim=-1).gather(1, predicted_index).squeeze(1).cpu().tolist()
            )
            irrelevant_prediction.extend(irrelevant["harm_logits"].argmax(dim=-1).cpu().tolist())

    result = {"predictions": collected}
    if compute_rationales:
        result["rationales"] = (rationales_predicted, rationales_reference)
    if compute_counterfactuals:
        result["counterfactuals"] = {
            "original_confidence": np.array(original_confidence),
            "relevant_removed_confidence": np.array(relevant_confidence),
            "original_prediction": np.array(original_prediction),
            "irrelevant_removed_prediction": np.array(irrelevant_prediction),
        }
    return result


def evaluate(model, loader, tokenizer, device, compute_bertscore: bool = True,
             compute_counterfactuals: bool = True, bertscore_model: str = "roberta-large",
             joint_decoding: bool = True) -> dict:
    outputs = predict(
        model, loader, tokenizer, device,
        compute_rationales=compute_bertscore, compute_counterfactuals=compute_counterfactuals,
        joint_decoding=joint_decoding,
    )
    return metric_lib.summarize(
        outputs["predictions"],
        rationales=outputs.get("rationales") if compute_bertscore else None,
        counterfactuals=outputs.get("counterfactuals"),
        bertscore_model=bertscore_model,
    )


@torch.no_grad()
def intervention_analysis(model, loader, device, interventions: list[str]) -> dict[str, dict[str, float]]:
    """Table `tab:counterfactual-analysis`: mean confidence drop and flip rate per intervention."""
    model.eval()
    drops = {name: [] for name in interventions}
    flips = {name: [] for name in interventions}
    for batch in loader:
        batch = {
            key: (value.to(device) if isinstance(value, torch.Tensor) else value)
            for key, value in batch.items()
        }
        output = model(
            pixel_values=batch["pixel_values"], input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"], compute_counterfactuals=False,
        )
        for name in interventions:
            result = model.intervene(output, batch["attention_mask"], name)
            drops[name].extend(result["confidence_drop"].cpu().tolist())
            flips[name].extend(result["flipped"].cpu().tolist())
    return {
        name: {
            "delta_confidence": float(np.mean(drops[name])) * 100.0,
            "flip_rate": float(np.mean(flips[name])) * 100.0,
        }
        for name in interventions
    }
