#!/usr/bin/env python3
"""Evaluate a prompted vision-language baseline. Independently runnable.

    python experiments/run_prompted_baseline.py --model qwen25_vl --limit 200

Prompted models are not trained, so there is no seed averaging: one deterministic greedy pass
over the split. Free-text targets are mapped through the same canonical inventory the trained
models are scored against, otherwise Tgt-F1 would not be comparable.

CF-Faith uses the identical text intervention applied to every model: the grounded
women-related target span is removed for the relevant condition, and an equally long distant
span is removed for the irrelevant control.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent))

from common import build_datasets, build_tokenizer_and_processor, load_config, resolve_device, write_results  # noqa: E402
from sheharm import metrics as metric_lib  # noqa: E402
from sheharm.baselines.prompted_vlm import PromptedVLM  # noqa: E402
from sheharm.data.counterfactuals import mask_distant_span, mask_span  # noqa: E402
from sheharm.labels import CAT2ID, HARM2ID, IGNORE_INDEX, NULL_CATEGORY  # noqa: E402


def load_images(frame, image_root: Path, indices):
    return [Image.open(image_root / frame.iloc[i]["image_path"]).convert("RGB") for i in indices]


def evaluate_prompted(key: str, config: dict, device, limit: int | None = None,
                      split: str = "test", batch_size: int = 4, cf_limit: int | None = 200) -> dict:
    import json

    tokenizer, processor = build_tokenizer_and_processor(config)
    datasets, inventory = build_datasets(config, tokenizer, processor)
    frame = datasets[split].frame
    if limit:
        frame = frame.head(limit)
    alias_map = json.loads(Path(config["data"]["target_inventory"]).read_text(encoding="utf-8"))["alias_map"]
    target2id = {name: index for index, name in enumerate(inventory["concepts"])}
    image_root = Path(config["data"]["image_root"])

    model = PromptedVLM(key, device=str(device))
    collected = {k: [] for k in
                 ("target_true", "target_pred", "harm_true", "harm_pred", "category_true", "category_pred")}
    rationales_predicted, rationales_reference = [], []

    for start in range(0, len(frame), batch_size):
        indices = range(start, min(start + batch_size, len(frame)))
        rows = [frame.iloc[i] for i in indices]
        images = load_images(frame, image_root, indices)
        predictions = model.generate(images, [str(row["ocr_text"]) for row in rows])

        for row, prediction in zip(rows, predictions):
            canonical = alias_map.get(prediction["women_related_target"], None)
            collected["target_pred"].append(target2id.get(canonical, -1) if canonical else -1)
            collected["target_true"].append(target2id.get(str(row["target_concept"]), IGNORE_INDEX))
            collected["harm_pred"].append(HARM2ID[prediction["harm_type"]])
            collected["harm_true"].append(HARM2ID[str(row["harmfulness"])])
            collected["category_pred"].append(CAT2ID.get(prediction["harm_category"], 0))
            category = str(row["harm_category"])
            collected["category_true"].append(IGNORE_INDEX if category == NULL_CATEGORY else CAT2ID[category])
            rationales_predicted.append(prediction["rationale"])
            rationales_reference.append(str(row["rationale"]))
        for image in images:
            image.close()
        print(f"  [{min(start + batch_size, len(frame))}/{len(frame)}]", flush=True)

    # CF-Faith on the subset whose target span is grounded in the OCR text.
    grounded = frame[frame["target_start"] >= 0]
    if cf_limit:
        grounded = grounded.head(cf_limit)
    original_confidence, relevant_confidence = [], []
    original_prediction, irrelevant_prediction = [], []
    for start in range(0, len(grounded), batch_size):
        indices = range(start, min(start + batch_size, len(grounded)))
        rows = [grounded.iloc[i] for i in indices]
        images = load_images(grounded, image_root, indices)
        texts = [str(row["ocr_text"]) for row in rows]
        relevant = [mask_span(t, int(r["target_start"]), int(r["target_end"])) for t, r in zip(texts, rows)]
        irrelevant = [mask_distant_span(t, int(r["target_start"]), int(r["target_end"])) for t, r in zip(texts, rows)]

        base = model.harm_distribution(images, texts)
        predicted = base.argmax(dim=-1)
        original_confidence.extend(base.gather(1, predicted.unsqueeze(1)).squeeze(1).cpu().tolist())
        original_prediction.extend(predicted.cpu().tolist())
        relevant_confidence.extend(
            model.harm_distribution(images, relevant).gather(1, predicted.unsqueeze(1)).squeeze(1).cpu().tolist()
        )
        irrelevant_prediction.extend(model.harm_distribution(images, irrelevant).argmax(dim=-1).cpu().tolist())
        for image in images:
            image.close()

    model.release()

    metrics = metric_lib.summarize(
        collected,
        rationales=(rationales_predicted, rationales_reference),
        counterfactuals={
            "original_confidence": np.array(original_confidence),
            "relevant_removed_confidence": np.array(relevant_confidence),
            "original_prediction": np.array(original_prediction),
            "irrelevant_removed_prediction": np.array(irrelevant_prediction),
        } if original_confidence else None,
        bertscore_model=config["eval"]["bertscore_model"],
    )
    return {
        "summary": {name: {"mean": value, "std": 0.0, "n": 1} for name, value in metrics.items()},
        "per_seed": [{"seed": None, "metrics": metrics}],
        "instances": int(len(frame)), "cf_instances": int(len(grounded)),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", type=Path, default=Path("configs/default.yaml"))
    parser.add_argument("--model", required=True, choices=["llava", "internvl", "llama32_vision", "qwen25_vl"])
    parser.add_argument("--split", default="test", choices=["dev", "test"])
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--cf-limit", type=int, default=200)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--device", default=None)
    args = parser.parse_args()

    config = load_config(args.config)
    result = evaluate_prompted(
        args.model, config, resolve_device(args.device), limit=args.limit,
        split=args.split, batch_size=args.batch_size, cf_limit=args.cf_limit,
    )
    write_results(f"prompted_{args.model}", result)
    for name, value in result["per_seed"][0]["metrics"].items():
        if not isinstance(value, float) or value == value:
            print(f"{name:24s} {value}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
