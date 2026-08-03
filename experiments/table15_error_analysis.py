#!/usr/bin/env python3
"""Table `tab:qualitative_error_analysis` — qualitative error cases with symbolic evidence.

    python experiments/table15_error_analysis.py --checkpoint runs/default/seed42/best_model.pt

For every misclassified test instance it records the gold and predicted labels, the generated
rationale, the top retrieved ontology concepts, the highest-activation rules, and the
confidence gate, then groups them into the recurring failure patterns the paper discusses:
implicit stereotype read as non-harm, salient violence overshadowing misogyny, and lexical
over-triggering on benign context.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parent))

from common import (  # noqa: E402
    build_datasets, build_model, build_tokenizer_and_processor, load_config, resolve_device, write_results,
)
from sheharm.knowledge import load_knowledge  # noqa: E402
from sheharm.labels import CATEGORY_LABELS, HARMFULNESS_LABELS, IGNORE_INDEX, NON_HARM_ID  # noqa: E402
from sheharm.trainer import enable_determinism  # noqa: E402


def failure_pattern(gold_harm: int, predicted_harm: int, gold_category, predicted_category) -> str:
    if gold_harm != NON_HARM_ID and predicted_harm == NON_HARM_ID:
        return "implicit harm read as non-harm"
    if gold_harm == NON_HARM_ID and predicted_harm != NON_HARM_ID:
        return "benign context over-triggered as harm"
    if gold_harm != predicted_harm:
        return "explicit/implicit boundary confusion"
    if gold_category != predicted_category:
        return "correct harmfulness, wrong category"
    return "correct"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", type=Path, default=Path("configs/default.yaml"))
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--split", default="test", choices=["dev", "test"])
    parser.add_argument("--device", default=None)
    parser.add_argument("--max-cases", type=int, default=40)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    config = load_config(args.config)
    device = resolve_device(args.device)
    enable_determinism(args.seed)

    tokenizer, processor = build_tokenizer_and_processor(config)
    datasets, _ = build_datasets(config, tokenizer, processor)
    knowledge = load_knowledge(config["knowledge"]["ontology"], config["knowledge"]["rules"])
    concept_names = [concept["name"] for concept in knowledge.retrieval_concepts]
    rule_names = [rule["name"] for rule in knowledge.rules]

    model = build_model(config, tokenizer, device)
    model.load_state_dict(torch.load(args.checkpoint, map_location=device))
    model.eval()

    frame = datasets[args.split].frame
    loader = DataLoader(datasets[args.split], batch_size=8, shuffle=False, num_workers=0)

    cases, patterns, position = [], Counter(), 0
    with torch.no_grad():
        for batch in loader:
            tensors = {k: (v.to(device) if isinstance(v, torch.Tensor) else v) for k, v in batch.items()}
            output = model(
                pixel_values=tensors["pixel_values"], input_ids=tensors["input_ids"],
                attention_mask=tensors["attention_mask"], compute_counterfactuals=False,
            )
            harm_pred = output.harm_logits.argmax(dim=-1)
            category_pred = output.category_logits.argmax(dim=-1)
            generated = model.generate_rationale(
                output.extras["memory"], tokenizer.bos_token_id or tokenizer.cls_token_id,
                tokenizer.eos_token_id or tokenizer.sep_token_id,
            )
            rationales = tokenizer.batch_decode(generated, skip_special_tokens=True)

            for row in range(harm_pred.size(0)):
                record = frame.iloc[position]
                gold_harm = int(tensors["harm_labels"][row])
                gold_category = int(tensors["cat_labels"][row])
                pattern = failure_pattern(
                    gold_harm, int(harm_pred[row]),
                    None if gold_category == IGNORE_INDEX else gold_category,
                    None if int(harm_pred[row]) == NON_HARM_ID else int(category_pred[row]),
                )
                patterns[pattern] += 1
                if pattern != "correct" and len(cases) < args.max_cases:
                    activations = output.activations[row]
                    top_rules = activations.topk(min(3, activations.numel())).indices.tolist()
                    cases.append({
                        "image": record["image_path"],
                        "ocr_text": record["ocr_text"][:200],
                        "gold": f"{HARMFULNESS_LABELS[gold_harm]}/"
                                f"{CATEGORY_LABELS[gold_category] if gold_category != IGNORE_INDEX else 'None'}",
                        "predicted": f"{HARMFULNESS_LABELS[int(harm_pred[row])]}/"
                                     f"{CATEGORY_LABELS[int(category_pred[row])] if int(harm_pred[row]) != NON_HARM_ID else 'None'}",
                        "pattern": pattern,
                        "generated_rationale": rationales[row],
                        "top_concepts": [concept_names[i] for i in output.top_concepts[row].tolist()
                                         if i < len(concept_names)],
                        "top_rules": [{"rule": rule_names[i], "activation": round(float(activations[i]), 3)}
                                      for i in top_rules if i < len(rule_names)],
                        "confidence_gate": round(float(output.gamma[row]), 3),
                        "max_rule_activation": round(float(output.extras["max_activation"][row]), 3),
                    })
                position += 1

    total = sum(patterns.values())
    results = {
        "checkpoint": str(args.checkpoint), "split": args.split, "instances": total,
        "pattern_counts": dict(patterns.most_common()),
        "pattern_share_percent": {k: round(v / total * 100, 2) for k, v in patterns.most_common()},
        "cases": cases,
    }
    write_results("table15_error_analysis", results)
    Path("results/table15_error_cases.json").write_text(json.dumps(cases, indent=2), encoding="utf-8")

    print(f"\n{'pattern':40s} {'n':>6s} {'share':>7s}")
    for pattern, count in patterns.most_common():
        print(f"{pattern:40s} {count:6d} {count/total*100:6.1f}%")
    print(f"\n{len(cases)} qualitative cases written to results/table15_error_cases.json")
    return 0


if __name__ == "__main__":
    sys.exit(main())
