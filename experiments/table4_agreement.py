#!/usr/bin/env python3
"""Table `tab:annotation-agreement` — inter-annotator agreement.

    python experiments/table4_agreement.py --a dataset/annotations_v2.csv --b dataset/annotations_pass2.csv

Cohen's kappa for the women-related target, harmfulness and harm category; token-level F1 for
rationales. Requires two independent annotation passes over the same images. Produce the
second pass with a different annotator - a human, or a different model:

    python meme_annotator.py --output dataset/annotations_pass2.csv --model <other-model>

Reporting kappa between two runs of the *same* model measures decoding variance, not
annotator agreement, so the script refuses that case unless --allow-same-annotator is given.

A second mode compares the annotations against the trained model's own predictions:

    python experiments/table4_agreement.py --vs-model --checkpoint runs/default/seed42/best_model.pt

which measures how far SheHarm-CAR has moved from the Qwen2.5-VL labels it was fitted on.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd
from sklearn.metrics import cohen_kappa_score

sys.path.insert(0, str(Path(__file__).resolve().parent))

from common import write_latex, write_results  # noqa: E402


def token_f1(predicted: str, reference: str) -> float:
    predicted_tokens, reference_tokens = str(predicted).lower().split(), str(reference).lower().split()
    if not predicted_tokens or not reference_tokens:
        return 0.0
    common = 0
    remaining = list(reference_tokens)
    for token in predicted_tokens:
        if token in remaining:
            remaining.remove(token)
            common += 1
    if common == 0:
        return 0.0
    precision, recall = common / len(predicted_tokens), common / len(reference_tokens)
    return 2 * precision * recall / (precision + recall)


def model_agreement(args) -> int:
    """Cohen's kappa between the Qwen-VL annotations and SheHarm-CAR's own predictions.

    This is not inter-*annotator* agreement: it measures how far the trained model has moved
    from the labels it was fitted on. High kappa on the test split means the model reproduces
    the annotator's decision function; the disagreements are where the two differ, and are
    the interesting cases to inspect by hand.
    """
    import torch
    from torch.utils.data import DataLoader

    from common import (build_datasets, build_model, build_tokenizer_and_processor,
                        load_config, resolve_device)
    from sheharm.evaluate import predict
    from sheharm.labels import CATEGORY_LABELS, HARMFULNESS_LABELS, IGNORE_INDEX
    from sheharm.trainer import enable_determinism

    config = load_config(args.config)
    device = resolve_device(args.device)
    enable_determinism(args.seed)
    tokenizer, processor = build_tokenizer_and_processor(config)
    datasets, inventory = build_datasets(config, tokenizer, processor)
    model = build_model(config, tokenizer, device)
    model.load_state_dict(torch.load(args.checkpoint, map_location=device))
    model.eval()

    loader = DataLoader(datasets[args.split], batch_size=16, shuffle=False, num_workers=0)
    outputs = predict(model, loader, tokenizer, device,
                      compute_rationales=True, compute_counterfactuals=False)
    collected = outputs["predictions"]

    def kappa(truth, prediction, drop_ignore=True):
        pairs = [(t, p) for t, p in zip(truth, prediction) if not (drop_ignore and t == IGNORE_INDEX)]
        if len(pairs) < 2 or len({t for t, _ in pairs}) < 2:
            return float("nan")
        return float(cohen_kappa_score([t for t, _ in pairs], [p for _, p in pairs]))

    predicted_rationales, reference_rationales = outputs["rationales"]
    results = {
        "comparison": "Qwen2.5-VL annotation vs SheHarm-CAR prediction",
        "split": args.split, "checkpoint": str(args.checkpoint), "n": len(collected["harm_true"]),
        "target_kappa": kappa(collected["target_true"], collected["target_pred"]),
        "harmfulness_kappa": kappa(collected["harm_true"], collected["harm_pred"]),
        "category_kappa": kappa(collected["category_true"], collected["category_pred"]),
        "rationale_token_f1": float(
            pd.Series([token_f1(p, r) for p, r in zip(predicted_rationales, reference_rationales)]).mean() * 100),
        "raw_agreement_harmfulness": float(
            sum(t == p for t, p in zip(collected["harm_true"], collected["harm_pred"]))
            / max(len(collected["harm_true"]), 1) * 100),
    }
    write_results("table4_agreement_model", results)
    write_latex("table4_agreement_model", ["Annotation Component", "Agreement"], [
        ["Women-related Target Cohen's $\\kappa$", f"{results['target_kappa']:.2f}"],
        ["Harmfulness Cohen's $\\kappa$", f"{results['harmfulness_kappa']:.2f}"],
        ["Harm Category Cohen's $\\kappa$", f"{results['category_kappa']:.2f}"],
        ["Rationale Token-F1", f"{results['rationale_token_f1']:.1f}"],
    ], caption="Agreement between the Qwen2.5-VL annotations and SheHarm-CAR predictions.")
    for name, value in results.items():
        if isinstance(value, float):
            print(f"{name:28s} {value:.3f}")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--a", type=Path, help="First annotation pass")
    parser.add_argument("--b", type=Path, help="Second, independent annotation pass")
    parser.add_argument("--inventory", type=Path, default=Path("dataset/target_inventory.json"))
    parser.add_argument("--allow-same-annotator", action="store_true")
    # model-vs-annotator mode
    parser.add_argument("--vs-model", action="store_true",
                        help="Compare the annotations against SheHarm-CAR's predictions instead")
    parser.add_argument("--config", type=Path, default=Path("configs/default.yaml"))
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--split", default="test", choices=["dev", "test"])
    parser.add_argument("--device", default=None)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    if args.vs_model:
        if not args.checkpoint:
            raise SystemExit("--vs-model needs --checkpoint")
        return model_agreement(args)
    if not (args.a and args.b):
        raise SystemExit("give --a and --b for two annotation passes, or --vs-model --checkpoint ...")

    first = pd.read_csv(args.a).set_index("filename")
    second = pd.read_csv(args.b).set_index("filename")
    shared = first.index.intersection(second.index)
    if len(shared) == 0:
        raise SystemExit("the two passes share no filenames")
    first, second = first.loc[shared], second.loc[shared]

    if first.equals(second) and not args.allow_same_annotator:
        raise SystemExit("the two files are identical — this measures nothing. Produce a genuinely "
                         "independent second pass, or pass --allow-same-annotator.")

    # Targets are compared after canonicalisation: free text would understate agreement.
    if args.inventory.exists():
        alias = json.loads(args.inventory.read_text(encoding="utf-8"))["alias_map"]
        target_a = first["women-related target"].astype(str).str.lower().str.strip().map(alias).fillna("other")
        target_b = second["women-related target"].astype(str).str.lower().str.strip().map(alias).fillna("other")
    else:
        target_a = first["women-related target"].astype(str).str.lower().str.strip()
        target_b = second["women-related target"].astype(str).str.lower().str.strip()

    results = {
        "n": int(len(shared)),
        "target_kappa": float(cohen_kappa_score(target_a, target_b)),
        "harmfulness_kappa": float(cohen_kappa_score(first["harm_type"], second["harm_type"])),
        "category_kappa": float(cohen_kappa_score(
            first["harm_category"].fillna("NULL"), second["harm_category"].fillna("NULL"))),
        "rationale_token_f1": float(
            pd.Series([token_f1(p, r) for p, r in zip(first["rationale"], second["rationale"])]).mean() * 100
        ),
        "raw_agreement_harmfulness": float((first["harm_type"] == second["harm_type"]).mean() * 100),
        "sources": [str(args.a), str(args.b)],
    }

    write_results("table4_agreement", results)
    write_latex("table4_agreement", ["Annotation Component", "Agreement"], [
        ["Women-related Target Cohen's $\\kappa$", f"{results['target_kappa']:.2f}"],
        ["Harmfulness Cohen's $\\kappa$", f"{results['harmfulness_kappa']:.2f}"],
        ["Harm Category Cohen's $\\kappa$", f"{results['category_kappa']:.2f}"],
        ["Rationale Token-F1", f"{results['rationale_token_f1']:.1f}"],
    ], caption="Inter-annotator agreement before expert adjudication.")

    for name, value in results.items():
        if isinstance(value, float):
            print(f"{name:28s} {value:.3f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
