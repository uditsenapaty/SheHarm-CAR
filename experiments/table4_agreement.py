#!/usr/bin/env python3
"""Table `tab:annotation-agreement` — inter-annotator agreement.

    python experiments/table4_agreement.py --a dataset/annotations_v2.csv --b dataset/annotations_pass2.csv

Cohen's kappa for the women-related target, harmfulness and harm category; token-level F1 for
rationales. Requires two independent annotation passes over the same images. Produce the
second pass with a different annotator - a human, or a different model:

    python meme_annotator.py --output dataset/annotations_pass2.csv --model <other-model>

Reporting kappa between two runs of the *same* model measures decoding variance, not
annotator agreement, so the script refuses that case unless --allow-same-annotator is given.
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


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--a", type=Path, required=True, help="First annotation pass")
    parser.add_argument("--b", type=Path, required=True, help="Second, independent annotation pass")
    parser.add_argument("--inventory", type=Path, default=Path("dataset/target_inventory.json"))
    parser.add_argument("--allow-same-annotator", action="store_true")
    args = parser.parse_args()

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
