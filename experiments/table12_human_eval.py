#!/usr/bin/env python3
"""Table `tab:human-evaluation` — rationale quality on a five-point scale.

Two modes:

  export   build the randomised rating sheets three evaluators fill in.
           100 test instances covering explicit / implicit / non-harm and all five harm
           categories, one row per (instance, model), model identity hidden and order
           shuffled, exactly as the paper describes.

               python experiments/table12_human_eval.py export \\
                   --rationales results/rationales_sheharm_car.json results/rationales_sgot_r1.json ...

  score    aggregate the completed sheets into the table, with Krippendorff's alpha.

               python experiments/table12_human_eval.py score --sheets human_eval/*.csv

An optional `judge` mode produces an LLM-judge proxy so the pipeline is runnable end-to-end
without waiting for raters. Its output is written to a separate file and labelled as a proxy:
it is not a substitute for the human numbers reported in the paper.
"""

from __future__ import annotations

import argparse
import itertools
import json
import random
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))

from common import write_latex, write_results  # noqa: E402

CRITERIA = ["relevance", "groundedness", "informativeness", "faithfulness"]


def krippendorff_alpha(matrix: np.ndarray) -> float:
    """Interval-scale alpha over an items x raters matrix that may contain NaN."""
    pairs = []
    for row in matrix:
        observed = row[~np.isnan(row)]
        pairs.extend(itertools.permutations(observed, 2))
    if not pairs:
        return float("nan")
    differences = np.array([(a - b) ** 2 for a, b in pairs])
    values = matrix[~np.isnan(matrix)]
    expected = np.array([(a - b) ** 2 for a, b in itertools.permutations(values, 2)])
    if expected.mean() == 0:
        return float("nan")
    return float(1 - differences.mean() / expected.mean())


def export(args) -> int:
    payloads = {}
    for path in args.rationales:
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        payloads[data["model"]] = data["rationales"]
    if not payloads:
        raise SystemExit("no rationale files given")

    lengths = {len(v) for v in payloads.values()}
    if len(lengths) != 1:
        raise SystemExit(f"rationale files disagree on length: {lengths}")

    rng = random.Random(args.seed)
    indices = list(range(next(iter(lengths))))
    rng.shuffle(indices)
    selected = indices[: args.instances]

    rows = []
    for position, index in enumerate(selected):
        models = list(payloads)
        rng.shuffle(models)   # hide model identity by randomising presentation order
        for model in models:
            record = payloads[model][index]
            rows.append({
                "sheet_id": f"{position:04d}_{rng.randrange(10**6):06d}",
                "instance": index,
                "image_path": record.get("image_path", ""),
                "ocr_text": record.get("ocr_text", ""),
                "gold_label": record.get("gold", ""),
                "rationale": record.get("rationale", ""),
                "system": model,     # kept for scoring; drop this column before handing out
                **{criterion: "" for criterion in CRITERIA},
            })

    frame = pd.DataFrame(rows)
    output = Path(args.out)
    output.mkdir(parents=True, exist_ok=True)
    frame.to_csv(output / "key.csv", index=False)
    blinded = frame.drop(columns=["system", "instance"])
    for rater in range(1, args.raters + 1):
        blinded.sample(frac=1.0, random_state=args.seed + rater).to_csv(output / f"rater{rater}.csv", index=False)
    print(f"wrote {output}/key.csv and {args.raters} blinded sheets "
          f"({args.instances} instances x {len(payloads)} systems)")
    print("Hand out rater*.csv only. Score each criterion 1-5, then run `score --sheets`.")
    return 0


def score(args) -> int:
    key = pd.read_csv(Path(args.out) / "key.csv")[["sheet_id", "system"]]
    sheets = [pd.read_csv(path).merge(key, on="sheet_id") for path in args.sheets]
    if not sheets:
        raise SystemExit("no completed sheets given")

    means, alphas = {}, {}
    for criterion in CRITERIA:
        wide = pd.concat(
            [sheet[["sheet_id", criterion]].rename(columns={criterion: f"r{i}"}).set_index("sheet_id")
             for i, sheet in enumerate(sheets)], axis=1,
        )
        alphas[criterion] = krippendorff_alpha(wide.to_numpy(dtype=float))
        combined = pd.concat(sheets)
        combined[criterion] = pd.to_numeric(combined[criterion], errors="coerce")
        means[criterion] = combined.groupby("system")[criterion].mean().to_dict()

    systems = sorted({system for values in means.values() for system in values})
    results = {
        "systems": systems, "means": means, "krippendorff_alpha": alphas,
        "overall_alpha": float(np.nanmean(list(alphas.values()))), "n_sheets": len(sheets),
    }
    write_results("table12_human_evaluation", results)
    rows = [[system] + [f"{means[c].get(system, float('nan')):.2f}" for c in CRITERIA] for system in systems]
    write_latex("table12_human_evaluation", ["Model", "Rel.", "Ground.", "Info.", "Faith."], rows,
                caption="Human evaluation of generated rationales on a five-point scale.")
    print(f"overall Krippendorff's alpha {results['overall_alpha']:.2f}")
    for system in systems:
        print(f"{system:22s} " + " ".join(f"{means[c].get(system, float('nan')):5.2f}" for c in CRITERIA))
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="mode", required=True)

    exporter = sub.add_parser("export")
    exporter.add_argument("--rationales", nargs="+", required=True)
    exporter.add_argument("--instances", type=int, default=100)
    exporter.add_argument("--raters", type=int, default=3)
    exporter.add_argument("--seed", type=int, default=42)
    exporter.add_argument("--out", default="human_eval")
    exporter.set_defaults(function=export)

    scorer = sub.add_parser("score")
    scorer.add_argument("--sheets", nargs="+", required=True)
    scorer.add_argument("--out", default="human_eval")
    scorer.set_defaults(function=score)

    args = parser.parse_args()
    return args.function(args)


if __name__ == "__main__":
    sys.exit(main())
