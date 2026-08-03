#!/usr/bin/env python3
"""Assemble the model-facing SheHarm-Meme CSV and the split table.

Joins the annotation pass, the OCR pass, and the canonical target inventory; normalises the
label surface forms to the paper's (`Non-Harmful` -> `Non-Harm`, `NULL` -> `None`); assigns
an 80/10/10 split stratified jointly over harmfulness and harm category; and emits
Table `tab:data-splits`.

Output columns:
    image_path, ocr_text, target_span, target_start, target_end, target_concept,
    harmfulness, harm_category, rationale, split
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from sheharm.data.splits import assign_splits, split_report  # noqa: E402
from sheharm.labels import CATEGORY_LABELS, NULL_CATEGORY, normalize_category, normalize_harmfulness  # noqa: E402

COLUMNS = [
    "image_path", "ocr_text", "target_span", "target_start", "target_end",
    "target_concept", "harmfulness", "harm_category", "rationale", "split",
]


def latex_table(table: pd.DataFrame) -> str:
    header = "\\textbf{Split} & \\textbf{Exp.} & \\textbf{Imp.} & \\textbf{Non.} & \\textbf{Total} \\\\"
    pretty = {"train": "Train", "dev": "Dev", "test": "Test", "Total": "\\textbf{Total}"}
    lines = []
    for name, row in table.iterrows():
        values = " & ".join(f"{int(value):,}" for value in row)
        if name == "Total":
            values = " & ".join(f"\\textbf{{{int(value):,}}}" for value in row)
        lines.append(f"{pretty.get(name, name)} & {values} \\\\")
    body = "\n".join(lines[:-1]) + "\n\\midrule\n" + lines[-1]
    return (
        "\\begin{tabular}{lrrrr}\n\\toprule\n" + header + "\n\\midrule\n" + body + "\n\\bottomrule\n\\end{tabular}\n"
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--annotations", type=Path, default=Path("dataset/annotations_v2.csv"))
    parser.add_argument("--ocr", type=Path, default=Path("dataset/ocr.csv"))
    parser.add_argument("--inventory", type=Path, default=Path("dataset/target_inventory.json"))
    parser.add_argument("--images-dir", type=Path, default=Path("dataset/images"))
    parser.add_argument("--output", type=Path, default=Path("dataset/sheharm_meme.csv"))
    parser.add_argument("--results-dir", type=Path, default=Path("results"))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--train-ratio", type=float, default=0.8)
    parser.add_argument("--dev-ratio", type=float, default=0.1)
    parser.add_argument("--allow-missing-ocr", action="store_true",
                        help="Keep rows that have no OCR row yet (for partial smoke runs)")
    parser.add_argument("--prior-labels", type=Path, default=Path("dataset/mami_sublabels.csv"),
                        help="Rows that arrive with their own harmfulness/category labels")
    args = parser.parse_args()

    annotations = pd.read_csv(args.annotations)
    inventory = json.loads(args.inventory.read_text(encoding="utf-8"))
    alias_map, no_target = inventory["alias_map"], inventory["no_target_label"]

    if args.ocr.exists():
        ocr = pd.read_csv(args.ocr, keep_default_na=False)
    elif args.allow_missing_ocr:
        ocr = pd.DataFrame(columns=["filename", "ocr_text", "target_span"])
    else:
        raise SystemExit(f"{args.ocr} not found. Run scripts/ocr_and_span.py first, or pass --allow-missing-ocr.")

    # Part of the corpus arrives with its own harmfulness and category labels. Those are
    # kept as-is; annotator 1 still supplies the women-related target and the rationale,
    # which the imported labels do not carry.
    prior = {}
    if args.prior_labels.exists():
        for _, row in pd.read_csv(args.prior_labels).iterrows():
            direct = any(int(row[k]) for k in ("shaming", "objectification", "violence"))
            prior[row["filename"]] = {
                "harmfulness": "Explicit-Harm" if direct else "Implicit-Harm",
                "harm_category": row["category"],
            }
        print(f"{len(prior)} rows carry pre-existing harmfulness/category labels")

    merged = annotations.merge(ocr, on="filename", how="left")
    before = len(merged)
    if not args.allow_missing_ocr:
        merged = merged[merged["ocr_text"].notna()]
        if len(merged) < before:
            print(f"dropped {before - len(merged)} rows without OCR")
    merged["ocr_text"] = merged["ocr_text"].fillna("")
    merged["target_span"] = merged.get("target_span", pd.Series(dtype=str)).fillna("")

    rows = []
    for _, row in merged.iterrows():
        if row["filename"] in prior:
            harmfulness = prior[row["filename"]]["harmfulness"]
            category = prior[row["filename"]]["harm_category"]
        else:
            harmfulness = normalize_harmfulness(row["harm_type"])
            category = normalize_category(row["harm_category"], harmfulness)
        raw_target = str(row["women-related target"]).lower().strip()
        concept = alias_map.get(raw_target, no_target)

        ocr_text, span = str(row["ocr_text"]), str(row["target_span"])
        start = ocr_text.find(span) if span else -1
        end = start + len(span) if start >= 0 else -1

        image_path = row["filename"]
        if not (args.images_dir / image_path).exists():
            raise SystemExit(f"image referenced by annotations is missing: {image_path}")

        rows.append({
            "image_path": image_path, "ocr_text": ocr_text, "target_span": span,
            "target_start": start, "target_end": end, "target_concept": concept,
            "harmfulness": harmfulness, "harm_category": category,
            "rationale": str(row["rationale"]), "split": "",
        })

    frame = pd.DataFrame(rows)
    splits, split_notes = assign_splits(frame, args.train_ratio, args.dev_ratio, args.seed)
    frame["split"] = splits.values
    frame = frame[COLUMNS]

    args.output.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(args.output, index=False)

    table = split_report(frame)
    args.results_dir.mkdir(parents=True, exist_ok=True)
    (args.results_dir / "table3_data_splits.tex").write_text(latex_table(table), encoding="utf-8")

    category_table = pd.crosstab(frame["split"], frame["harm_category"])
    grounded = int((frame["target_start"] >= 0).sum())
    statistics = {
        "total": int(len(frame)),
        "splits": {name: int(count) for name, count in frame["split"].value_counts().items()},
        "harmfulness": {name: int(count) for name, count in frame["harmfulness"].value_counts().items()},
        "harm_category": {name: int(count) for name, count in frame["harm_category"].value_counts().items()},
        "split_by_harmfulness": table.to_dict(),
        "split_by_category": category_table.to_dict(),
        "target_concepts_attested": int(frame.loc[frame.target_concept != no_target, "target_concept"].nunique()),
        "rows_without_target_concept": int((frame["target_concept"] == no_target).sum()),
        "rows_with_grounded_span": grounded,
        "span_grounding_rate": round(grounded / max(len(frame), 1) * 100, 2),
        "rows_with_empty_ocr": int((frame["ocr_text"].str.len() == 0).sum()),
        "rows_with_prior_labels": int(sum(1 for r in rows if r["image_path"] in prior)),
        **split_notes,
    }
    (args.results_dir / "table3_data_splits.json").write_text(json.dumps(statistics, indent=2), encoding="utf-8")

    print(f"\nwrote {args.output} ({len(frame)} rows)\n")
    print(table.to_string())
    print("\nharm category by split:")
    print(category_table.reindex(columns=[c for c in CATEGORY_LABELS + [NULL_CATEGORY] if c in category_table.columns]).to_string())
    if split_notes["strata_too_small_for_dev_test"]:
        print(f"\nstrata too small to reach dev/test (train only): {split_notes['strata_too_small_for_dev_test']}")
    print(f"\ntarget concepts attested: {statistics['target_concepts_attested']}"
          f" | rows without target: {statistics['rows_without_target_concept']}"
          f" | span grounded: {statistics['span_grounding_rate']}%")
    return 0


if __name__ == "__main__":
    sys.exit(main())
