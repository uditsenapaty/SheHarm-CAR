#!/usr/bin/env python3
"""Merge an annotation shard produced on another machine into the main annotation file.

    python scripts/merge_annotations.py --into dataset/annotations_v2.csv \\
                                        --from dataset/annotations_part2.csv --apply

Validates before merging, because a shard produced with a stale copy of meme_annotator.py
would silently mix two different labelling policies into one dataset:

  * the header must be exactly the annotator schema
  * every filename must exist in dataset/images/
  * harm_type and harm_category must be in the taxonomy, and NULL must pair with Non-Harmful
  * overlapping filenames are reported; --prefer decides which side wins

Prints the label distribution of each side so a policy mismatch is visible immediately: if
the shard's category mix looks nothing like the local one, the shards were not produced by
the same prompt.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

FIELDNAMES = ["filename", "women-related target", "harm_type", "harm_category", "rationale"]
HARM_TYPES = {"Explicit-Harm", "Implicit-Harm", "Non-Harmful"}
CATEGORIES = {"Misogyny", "Sexual-Harassment", "Violence", "Appearance-Attack",
              "Character-Assassination", "NULL"}


def validate(frame: pd.DataFrame, name: str, images: Path) -> list[str]:
    problems = []
    if list(frame.columns) != FIELDNAMES:
        problems.append(f"{name}: header must be exactly {FIELDNAMES}, got {list(frame.columns)}")
        return problems
    missing = [f for f in frame["filename"] if not (images / str(f)).exists()]
    if missing:
        problems.append(f"{name}: {len(missing)} filenames not in {images} (e.g. {missing[:3]})")
    bad_harm = sorted(set(frame["harm_type"].astype(str)) - HARM_TYPES)
    if bad_harm:
        problems.append(f"{name}: invalid harm_type values {bad_harm}")
    bad_category = sorted(set(frame["harm_category"].astype(str)) - CATEGORIES)
    if bad_category:
        problems.append(f"{name}: invalid harm_category values {bad_category}")
    mismatched = frame[
        (frame["harm_type"].astype(str) == "Non-Harmful") ^ (frame["harm_category"].astype(str) == "NULL")
    ]
    if len(mismatched):
        problems.append(f"{name}: {len(mismatched)} rows where NULL and Non-Harmful disagree")
    if frame["filename"].duplicated().any():
        problems.append(f"{name}: {int(frame['filename'].duplicated().sum())} duplicate filenames")
    return problems


def distribution(frame: pd.DataFrame) -> str:
    harm = frame["harm_type"].value_counts(normalize=True).mul(100).round(1).to_dict()
    harmful = frame[frame["harm_type"] != "Non-Harmful"]
    category = harmful["harm_category"].value_counts(normalize=True).mul(100).round(1).to_dict() if len(harmful) else {}
    return f"    harm_type {harm}\n    category  {category}"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--into", type=Path, default=Path("dataset/annotations_v2.csv"))
    parser.add_argument("--from", dest="source", type=Path, required=True)
    parser.add_argument("--images", type=Path, default=Path("dataset/images"))
    parser.add_argument("--prefer", choices=["into", "from"], default="into",
                        help="Which side wins on overlapping filenames")
    parser.add_argument("--apply", action="store_true")
    args = parser.parse_args()

    incoming = pd.read_csv(args.source, keep_default_na=False)
    existing = pd.read_csv(args.into, keep_default_na=False) if args.into.exists() else pd.DataFrame(columns=FIELDNAMES)

    problems = validate(incoming, args.source.name, args.images)
    if args.into.exists():
        problems += validate(existing, args.into.name, args.images)
    if problems:
        for problem in problems:
            print(f"  PROBLEM  {problem}")
        raise SystemExit("refusing to merge: fix the problems above")

    overlap = set(existing["filename"]) & set(incoming["filename"])
    print(f"{args.into.name}: {len(existing)} rows")
    print(distribution(existing) if len(existing) else "    (empty)")
    print(f"{args.source.name}: {len(incoming)} rows")
    print(distribution(incoming))
    print(f"\noverlap: {len(overlap)} filenames (kept from '{args.prefer}')")

    frames = [existing, incoming] if args.prefer == "from" else [incoming, existing]
    merged = pd.concat(frames, ignore_index=True).drop_duplicates("filename", keep="last")
    merged["_id"] = merged["filename"].str.extract(r"img(\d+)").astype(int)
    merged = merged.sort_values("_id").drop(columns="_id")

    total = len(list(args.images.iterdir()))
    print(f"merged: {len(merged)} rows   coverage {len(merged)}/{total} images")
    gaps = sorted(set(p.name for p in args.images.iterdir()) - set(merged["filename"]))
    if gaps:
        print(f"still unannotated: {len(gaps)} (e.g. {gaps[:3]})")

    if not args.apply:
        print("\n(dry run — pass --apply to write)")
        return 0
    merged.to_csv(args.into, index=False)
    print(f"\nwrote {args.into}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
