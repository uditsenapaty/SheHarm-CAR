#!/usr/bin/env python3
"""Rebalance SheHarm-Meme: drop non-harmful memes, import the MAMI harmful set.

The corpus was ~75% Non-Harm, which is the inverse of the paper's distribution and starves
every harmful class. This removes `--drop` non-harmful memes and imports the supplied
MAMI (SemEval-2022 Task 5) misogynous subset, which ships human sub-labels *and* official
transcriptions, so the imported rows need no OCR pass.

Selection order for removal (least useful first, so the surviving negatives stay hard):
    1. Non-Harm rows whose annotated target is not women-related at all ("man", "cat", ...).
       The paper wants same-entity hard negatives, not generic humour.
    2. Non-Harm rows with no OCR text (nothing for the text encoder to use).
    3. A seeded random sample of the remainder.

Everything is renumbered to a contiguous img00001..imgNNNNN afterwards, and the filename
keys inside annotations and OCR are rewritten through the same mapping so no prior work is
lost. `mapping.csv` records old -> new for traceability.
"""

from __future__ import annotations

import argparse
import csv
import json
import random
import re
import shutil
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# MAMI sub-labels -> our five categories. Ordered by specificity: violence is the most
# specific signal, stereotype the most generic, so a meme flagged both lands on violence.
MAMI_CATEGORY_PRECEDENCE = [
    ("violence", "Violence"),
    ("objectification", "Sexual-Harassment"),
    ("shaming", "Appearance-Attack"),
    ("stereotype", "Misogyny"),
]
WOMAN_MARKERS = ("wom", "fem", "girl", "lad", "wife", "wive", "mother", "mom", "daughter",
                 "sister", "aunt", "niece", "grandma", "bride", "she", "her", "nurse",
                 "actress", "waitress", "queen", "princess", "feminist", "pregnant")


def is_women_related(target: str) -> bool:
    lowered = str(target).lower()
    return any(marker in lowered for marker in WOMAN_MARKERS)


def numeric_id(name: str) -> int:
    return int(re.match(r"img(\d+)", name).group(1))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--source", type=Path, default=Path("WomenHarmfulOnly1000"))
    parser.add_argument("--images", type=Path, default=Path("dataset/images"))
    parser.add_argument("--labels-for-selection", type=Path, default=Path("annotations.csv"),
                        help="Used only to decide WHICH memes to drop, never as final labels")
    parser.add_argument("--annotations", type=Path, default=Path("dataset/annotations_v2.csv"))
    parser.add_argument("--ocr", type=Path, default=Path("dataset/ocr.csv"))
    parser.add_argument("--drop", type=int, default=1500)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--apply", action="store_true", help="Without this the script only reports")
    args = parser.parse_args()

    images = {p.name: p for p in args.images.iterdir() if p.is_file()}
    selection = pd.read_csv(args.labels_for_selection)
    ocr = pd.read_csv(args.ocr, keep_default_na=False) if args.ocr.exists() else pd.DataFrame(columns=["filename", "ocr_text", "target_span"])
    ocr_text = dict(zip(ocr["filename"], ocr["ocr_text"]))

    non_harm = selection[selection["harm_type"].astype(str).str.strip() == "Non-Harmful"]
    non_harm = non_harm[non_harm["filename"].isin(images)]

    tier1, tier2, tier3 = [], [], []
    for _, row in non_harm.iterrows():
        name = row["filename"]
        if not is_women_related(row["women-related target"]):
            tier1.append(name)
        elif name in ocr_text and not str(ocr_text[name]).strip():
            # Only memes that were actually transcribed and came back empty. A missing OCR
            # row means "not reached yet" — and since OCR runs in index order, treating that
            # as "no text" would systematically delete the high-numbered half of the corpus.
            tier2.append(name)
        else:
            tier3.append(name)
    rng = random.Random(args.seed)
    rng.shuffle(tier2)
    rng.shuffle(tier3)
    ordered = tier1 + tier2 + tier3
    to_drop = ordered[: args.drop]

    imported = pd.read_csv(args.source / "annotations.tsv", sep="\t")
    source_images = {p.name: p for p in (args.source / "mami_train_harmful_1000").iterdir() if p.is_file()}
    imported = imported[imported["file_name"].isin(source_images)]

    print(f"existing images            : {len(images)}")
    print(f"non-harm candidates        : {len(non_harm)}")
    print(f"  tier1 target not a woman : {len(tier1)}")
    print(f"  tier2 no OCR text        : {len(tier2)}")
    print(f"  tier3 remainder          : {len(tier3)}")
    print(f"to drop                    : {len(to_drop)}")
    if len(to_drop) < args.drop:
        print(f"  WARNING only {len(to_drop)} candidates available, wanted {args.drop}")
    print(f"to import (MAMI harmful)   : {len(imported)}")
    print(f"resulting corpus           : {len(images) - len(to_drop) + len(imported)}")

    counts = {name: int(imported[key].sum()) for key, name in MAMI_CATEGORY_PRECEDENCE}
    print(f"\nMAMI sub-label totals (multi-label): {counts}")
    assigned = imported.apply(
        lambda row: next((name for key, name in MAMI_CATEGORY_PRECEDENCE if row[key] == 1), "Misogyny"), axis=1
    )
    print(f"after precedence collapse         : {assigned.value_counts().to_dict()}")

    if not args.apply:
        print("\n(dry run — pass --apply to execute)")
        return 0

    # ---- execute -----------------------------------------------------------
    backup = Path("dataset/_backup")
    backup.mkdir(parents=True, exist_ok=True)
    for path in (args.annotations, args.ocr):
        if path.exists():
            shutil.copy2(path, backup / f"pre_swap_{path.name}")

    # Moved aside rather than deleted: the images are gitignored, so an unlink would be the
    # only copy gone. dataset/_removed/ is also gitignored and can be emptied at any time.
    removed_dir = Path("dataset/_removed")
    removed_dir.mkdir(parents=True, exist_ok=True)
    dropped = set(to_drop)
    for name in dropped:
        shutil.move(str(images[name]), str(removed_dir / name))
    survivors = sorted((n for n in images if n not in dropped), key=numeric_id)

    # Renumber survivors first (order preserved), then append the imported set.
    mapping, rows_out = [], []
    index = 0
    for name in survivors:
        index += 1
        new = f"img{index:05d}{Path(name).suffix.lower()}"
        mapping.append({"old": name, "new": new, "origin": "sheharm"})
    for _, row in imported.iterrows():
        index += 1
        new = f"img{index:05d}.jpg"
        mapping.append({"old": row["file_name"], "new": new, "origin": "mami"})
        rows_out.append({
            "filename": new,
            "ocr_text": " ".join(str(row["text"]).split()),
            "category": next((name for key, name in MAMI_CATEGORY_PRECEDENCE if row[key] == 1), "Misogyny"),
            "shaming": int(row["shaming"]), "stereotype": int(row["stereotype"]),
            "objectification": int(row["objectification"]), "violence": int(row["violence"]),
        })

    # Rename via a temporary prefix so a survivor never overwrites a not-yet-moved file.
    rename = {m["old"]: m["new"] for m in mapping if m["origin"] == "sheharm"}
    for old, new in rename.items():
        (args.images / old).rename(args.images / f"__tmp__{new}")
    for new in rename.values():
        (args.images / f"__tmp__{new}").rename(args.images / new)
    for entry in (m for m in mapping if m["origin"] == "mami"):
        shutil.copy2(source_images[entry["old"]], args.images / entry["new"])

    pd.DataFrame(mapping).to_csv("dataset/mapping.csv", index=False)

    # Rewrite filename keys so the 995 annotations and 1478 OCR rows survive the renumber.
    for path in (args.annotations, args.ocr):
        if not path.exists():
            continue
        frame = pd.read_csv(path, keep_default_na=False)
        frame = frame[frame["filename"].isin(rename)]
        frame["filename"] = frame["filename"].map(rename)
        frame = frame.sort_values("filename")
        frame.to_csv(path, index=False)
        print(f"rewrote {path}: {len(frame)} rows kept")

    # Imported memes arrive with their official transcription, so they skip the OCR pass.
    ocr_frame = pd.read_csv(args.ocr, keep_default_na=False) if args.ocr.exists() else pd.DataFrame(columns=["filename", "ocr_text", "target_span"])
    added = pd.DataFrame([{"filename": r["filename"], "ocr_text": r["ocr_text"], "target_span": ""} for r in rows_out])
    combined = pd.concat([ocr_frame, added], ignore_index=True).drop_duplicates("filename", keep="last")
    combined = combined.sort_values("filename")
    combined.to_csv(args.ocr, index=False)

    Path("dataset/mami_sublabels.csv").write_text(
        pd.DataFrame(rows_out).drop(columns=["ocr_text"]).to_csv(index=False), encoding="utf-8")

    print(f"\nimages now                 : {len(list(args.images.iterdir()))}")
    print(f"ocr.csv now                : {len(combined)} rows")
    print("wrote dataset/mapping.csv and dataset/mami_sublabels.csv")
    return 0


if __name__ == "__main__":
    sys.exit(main())
