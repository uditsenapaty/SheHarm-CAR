#!/usr/bin/env python3
"""Fill target spans offline, from the OCR text and the annotated target.

OCR runs ahead of annotation, so rows transcribed before their label existed carry an empty
span. Recovering the span needs no image and no GPU: the transcription and the target string
are both text, so the span is a string-alignment problem.

    exact / case / punctuation / whitespace match   (snap_span, shared with the OCR pass)
    else the best contiguous word window in ocr_text by content-word overlap
    else the head noun of the target on its own
    else empty - the target is expressed visually, which is expected and fine

Never invents text: every span written is verified to be a literal substring of ocr_text.
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))

from ocr_and_span import snap_span  # noqa: E402

STOPWORDS = {"a", "an", "the", "of", "in", "on", "at", "with", "and", "or", "to", "for",
             "s", "her", "his", "their", "this", "that", "is", "are", "who", "which"}


def content_words(text: str) -> list[str]:
    return [w for w in re.split(r"[^a-z0-9]+", text.lower()) if w and w not in STOPWORDS]


def best_window(ocr_text: str, target: str, min_overlap: float = 0.34) -> str:
    """Highest-overlap contiguous word window; ties break toward the shorter window."""
    wanted = set(content_words(target))
    if not wanted:
        return ""
    tokens = [(m.group(0), m.start(), m.end()) for m in re.finditer(r"\S+", ocr_text)]
    if not tokens:
        return ""

    best_score, best_span = 0.0, ""
    for start in range(len(tokens)):
        found = set()
        for end in range(start, min(start + len(wanted) + 2, len(tokens))):
            found |= set(content_words(tokens[end][0]))
            hit = len(found & wanted)
            if not hit:
                continue
            score = hit / len(wanted | found)
            length = tokens[end][2] - tokens[start][1]
            if score > best_score or (score == best_score and best_span and length < len(best_span)):
                best_score, best_span = score, ocr_text[tokens[start][1] : tokens[end][2]]
    return best_span if best_score >= min_overlap else ""


def resolve_span(ocr_text: str, target: str) -> tuple[str, str]:
    if not ocr_text or not target:
        return "", "no_text_or_target"
    snapped = snap_span(ocr_text, target)
    if snapped:
        return snapped, "exact"
    window = best_window(ocr_text, target)
    if window:
        return window, "window"
    words = content_words(target)
    if words:
        head = words[-1]                      # "female shoppers" -> "shoppers"
        match = re.search(rf"\b{re.escape(head)}\w*", ocr_text, re.IGNORECASE)
        if match:
            return match.group(0), "head_noun"
    return "", "unmatched"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--annotations", type=Path, default=Path("dataset/annotations_v2.csv"))
    parser.add_argument("--ocr", type=Path, default=Path("dataset/ocr.csv"))
    parser.add_argument("--overwrite", action="store_true", help="Also recompute spans that already exist")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    ocr = pd.read_csv(args.ocr, keep_default_na=False)
    targets = {row["filename"]: str(row["women-related target"])
               for _, row in pd.read_csv(args.annotations).iterrows()}

    reasons, filled = {}, 0
    spans = []
    for _, row in ocr.iterrows():
        existing = str(row["target_span"])
        if existing and not args.overwrite:
            spans.append(existing)
            reasons["kept"] = reasons.get("kept", 0) + 1
            continue
        span, reason = resolve_span(str(row["ocr_text"]), targets.get(row["filename"], ""))
        if span and span not in str(row["ocr_text"]):
            span, reason = "", "rejected_not_substring"
        spans.append(span)
        reasons[reason] = reasons.get(reason, 0) + 1
        filled += bool(span)

    print(f"rows                : {len(ocr)}")
    for reason, count in sorted(reasons.items(), key=lambda kv: -kv[1]):
        print(f"  {reason:24s}: {count}")
    before = int((ocr['target_span'].str.len() > 0).sum())
    after = sum(1 for s in spans if s)
    print(f"spans {before} -> {after}  ({after / max(len(ocr), 1) * 100:.1f}% grounded)")

    if args.dry_run:
        print("\n(dry run — nothing written)")
        return 0
    ocr["target_span"] = spans
    ocr.to_csv(args.ocr, index=False)
    print(f"\nwrote {args.ocr}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
