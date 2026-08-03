#!/usr/bin/env python3
"""Canonicalise annotated target strings onto the ontology's controlled inventory.

The paper predicts the women-related target "over a controlled inventory of ontology-linked
target concepts", but the annotator writes free text (1,286 distinct strings, most of them
singletons). This maps every raw string to one of the 126 ontology target concepts.

Matching is deterministic and hybrid:
    score = 0.5 * cosine(RoBERTa gloss embeddings) + 0.5 * Jaccard(content words)

Strings that clear `--threshold` are mapped. Strings that contain no women-related token at
all (the annotator sometimes writes "man" or "cat" for non-harmful memes) are labelled
`no-women-target`; the dataset builder turns those into an ignored target label so they
contribute to harmfulness training without inventing a fake target.

Writes dataset/target_inventory.json: the 126 concepts, the alias map, and a coverage report.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter
from pathlib import Path

import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from sheharm.knowledge import encode_concepts, load_knowledge  # noqa: E402

NO_TARGET = "no-women-target"

WOMAN_TOKENS = {
    "woman", "women", "womans", "womens", "female", "females", "girl", "girls", "lady",
    "ladies", "she", "her", "hers", "wife", "wives", "girlfriend", "mother", "mom", "mum",
    "mothers", "daughter", "daughters", "sister", "sisters", "aunt", "niece", "grandmother",
    "grandma", "bride", "fiancee", "widow", "housewife", "homemaker", "nurse", "actress",
    "waitress", "queen", "princess", "feminist", "feminists", "mistress", "stepmother",
    "mother-in-law", "motherinlaw", "maternal", "pregnant", "makeup", "boobs", "breast",
    "breasts", "hijab", "bikini", "dress", "wonder",
}
STOPWORDS = {"a", "an", "the", "of", "in", "on", "at", "with", "and", "or", "to", "for", "s"}

# Tokens that make a string explicitly *not* a women-related target. Checked before
# similarity, because "male character" is lexically close to "female character".
MALE_TOKENS = {
    "man", "men", "male", "males", "boy", "boys", "guy", "guys", "husband", "husbands",
    "father", "fathers", "dad", "brother", "brothers", "son", "sons", "uncle", "nephew",
    "he", "his", "him", "gentleman", "groom", "boyfriend", "king", "prince",
}
NON_PERSON_TOKENS = {"cat", "dog", "animal", "object", "car", "food", "meme", "text", "none", "nan"}

# The highest-frequency strings decide most of the label distribution, so they are pinned
# rather than left to similarity search.
MANUAL_ALIASES = {
    "woman's appearance": "her physical appearance",
    "womans appearance": "her physical appearance",
    "woman's body": "her body",
    "woman's face": "her face",
    "woman's hair": "her hair",
    "woman's clothing": "her clothing",
    "woman's weight": "her weight",
    "woman's makeup": "her makeup",
    "woman driver": "female driver",
    "women drivers": "women drivers",
    "female characters": "female character",
    "female role": "woman",
    "woman's behaviour": "woman working",
    "woman's behavior": "woman working",
}


# Bare head nouns. "woman" is a subset of "woman in bikini", "fat woman", "angry woman" and
# hundreds more, so on word overlap it wins every one of them and swallows the modifier that
# carries the actual information. When a string says more than the head noun, the generic
# concepts are removed from candidacy so the modifier decides the match.
GENERIC_CONCEPTS = {
    "woman", "women", "womans", "womens", "female", "females", "girl", "girls",
    "lady", "ladies", "chick", "chicks", "gal", "gals", "babe", "babes", "she", "her",
}


def content_words(text: str) -> set[str]:
    return {w for w in re.split(r"[^a-z0-9]+", text.lower()) if w and w not in STOPWORDS}


def jaccard(left: set[str], right: set[str]) -> float:
    if not left or not right:
        return 0.0
    return len(left & right) / len(left | right)


def is_women_related(text: str) -> bool:
    words = content_words(text)
    if words & WOMAN_TOKENS or any(word.startswith(("wom", "fem", "girl", "lad")) for word in words):
        return True
    return False


def is_explicitly_not_women(text: str) -> bool:
    """A male or non-person subject with no women-related token."""
    words = content_words(text)
    return bool(words & (MALE_TOKENS | NON_PERSON_TOKENS)) and not is_women_related(text)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--annotations", type=Path, default=Path("dataset/annotations_v2.csv"))
    parser.add_argument("--ontology", type=Path, default=Path("knowledge/ontology.json"))
    parser.add_argument("--rules", type=Path, default=Path("knowledge/rules.json"))
    parser.add_argument("--output", type=Path, default=Path("dataset/target_inventory.json"))
    parser.add_argument("--threshold", type=float, default=0.55)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--report-top", type=int, default=20)
    parser.add_argument("--concept-floor", type=float, default=0.45,
                        help="Below this match quality a modified string takes its target type "
                             "rather than the best of a bad set of concepts")
    parser.add_argument("--min-support", type=int, default=20,
                        help="Concepts with fewer rows back off to their ontology target type. "
                             "0 keeps the full 126-way inventory.")
    args = parser.parse_args()

    knowledge = load_knowledge(args.ontology, args.rules)
    concepts = knowledge.target_concepts
    names = [concept["name"] for concept in concepts]

    frame = pd.read_csv(args.annotations)
    raw_values = frame["women-related target"].astype(str).str.lower().str.strip()
    counts = Counter(raw_values)
    unique = sorted(counts)
    print(f"{len(frame)} rows, {len(unique)} distinct target strings, {len(names)} inventory concepts")

    concept_vectors = torch.nn.functional.normalize(
        encode_concepts(concepts, device=args.device), dim=-1
    )
    raw_vectors = torch.nn.functional.normalize(
        encode_concepts([{"name": value, "text": value} for value in unique], device=args.device), dim=-1
    )
    cosine = raw_vectors @ concept_vectors.t()

    concept_words = [content_words(name) for name in names]
    # Modifier-only view of each concept: "woman on the street" -> {street}. Matching on the
    # head noun is what let one concept swallow hundreds of unrelated strings.
    concept_modifiers = [words - GENERIC_CONCEPTS for words in concept_words]
    parent_of = {concept["name"]: concept["target_type"] for concept in concepts}
    alias_map: dict[str, str] = {}
    scores_report: dict[str, float] = {}
    generic_index = torch.tensor([name in GENERIC_CONCEPTS for name in names])
    for row, value in enumerate(unique):
        words = content_words(value)
        lexical = torch.tensor([jaccard(words, target) for target in concept_words])
        combined = 0.5 * cosine[row] + 0.5 * lexical
        modifiers = words - GENERIC_CONCEPTS
        specific = bool(modifiers)
        if specific:
            # A concept is only a candidate if its own modifier overlaps this one's, so
            # "wonder woman" cannot match "woman on the street" on the shared head noun.
            candidate = torch.tensor([bool(modifiers & concept) for concept in concept_modifiers])
            combined = combined.masked_fill(generic_index | ~candidate, -1.0)
        best = int(combined.argmax())
        score = float(combined[best])
        if value in MANUAL_ALIASES:              # pinned high-frequency string
            alias_map[value], scores_report[value] = MANUAL_ALIASES[value], 1.0
        elif is_explicitly_not_women(value):     # "man", "male character", "cat"
            alias_map[value], scores_report[value] = NO_TARGET, score
        elif specific and score < args.concept_floor:
            # Nothing matched the modifier. Fall back to the type of the semantically nearest
            # concept rather than forcing a bad concept-level answer.
            nearest = int(cosine[row].masked_fill(generic_index, -1.0).argmax())
            alias_map[value], scores_report[value] = parent_of[names[nearest]], score
        elif value in names:                     # exact inventory hit
            alias_map[value], scores_report[value] = value, 1.0
        elif score >= args.threshold:
            alias_map[value], scores_report[value] = names[best], score
        elif is_women_related(value):            # unusual phrasing, still a woman target
            alias_map[value], scores_report[value] = names[best], score
        else:                                    # ambiguous and weakly matched
            alias_map[value], scores_report[value] = NO_TARGET, score

    # Back off rare concepts to their ontology parent type. The annotator writes "woman" for
    # over half the corpus, so a 126-way label space leaves most classes with one or two
    # examples and macro-F1 measures noise rather than skill. The ontology already defines
    # the hierarchy (Female-Role, Female-Relationship, ...), so the fallback is principled
    # rather than an arbitrary bucket. The ontology itself is unchanged.
    parent = parent_of
    if args.min_support > 0:
        support = Counter(raw_values.map(alias_map))
        collapsed = {name for name, count in support.items()
                     if count < args.min_support and name != NO_TARGET}
        alias_map = {raw: (label if label not in collapsed else parent.get(label, label))
                     for raw, label in alias_map.items()}
        label_space = sorted({label for label in alias_map.values() if label != NO_TARGET})
        print(f"backoff: {len(collapsed)} concepts below {args.min_support} rows folded into "
              f"their target type -> {len(label_space)} classes")
    else:
        label_space = names

    mapped = raw_values.map(alias_map)
    coverage = (mapped != NO_TARGET).mean() * 100
    used = mapped[mapped != NO_TARGET].value_counts()
    report = {
        "rows": int(len(frame)),
        "distinct_raw_strings": len(unique),
        "inventory_size": len(names),
        "label_space_size": len(label_space),
        "rows_with_target": int((mapped != NO_TARGET).sum()),
        "rows_without_target": int((mapped == NO_TARGET).sum()),
        "coverage_percent": round(float(coverage), 2),
        "concepts_attested": int(used.size),
        "concepts_unattested": int(len(names) - used.size),
        "median_rows_per_attested_concept": float(used.median()) if used.size else 0.0,
        "threshold": args.threshold,
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps({
        "concepts": label_space,
        "full_inventory": names,
        "min_support": args.min_support,
        "no_target_label": NO_TARGET,
        "alias_map": alias_map,
        "match_scores": {k: round(v, 4) for k, v in scores_report.items()},
        "report": report,
        "attested_counts": {str(k): int(v) for k, v in used.items()},
    }, indent=2), encoding="utf-8")

    print(json.dumps(report, indent=2))
    print(f"\ntop {args.report_top} canonical targets:")
    for name, count in used.head(args.report_top).items():
        print(f"  {count:5d}  {name}")
    unattested = [name for name in names if name not in used.index]
    print(f"\n{len(unattested)} concepts unattested, e.g. {unattested[:8]}")
    print(f"\nwrote {args.output}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
