#!/usr/bin/env python3
"""Correct harm-category bias in the annotator-1 (Qwen) pass.

The prompted annotator collapses toward whichever category is most salient: in the first
pass 72% of all harmful memes were labelled Sexual-Harassment while Character-Assassination
received 2 instances in 3,843 rows. Better prompting fixes most of this going forward; this
pass repairs what the prompt still misses, using lexical evidence rather than another model.

Method. Each category has a high-precision trigger lexicon (below) plus the harm-cue
concepts already defined for that category in the Women-Harm Knowledge Ontology. A row is
reassigned only under a deliberately conservative rule:

    the assigned category has NO trigger anywhere in the meme text or rationale,
    AND exactly one other category fires,
    AND that category fires at least `--min-hits` times.

So a meme genuinely about sexual harassment keeps its label even when other words appear,
and a meme with no lexical evidence either way is left alone. Every change is logged.

ONLY annotator 1 is adjudicated. The SheHarm-CAR pass (annotator 2) is the model's own
prediction and must stay untouched, or both the gold labels and the kappa become circular.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter
from pathlib import Path

import pandas as pd

# High-precision, deliberately general triggers. A term belongs here only if its presence
# is strong evidence for that category and weak evidence for the others.
TRIGGERS: dict[str, list[str]] = {
    "Character-Assassination": [
        "slut", "sluts", "slutty", "whore", "whores", "hoe", "hoes", "thot", "thots",
        "body count", "bodycount", "characterless", "no character", "loose woman",
        "easy girl", "used goods", "run through", "for the streets", "homewrecker",
        "home wrecker", "cheat", "cheats", "cheated", "cheating", "cheater", "unfaithful",
        "affair", "side chick", "mistress", "gold digger", "golddigger", "after his money",
        "alimony", "child support", "two faced", "manipulative", "attention seeking",
        "attention whore", "drama queen", "crazy ex", "bad mother", "unfit mother",
        "immoral", "shameless", "disloyal", "promiscuous", "sleeping around",
        "slept with", "ruined her reputation", "no morals",
    ],
    "Appearance-Attack": [
        "ugly", "hideous", "busted", "butterface", "fat", "obese", "chubby", "whale",
        "cow", "pig", "landwhale", "skinny", "anorexic", "flat chest", "wrinkle",
        "wrinkles", "old hag", "botox", "filler", "plastic surgery", "nose job",
        "makeup", "make up", "without makeup", "no makeup", "catfish", "filter",
        "photoshop", "facetune", "take her swimming", "acne", "double chin", "cellulite",
        "stretch marks", "bad hair", "looks like a man", "beat face", "no makeup on",
    ],
    "Violence": [
        "beat her", "beat his wife", "hit her", "hits her", "slap", "slapped", "punch",
        "punched", "choke", "choked", "strangle", "kill her", "murder", "stab", "shoot her",
        "acid", "burn her", "domestic violence", "black eye", "bruise", "bruises",
        "back of my hand", "knock her", "put her in her place", "dowry",
        "honour killing", "honor killing", "beating",
    ],
    "Sexual-Harassment": [
        "rape", "raped", "rapist", "molest", "grope", "groped", "boobs", "tits", "titties",
        "ass", "booty", "cleavage", "nudes", "nude", "naked", "sexy", "sexual", "sex",
        "blowjob", "horny", "thirsty", "smash", "hit that", "catcall", "whistle",
        "objectify", "objectified", "objectification", "sextape", "onlyfans", "stripper",
        "lingerie", "bikini", "porn", "porno", "pornography", "deepfake", "nudify",
        "upskirt", "revenge porn", "sexualise", "sexualize", "sexualised", "sexualized",
        "sexually", "consent", "assault", "harass", "harassment", "creep", "creepy",
    ],
    "Misogyny": [
        "kitchen", "make me a sandwich", "belongs in the kitchen", "make a sandwich",
        "clean the house", "women can't drive", "women cant drive", "woman driver",
        "female driver", "bad driver", "park the car", "too emotional", "hormonal",
        "her period", "pms", "women are inferior", "know your place", "obey",
        "shut up woman", "feminist", "feminism", "equal rights equal fights",
        "women belong", "a woman's job", "womens job", "get back in the kitchen",
        "can't do math", "cant do math", "nagging", "talks too much",
    ],
}
CATEGORIES = list(TRIGGERS)


def compile_patterns(ontology_path: Path | None) -> dict[str, list[re.Pattern]]:
    """Trigger lexicon plus the ontology's own harm cues for the same category."""
    lexicon = {name: list(terms) for name, terms in TRIGGERS.items()}
    if ontology_path and ontology_path.exists():
        ontology = json.loads(ontology_path.read_text(encoding="utf-8"))
        for concept in ontology["concepts"]:
            if concept["type"] == "harm_cue" and concept["category"] in lexicon:
                phrase = concept["name"]
                # Multi-word cues only: single generic words would fire everywhere.
                if len(phrase.split()) >= 2:
                    lexicon[concept["category"]].append(phrase)
    return {
        name: [re.compile(rf"\b{re.escape(term)}\b", re.IGNORECASE) for term in sorted(set(terms))]
        for name, terms in lexicon.items()
    }


def hits(text: str, patterns: dict[str, list[re.Pattern]]) -> dict[str, int]:
    return {name: sum(1 for pattern in group if pattern.search(text)) for name, group in patterns.items()}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--annotations", type=Path, default=Path("dataset/annotations_v2.csv"),
                        help="Annotator 1 only")
    parser.add_argument("--ocr", type=Path, default=Path("dataset/ocr.csv"))
    parser.add_argument("--ontology", type=Path, default=Path("knowledge/ontology.json"))
    parser.add_argument("--min-hits", type=int, default=1)
    parser.add_argument("--report", type=Path, default=Path("results/category_adjudication.json"))
    parser.add_argument("--apply", action="store_true")
    args = parser.parse_args()

    if "sheharm" in args.annotations.name.lower():
        raise SystemExit(
            "refusing to adjudicate the SheHarm-CAR pass: annotator 2 is the model's own "
            "prediction and must stay untouched, or the gold labels and kappa become circular."
        )

    # keep_default_na=False: "NULL" is a real category value, not a missing one.
    frame = pd.read_csv(args.annotations, keep_default_na=False)
    ocr_text = {}
    if args.ocr.exists():
        ocr = pd.read_csv(args.ocr, keep_default_na=False)
        ocr_text = dict(zip(ocr["filename"], ocr["ocr_text"]))
    patterns = compile_patterns(args.ontology)

    changes, before, after = [], Counter(), Counter()
    new_categories = []
    for _, row in frame.iterrows():
        current = str(row["harm_category"]).strip()
        before[current] += 1
        if current not in CATEGORIES:          # Non-Harmful rows are left alone
            new_categories.append(current)
            after[current] += 1
            continue

        text = " ".join([
            str(ocr_text.get(row["filename"], "")),
            str(row["rationale"]),
            str(row["women-related target"]),
        ])
        counted = hits(text, patterns)
        firing = [name for name, count in counted.items() if count >= args.min_hits]

        if counted.get(current, 0) == 0 and len(firing) == 1 and firing[0] != current:
            target = firing[0]
            changes.append({
                "filename": row["filename"], "from": current, "to": target,
                "hits": counted[target], "evidence": text[:160],
            })
            new_categories.append(target)
            after[target] += 1
        else:
            new_categories.append(current)
            after[current] += 1

    order = CATEGORIES + [c for c in before if c not in CATEGORIES]
    print(f"rows: {len(frame)}   reassigned: {len(changes)}\n")
    print(f"{'category':26s} {'before':>7s} {'after':>7s} {'delta':>7s}")
    for name in order:
        delta = after[name] - before[name]
        print(f"{name:26s} {before[name]:7d} {after[name]:7d} {delta:+7d}")

    flow = Counter((c["from"], c["to"]) for c in changes)
    if flow:
        print("\nreassignments:")
        for (source, target), count in flow.most_common():
            print(f"  {count:5d}  {source} -> {target}")
        print("\nexamples:")
        for change in changes[:5]:
            print(f"  {change['filename']}: {change['from']} -> {change['to']}  | {change['evidence'][:95]}")

    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps({
        "source": str(args.annotations), "rows": len(frame), "reassigned": len(changes),
        "before": dict(before), "after": dict(after),
        "flow": {f"{s}->{t}": c for (s, t), c in flow.items()}, "changes": changes,
    }, indent=2), encoding="utf-8")
    print(f"\nwrote {args.report}")

    if not args.apply:
        print("(dry run — pass --apply to write the corrected categories)")
        return 0
    frame["harm_category"] = new_categories
    frame.to_csv(args.annotations, index=False)
    print(f"updated {args.annotations}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
