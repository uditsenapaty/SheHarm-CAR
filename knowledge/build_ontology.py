#!/usr/bin/env python3
"""Compose the Women-Harm Knowledge Ontology and the soft rule inventory.

Reads the flat seed lists in `ontology_seed.py` and emits:
    knowledge/ontology.json   611 concepts, 14 relation types, 1,287 triples
    knowledge/rules.json      36 soft reasoning rules (R+ / R- / gate control)
    results/table13_ontology_triples.{json,tex}
    results/table14_ontology_statistics.{json,tex}

Triple composition (documented so the count is reproducible, not accidental):

    126  target                --is_women_related_role-> target type
      6  target type           --is_a-----------------> Women-Related-Target
    438  harm cue              --expresses------------> harm category
    438  harm cue              --indicates------------> harmfulness tendency
      5  harm category         --is_a-----------------> Harm-Category
     10  harm category         --supported_by---------> evidence (textual, visual)
     42  context concept       --is_a-----------------> context type
      6  context type          --supports-------------> Non-Harm
     12  target type           --evidenced_by---------> evidence
      5  harm category         --co_occurs_with-------> harm category
     12  target type           --refers_to------------> modality
    115  mitigating context    --mitigates------------> harm category
     30  harm category         --negated_by-----------> negation context
     35  harm category         --quoted_in------------> quotation context
      2  severe cue            --escalates------------> harm category
      5  harm category         --targets--------------> target type
    ----
   1287
"""

from __future__ import annotations

import argparse
import json
import sys
from itertools import combinations
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from ontology_seed import (  # noqa: E402
    CATEGORY_CONCEPTS,
    CONTEXT_CONCEPTS,
    EVIDENCE_CONCEPTS,
    HARM_CUES,
    HARMFULNESS,
    IMPLICIT_CATEGORIES,
    IMPLICIT_MARKERS,
    MODALITIES,
    RELATIONS,
    TARGET_CONCEPTS,
)

# Predicates whose truth values the reasoner estimates from (z, t_soft, k_retrieved).
PREDICATES = [
    "Targets-Woman",
    "Invokes-Sexual-Cue", "Invokes-Violence-Cue", "Invokes-Misogyny-Cue",
    "Attacks-Appearance", "Attacks-Character",
    "Direct-Reference", "Indirect-Implication", "Sarcasm", "Cultural-Reference",
    "Image-Text-Association", "Explicit-Threat", "Weapon-Present", "Visual-Evidence",
    "Condemns-Harm", "Quotes-Harm", "Raises-Awareness", "Negates-Stereotype",
    "Empowers-Women", "Non-Targeted-Content",
    "Image-Text-Conflict", "Weak-Retrieval", "Rule-Conflict", "No-Target-Detected",
    "Low-Evidence",
]

CUE_PREDICATE = {
    "Sexual-Harassment": "Invokes-Sexual-Cue",
    "Violence": "Invokes-Violence-Cue",
    "Misogyny": "Invokes-Misogyny-Cue",
    "Appearance-Attack": "Attacks-Appearance",
    "Character-Assassination": "Attacks-Character",
}

MITIGATING_TYPES = ["Counter-Speech", "Awareness", "Empowerment"]
CATEGORY_TARGET_TYPE = {
    "Sexual-Harassment": "Female-Role",
    "Violence": "Female-Relationship",
    "Misogyny": "Female-Group",
    "Appearance-Attack": "Female-Appearance",
    "Character-Assassination": "Female-Role",
}
CO_OCCURRENCE = [
    ("Sexual-Harassment", "Violence"),
    ("Misogyny", "Appearance-Attack"),
    ("Misogyny", "Character-Assassination"),
    ("Appearance-Attack", "Character-Assassination"),
    ("Sexual-Harassment", "Character-Assassination"),
]
TYPE_MODALITIES = {
    "Female-Role": ["textual", "visual"],
    "Female-Relationship": ["textual", "multimodal"],
    "Female-Profession": ["textual", "multimodal"],
    "Female-Appearance": ["visual", "multimodal"],
    "Female-Group": ["textual", "multimodal"],
    "Female-Behaviour": ["visual", "multimodal"],
}


def slug(text: str) -> str:
    return "".join(character if character.isalnum() else "_" for character in text.lower()).strip("_")


def harmfulness_of(cue: str, category: str) -> str:
    """Indirect realizations (stereotype, joke, framing) read as implicit harm."""
    if category in IMPLICIT_CATEGORIES or any(marker in cue for marker in IMPLICIT_MARKERS):
        return "Implicit-Harm"
    return "Explicit-Harm"


def build_concepts() -> list[dict]:
    concepts: list[dict] = []
    for target_type, names in TARGET_CONCEPTS.items():
        for name in names:
            concepts.append({
                "id": f"tgt_{slug(name)}", "name": name, "type": "target",
                "target_type": target_type, "category": None,
                "text": f"{name}, a women-related target of type {target_type.replace('-', ' ').lower()}",
            })
    for category, cues in HARM_CUES.items():
        for cue in cues:
            concepts.append({
                "id": f"cue_{slug(cue)}", "name": cue, "type": "harm_cue",
                "target_type": None, "category": category,
                "text": f"{cue}, an expression of {category.replace('-', ' ').lower()} directed at women",
            })
    for category in CATEGORY_CONCEPTS:
        concepts.append({
            "id": f"cat_{slug(category)}", "name": category, "type": "harm_category",
            "target_type": None, "category": category,
            "text": f"{category.replace('-', ' ').lower()}, a category of women-targeted harm",
        })
    for context_type, names in CONTEXT_CONCEPTS.items():
        for name in names:
            concepts.append({
                "id": f"ctx_{slug(name)}", "name": name, "type": "context",
                "target_type": None, "category": None, "context_type": context_type,
                "text": f"{name}, a contextual exception of type {context_type.replace('-', ' ').lower()} that withholds harm",
            })
    return concepts


def build_triples() -> list[dict]:
    triples: list[dict] = []

    def add(head: str, relation: str, tail: str, role: str) -> None:
        triples.append({"head": head, "relation": relation, "tail": tail, "reasoning_role": role})

    for target_type, names in TARGET_CONCEPTS.items():
        for name in names:
            add(name, "is_women_related_role", target_type, "Identifies the women-related target.")
    for target_type in TARGET_CONCEPTS:
        add(target_type, "is_a", "Women-Related-Target", "Places the target type in the target hierarchy.")

    for category, cues in HARM_CUES.items():
        for cue in cues:
            add(cue, "expresses", category, "Associates a harmful expression with its category.")
    for category, cues in HARM_CUES.items():
        for cue in cues:
            add(cue, "indicates", harmfulness_of(cue, category), "Signals whether the harm is direct or implied.")

    for category in CATEGORY_CONCEPTS:
        add(category, "is_a", "Harm-Category", "Places the category in the harm hierarchy.")
    for category in CATEGORY_CONCEPTS:
        for evidence in EVIDENCE_CONCEPTS:
            add(category, "supported_by", evidence, "Connects the predicted category with multimodal evidence.")

    for context_type, names in CONTEXT_CONCEPTS.items():
        for name in names:
            add(name, "is_a", context_type, "Places the contextual concept in the exception hierarchy.")
    for context_type in CONTEXT_CONCEPTS:
        add(context_type, "supports", "Non-Harm", "Withholds harm support under this context.")

    for target_type in TARGET_CONCEPTS:
        for evidence in EVIDENCE_CONCEPTS:
            add(target_type, "evidenced_by", evidence, "Grounds the target type in observable evidence.")
    for first, second in CO_OCCURRENCE:
        add(first, "co_occurs_with", second, "Captures categories that frequently appear together.")
    for target_type, modalities in TYPE_MODALITIES.items():
        for modality in modalities:
            add(target_type, "refers_to", modality, "Records how the target is expressed.")

    for context_type in MITIGATING_TYPES:
        for name in CONTEXT_CONCEPTS[context_type]:
            for category in CATEGORY_CONCEPTS:
                add(name, "mitigates", category, "Reduces harm support for this category.")
    for category in CATEGORY_CONCEPTS:
        for name in CONTEXT_CONCEPTS["Negation"]:
            add(category, "negated_by", name, "Reduces harm support when the statement is explicitly rejected.")
    for category in CATEGORY_CONCEPTS:
        for name in CONTEXT_CONCEPTS["Quotation"]:
            add(category, "quoted_in", name, "Prevents quoted harmful language from being treated as endorsement.")

    add("rape threat", "escalates", "Violence", "Sexual threat that escalates into physical harm.")
    add("acid attack", "escalates", "Appearance-Attack", "Physical violence that also destroys appearance.")
    for category, target_type in CATEGORY_TARGET_TYPE.items():
        add(category, "targets", target_type, "Records the target type the category typically attacks.")
    return triples


def build_rules() -> list[dict]:
    harm_index = {label: i for i, label in enumerate(HARMFULNESS)}
    category_index = {label: i for i, label in enumerate(CATEGORY_CONCEPTS)}
    rules: list[dict] = []

    def add(name, text, polarity, predicates, harm=None, category=None, weight=1.0):
        rules.append({
            "name": name, "text": text, "polarity": polarity, "predicates": predicates,
            "harm_class": None if harm is None else harm_index[harm],
            "category_class": None if category is None else category_index[category],
            "weight": weight,
        })

    descriptions = {
        "Sexual-Harassment": "sexualized remark or objectification",
        "Violence": "threat, violent intent, or encouragement of physical harm",
        "Misogyny": "gender-based inferiority, exclusion, or stereotype",
        "Appearance-Attack": "ridicule of face, body, clothing, or physical appearance",
        "Character-Assassination": "reputation attack, moral judgement, or derogatory allegation",
    }
    for index, category in enumerate(CATEGORY_CONCEPTS, start=1):
        add(f"R{index}", f"IF women-related target AND {descriptions[category]} THEN {category}.",
            "positive", ["Targets-Woman", CUE_PREDICATE[category]], category=category)
    # R6-R10 are the paper's Table 2 identifiers: generic explicit, generic implicit, the two
    # contextual exceptions, and confidence control. Category-specific variants follow at R11+.
    add("R6", "IF a harmful cue directly refers to the identified women-related target THEN increase Explicit-Harm support.",
        "positive", ["Targets-Woman", "Direct-Reference"], harm="Explicit-Harm")
    add("R7", "IF the harmful interpretation requires image-text association, cultural knowledge, sarcasm, or indirect implication THEN increase Implicit-Harm support.",
        "positive", ["Targets-Woman", "Indirect-Implication"], harm="Implicit-Harm")
    add("R8", "IF a harmful expression is quoted or depicted AND explicitly criticized, condemned, or challenged THEN support Non-Harm.",
        "exception", ["Quotes-Harm", "Condemns-Harm"], harm="Non-Harm")
    add("R9", "IF the meme raises awareness, provides counter-speech, or reports harm without endorsement THEN support Non-Harm.",
        "exception", ["Raises-Awareness"], harm="Non-Harm")
    add("R10", "IF image, OCR text, retrieved knowledge, and activated rules conflict THEN reduce the neural-symbolic confidence gate.",
        "gate", ["Image-Text-Conflict"])

    for index, category in enumerate(CATEGORY_CONCEPTS, start=11):
        add(f"R{index}", f"IF a {category} cue directly refers to the identified women-related target THEN Explicit-Harm.",
            "positive", ["Targets-Woman", CUE_PREDICATE[category], "Direct-Reference"],
            harm="Explicit-Harm", category=category)
    for index, category in enumerate(CATEGORY_CONCEPTS, start=16):
        add(f"R{index}", f"IF a {category} interpretation requires implication or association THEN Implicit-Harm.",
            "positive", ["Targets-Woman", CUE_PREDICATE[category], "Indirect-Implication"],
            harm="Implicit-Harm", category=category)
    add("R21", "IF women-related target AND explicit threat AND violent cue THEN Explicit-Harm / Violence.",
        "positive", ["Targets-Woman", "Explicit-Threat", "Invokes-Violence-Cue"], harm="Explicit-Harm", category="Violence")
    add("R22", "IF women-related target AND weapon present AND visual evidence THEN Explicit-Harm / Violence.",
        "positive", ["Targets-Woman", "Weapon-Present", "Visual-Evidence"], harm="Explicit-Harm", category="Violence")
    add("R23", "IF women-related target AND sarcasm AND misogynistic cue THEN Implicit-Harm / Misogyny.",
        "positive", ["Targets-Woman", "Sarcasm", "Invokes-Misogyny-Cue"], harm="Implicit-Harm", category="Misogyny")

    add("E1", "IF harmful language is quoted or depicted without endorsement THEN Non-Harm.",
        "exception", ["Quotes-Harm"], harm="Non-Harm", weight=0.8)
    add("E3", "IF a stereotype is explicitly rejected or corrected THEN Non-Harm.",
        "exception", ["Negates-Stereotype"], harm="Non-Harm")
    add("E5", "IF the meme praises, empowers, or affirms equality for women THEN Non-Harm.",
        "exception", ["Empowers-Women"], harm="Non-Harm")
    add("E6", "IF no women-related target is addressed THEN Non-Harm.",
        "exception", ["Non-Targeted-Content"], harm="Non-Harm")
    add("E7", "IF misogyny is mentioned AND condemned THEN Non-Harm.",
        "exception", ["Invokes-Misogyny-Cue", "Condemns-Harm"], harm="Non-Harm")
    add("E8", "IF a sexual cue is mentioned AND condemned THEN Non-Harm.",
        "exception", ["Invokes-Sexual-Cue", "Condemns-Harm"], harm="Non-Harm")
    add("E9", "IF violence is depicted AND awareness is raised THEN Non-Harm.",
        "exception", ["Invokes-Violence-Cue", "Raises-Awareness"], harm="Non-Harm")
    add("E10", "IF an appearance stereotype is challenged THEN Non-Harm.",
        "exception", ["Attacks-Appearance", "Negates-Stereotype"], harm="Non-Harm")
    add("E11", "IF a character attack is quoted AND criticized THEN Non-Harm.",
        "exception", ["Attacks-Character", "Quotes-Harm", "Condemns-Harm"], harm="Non-Harm")

    add("G2", "IF retrieved knowledge is weakly matched THEN reduce the confidence gate.", "gate", ["Weak-Retrieval"])
    add("G3", "IF harm-supporting and exception rules conflict THEN reduce the confidence gate.", "gate", ["Rule-Conflict"])
    add("G4", "IF no women-related target is detected THEN reduce the confidence gate.", "gate", ["No-Target-Detected"])
    add("G5", "IF multimodal evidence is weak THEN reduce the confidence gate.", "gate", ["Low-Evidence"])
    return rules


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", type=Path, default=Path("knowledge"))
    parser.add_argument("--results-dir", type=Path, default=Path("results"))
    parser.add_argument("--strict", action="store_true", help="Fail if counts deviate from the paper inventory")
    args = parser.parse_args()

    concepts, triples, rules = build_concepts(), build_triples(), build_rules()
    counts = {
        "Women-related target concepts": sum(1 for c in concepts if c["type"] == "target"),
        "Harm-cue concepts": sum(1 for c in concepts if c["type"] == "harm_cue"),
        "Harm-category concepts": sum(1 for c in concepts if c["type"] == "harm_category"),
        "Context and exception concepts": sum(1 for c in concepts if c["type"] == "context"),
        "Relation types": len({t["relation"] for t in triples}),
        "Ontology triples": len(triples),
        "Soft reasoning rules": len(rules),
    }
    expected = {
        "Women-related target concepts": 126, "Harm-cue concepts": 438,
        "Harm-category concepts": 5, "Context and exception concepts": 42,
        "Relation types": 14, "Ontology triples": 1287, "Soft reasoning rules": 36,
    }

    identifiers = [c["id"] for c in concepts]
    if len(identifiers) != len(set(identifiers)):
        raise SystemExit("duplicate concept ids in the seed inventory")
    declared = {r["name"] for r in RELATIONS}
    used = {t["relation"] for t in triples}
    if used - declared:
        raise SystemExit(f"triples use undeclared relations: {sorted(used - declared)}")
    if declared - used:
        raise SystemExit(f"declared relations never used: {sorted(declared - used)}")
    unknown = {p for rule in rules for p in rule["predicates"]} - set(PREDICATES)
    if unknown:
        raise SystemExit(f"rules use undeclared predicates: {sorted(unknown)}")

    args.out_dir.mkdir(parents=True, exist_ok=True)
    args.results_dir.mkdir(parents=True, exist_ok=True)
    (args.out_dir / "ontology.json").write_text(json.dumps({
        "concepts": concepts, "relations": RELATIONS, "triples": triples,
        "predicates": PREDICATES, "statistics": counts,
    }, indent=2), encoding="utf-8")
    (args.out_dir / "rules.json").write_text(json.dumps({
        "predicates": PREDICATES, "rules": rules,
        "harmfulness_labels": HARMFULNESS, "category_labels": CATEGORY_CONCEPTS,
    }, indent=2), encoding="utf-8")

    (args.results_dir / "table14_ontology_statistics.json").write_text(json.dumps(counts, indent=2), encoding="utf-8")
    rows = "\n".join(f"{name} & {value:,} \\\\" for name, value in counts.items())
    (args.results_dir / "table14_ontology_statistics.tex").write_text(
        "\\begin{tabular}{lr}\n\\toprule\n\\textbf{Ontology Component} & \\textbf{Count} \\\\\n\\midrule\n"
        f"{rows}\n\\bottomrule\n\\end{{tabular}}\n", encoding="utf-8")

    # Table 1: representative concepts and relations, one per reasoning role.
    examples = []
    for relation, role in (("is_women_related_role", "Target"), ("expresses", "Harm cue"),
                           ("negated_by", "Context"), ("quoted_in", "Context"),
                           ("supported_by", "Evidence")):
        for triple in [x for x in triples if x["relation"] == relation][:2]:
            examples.append({"type": role, "head": triple["head"],
                             "relation_tail": f'{triple["relation"]} {triple["tail"]}',
                             "reasoning_role": triple["reasoning_role"]})
    (args.results_dir / "table1_ontology_examples.json").write_text(
        json.dumps(examples, indent=2), encoding="utf-8")

    # Table 2: the soft rule inventory, grouped by polarity.
    (args.results_dir / "table2_symbolic_rules.json").write_text(json.dumps(
        [{"rule": r["name"], "polarity": r["polarity"], "description": r["text"]} for r in rules],
        indent=2), encoding="utf-8")
    rule_rows = "\n".join(
        f'{r["name"]} & {r["text"]} \\\\' for r in rules if r["name"] in
        ("R1", "R2", "R3", "R4", "R5", "R6", "R11", "E1", "E3", "G1"))
    (args.results_dir / "table2_symbolic_rules.tex").write_text(
        "\\begin{tabularx}{\\columnwidth}{@{}lX@{}}\n\\toprule\n"
        "\\textbf{Rule} & \\textbf{Description} \\\\\n\\midrule\n"
        f"{rule_rows}\n\\bottomrule\n\\end{{tabularx}}\n", encoding="utf-8")

    representative = [t for t in triples if t["relation"] in
                      ("is_women_related_role", "expresses", "quoted_in", "negated_by", "supported_by")]
    sample = [representative[0], representative[70],
              *[t for t in triples if t["relation"] == "expresses"][:5],
              *[t for t in triples if t["relation"] == "quoted_in"][:1],
              *[t for t in triples if t["relation"] == "negated_by"][:1],
              *[t for t in triples if t["relation"] == "supported_by"][:1]]
    (args.results_dir / "table13_ontology_triples.json").write_text(json.dumps(sample, indent=2), encoding="utf-8")

    print(f"{'component':32s} {'built':>7s} {'paper':>7s}")
    ok = True
    for name, value in counts.items():
        flag = "" if value == expected[name] else "  <-- MISMATCH"
        ok &= value == expected[name]
        print(f"{name:32s} {value:7,d} {expected[name]:7,d}{flag}")
    print(f"\nconcepts total: {len(concepts)} (paper 611) | predicates: {len(PREDICATES)}")
    print(f"rules: {sum(1 for r in rules if r['polarity']=='positive')} positive, "
          f"{sum(1 for r in rules if r['polarity']=='exception')} exception, "
          f"{sum(1 for r in rules if r['polarity']=='gate')} gate")
    if args.strict and not ok:
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
