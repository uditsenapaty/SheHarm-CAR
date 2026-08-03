#!/usr/bin/env python3
"""Draw a representative fragment of the Women-Harm Knowledge Ontology.

    python knowledge/draw_kg.py

Every triple rendered is pulled from `knowledge/ontology.json` and verified to exist there,
so the figure cannot drift from the artefact it illustrates. Writes:

    results/kg_fragment.png     figure for slides / the paper
    results/kg_fragment.mmd     mermaid source (renders in GitHub / Markdown)

The fragment covers one triple per relation family in Table 1: a women-related target linked
to its type, harm cues linked to their categories, and the contextual relations that withhold
harm - which are precisely the relations no general-purpose commonsense graph contains.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

# (head, relation, tail, family) - each is checked against the ontology before drawing.
FRAGMENT = [
    ("woman passenger", "is_women_related_role", "Female-Role", "target"),
    ("her face", "is_women_related_role", "Female-Appearance", "target"),
    ("belongs in the kitchen", "expresses", "Misogyny", "cue"),
    ("too ugly", "expresses", "Appearance-Attack", "cue"),
    ("beat her", "expresses", "Violence", "cue"),
    ("Misogyny", "quoted_in", "quotation", "context"),
    ("Misogyny", "negated_by", "negation cue", "context"),
    ("Violence", "supported_by", "Visual Evidence", "evidence"),
]

COLOURS = {
    "target": "#8ecae6", "target_type": "#219ebc", "cue": "#ffb703",
    "category": "#fb8500", "context": "#a7c957", "evidence": "#bdb2ff",
}


def verify(triples, ontology) -> list[tuple]:
    index = {(t["head"], t["relation"], t["tail"]) for t in ontology["triples"]}
    verified, missing = [], []
    for head, relation, tail, family in triples:
        (verified if (head, relation, tail) in index else missing).append((head, relation, tail, family))
    if missing:
        print("NOT IN ONTOLOGY (skipped):")
        for item in missing:
            print(f"  {item[0]} --{item[1]}--> {item[2]}")
    return verified


def node_kind(name: str, ontology) -> str:
    for concept in ontology["concepts"]:
        if concept["name"] == name:
            return {"target": "target", "harm_cue": "cue",
                    "harm_category": "category", "context": "context"}[concept["type"]]
    if name.startswith("Female-"):
        return "target_type"
    return "evidence"


def draw_png(triples, ontology, path: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.patches as patches
    import matplotlib.pyplot as plt

    rows = len(triples)
    figure, axes = plt.subplots(figsize=(11.5, 1.35 * rows + 0.9))
    axes.set_xlim(0, 10); axes.set_ylim(0, rows * 1.35 + 0.6); axes.axis("off")

    def box(x, y, text, kind, width=3.15):
        axes.add_patch(patches.FancyBboxPatch(
            (x, y - 0.24), width, 0.52, boxstyle="round,pad=0.06",
            facecolor=COLOURS[kind], edgecolor="#22333b", linewidth=1.1))
        axes.text(x + width / 2, y, text, ha="center", va="center",
                  fontsize=9.2, weight="medium", color="#0b1416")

    for index, (head, relation, tail, _) in enumerate(reversed(triples)):
        y = 0.75 + index * 1.35
        box(0.35, y, head, node_kind(head, ontology))
        box(6.4, y, tail, node_kind(tail, ontology))
        axes.annotate("", xy=(6.35, y), xytext=(3.55, y),
                      arrowprops=dict(arrowstyle="-|>", linewidth=1.5, color="#22333b"))
        axes.text(4.95, y + 0.2, relation, ha="center", va="bottom",
                  fontsize=8.6, style="italic", color="#22333b")

    axes.set_title("Women-Harm Knowledge Ontology — representative triples",
                   fontsize=12.5, weight="bold", pad=14)
    handles = [patches.Patch(facecolor=c, edgecolor="#22333b", label=k.replace("_", " "))
               for k, c in COLOURS.items()]
    axes.legend(handles=handles, loc="lower center", ncol=6, frameon=False,
                bbox_to_anchor=(0.5, -0.055), fontsize=8.5)
    figure.tight_layout()
    figure.savefig(path, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(figure)


def draw_mermaid(triples, ontology, path: Path) -> None:
    style = {"target": "fill:#8ecae6", "target_type": "fill:#219ebc", "cue": "fill:#ffb703",
             "category": "fill:#fb8500", "context": "fill:#a7c957", "evidence": "fill:#bdb2ff"}
    identifiers, lines = {}, ["graph LR"]
    for head, relation, tail, _ in triples:
        for name in (head, tail):
            if name not in identifiers:
                identifiers[name] = f"n{len(identifiers)}"
                lines.append(f'    {identifiers[name]}["{name}"]')
        lines.append(f"    {identifiers[head]} -- {relation} --> {identifiers[tail]}")
    for name, identifier in identifiers.items():
        lines.append(f"    style {identifier} {style[node_kind(name, ontology)]},stroke:#22333b")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--ontology", type=Path, default=Path("knowledge/ontology.json"))
    parser.add_argument("--out-dir", type=Path, default=Path("results"))
    args = parser.parse_args()

    ontology = json.loads(args.ontology.read_text(encoding="utf-8"))
    triples = verify(FRAGMENT, ontology)
    if not triples:
        raise SystemExit("no fragment triple exists in the ontology")
    args.out_dir.mkdir(parents=True, exist_ok=True)

    draw_png(triples, ontology, args.out_dir / "kg_fragment.png")
    draw_mermaid(triples, ontology, args.out_dir / "kg_fragment.mmd")
    print(f"{len(triples)} triples verified against the ontology and drawn:")
    for head, relation, tail, _ in triples:
        print(f"  ({head}) --{relation}--> ({tail})")
    print(f"\nwrote {args.out_dir/'kg_fragment.png'} and {args.out_dir/'kg_fragment.mmd'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
