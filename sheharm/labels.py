"""Label spaces for SheHarm-Meme, matching the paper's task formulation."""

from __future__ import annotations

HARMFULNESS_LABELS = ["Explicit-Harm", "Implicit-Harm", "Non-Harm"]
CATEGORY_LABELS = [
    "Sexual-Harassment",
    "Violence",
    "Misogyny",
    "Appearance-Attack",
    "Character-Assassination",
]
NULL_CATEGORY = "None"

HARM2ID = {label: index for index, label in enumerate(HARMFULNESS_LABELS)}
CAT2ID = {label: index for index, label in enumerate(CATEGORY_LABELS)}
ID2HARM = {index: label for label, index in HARM2ID.items()}
ID2CAT = {index: label for label, index in CAT2ID.items()}

NON_HARM_ID = HARM2ID["Non-Harm"]
IGNORE_INDEX = -100

# The annotator writes the pre-revision surface forms; the paper uses these.
ANNOTATOR_HARM_ALIASES = {
    "Non-Harmful": "Non-Harm",
    "Explicit-Harm": "Explicit-Harm",
    "Implicit-Harm": "Implicit-Harm",
}
ANNOTATOR_CATEGORY_ALIASES = {"NULL": NULL_CATEGORY, "None": NULL_CATEGORY, "": NULL_CATEGORY}


def normalize_harmfulness(value: str) -> str:
    label = ANNOTATOR_HARM_ALIASES.get(str(value).strip(), str(value).strip())
    if label not in HARM2ID:
        raise ValueError(f"unknown harmfulness label: {value!r}")
    return label


def normalize_category(value: str, harmfulness: str) -> str:
    label = ANNOTATOR_CATEGORY_ALIASES.get(str(value).strip(), str(value).strip())
    if harmfulness == "Non-Harm":
        return NULL_CATEGORY
    if label not in CAT2ID:
        raise ValueError(f"unknown harm category: {value!r}")
    return label
