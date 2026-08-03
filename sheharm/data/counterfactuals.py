"""Counterfactual construction for faithfulness and robustness evaluation.

Two families, used by different tables:

1. Evidence masking (Table `tab:main-results` CF-Faith., Table `tab:counterfactual-analysis`).
   The relevant intervention removes the grounded women-related target span from the OCR
   text; the irrelevant control removes an equally long span as far from it as possible.
   Both are text edits, so every model in the comparison - encoder baselines and prompted
   VLMs alike - can be scored through the identical intervention.

2. Lexical substitution (Table `tab:lexical-robustness`).
   - meaning-preserving: swap a harm cue for a synonymous cue. A faithful model keeps its
     prediction; a lexically-driven model flips (CFR).
   - harm-removing: replace the harm cue with a benign phrase, or wrap it in counter-speech.
     The gold label becomes Non-Harm, so any harmful prediction is a false positive (FPR-H).
   Rationale stability (RS) compares the rationale before and after the perturbation.
"""

from __future__ import annotations

import random
import re

MASK_TOKEN = "[...]"

# Meaning-preserving swaps: same harm, different words.
SYNONYM_SUBSTITUTIONS = {
    "ugly": ["hideous", "unattractive"],
    "fat": ["overweight", "obese"],
    "stupid": ["dumb", "brainless"],
    "kitchen": ["stove", "cooking area"],
    "sandwich": ["meal", "lunch"],
    "wife": ["spouse", "missus"],
    "girlfriend": ["partner", "girl"],
    "woman": ["female", "lady"],
    "women": ["females", "ladies"],
    "girl": ["lass", "young woman"],
    "slut": ["tramp", "hussy"],
    "whore": ["hooker", "harlot"],
    "beat": ["hit", "strike"],
    "kill": ["murder", "slay"],
    "nagging": ["complaining", "whining"],
    "gold digger": ["money chaser", "fortune hunter"],
    "makeup": ["cosmetics", "foundation"],
    "drive": ["steer", "operate a car"],
}

# Harm-removing rewrites: the resulting meme should be judged Non-Harm.
BENIGN_SUBSTITUTIONS = {
    "ugly": "kind", "fat": "cheerful", "stupid": "clever", "slut": "friend",
    "whore": "neighbour", "beat": "greet", "kill": "help", "nagging": "supportive",
    "gold digger": "hard worker", "worthless": "valuable", "useless": "capable",
    "belongs in the kitchen": "leads the meeting", "belong in the kitchen": "lead the meeting",
    "shut up": "speak up", "nagging": "supportive", "bitch": "person", "dumb": "smart",
    "hideous": "lovely", "obese": "healthy", "worthless woman": "valued colleague",
}

# Wrapping a harmful phrase in condemnation is the exception case rules R8/R9 must catch.
COUNTER_SPEECH_TEMPLATES = [
    'Someone actually said "{text}" - this is exactly the sexism we need to call out.',
    'Stop saying things like "{text}". It is harmful and wrong.',
    'Quoting this only to condemn it: "{text}". Nobody should talk about women this way.',
]


def mask_span(text: str, start: int, end: int, placeholder: str = MASK_TOKEN) -> str:
    if start < 0 or end <= start or end > len(text):
        return text
    return text[:start] + placeholder + text[end:]


def mask_distant_span(text: str, start: int, end: int, placeholder: str = MASK_TOKEN) -> str:
    """Remove a same-length region as far from the target span as the text allows."""
    if start < 0 or end <= start or end > len(text):
        return text
    length = end - start
    if len(text) <= length:
        return text
    candidates = [0, len(text) - length]
    # Pick whichever candidate window overlaps the target span least.
    best = max(candidates, key=lambda position: abs(position - start))
    if best < start < best + length or best < end <= best + length:
        return text
    return text[:best] + placeholder + text[best + length:]


def evidence_counterfactuals(text: str, start: int, end: int) -> dict[str, str]:
    return {
        "original": text,
        "relevant_removed": mask_span(text, start, end),
        "irrelevant_removed": mask_distant_span(text, start, end),
        "grounded": start >= 0 and end > start,
    }


def _replace_first(text: str, needle: str, replacement: str) -> tuple[str, bool]:
    pattern = re.compile(rf"\b{re.escape(needle)}\b", re.IGNORECASE)
    if not pattern.search(text):
        return text, False

    def keep_case(match: re.Match) -> str:
        found = match.group(0)
        if found.isupper():
            return replacement.upper()
        if found[:1].isupper():
            return replacement[:1].upper() + replacement[1:]
        return replacement

    return pattern.sub(keep_case, text, count=1), True


def synonym_counterfactual(text: str, seed: int = 0) -> tuple[str, bool]:
    """Meaning-preserving lexical swap; the label must not change."""
    rng = random.Random(seed)
    for needle in sorted(SYNONYM_SUBSTITUTIONS, key=len, reverse=True):
        replacement = rng.choice(SYNONYM_SUBSTITUTIONS[needle])
        edited, changed = _replace_first(text, needle, replacement)
        if changed:
            return edited, True
    return text, False


def benign_counterfactual(text: str) -> tuple[str, bool]:
    """Harm-removing rewrite; the label becomes Non-Harm.

    Every matching cue is replaced, not just the first: leaving a second harmful cue in
    place would keep the instance genuinely harmful and make FPR-H meaningless.
    """
    edited, any_change = text, False
    for needle in sorted(BENIGN_SUBSTITUTIONS, key=len, reverse=True):
        replacement = BENIGN_SUBSTITUTIONS[needle]
        for _ in range(8):  # bounded: a replacement containing its own needle would not terminate
            edited, changed = _replace_first(edited, needle, replacement)
            any_change |= changed
            if not changed:
                break
    return edited, any_change


def counter_speech_counterfactual(text: str, seed: int = 0) -> tuple[str, bool]:
    """Quote the harmful text inside explicit condemnation; the label becomes Non-Harm."""
    if not text.strip():
        return text, False
    template = COUNTER_SPEECH_TEMPLATES[seed % len(COUNTER_SPEECH_TEMPLATES)]
    return template.format(text=text.strip()), True


def build_lexical_counterfactuals(frame, seed: int = 42) -> dict[str, list]:
    """Split the frame into the three perturbation sets Table 8 needs."""
    preserving, harm_removing = [], []
    for position, (_, row) in enumerate(frame.iterrows()):
        text = str(row["ocr_text"])
        harmful = str(row["harmfulness"]) != "Non-Harm"

        edited, changed = synonym_counterfactual(text, seed + position)
        if changed:
            preserving.append({"index": position, "text": edited, "original": text})

        if harmful:
            edited, changed = benign_counterfactual(text)
            if changed:
                harm_removing.append({"index": position, "text": edited, "original": text, "kind": "benign"})
            edited, changed = counter_speech_counterfactual(text, seed + position)
            if changed:
                harm_removing.append({"index": position, "text": edited, "original": text, "kind": "counter_speech"})
    return {"meaning_preserving": preserving, "harm_removing": harm_removing}
