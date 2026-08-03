"""Stratified 80/10/10 partitioning.

Paper Section `subsec:dataset`: "We partition the dataset into training, validation, and
test sets using an approximately 80/10/10 split... Stratification is performed jointly over
the harmfulness and harm-category labels. Duplicate and near-duplicate memes are kept within
the same split to prevent information leakage."

Joint stratification is done on the string `harmfulness|harm_category`. Strata too small to
place in every split fall back to train-only, which is reported rather than hidden.
"""

from __future__ import annotations

import hashlib

import pandas as pd


def stratum_key(row) -> str:
    return f"{row['harmfulness']}|{row['harm_category']}"


def duplicate_group(frame: pd.DataFrame, column: str = "ocr_text") -> pd.Series:
    """Near-duplicate grouping by normalized OCR text; unique rows group with themselves."""
    normalized = (
        frame[column].fillna("").astype(str).str.lower().str.replace(r"[^a-z0-9 ]", "", regex=True).str.strip()
    )
    digest = normalized.map(lambda value: hashlib.md5(value.encode()).hexdigest() if value else "")
    counts = digest.value_counts()
    singleton = digest.map(lambda value: value == "" or counts.get(value, 0) < 2)
    return digest.where(~singleton, other=pd.Series(frame.index, index=frame.index).map(lambda i: f"row{i}"))


def assign_splits(
    frame: pd.DataFrame,
    train_ratio: float = 0.8,
    dev_ratio: float = 0.1,
    seed: int = 42,
    respect_duplicates: bool = True,
) -> tuple[pd.Series, dict]:
    """Return the split column plus a report of any stratum that could not be split."""
    frame = frame.reset_index(drop=True)
    strata = frame.apply(stratum_key, axis=1)
    groups = duplicate_group(frame) if respect_duplicates else pd.Series(
        [f"row{i}" for i in range(len(frame))], index=frame.index
    )

    splits = pd.Series(["train"] * len(frame), index=frame.index, dtype=object)
    undersized: dict[str, int] = {}

    for stratum in sorted(strata.unique()):
        member_index = frame.index[strata == stratum]
        # A duplicate group must land in exactly one split, so shuffle groups, not rows.
        unique_groups = sorted(groups.loc[member_index].unique())
        rng = _rng(seed, stratum)
        rng.shuffle(unique_groups)

        sizes = {g: int((groups.loc[member_index] == g).sum()) for g in unique_groups}
        total = sum(sizes.values())
        if total < 3:
            undersized[stratum] = total
            continue

        dev_quota = max(1, round(total * dev_ratio))
        test_quota = max(1, round(total * (1.0 - train_ratio - dev_ratio)))
        assigned_dev = assigned_test = 0
        for group in unique_groups:
            size = sizes[group]
            members = member_index[groups.loc[member_index] == group]
            if assigned_dev < dev_quota:
                splits.loc[members] = "dev"
                assigned_dev += size
            elif assigned_test < test_quota:
                splits.loc[members] = "test"
                assigned_test += size
            else:
                splits.loc[members] = "train"
    return splits, {"strata_too_small_for_dev_test": undersized}


def _rng(seed: int, stratum: str):
    import random

    digest = hashlib.md5(f"{seed}:{stratum}".encode()).hexdigest()
    return random.Random(int(digest[:8], 16))


def split_report(frame: pd.DataFrame, split_column: str = "split") -> pd.DataFrame:
    """Table 1: split x harmfulness counts."""
    table = pd.crosstab(frame[split_column], frame["harmfulness"])
    order = [name for name in ("train", "dev", "test") if name in table.index]
    columns = [c for c in ("Explicit-Harm", "Implicit-Harm", "Non-Harm") if c in table.columns]
    table = table.loc[order, columns]
    table["Total"] = table.sum(axis=1)
    table.loc["Total"] = table.sum(axis=0)
    return table
