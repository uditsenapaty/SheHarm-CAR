"""SheHarm-Meme torch dataset.

Consumes the final CSV produced by `scripts/build_dataset.py`:

    image_path, ocr_text, target_span, target_start, target_end, target_concept,
    harmfulness, harm_category, rationale, split

Only `image_path`, `ocr_text`, `target_concept`, `harmfulness`, `harm_category` and
`rationale` reach the model — the span columns are kept for qualitative analysis.
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset

from ..labels import CAT2ID, HARM2ID, IGNORE_INDEX, NULL_CATEGORY

REQUIRED_COLUMNS = {
    "image_path", "ocr_text", "target_concept", "harmfulness", "harm_category", "rationale", "split",
}


class SheHarmDataset(Dataset):
    def __init__(
        self,
        frame: pd.DataFrame,
        image_root: str | Path,
        tokenizer,
        image_processor,
        target_concepts: list[str],
        max_text_len: int = 128,
        max_rationale_len: int = 64,
    ):
        missing = REQUIRED_COLUMNS - set(frame.columns)
        if missing:
            raise ValueError(f"missing columns: {sorted(missing)}")
        self.frame = frame.reset_index(drop=True)
        self.image_root = Path(image_root)
        self.tokenizer = tokenizer
        self.image_processor = image_processor
        self.target2id = {name: index for index, name in enumerate(target_concepts)}
        self.max_text_len = max_text_len
        self.max_rationale_len = max_rationale_len

    def __len__(self) -> int:
        return len(self.frame)

    def _load_image(self, image_path: str) -> Image.Image:
        path = Path(str(image_path))
        if not path.is_absolute():
            path = self.image_root / path
        if not path.exists():
            raise FileNotFoundError(f"image not found: {path}")
        return Image.open(path).convert("RGB")

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        row = self.frame.iloc[index]
        text = "" if pd.isna(row["ocr_text"]) else str(row["ocr_text"])

        encoded = self.tokenizer(
            text, truncation=True, padding="max_length",
            max_length=self.max_text_len, return_tensors="pt",
        )
        pixel_values = self.image_processor(
            images=self._load_image(row["image_path"]), return_tensors="pt"
        )["pixel_values"].squeeze(0)

        rationale = "" if pd.isna(row["rationale"]) else str(row["rationale"])
        rationale_encoded = self.tokenizer(
            rationale, truncation=True, padding="max_length",
            max_length=self.max_rationale_len, return_tensors="pt",
        )

        harmfulness = HARM2ID[str(row["harmfulness"])]
        category = str(row["harm_category"])
        category_label = IGNORE_INDEX if category == NULL_CATEGORY else CAT2ID.get(category, IGNORE_INDEX)
        target_label = self.target2id.get(str(row["target_concept"]), IGNORE_INDEX)

        return {
            # Raw text travels with the batch: ViLT and CLIP baselines must tokenize with
            # their own vocabularies (BERT / CLIP-BPE), not RoBERTa's.
            "ocr_text": text,
            "pixel_values": pixel_values,
            "input_ids": encoded["input_ids"].squeeze(0),
            "attention_mask": encoded["attention_mask"].squeeze(0),
            "target_labels": torch.tensor(target_label, dtype=torch.long),
            "harm_labels": torch.tensor(harmfulness, dtype=torch.long),
            "cat_labels": torch.tensor(category_label, dtype=torch.long),
            "rationale_ids": rationale_encoded["input_ids"].squeeze(0),
        }


def load_splits(csv_path: str | Path) -> dict[str, pd.DataFrame]:
    frame = pd.read_csv(csv_path, keep_default_na=False, na_values=[""])
    frame["split"] = frame["split"].astype(str).str.lower()
    aliases = {"val": "dev", "validation": "dev"}
    frame["split"] = frame["split"].replace(aliases)
    return {name: frame[frame["split"] == name].copy() for name in ("train", "dev", "test")}


def load_target_inventory(path: str | Path) -> list[str]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    return payload["concepts"] if isinstance(payload, dict) else list(payload)
