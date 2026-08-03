#!/usr/bin/env python3
"""Table `tab:cross_dataset` — public harmful-meme benchmarks (FHM, MAMI, HarMeme).

    python experiments/table11_cross_dataset.py --dataset harmeme
    python experiments/table11_cross_dataset.py --dataset fhm --seeds 42 43 44

These benchmarks are binary and carry no target, category, or rationale annotation, so per
the paper we "retain the multimodal, ontology, rule-reasoning, and confidence-gated
components and optimize only the supported classification objectives" — the model is built
with a two-way harm head and the target/category/rationale objectives disabled.

DATA
Each benchmark needs a licence acceptance and cannot be fetched automatically. Place them as:

    dataset/public/<name>/images/...
    dataset/public/<name>/{train,dev,test}.jsonl     one JSON object per line:
        {"image": "img/00001.png", "text": "...", "label": 0}

  fhm      https://hatefulmemeschallenge.com           (registration required)
  mami     https://competitions.codalab.org/competitions/34175   (SemEval-2022 Task 5)
  harmeme  https://github.com/di-dimitrov/harmeme      (Findings of EMNLP 2021)
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image
from sklearn.metrics import accuracy_score, roc_auc_score
from torch.utils.data import DataLoader, Dataset

sys.path.insert(0, str(Path(__file__).resolve().parent))

from common import (  # noqa: E402
    build_tokenizer_and_processor, deep_update, load_config, resolve_device, write_latex, write_results,
)
from sheharm.knowledge import build_embeddings, load_knowledge  # noqa: E402
from sheharm.models import SheHarmCAR  # noqa: E402
from sheharm.trainer import TrainConfig, enable_determinism, build_optimizer, move_batch  # noqa: E402

BENCHMARKS = {"fhm": "FHM", "mami": "MAMI", "harmeme": "HarMeme"}


class BinaryMemeDataset(Dataset):
    def __init__(self, records, image_root, tokenizer, processor, max_text_len=128):
        self.records = records
        self.image_root = Path(image_root)
        self.tokenizer, self.processor = tokenizer, processor
        self.max_text_len = max_text_len

    def __len__(self):
        return len(self.records)

    def __getitem__(self, index):
        record = self.records[index]
        image = Image.open(self.image_root / record["image"]).convert("RGB")
        encoded = self.tokenizer(str(record.get("text", "")), truncation=True, padding="max_length",
                                 max_length=self.max_text_len, return_tensors="pt")
        pixel_values = self.processor(images=image, return_tensors="pt")["pixel_values"].squeeze(0)
        image.close()
        return {
            "pixel_values": pixel_values,
            "input_ids": encoded["input_ids"].squeeze(0),
            "attention_mask": encoded["attention_mask"].squeeze(0),
            "harm_labels": torch.tensor(int(record["label"]), dtype=torch.long),
        }


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def build_binary_model(config, tokenizer, device):
    knowledge = load_knowledge(config["knowledge"]["ontology"], config["knowledge"]["rules"])
    target_embeddings, concept_embeddings = build_embeddings(
        knowledge, config["model"]["text_model"], device="cpu", cache=config["knowledge"].get("embedding_cache"))
    m = config["model"]
    return SheHarmCAR(
        target_embeddings=target_embeddings, concept_embeddings=concept_embeddings,
        rules=knowledge.rules, predicates=knowledge.predicates,
        vocab_size=len(tokenizer), pad_token_id=tokenizer.pad_token_id,
        hidden_size=m["hidden_size"], dropout=m["dropout"], top_k=m["top_k"],
        temperature=m["temperature"], num_hard_negatives=m["num_hard_negatives"],
        rule_threshold=m["rule_threshold"], vision_model=m["vision_model"], text_model=m["text_model"],
        cross_modal_layers=m.get("cross_modal_layers", 2),
        num_harm_classes=2, use_target_loss=False, use_category_loss=False, use_rationale_loss=False,
        use_consistency_loss=False, use_counterfactual_loss=False,
    ).to(device)


@torch.no_grad()
def score(model, loader, device):
    model.eval()
    probabilities, labels = [], []
    for batch in loader:
        batch = move_batch(batch, device)
        output = model(pixel_values=batch["pixel_values"], input_ids=batch["input_ids"],
                       attention_mask=batch["attention_mask"], compute_counterfactuals=False)
        probabilities.extend(torch.softmax(output.harm_logits, dim=-1)[:, 1].cpu().tolist())
        labels.extend(batch["harm_labels"].cpu().tolist())
    predictions = [int(p >= 0.5) for p in probabilities]
    return {
        "auroc": float(roc_auc_score(labels, probabilities)) * 100 if len(set(labels)) > 1 else float("nan"),
        "accuracy": float(accuracy_score(labels, predictions)) * 100,
        "n": len(labels),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", type=Path, default=Path("configs/default.yaml"))
    parser.add_argument("--dataset", nargs="*", default=list(BENCHMARKS), choices=list(BENCHMARKS))
    parser.add_argument("--root", type=Path, default=Path("dataset/public"))
    parser.add_argument("--seeds", type=int, nargs="*", default=None)
    parser.add_argument("--device", default=None)
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()

    config = load_config(args.config)
    if args.smoke:
        config = deep_update(config, {"train": {"epochs": 1, "batch_size": 8, "num_workers": 0}})
    seeds = args.seeds if args.seeds else config["eval"]["seeds"]
    device = resolve_device(args.device)
    tokenizer, processor = build_tokenizer_and_processor(config)

    results, missing = {}, []
    for name in args.dataset:
        directory = args.root / name
        if not (directory / "train.jsonl").exists():
            missing.append(name)
            print(f"skipping {name}: {directory}/train.jsonl not found — see this file's docstring")
            continue

        splits = {
            split: BinaryMemeDataset(read_jsonl(directory / f"{split}.jsonl"), directory / "images",
                                     tokenizer, processor, config["data"]["max_text_len"])
            for split in ("train", "dev", "test") if (directory / f"{split}.jsonl").exists()
        }
        per_seed = []
        for seed in seeds:
            enable_determinism(seed)
            train_config = TrainConfig(**config["train"])
            train_config.seed = seed
            model = build_binary_model(config, tokenizer, device)
            optimizer = build_optimizer(model, train_config)
            loader = DataLoader(splits["train"], batch_size=train_config.batch_size, shuffle=True,
                                num_workers=train_config.num_workers)
            scaler = torch.amp.GradScaler("cuda", enabled=train_config.amp and device.type == "cuda")
            for epoch in range(train_config.epochs):
                model.train()
                for batch in loader:
                    batch = move_batch(batch, device)
                    optimizer.zero_grad(set_to_none=True)
                    with torch.autocast(device_type=device.type, dtype=torch.float16,
                                        enabled=train_config.amp and device.type == "cuda"):
                        loss = model(**batch).total_loss
                    scaler.scale(loss).backward()
                    scaler.step(optimizer)
                    scaler.update()
                print(f"  {name} seed {seed} epoch {epoch+1}/{train_config.epochs}", flush=True)
            evaluation = splits.get("test") or splits.get("dev")
            per_seed.append(score(model, DataLoader(evaluation, batch_size=train_config.batch_size), device))
            del model
            torch.cuda.empty_cache()

        results[name] = {
            "auroc": float(np.nanmean([r["auroc"] for r in per_seed])),
            "accuracy": float(np.mean([r["accuracy"] for r in per_seed])),
            "per_seed": per_seed,
        }
        print(f"{name}: AUROC {results[name]['auroc']:.2f} Acc {results[name]['accuracy']:.2f}")

    write_results("table11_cross_dataset", {"seeds": seeds, "results": results, "missing_datasets": missing})
    if results:
        header = ["Model"] + [f"{BENCHMARKS[n]} {m}" for n in results for m in ("AUROC", "Acc.")]
        row = ["\\textbf{SheHarm-CAR}"] + [
            f"\\textbf{{{results[n][m]:.2f}}}" for n in results for m in ("auroc", "accuracy")
        ]
        write_latex("table11_cross_dataset", header, [row],
                    caption="Cross-dataset evaluation on public harmful-meme benchmarks.")
    if missing:
        print(f"\nnot evaluated (data absent): {', '.join(missing)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
