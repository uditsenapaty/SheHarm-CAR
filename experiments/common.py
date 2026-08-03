"""Shared setup for every table script: config loading, model/data construction, result IO.

Each experiment script is independently runnable and depends only on this module, so a
single table can be reproduced without touching the others.
"""

from __future__ import annotations

import copy
import json
import sys
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, CLIPImageProcessor

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from sheharm.data.dataset import SheHarmDataset, load_splits  # noqa: E402
from sheharm.evaluate import evaluate  # noqa: E402
from sheharm.knowledge import build_embeddings, load_knowledge  # noqa: E402
from sheharm.models import SheHarmCAR  # noqa: E402
from sheharm.trainer import TrainConfig, enable_determinism, fit  # noqa: E402


def load_config(path: str | Path) -> dict:
    with open(path, encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def deep_update(base: dict, overrides: dict) -> dict:
    result = copy.deepcopy(base)
    for key, value in overrides.items():
        if isinstance(value, dict) and isinstance(result.get(key), dict):
            result[key] = deep_update(result[key], value)
        else:
            result[key] = value
    return result


def resolve_device(name: str | None = None) -> torch.device:
    if name:
        return torch.device(name)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def build_tokenizer_and_processor(config: dict):
    tokenizer = AutoTokenizer.from_pretrained(config["model"]["text_model"], use_fast=True)
    processor = CLIPImageProcessor.from_pretrained(config["model"]["vision_model"])
    return tokenizer, processor


def build_datasets(config: dict, tokenizer, processor):
    inventory = json.loads(Path(config["data"]["target_inventory"]).read_text(encoding="utf-8"))
    splits = load_splits(config["data"]["csv"])
    datasets = {
        name: SheHarmDataset(
            frame, config["data"]["image_root"], tokenizer, processor,
            inventory["concepts"], config["data"]["max_text_len"], config["data"]["max_rationale_len"],
        )
        for name, frame in splits.items()
    }
    return datasets, inventory


def build_model(config: dict, tokenizer, device: torch.device) -> SheHarmCAR:
    knowledge = load_knowledge(config["knowledge"]["ontology"], config["knowledge"]["rules"])
    target_embeddings, concept_embeddings = build_embeddings(
        knowledge, config["model"]["text_model"], device="cpu", cache=config["knowledge"].get("embedding_cache")
    )
    model_config, loss_config = config["model"], config["loss"]
    model = SheHarmCAR(
        target_embeddings=target_embeddings,
        concept_embeddings=concept_embeddings,
        rules=knowledge.rules,
        predicates=knowledge.predicates,
        vocab_size=len(tokenizer),
        pad_token_id=tokenizer.pad_token_id,
        hidden_size=model_config["hidden_size"],
        dropout=model_config["dropout"],
        top_k=model_config["top_k"],
        temperature=model_config["temperature"],
        num_hard_negatives=model_config["num_hard_negatives"],
        rule_threshold=model_config["rule_threshold"],
        max_rationale_len=config["data"]["max_rationale_len"],
        beam_size=model_config["beam_size"],
        vision_model=model_config["vision_model"],
        text_model=model_config["text_model"],
        freeze_encoders=model_config["freeze_encoders"],
        cross_modal_layers=model_config.get("cross_modal_layers", 2),
        tied_target_head=model_config.get("tied_target_head", True),
        rationale_decoder=model_config.get("rationale_decoder", "bart"),
        rationale_model=model_config.get("rationale_model", "facebook/bart-base"),
        cf_margin=model_config["cf_margin"],
        use_target_conditioning=model_config["use_target_conditioning"],
        use_ontology_retrieval=model_config["use_ontology_retrieval"],
        use_exception_rules=model_config["use_exception_rules"],
        use_confidence_gate=model_config["use_confidence_gate"],
        use_consistency_loss=model_config["use_consistency_loss"],
        use_counterfactual_loss=model_config["use_counterfactual_loss"],
        **{f"lambda_{k}": v for k, v in ((key.replace("lambda_", ""), value)
                                         for key, value in loss_config.items())},
    )
    return model.to(device)


def make_train_config(config: dict, seed: int, output_dir: str | None = None) -> TrainConfig:
    train_config = TrainConfig(**config["train"])
    train_config.seed = seed
    if output_dir:
        train_config.output_dir = output_dir
    return train_config


def build_baseline(kind: str, config: dict, tokenizer, device: torch.device):
    """Encoder / knowledge baselines with the same task heads (paper: `subsec:baselines`)."""
    from sheharm.baselines.encoder_baselines import EncoderBaseline
    from sheharm.baselines.knowledge_baselines import KnowledgeBaseline

    inventory = json.loads(Path(config["data"]["target_inventory"]).read_text(encoding="utf-8"))
    shared = dict(
        num_targets=len(inventory["concepts"]),
        vocab_size=len(tokenizer),
        pad_token_id=tokenizer.pad_token_id,
        hidden_size=config["model"]["hidden_size"],
        dropout=config["model"]["dropout"],
        max_rationale_len=config["data"]["max_rationale_len"],
        beam_size=config["model"]["beam_size"],
    )
    if kind in ("roberta_text", "vilt", "hate_clipper", "clip"):
        return EncoderBaseline(kind, **shared).to(device)

    knowledge = load_knowledge(config["knowledge"]["ontology"], config["knowledge"]["rules"])
    _, concept_embeddings = build_embeddings(
        knowledge, config["model"]["text_model"], device="cpu", cache=config["knowledge"].get("embedding_cache")
    )
    return KnowledgeBaseline(kind, concept_embeddings=concept_embeddings, **shared).to(device)


def train_and_evaluate(config: dict, seed: int, device: torch.device, output_dir: str | None = None,
                       evaluate_split: str = "test", model_builder=None) -> dict:
    """One full train + test cycle at one seed."""
    enable_determinism(seed)
    tokenizer, processor = build_tokenizer_and_processor(config)
    datasets, _ = build_datasets(config, tokenizer, processor)
    model = (model_builder or build_model)(config, tokenizer, device)
    train_config = make_train_config(config, seed, output_dir)

    summary = fit(model, datasets["train"], datasets["dev"], tokenizer, train_config, device)
    loader = DataLoader(
        datasets[evaluate_split], batch_size=train_config.batch_size, shuffle=False,
        num_workers=train_config.num_workers, pin_memory=(device.type == "cuda"),
    )
    metrics = evaluate(
        model, loader, tokenizer, device,
        compute_bertscore=True, compute_counterfactuals=True,
        bertscore_model=config["eval"]["bertscore_model"],
    )
    metrics["best_dev_mean_f1"] = summary["best_dev_mean_f1"]
    return {"metrics": metrics, "checkpoint": summary["checkpoint"], "seed": seed,
            "train_config": asdict(train_config)}


def aggregate_seeds(runs: list[dict]) -> dict[str, dict[str, float]]:
    """Mean and standard deviation over seeds, as the paper reports."""
    keys = sorted({key for run in runs for key in run["metrics"]})
    aggregated = {}
    for key in keys:
        values = [run["metrics"][key] for run in runs if key in run["metrics"]]
        values = [v for v in values if v is not None and not (isinstance(v, float) and np.isnan(v))]
        if values:
            aggregated[key] = {"mean": float(np.mean(values)), "std": float(np.std(values)), "n": len(values)}
    return aggregated


def write_results(name: str, payload: dict, results_dir: str | Path = "results") -> Path:
    directory = Path(results_dir)
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"{name}.json"
    path.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
    print(f"\nwrote {path}")
    return path


def write_latex(name: str, header: list[str], rows: list[list[str]], caption: str = "",
                results_dir: str | Path = "results") -> Path:
    directory = Path(results_dir)
    directory.mkdir(parents=True, exist_ok=True)
    alignment = "l" + "r" * (len(header) - 1)
    lines = [f"\\begin{{tabular}}{{{alignment}}}", "\\toprule",
             " & ".join(f"\\textbf{{{column}}}" for column in header) + " \\\\", "\\midrule"]
    lines.extend(" & ".join(cell for cell in row) + " \\\\" for row in rows)
    lines.extend(["\\bottomrule", "\\end{tabular}"])
    if caption:
        lines.append(f"% {caption}")
    path = directory / f"{name}.tex"
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"wrote {path}")
    return path


def format_metric(value: float | None, decimals: int = 2) -> str:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return "--"
    return f"{value:.{decimals}f}"
