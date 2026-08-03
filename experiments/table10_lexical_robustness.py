#!/usr/bin/env python3
"""Table `tab:lexical-robustness` — robustness to counterfactual lexical substitution.

    python experiments/table10_lexical_robustness.py --checkpoint runs/default/seed42/best_model.pt

    CFR    counterfactual flip rate: prediction changes under a *meaning-preserving* swap.
           A faithful model keeps its label; a lexically-driven one flips.  Lower is better.
    FPR-H  false-positive rate on *harm-removing* counterfactuals: the harmful cue is either
           replaced by a benign phrase or quoted inside explicit condemnation, so the label
           becomes Non-Harm. Predicting harm is a false positive.  Lower is better.
    RS     rationale stability: BERTScore between the rationale before and after a
           meaning-preserving swap.  Higher is better.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent))

from common import (  # noqa: E402
    build_datasets, build_model, build_tokenizer_and_processor, load_config,
    resolve_device, write_latex, write_results,
)
from sheharm.data.counterfactuals import build_lexical_counterfactuals  # noqa: E402
from sheharm.labels import NON_HARM_ID  # noqa: E402
from sheharm.metrics import bertscore  # noqa: E402
from sheharm.trainer import enable_determinism  # noqa: E402


@torch.no_grad()
def run_texts(model, frame, indices, texts, tokenizer, processor, image_root, device,
              max_text_len, batch_size=16, want_rationales=False):
    """Score a list of (row, replacement text) pairs; optionally decode rationales."""
    predictions, rationales = [], []
    for start in range(0, len(indices), batch_size):
        chunk = list(zip(indices[start : start + batch_size], texts[start : start + batch_size]))
        images = [Image.open(image_root / frame.iloc[i]["image_path"]).convert("RGB") for i, _ in chunk]
        pixel_values = processor(images=images, return_tensors="pt")["pixel_values"].to(device)
        encoded = tokenizer([t for _, t in chunk], truncation=True, padding="max_length",
                            max_length=max_text_len, return_tensors="pt").to(device)
        output = model(
            pixel_values=pixel_values, input_ids=encoded["input_ids"],
            attention_mask=encoded["attention_mask"], compute_counterfactuals=False,
            ocr_text=[t for _, t in chunk],
        )
        predictions.extend(output.harm_logits.argmax(dim=-1).cpu().tolist())
        if want_rationales:
            generated = model.generate_rationale(
                output.extras["memory"], tokenizer.bos_token_id or tokenizer.cls_token_id,
                tokenizer.eos_token_id or tokenizer.sep_token_id,
            )
            rationales.extend(tokenizer.batch_decode(generated, skip_special_tokens=True))
        for image in images:
            image.close()
    return predictions, rationales


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", type=Path, default=Path("configs/default.yaml"))
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--split", default="test", choices=["dev", "test"])
    parser.add_argument("--device", default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--label", default="SheHarm-CAR")
    args = parser.parse_args()

    config = load_config(args.config)
    device = resolve_device(args.device)
    enable_determinism(args.seed)

    tokenizer, processor = build_tokenizer_and_processor(config)
    datasets, _ = build_datasets(config, tokenizer, processor)
    frame = datasets[args.split].frame
    model = build_model(config, tokenizer, device)
    model.load_state_dict(torch.load(args.checkpoint, map_location=device))
    model.eval()

    image_root = Path(config["data"]["image_root"])
    max_text_len = config["data"]["max_text_len"]
    counterfactuals = build_lexical_counterfactuals(frame, args.seed)

    common = dict(tokenizer=tokenizer, processor=processor, image_root=image_root,
                  device=device, max_text_len=max_text_len)

    preserving = counterfactuals["meaning_preserving"]
    indices = [item["index"] for item in preserving]
    base_predictions, base_rationales = run_texts(
        model, frame, indices, [item["original"] for item in preserving], want_rationales=True, **common)
    edited_predictions, edited_rationales = run_texts(
        model, frame, indices, [item["text"] for item in preserving], want_rationales=True, **common)
    flip_rate = float(np.mean([a != b for a, b in zip(base_predictions, edited_predictions)])) * 100
    stability = bertscore(edited_rationales, base_rationales, config["eval"]["bertscore_model"], str(device))

    removing = counterfactuals["harm_removing"]
    removed_predictions, _ = run_texts(
        model, frame, [item["index"] for item in removing], [item["text"] for item in removing], **common)
    false_positive_rate = float(np.mean([p != NON_HARM_ID for p in removed_predictions])) * 100

    results = {
        "model": args.label, "split": args.split, "checkpoint": str(args.checkpoint),
        "CFR": flip_rate, "FPR_H": false_positive_rate, "RS": stability,
        "n_meaning_preserving": len(preserving), "n_harm_removing": len(removing),
    }
    write_results("table10_lexical_robustness", results)
    write_latex("table10_lexical_robustness",
                ["Model", "CFR$\\downarrow$", "FPR-H$\\downarrow$", "RS$\\uparrow$"],
                [[f"\\textbf{{{args.label}}}", f"{flip_rate:.1f}", f"{false_positive_rate:.1f}", f"{stability:.1f}"]],
                caption="Robustness under lexical counterfactuals.")

    print(f"\nCFR   {flip_rate:6.1f}  (lower better, n={len(preserving)})")
    print(f"FPR-H {false_positive_rate:6.1f}  (lower better, n={len(removing)})")
    print(f"RS    {stability:6.1f}  (higher better)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
