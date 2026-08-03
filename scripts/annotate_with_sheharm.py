#!/usr/bin/env python3
"""Annotator 2: re-annotate the whole corpus with a trained SheHarm-CAR checkpoint.

    python scripts/annotate_with_sheharm.py --checkpoint runs/default/seed42/best_model.pt

Annotator 1 is the Qwen2.5-VL pass that bootstrapped training. This runs the trained model
over every meme and writes a second, independent annotation in the *same* CSV schema, so the
two passes are directly comparable:

    python experiments/table4_agreement.py --a dataset/annotations_v2.csv \\
                                           --b dataset/annotations_sheharm.csv

Output columns match the annotator exactly:
    filename, women-related target, harm_type, harm_category, rationale

The women-related target is written back as its canonical inventory name, and harm_category
is forced to NULL whenever harm_type is Non-Harmful, so the file satisfies the same schema
contract meme_annotator.py enforces.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd
import torch
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "experiments"))

from common import build_datasets, build_model, build_tokenizer_and_processor, load_config, resolve_device  # noqa: E402
from sheharm.decoding import decode  # noqa: E402
from sheharm.labels import CATEGORY_LABELS, HARMFULNESS_LABELS, NON_HARM_ID  # noqa: E402
from sheharm.trainer import enable_determinism  # noqa: E402

# The annotator CSV uses the pre-revision surface forms; keep them so both passes match.
ANNOTATOR_HARM = {"Explicit-Harm": "Explicit-Harm", "Implicit-Harm": "Implicit-Harm",
                  "Non-Harm": "Non-Harmful"}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", type=Path, default=Path("configs/default.yaml"))
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path("dataset/annotations_sheharm.csv"))
    parser.add_argument("--splits", nargs="*", default=["train", "dev", "test"])
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--device", default=None)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    config = load_config(args.config)
    device = resolve_device(args.device)
    enable_determinism(args.seed)

    tokenizer, processor = build_tokenizer_and_processor(config)
    datasets, inventory = build_datasets(config, tokenizer, processor)
    concepts = inventory["concepts"]

    model = build_model(config, tokenizer, device)
    model.load_state_dict(torch.load(args.checkpoint, map_location=device))
    model.eval()

    rows = []
    for split in args.splits:
        dataset = datasets[split]
        if len(dataset) == 0:
            continue
        frame = dataset.frame
        loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=0)
        position = 0
        with torch.no_grad():
            for batch in loader:
                tensors = {k: (v.to(device) if isinstance(v, torch.Tensor) else v) for k, v in batch.items()}
                output = model(
                    pixel_values=tensors["pixel_values"], input_ids=tensors["input_ids"],
                    attention_mask=tensors["attention_mask"], compute_counterfactuals=False,
                )
                harm, category = decode(output.harm_logits, output.category_logits, joint=True)
                target = output.target_logits.argmax(dim=-1)
                generated = model.generate_rationale(
                    output.extras["memory"], tokenizer.bos_token_id or tokenizer.cls_token_id,
                    tokenizer.eos_token_id or tokenizer.sep_token_id,
                )
                rationales = tokenizer.batch_decode(generated, skip_special_tokens=True)

                for index in range(harm.size(0)):
                    harm_label = HARMFULNESS_LABELS[int(harm[index])]
                    is_harmful = int(harm[index]) != NON_HARM_ID
                    rationale = " ".join(str(rationales[index]).split()) or "No rationale generated."
                    rows.append({
                        "filename": frame.iloc[position]["image_path"],
                        "women-related target": concepts[int(target[index])],
                        "harm_type": ANNOTATOR_HARM[harm_label],
                        "harm_category": CATEGORY_LABELS[int(category[index])] if is_harmful else "NULL",
                        "rationale": rationale,
                    })
                    position += 1
                print(f"  {split}: {position}/{len(dataset)}", end="\r", flush=True)
        print()

    out = pd.DataFrame(rows).sort_values("filename")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(args.output, index=False)

    print(f"\nwrote {args.output} ({len(out)} rows)\n")
    print("harm_type:")
    print(out["harm_type"].value_counts().to_string())
    print("\nharm_category:")
    print(out["harm_category"].value_counts().to_string())
    print(f"\ndistinct targets predicted: {out['women-related target'].nunique()}")
    print(f"\nNext: python experiments/table4_agreement.py --a dataset/annotations_v2.csv --b {args.output}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
