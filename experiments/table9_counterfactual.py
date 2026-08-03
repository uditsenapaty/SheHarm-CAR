#!/usr/bin/env python3
"""Table `tab:counterfactual-analysis` — evidence interventions. Independently runnable.

    python experiments/table9_counterfactual.py --checkpoint runs/default/seed42/best_model.pt

Reports the mean confidence decrease and prediction-flip rate after suppressing the target,
masking the most relevant visual region, removing the top retrieved concept, deactivating
the strongest rule, and masking an irrelevant region (the control).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parent))

from common import (  # noqa: E402
    build_datasets, build_model, build_tokenizer_and_processor, load_config,
    resolve_device, write_latex, write_results,
)
from sheharm.evaluate import intervention_analysis  # noqa: E402
from sheharm.trainer import enable_determinism  # noqa: E402

INTERVENTIONS = [
    ("suppress_target", "Suppress target"),
    ("mask_relevant_region", "Mask relevant region"),
    ("remove_top_concept", "Remove top concept"),
    ("deactivate_strongest_rule", "Deactivate strongest rule"),
    ("mask_irrelevant_region", "Mask irrelevant region"),
]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", type=Path, default=Path("configs/default.yaml"))
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--split", default="test", choices=["dev", "test"])
    parser.add_argument("--device", default=None)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    config = load_config(args.config)
    device = resolve_device(args.device)
    enable_determinism(args.seed)

    tokenizer, processor = build_tokenizer_and_processor(config)
    datasets, _ = build_datasets(config, tokenizer, processor)
    model = build_model(config, tokenizer, device)
    model.load_state_dict(torch.load(args.checkpoint, map_location=device))
    model.eval()

    loader = DataLoader(datasets[args.split], batch_size=args.batch_size, shuffle=False, num_workers=0)
    analysis = intervention_analysis(model, loader, device, [name for name, _ in INTERVENTIONS])

    write_results("table9_counterfactual", {
        "config": str(args.config), "checkpoint": str(args.checkpoint),
        "split": args.split, "interventions": analysis,
    })
    rows = [
        [label, f"{analysis[name]['delta_confidence']:.1f}", f"{analysis[name]['flip_rate']:.1f}"]
        for name, label in INTERVENTIONS
    ]
    write_latex("table9_counterfactual", ["Intervention", "$\\Delta$ Conf.", "Flip Rate"], rows,
                caption="Counterfactual evidence analysis.")

    print(f"\n{'intervention':30s} {'dConf':>8s} {'flip%':>8s}")
    for name, label in INTERVENTIONS:
        print(f"{label:30s} {analysis[name]['delta_confidence']:8.1f} {analysis[name]['flip_rate']:8.1f}")
    print("\nSanity: relevant interventions should exceed the irrelevant-region control on both columns.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
