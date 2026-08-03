#!/usr/bin/env python3
"""Train and evaluate SheHarm-CAR. Independently runnable.

    python experiments/train_sheharm.py --config configs/default.yaml
    python experiments/train_sheharm.py --config configs/default.yaml --seeds 42 43 44
    python experiments/train_sheharm.py --config configs/default.yaml --smoke   # 2 epochs, tiny

Writes results/<name>.json with per-seed metrics and the seed-averaged summary.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from common import (  # noqa: E402
    aggregate_seeds, deep_update, format_metric, load_config, resolve_device,
    train_and_evaluate, write_latex, write_results,
)

REPORT_KEYS = ["target_f1", "harm_f1", "category_f1", "joint", "bertscore", "cf_faithfulness"]


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", type=Path, default=Path("configs/default.yaml"))
    parser.add_argument("--seeds", type=int, nargs="*", default=None, help="Overrides eval.seeds")
    parser.add_argument("--device", default=None)
    parser.add_argument("--name", default=None, help="Results file stem")
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--split", default="test", choices=["dev", "test"])
    parser.add_argument("--smoke", action="store_true", help="2 epochs, batch 8, one seed, no BERTScore")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    config = load_config(args.config)
    name = args.name or f"train_{args.config.stem}"

    if args.smoke:
        config = deep_update(config, {
            "train": {"epochs": 2, "batch_size": 8, "patience": 2, "num_workers": 0, "log_every": 10},
            "eval": {"seeds": [42]},
        })
        name = f"smoke_{args.config.stem}"

    seeds = args.seeds if args.seeds else config["eval"]["seeds"]
    device = resolve_device(args.device)
    print(f"config={args.config} device={device} seeds={seeds} split={args.split}")

    runs = []
    for seed in seeds:
        print(f"\n===== seed {seed} =====")
        output_dir = args.output_dir or f"{config['train']['output_dir']}/seed{seed}"
        runs.append(train_and_evaluate(config, seed, device, output_dir, evaluate_split=args.split))
        printable = {key: round(runs[-1]["metrics"].get(key, float("nan")), 2) for key in REPORT_KEYS}
        print(f"seed {seed} {args.split}: {printable}")

    summary = aggregate_seeds(runs)
    write_results(name, {
        "config": str(args.config), "seeds": seeds, "split": args.split,
        "per_seed": [{"seed": run["seed"], "metrics": run["metrics"]} for run in runs],
        "summary": summary,
    })
    write_latex(
        name,
        ["Model", "Tgt.-F1", "Harm-F1", "Cat.-F1", "Joint-F1", "BERTScore", "CF-Faith."],
        [["\\textbf{SheHarm-CAR}"] + [
            f"\\textbf{{{format_metric(summary.get(key, {}).get('mean'))}}}" for key in REPORT_KEYS
        ]],
        caption=f"seeds {seeds}, split {args.split}",
    )

    print(f"\n{'metric':20s} {'mean':>8s} {'std':>7s}")
    for key in REPORT_KEYS:
        if key in summary:
            print(f"{key:20s} {summary[key]['mean']:8.2f} {summary[key]['std']:7.2f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
