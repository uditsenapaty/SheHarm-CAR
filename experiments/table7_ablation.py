#!/usr/bin/env python3
"""Table `tab:ablation` — component ablations. Independently runnable.

    python experiments/table7_ablation.py --config configs/default.yaml
    python experiments/table7_ablation.py --variants no_exception_rules full --seeds 42

Trains one model per variant and reports Harm-F1, Joint-F1 and CF-Faith.
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

VARIANTS = {
    "no_target_conditioning": ("w/o Target Conditioning", {"model": {"use_target_conditioning": False}}),
    "no_ontology_retrieval": ("w/o Ontology Retrieval", {"model": {"use_ontology_retrieval": False}}),
    "no_exception_rules": ("w/o Exception Rules", {"model": {"use_exception_rules": False}}),
    "no_confidence_gate": ("w/o Confidence Gate", {"model": {"use_confidence_gate": False}}),
    "no_consistency": (r"w/o $\mathcal{L}_{\mathrm{cons}}$", {"model": {"use_consistency_loss": False}}),
    "no_counterfactual": (r"w/o $\mathcal{L}_{\mathrm{cf}}$", {"model": {"use_counterfactual_loss": False}}),
    "full": (r"\textbf{Full Model}", {}),
}
REPORT_KEYS = ["harm_f1", "joint", "cf_faithfulness"]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", type=Path, default=Path("configs/default.yaml"))
    parser.add_argument("--variants", nargs="*", default=list(VARIANTS), choices=list(VARIANTS))
    parser.add_argument("--seeds", type=int, nargs="*", default=None)
    parser.add_argument("--device", default=None)
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()

    base = load_config(args.config)
    if args.smoke:
        base = deep_update(base, {"train": {"epochs": 2, "batch_size": 8, "patience": 2, "num_workers": 0}})
    seeds = args.seeds if args.seeds else base["eval"]["seeds"]
    device = resolve_device(args.device)

    results = {}
    for key in args.variants:
        label, overrides = VARIANTS[key]
        print(f"\n===== {key} =====")
        config = deep_update(base, overrides)
        runs = [
            train_and_evaluate(config, seed, device, f"runs/ablation/{key}/seed{seed}")
            for seed in seeds
        ]
        results[key] = {"label": label, "summary": aggregate_seeds(runs),
                        "per_seed": [{"seed": r["seed"], "metrics": r["metrics"]} for r in runs]}
        print(f"{key}: " + " ".join(
            f"{m}={format_metric(results[key]['summary'].get(m, {}).get('mean'))}" for m in REPORT_KEYS
        ))

    write_results("table7_ablation", {"config": str(args.config), "seeds": seeds, "variants": results})
    rows = [
        [results[key]["label"]] + [
            format_metric(results[key]["summary"].get(metric, {}).get("mean"), 2 if metric != "cf_faithfulness" else 1)
            for metric in REPORT_KEYS
        ]
        for key in args.variants if key in results
    ]
    write_latex("table7_ablation", ["Variant", "Harm-F1", "Joint-F1", "CF-Faith."], rows,
                caption="Ablation results for SheHarm-CAR.")

    print(f"\n{'variant':32s} {'Harm-F1':>8s} {'Joint-F1':>9s} {'CF-Faith':>9s}")
    for key in args.variants:
        if key in results:
            summary = results[key]["summary"]
            print(f"{key:32s} " + " ".join(
                f"{format_metric(summary.get(m, {}).get('mean')):>8s}" for m in REPORT_KEYS
            ))
    return 0


if __name__ == "__main__":
    sys.exit(main())
