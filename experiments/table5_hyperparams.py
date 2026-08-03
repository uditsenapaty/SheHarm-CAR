#!/usr/bin/env python3
"""Table `tab:hyperparameters` — emit the settings actually used, straight from the config.

    python experiments/table5_hyperparams.py --config configs/default.yaml

Generating this from the config rather than transcribing it by hand means the reported table
cannot drift away from the code that produced the numbers.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from common import load_config, write_latex, write_results  # noqa: E402


def scientific(value: float) -> str:
    """1e-05 -> $1\\times10^{-5}$"""
    mantissa, exponent = f"{value:.0e}".split("e")
    return f"${mantissa}\\times10^{{{int(exponent)}}}$"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", type=Path, default=Path("configs/default.yaml"))
    args = parser.parse_args()
    config = load_config(args.config)
    model, data, loss, train, evaluation = (
        config["model"], config["data"], config["loss"], config["train"], config["eval"]
    )

    rows = [
        ("Encoders", f"{model['vision_model'].split('/')[-1]}; {model['text_model']}"),
        ("Input size", f"$224\\times224$; {data['max_text_len']} OCR tokens"),
        ("Hidden size / dropout", f"{model['hidden_size']} / {model['dropout']}"),
        ("Cross-modal layers", str(model.get("cross_modal_layers", 1))),
        ("Ontology retrieval",
         f"$K={model['top_k']}$, $\\tau={model['temperature']}$, {model['num_hard_negatives']} hard negatives"),
        ("Rule threshold", str(model["rule_threshold"])),
        ("Encoder learning rate", scientific(train["encoder_lr"])),
        ("New-layer learning rate", scientific(train["new_lr"])),
        ("Weight decay", str(train["weight_decay"])),
        ("Batch / epochs / patience", f"{train['batch_size']} / {train['epochs']} / {train['patience']}"),
        ("Rationale decoding", f"Beam size {model['beam_size']}; maximum {data['max_rationale_len']} tokens"),
        ("Task-loss weights",
         f"$\\lambda_{{\\mathrm{{harm}}}}={loss['lambda_harm']}$, "
         f"$\\lambda_{{\\mathrm{{cat}}}}={loss['lambda_cat']}$, $\\lambda_{{\\mathrm{{rat}}}}={loss['lambda_rat']}$"),
        ("Auxiliary-loss weights",
         f"$\\lambda_{{\\mathrm{{align}}}}={loss['lambda_align']}$, "
         f"$\\lambda_{{\\mathrm{{cons}}}}={loss['lambda_cons']}$, $\\lambda_{{\\mathrm{{cf}}}}={loss['lambda_cf']}$"),
        ("Counterfactual invariance", f"$\\lambda_{{\\mathrm{{inv}}}}={loss['lambda_inv']}$"),
        ("Optimizer / precision", "AdamW / mixed precision" if train["amp"] else "AdamW / fp32"),
        ("Balanced sampling", "yes" if train.get("balanced_sampling") else "no"),
        ("Weight EMA", str(train.get("ema_decay") or "off")),
        ("Runs / hardware", f"{len(evaluation['seeds'])} seeds / single GPU"),
    ]

    write_results("table5_hyperparameters", {"config": str(args.config), "settings": dict(rows)})
    write_latex("table5_hyperparameters", ["Setting", "Value"], [list(row) for row in rows],
                caption="Hyperparameter settings for SheHarm-CAR.")
    for name, value in rows:
        print(f"{name:28s} {value}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
