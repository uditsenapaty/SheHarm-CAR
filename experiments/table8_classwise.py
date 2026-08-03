#!/usr/bin/env python3
"""Table `tab:classwise-harm` — per-class harmfulness F1.

Derived from results/table6_main_results.json: `sheharm.metrics.summarize` already emits
`harm_f1_<label>` for every model, so this needs no additional training.

    python experiments/table8_classwise.py
    python experiments/table8_classwise.py --models vilt hate_clipper qwen25_vl sheharm_car
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from common import format_metric, write_latex, write_results  # noqa: E402
from sheharm.labels import HARMFULNESS_LABELS  # noqa: E402

DEFAULT_MODELS = ["vilt", "hate_clipper", "qwen25_vl", "kid_vlm", "explainhm", "sgot_r1", "sheharm_car"]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--source", type=Path, default=Path("results/table6_main_results.json"))
    parser.add_argument("--models", nargs="*", default=DEFAULT_MODELS)
    args = parser.parse_args()

    if not args.source.exists():
        raise SystemExit(f"{args.source} not found — run experiments/table6_main_results.py first.")
    payload = json.loads(args.source.read_text(encoding="utf-8"))
    available = payload["models"]

    keys = [f"harm_f1_{label}" for label in HARMFULNESS_LABELS]
    rows, collected = [], {}
    for model in args.models:
        if model not in available:
            print(f"skipping {model}: absent from {args.source}")
            continue
        summary = available[model]["summary"]
        values = [summary.get(key, {}).get("mean") for key in keys]
        collected[model] = dict(zip(HARMFULNESS_LABELS, values))
        cells = [format_metric(value, 1) for value in values]
        if model == "sheharm_car":
            cells = [f"\\textbf{{{cell}}}" for cell in cells]
        rows.append([available[model]["label"]] + cells)

    write_results("table8_classwise", {"source": str(args.source), "per_class_f1": collected})
    write_latex("table8_classwise", ["Model", "Expl.-F1", "Impl.-F1", "Non-Hm.-F1"], rows,
                caption="Class-wise harmfulness performance on SheHarm-Meme.")

    print(f"\n{'model':22s} " + " ".join(f"{label[:9]:>10s}" for label in HARMFULNESS_LABELS))
    for model, values in collected.items():
        print(f"{model:22s} " + " ".join(f"{format_metric(v, 1):>10s}" for v in values.values()))
    return 0


if __name__ == "__main__":
    sys.exit(main())
