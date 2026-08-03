#!/usr/bin/env python3
"""Table `tab:main-results` — comparison against all twelve baselines. Independently runnable.

    python experiments/table6_main_results.py --models sheharm_car roberta vilt
    python experiments/table6_main_results.py                    # everything
    python experiments/table6_main_results.py --list             # show the roster and status

Trained baselines get the same task heads, the same splits, the same recipe. Prompted VLMs
follow a fixed output schema with identical label definitions. Rows whose upstream code does
not cover this four-task setting are marked as reimplementations in the emitted table.
"""

from __future__ import annotations

import argparse
import functools
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from common import (  # noqa: E402
    aggregate_seeds, build_baseline, deep_update, format_metric, load_config,
    resolve_device, train_and_evaluate, write_latex, write_results,
)

REPORT_KEYS = ["target_f1", "harm_f1", "category_f1", "joint", "bertscore", "cf_faithfulness"]

# key -> (table label, kind, note)
ROSTER = {
    "roberta":       (r"Text-only RoBERTa \citep{liu2019roberta}", "encoder:roberta_text", ""),
    "vilt":          (r"ViLT \citep{kim2021vilt}", "encoder:vilt", ""),
    "hate_clipper":  (r"Hate-CLIPper \citep{kumar2022hate}", "encoder:hate_clipper", ""),
    "llava":         (r"LLaVA \citep{liu2023llava}", "prompted:llava", ""),
    "internvl":      (r"InternVL \citep{chen2024internvl}", "prompted:internvl", ""),
    "llama32_vision": ("Llama-3.2-Vision-11B", "prompted:llama32_vision", "gated weights"),
    "qwen25_vl":     (r"Qwen2.5-VL-7B \citep{qwen2025vl}", "prompted:qwen25_vl", "ANNOTATED OUR LABELS"),
    "kermit":        (r"KERMIT \citep{grasso2024kermit}", "knowledge:kermit", "reimplementation"),
    "kid_vlm":       (r"KID-VLM \citep{garg2025just}", "knowledge:kid_vlm", "reimplementation"),
    "intmeme":       (r"IntMeme$_{\textsc{InstructBLIP}}$ \citep{hee2025demystifying}", "knowledge:intmeme", "reimplementation"),
    "explainhm":     (r"ExplainHM++ \citep{lin2025explainhm++}", "knowledge:explainhm", "reimplementation"),
    "sgot_r1":       (r"SGoT-R1 \citep{wang2026sgotr1}", "knowledge:sgot_r1", "reimplementation"),
    "sheharm_car":   (r"\textbf{SheHarm-CAR}", "ours", ""),
}


def run_trained(key: str, kind: str, config: dict, seeds: list[int], device) -> dict:
    builder = None if kind == "ours" else functools.partial(_baseline_builder, kind.split(":", 1)[1])
    runs = [
        train_and_evaluate(config, seed, device, f"runs/table4/{key}/seed{seed}", model_builder=builder)
        for seed in seeds
    ]
    return {"summary": aggregate_seeds(runs),
            "per_seed": [{"seed": r["seed"], "metrics": r["metrics"]} for r in runs]}


def _baseline_builder(baseline_kind: str, config: dict, tokenizer, device):
    return build_baseline(baseline_kind, config, tokenizer, device)


def run_prompted(key: str, config: dict, device, limit: int | None) -> dict:
    from run_prompted_baseline import evaluate_prompted

    return evaluate_prompted(key, config, device, limit=limit)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", type=Path, default=Path("configs/default.yaml"))
    parser.add_argument("--models", nargs="*", default=None, choices=list(ROSTER))
    parser.add_argument("--seeds", type=int, nargs="*", default=None)
    parser.add_argument("--device", default=None)
    parser.add_argument("--limit", type=int, default=None, help="Cap prompted-VLM instances (cost control)")
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--list", action="store_true")
    args = parser.parse_args()

    if args.list:
        print(f"{'key':16s} {'kind':22s} note")
        for key, (_, kind, note) in ROSTER.items():
            print(f"{key:16s} {kind:22s} {note}")
        return 0

    base = load_config(args.config)
    if args.smoke:
        base = deep_update(base, {"train": {"epochs": 2, "batch_size": 8, "patience": 2, "num_workers": 0}})
    seeds = args.seeds if args.seeds else base["eval"]["seeds"]
    device = resolve_device(args.device)
    selected = args.models or list(ROSTER)

    results, failures = {}, {}
    for key in selected:
        label, kind, note = ROSTER[key]
        print(f"\n===== {key} ({kind}) =====")
        try:
            if kind.startswith("prompted:"):
                results[key] = run_prompted(key, base, device, args.limit)
            else:
                results[key] = run_trained(key, kind, base, seeds, device)
            results[key].update({"label": label, "kind": kind, "note": note})
            print(f"{key}: " + " ".join(
                f"{m}={format_metric(results[key]['summary'].get(m, {}).get('mean'))}" for m in REPORT_KEYS
            ))
        except Exception as error:  # noqa: BLE001
            failures[key] = f"{type(error).__name__}: {error}"
            print(f"{key} FAILED: {failures[key]}", file=sys.stderr)

    write_results("table6_main_results", {
        "config": str(args.config), "seeds": seeds, "models": results, "failures": failures,
    })
    rows = []
    for key in selected:
        if key not in results:
            continue
        summary = results[key]["summary"]
        cells = [format_metric(summary.get(metric, {}).get("mean"),
                               1 if metric == "cf_faithfulness" else 2) for metric in REPORT_KEYS]
        if key == "sheharm_car":
            cells = [f"\\textbf{{{cell}}}" for cell in cells]
        rows.append([results[key]["label"]] + cells)
    write_latex("table6_main_results",
                ["Model", "Tgt.-F1", "Harm-F1", "Cat.-F1", "Joint-F1", "BERTScore", "CF-Faith."], rows,
                caption="Comparison on SheHarm-Meme. Rows marked reimplementation follow the "
                        "published method description; see referred_clones/MANIFEST.md.")

    print(f"\n{'model':22s} " + " ".join(f"{k[:9]:>9s}" for k in REPORT_KEYS))
    for key in selected:
        if key in results:
            summary = results[key]["summary"]
            print(f"{key:22s} " + " ".join(
                f"{format_metric(summary.get(m, {}).get('mean')):>9s}" for m in REPORT_KEYS))
    if failures:
        print("\nfailed:", ", ".join(f"{k} ({v[:60]})" for k, v in failures.items()))
    return 0


if __name__ == "__main__":
    sys.exit(main())
