#!/usr/bin/env bash
# Table 6 (main results) + Cohen's kappa against the gold annotation.
# Trained models only; prompted VLMs are a separate, download-heavy step.
set -u
cd "$(dirname "$0")/.."
stamp() { date -u +%H:%M; }

echo "=== Table 6: 9 trained models ($(stamp)) ==="
python3 -u experiments/table6_main_results.py --config configs/default.yaml --seeds 42 \
    --models sheharm_car roberta vilt hate_clipper kermit kid_vlm intmeme explainhm sgot_r1 \
    2>&1 | grep -vE "LOAD REPORT|UNEXPECTED|MISSING|^-|^Key|Notes:|it/s\]|Warning:"

echo "=== Table 8: class-wise, from Table 6 ($(stamp)) ==="
python3 -u experiments/table8_classwise.py 2>&1 | tail -12

echo "=== annotator 2: SheHarm-CAR over the whole corpus ($(stamp)) ==="
python3 -u scripts/annotate_with_sheharm.py --checkpoint runs/table6/sheharm_car/seed42/best_model.pt \
    2>&1 | grep -vE "LOAD REPORT|UNEXPECTED|MISSING|^-|^Key|Notes:|it/s\]|Warning:" | tail -20

echo "=== Cohen's kappa: gold vs SheHarm-CAR ($(stamp)) ==="
python3 -u experiments/table4_agreement.py --a dataset/gold_annotations.csv \
    --b dataset/annotations_sheharm.csv 2>&1 | tail -10

echo "=== DONE ($(stamp)) ==="
