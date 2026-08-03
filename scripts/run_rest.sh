#!/usr/bin/env bash
set -u
cd "$(dirname "$0")/.."
stamp() { date -u +%H:%M; }
echo "=== annotator 2 over 3500 ($(stamp)) ==="
python3 -u scripts/annotate_with_sheharm.py --checkpoint runs/table4/sheharm_car/seed42/best_model.pt \
  2>&1 | grep -vE "LOAD REPORT|UNEXPECTED|MISSING|^-|^Key|Notes:|it/s\]|Warning:" | tail -22
echo "=== Cohen's kappa: gold vs SheHarm-CAR ($(stamp)) ==="
python3 -u experiments/table4_agreement.py --a dataset/gold_annotations.csv \
  --b dataset/annotations_sheharm.csv --allow-same-annotator 2>&1 | tail -10
echo "=== 7 remaining Table 6 baselines ($(stamp)) ==="
python3 -u experiments/table6_main_results.py --config configs/default.yaml --seeds 42 \
  --models vilt hate_clipper kermit kid_vlm intmeme explainhm sgot_r1 \
  2>&1 | grep -vE "LOAD REPORT|UNEXPECTED|MISSING|^-|^Key|Notes:|it/s\]|Warning:" | tail -30
echo "=== DONE ($(stamp)) ==="
