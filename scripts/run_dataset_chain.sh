#!/usr/bin/env bash
# Finish the dataset unattended. Every stage is resumable: re-running the script picks up
# where it stopped, so killing it costs at most one batch.
set -u
cd "$(dirname "$0")/.."
echo "=== [1/4] annotation ($(date -u +%H:%M)) ==="
python3 meme_annotator.py --images-dir dataset/images --output dataset/annotations_v2.csv \
    --batch-size 5 --retries 3 2>&1 | tail -3
echo "=== [2/4] OCR + target span ($(date -u +%H:%M)) ==="
python3 scripts/ocr_and_span.py --annotations dataset/annotations_v2.csv --batch-size 8 2>&1 | tail -3
echo "=== [3/4] target inventory ($(date -u +%H:%M)) ==="
python3 scripts/build_target_inventory.py --annotations dataset/annotations_v2.csv 2>&1 | tail -12
echo "=== [4/4] final dataset + Table 3 ($(date -u +%H:%M)) ==="
python3 scripts/build_dataset.py --annotations dataset/annotations_v2.csv 2>&1 | tail -20
echo "=== chain complete ($(date -u +%H:%M)) ==="
