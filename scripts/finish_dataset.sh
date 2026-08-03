#!/usr/bin/env bash
# Finish SheHarm-Meme unattended: OCR -> annotation -> spans -> inventory -> final CSV.
# Every stage is resumable; re-running picks up where it stopped.
set -u
cd "$(dirname "$0")/.."
stamp() { date -u +%H:%M; }

echo "=== [1/5] waiting for OCR ($(stamp)) ==="
while pgrep -f "[o]cr_and_span" >/dev/null; do sleep 30; done
python3 -u scripts/ocr_and_span.py --annotations dataset/annotations_v2.csv --batch-size 8 \
    >> dataset/_backup/ocr_3500.log 2>&1
echo "OCR done ($(stamp)): $(python3 -c "import pandas;print(len(pandas.read_csv('dataset/ocr.csv',keep_default_na=False)))") rows"

echo "=== [2/5] annotation ($(stamp)) ==="
python3 -u meme_annotator.py --images-dir dataset/images --output dataset/annotations_v2.csv \
    --batch-size 5 --retries 3 > dataset/_backup/annotate_3500.log 2>&1
echo "annotation done ($(stamp)): $(python3 -c "import pandas;print(len(pandas.read_csv('dataset/annotations_v2.csv')))") rows"

echo "=== [3/5] target spans ($(stamp)) ==="
python3 -u scripts/fill_target_spans.py 2>&1 | tail -12

echo "=== [4/5] target inventory ($(stamp)) ==="
python3 -u scripts/build_target_inventory.py --annotations dataset/annotations_v2.csv 2>&1 | grep -vE "LOAD REPORT|UNEXPECTED|MISSING|^-|^Key|Notes:|it/s\]|Warning:" | tail -16

echo "=== [5/5] final dataset + Table 3 ($(stamp)) ==="
python3 -u scripts/build_dataset.py --annotations dataset/annotations_v2.csv 2>&1 | tail -22

echo "=== COMPLETE ($(stamp)) ==="
