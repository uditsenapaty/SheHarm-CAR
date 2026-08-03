#!/usr/bin/env bash
# Finish SheHarm-Meme unattended:
#   OCR -> annotation (annotator 1) -> category adjudication -> spans -> inventory -> CSV
# Every stage is resumable; re-running picks up where it stopped.
set -u
cd "$(dirname "$0")/.."
stamp() { date -u +%H:%M; }
rows() { python3 -c "import pandas,sys;print(len(pandas.read_csv(sys.argv[1],keep_default_na=False)))" "$1" 2>/dev/null || echo 0; }

echo "=== [1/6] OCR ($(stamp)) ==="
while pgrep -f "[o]cr_and_span" >/dev/null; do sleep 30; done
python3 -u scripts/ocr_and_span.py --annotations dataset/annotations_v2.csv --batch-size 8 \
    >> dataset/_backup/ocr_3500.log 2>&1
echo "OCR done ($(stamp)): $(rows dataset/ocr.csv) rows"

echo "=== [2/6] annotation, annotator 1, images 1-1500 ($(stamp)) ==="
python3 -u meme_annotator.py --images-dir dataset/images --output dataset/annotations_v2.csv \
    --start 1 --end 1500 --batch-size 5 --retries 3 > dataset/_backup/annotate_1_1500.log 2>&1
echo "annotation done ($(stamp)): $(rows dataset/annotations_v2.csv) rows"

if [ -f dataset/annotations_part2.csv ]; then
  echo "=== [2b] merging partner annotations ($(stamp)) ==="
  python3 -u scripts/merge_annotations.py --into dataset/annotations_v2.csv \
      --from dataset/annotations_part2.csv --apply 2>&1 | tail -8
fi

echo "=== [3/6] category adjudication, annotator 1 only ($(stamp)) ==="
python3 -u scripts/adjudicate_categories.py --apply 2>&1 | tail -18

echo "=== [4/6] target spans ($(stamp)) ==="
python3 -u scripts/fill_target_spans.py 2>&1 | tail -10

echo "=== [5/6] target inventory ($(stamp)) ==="
python3 -u scripts/build_target_inventory.py --annotations dataset/annotations_v2.csv 2>&1 \
    | grep -vE "LOAD REPORT|UNEXPECTED|MISSING|^-|^Key|Notes:|it/s\]|Warning:" | tail -16

echo "=== [6/6] final dataset + Table 3 ($(stamp)) ==="
python3 -u scripts/build_dataset.py --annotations dataset/annotations_v2.csv 2>&1 | tail -24

echo "=== COMPLETE ($(stamp)) ==="
