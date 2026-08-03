# Walkthrough

End-to-end, in the order you would actually run it. Every step says what it costs, what it
writes, and how to tell it worked.

---

## 0 · Install

```bash
pip install -r requirements.txt
nvidia-smi          # a 16 GB card is enough; the paper used an A100
```

The annotation and OCR passes need a CUDA GPU (4-bit Qwen2.5-VL). Everything else runs on CPU,
slowly.

---

## 1 · Knowledge layer — seconds

```bash
python knowledge/build_ontology.py --strict
```

Composes the Women-Harm Knowledge Ontology from the flat lists in `knowledge/ontology_seed.py`
and validates it against the paper's inventory. `--strict` makes a count mismatch a non-zero
exit, so this doubles as a regression test.

**Expect:**

```
Women-related target concepts        126     126
Harm-cue concepts                    438     438
Harm-category concepts                 5       5
Context and exception concepts        42      42
Relation types                        14      14
Ontology triples                   1,287   1,287
Soft reasoning rules                  36      36
```

**Writes:** `knowledge/ontology.json`, `knowledge/rules.json`,
`results/table13_ontology_triples.json`, `results/table14_ontology_statistics.{json,tex}`

To change the knowledge content, edit `ontology_seed.py` — the builder owns composition and
counting, the seed owns content.

---

## 2 · Deterministic battery — about a minute

```bash
python scripts/check_pipeline.py                # add --skip-overfit to halve the runtime
```

Twelve checks, cheapest-failing-first: ontology counts, embedding shapes, forward shapes,
loss finiteness, gradient reachability, all six ablation switches, beam search, all five
counterfactual interventions, metric behaviour on synthetic perfect/worst predictions, seed
determinism, baseline forward/backward, and an eight-sample overfit.

Check 4 asserts that each cross-entropy head starts at `ln(num_classes)`. That single
assertion is what caught the tied `lm_head` inheriting `nn.Embedding`'s N(0,1) default, which
had put the initial rationale loss at 378 instead of 10.8 and was swamping every other term.

**Run this before every GPU job.** It turns a multi-hour failure into a one-minute one.

---

## 3 · Dataset

### 3a · Annotate — about 3.5 h for 4,000 memes

```bash
python meme_annotator.py --output dataset/annotations_v2.csv --batch-size 5 --retries 3
```

Resumable: it skips filenames already present in the output and checkpoints after every batch,
so `Ctrl-C` costs at most one batch. Failures land in `dataset/annotation_failures.jsonl`.

Three things this script gets right that are easy to get wrong:

- `padding_side="left"`. Decoder-only batched generation with right padding makes every
  sequence shorter than the longest continue from a *pad* position. Measured effect on this
  data: **90.7 %** agreement between the corrected and uncorrected passes over 888 paired
  rows, with the disagreements concentrated on the Explicit/Implicit boundary.
- `--max-pixels` caps Qwen's dynamic resolution. Uncapped, a single large meme tried to
  allocate 14.35 GiB and took 158 images down with it; capped, that same image costs 756
  visual tokens.
- Repair retries name the specific violation and switch to sampling on the second attempt,
  because greedy decoding reproduces its own mistake verbatim.

### 3b · OCR and target span — about 8 h for 4,000 memes

```bash
python scripts/ocr_and_span.py --annotations dataset/annotations_v2.csv --batch-size 4
```

One pass produces both the transcription and the shortest verbatim substring naming the
women-related target. They must come from the same pass: a span produced against a different
transcription cannot be converted into character offsets. The validator rejects any span that
is not literally inside the returned `ocr_text`; if only the span fails, the transcription is
still kept and the span is left empty.

**Expect** roughly 60–65 % of rows to carry a grounded span. The rest are targets expressed
only visually, which is exactly why target identification is concept classification rather
than span tagging.

### 3c · Canonical target inventory — about a minute

```bash
python scripts/build_target_inventory.py --annotations dataset/annotations_v2.csv
```

Maps free-text targets onto the 126 ontology concepts by
`0.5 · cosine(gloss embeddings) + 0.5 · Jaccard(content words)`, with pinned aliases for the
highest-frequency strings and an explicit male/non-person gate so `male character` cannot land
on `female character`. Strings with no women-related token become `no-women-target`, which the
dataset builder turns into an ignored target label — those rows still train harmfulness.

**Check** `coverage_percent` and `concepts_attested` in the printed report before moving on.

### 3d · Assemble — seconds

```bash
python scripts/build_dataset.py
```

Normalises labels to the paper's surface forms (`Non-Harmful` → `Non-Harm`, `NULL` → `None`),
computes span offsets, and assigns an 80/10/10 split stratified jointly over harmfulness and
harm category. Near-duplicate memes (matched on normalised OCR text) are kept in the same
split to prevent leakage. Strata too small to reach dev and test are reported explicitly
rather than silently folded into train.

**Writes:** `dataset/sheharm_meme.csv`, `results/table3_data_splits.{json,tex}`

---

## 4 · Train

```bash
python experiments/train_sheharm.py --config configs/default.yaml --smoke   # shakedown
python experiments/train_sheharm.py --config configs/default.yaml           # 3 seeds
```

Per-epoch line: train loss, then dev Tgt-F1 / Harm-F1 / Cat-F1 and their mean, which is the
checkpoint-selection criterion. Early stopping at patience 5.

To measure what the paper-silent knobs buy you:

```bash
python experiments/train_sheharm.py --config configs/paper_literal.yaml --name literal
python experiments/train_sheharm.py --config configs/default.yaml       --name tuned
```

`paper_literal.yaml` differs only in cross-modal depth (1 vs 2), sampler (shuffle vs
class-balanced) and EMA (off vs 0.999). Nothing the paper states is different between them.

---

## 5 · Tables

Order matters only where a script consumes another's output.

```bash
python experiments/table5_hyperparams.py
python experiments/table6_main_results.py                       # trains every baseline
python experiments/table8_classwise.py                          # reads table4's JSON
python experiments/table7_ablation.py
python experiments/table9_counterfactual.py  --checkpoint runs/default/seed42/best_model.pt
python experiments/table10_lexical_robustness.py --checkpoint runs/default/seed42/best_model.pt
python experiments/table15_error_analysis.py --checkpoint runs/default/seed42/best_model.pt
```

Cost control: `--models` restricts Table 4 to a subset, `--variants` does the same for Table 5,
and `--limit` caps how many instances the prompted VLMs see.

Sanity check for Table 9: every target-relevant intervention should move `ΔConf` and
`Flip Rate` further than the `Mask irrelevant region` control. If it does not, the model is
not using the evidence it claims to use, and the CF-Faith number is not meaningful.

---

## 6 · Things that will bite you

| Symptom | Cause | Fix |
|---|---|---|
| CUDA OOM during annotation/OCR | uncapped Qwen dynamic resolution | lower `--max-pixels`, then `--batch-size` |
| CUDA OOM during training | another GPU job is resident | jobs are resumable — stop one, or drop `batch_size` |
| Every baseline scores identically | RoBERTa ids fed to ViLT/CLIP | already handled: raw `ocr_text` rides in the batch and each backbone retokenises |
| `qwen25_vl` beats everything | it annotated the labels | expected, and not a fair comparison — see README limitations |
| Tgt-F1 far below the other metrics | long tail of rare target concepts | inspect `attested_counts` in `dataset/target_inventory.json` |
| Table 11 skips a dataset | licensed data absent | see the docstring in `table11_cross_dataset.py` |
