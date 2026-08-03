# SheHarm-CAR — Implementation Plan

Source of truth: `SheHarm-CAR_latex/latex/main.tex` (current, uncommented sections only) +
`SheHarm-CAR_UPTODATE.png` (current architecture diagram). The prototype script supplied at
kickoff is **not** authoritative — where it disagrees with the paper, the paper wins (see §7).

Everything below is built so that **each table is produced by one independently runnable script**
against **our own dataset**. Numbers printed in the paper are placeholders and are never copied
into results.

---

## 1. Resolved conflicts (decided before any code)

| # | Conflict | Decision | Evidence |
|---|---|---|---|
| C1 | Target task: concept classification vs. BIO span tagging | **Classification over a controlled inventory of ontology-linked target concepts** | §Task Formulation, §Multimodal-and-Target-Representation (Eq. target-distribution/target-loss/target-representation), diagram block ②, Table 6 caption "Tgt.-F1 … macro-F1"; BIO only appears in *commented-out* §Women-Targeted Aspect-Span Extraction |
| C2 | §Evaluation Metrics still says "token-level F1 and exact-span F1" | Stale text from the pre-revision task. Target is scored with **macro-F1** | Table 6 caption is the current revision |
| C3 | Diagram says ViT + XLM-R | **CLIP ViT-B/32 + RoBERTa-base** | §Implementation Details + Table 5 (`Encoders: CLIP ViT-B/32; RoBERTa-base`). XLM-R is a leftover from the CL-AOCS-XABSA lineage in the same LaTeX folder |
| C4 | Diagram locks both encoders as frozen, paper gives an encoder LR | **Fine-tuned at 1e-5** (config flag `freeze_encoders` available) | Table 5 `Encoder learning rate 1e-5` — a frozen encoder cannot have one |
| C5 | Naming | **"women-related target"** everywhere — `target_*`, `B-TGT`, `L_tgt`. No "aspect" | User instruction + Table 7 already reads "w/o Target Conditioning" |

## 2. Folder structure

```
SheHarm-CAR/
├── implementation-plan.md          ← this file
├── README.md · walkthrough.md · requirements.txt
├── configs/
│   ├── default.yaml                # Table 5 hyperparameters verbatim
│   ├── ablations/*.yaml            # 6 variants of Table 7
│   └── baselines/*.yaml
├── dataset/
│   ├── images/                     # img00001..img04000 (4000, one per ID)
│   ├── annotations_raw.csv         # original Qwen pass (kept, superseded)
│   ├── annotations_v2.csv          # padding-fixed re-annotation
│   ├── ocr.csv                     # filename, ocr_text, target_span
│   ├── target_inventory.json       # 126 canonical concepts + alias map
│   ├── sheharm_meme.csv            # FINAL: model-facing schema + splits
│   └── _backup/
├── knowledge/
│   ├── build_ontology.py           # emits ontology.json + rules.json
│   ├── ontology.json               # 611 concepts / 14 relations / 1287 triples
│   └── rules.json                  # 36 soft rules (R+ / R- / gate-control)
├── sheharm/
│   ├── data/       dataset.py · splits.py · counterfactuals.py
│   ├── models/     encoders.py · target_head.py · retriever.py ·
│   │               reasoner.py · fusion.py · rationale.py · sheharm_car.py
│   ├── losses.py · metrics.py · trainer.py · evaluate.py
│   └── baselines/  roberta.py · vilt.py · hate_clipper.py ·
│                   prompted_vlm.py · repo_adapters/*.py
├── experiments/                    # one script per table, independently runnable
├── scripts/                        # annotate · ocr · build_dataset · fetch_referred
├── referred_papers/                # PDFs of every cited model
├── referred_clones/                # their code, git metadata stripped
└── results/                        # <table>.json + <table>.tex
```

## 3. Data pipeline (stage → artifact)

| Stage | Script | Output | Status |
|---|---|---|---|
| D1 | `meme_annotator.py` | `annotations_v2.csv` — target, harm_type, harm_category, rationale | running (fixed) |
| D2 | `scripts/ocr_and_span.py` | `ocr.csv` — OCR text + verbatim target span | ready, queued behind D1 |
| D3 | `scripts/build_target_inventory.py` | `target_inventory.json` — 1286 raw strings → **126 canonical ontology-linked concepts** | to build |
| D4 | `scripts/build_dataset.py` | `sheharm_meme.csv` + `results/table3_data_splits.*` | to build |

**Final schema** (`sheharm_meme.csv`):
`image_path, ocr_text, target_span, target_start, target_end, target_concept,
harmfulness, harm_category, rationale, split`

- `harmfulness` ∈ {Explicit-Harm, Implicit-Harm, Non-Harm} (annotator's `Non-Harmful` → `Non-Harm`)
- `harm_category` ∈ 5 categories ∪ {None} (annotator's `NULL` → `None`)
- `target_concept` = the D3 canonical label; drives `L_tgt`
- `target_start/end` retained for the qualitative/error tables only — the model no longer consumes them
- `split` = 80/10/10 **stratified jointly over harmfulness × harm-category**, seed-fixed

## 4. Model (paper equation → module)

| Paper | Module | Note |
|---|---|---|
| Bidirectional cross-modal attention, pooled `z` | `models/encoders.py` | CLIP ViT-B/32 + RoBERTa-base, 224², 128 tokens, d=768, dropout 0.2 |
| `p^t = softmax(W_t z + b_t)`, `L_tgt` | `models/target_head.py` | M=126 concepts |
| `t̃ = Σ_m p^t_m e^t_m` | `models/target_head.py` | soft target — used by retrieval, rules, fusion, decoder |
| `q^K = LayerNorm(W_q(z ‖ t̃) + b_q)`, top-K, `α = softmax(s/τ)` | `models/retriever.py` | K=5, τ=0.07, 10 hard negatives for `L_align` |
| `ρ_ij = Π_l p_ijl` | `models/reasoner.py` | predicate truths estimated from `z`, `t̃`, `k̃` (**not** raw cosine similarity as in the prototype) |
| `q^Δ = Σ_{R+} ρ e_r − Σ_{R−} ρ e_r` | `models/reasoner.py` | explicit R⁺/R⁻ split |
| `γ = σ(w_g^T(z ‖ t̃ ‖ k̃ ‖ q^Δ) + b_g)` | `models/fusion.py` | |
| `u = LayerNorm(z + W_T t̃ + γ(W_K k̃ + W_R q^Δ))` | `models/fusion.py` | matches diagram ⑥ verbatim |
| `p^y`, `p^c`, masked `L_cat` | `models/fusion.py` | `m_i = 1[y ≠ Non-Harm]` |
| Rationale decoder on `u, t̃, k̃, activated rules` | `models/rationale.py` | **beam search, beam=4, ≤64 tokens** |
| `L_cons = Σ KL(s^y ‖ p^y) + Σ m_i KL(s^c ‖ p^c)` | `losses.py` | applied only when max activation ≥ 0.60 |
| `L_cf = L_nec + λ_inv L_inv` | `losses.py` + `data/counterfactuals.py` | 4 relevant interventions + 1 irrelevant (§5, Table 9) |
| `L_total = L_tgt + λ_harm L_harm + λ_cat L_cat + λ_rat L_rat + λ_align L_align + λ_cons L_cons + λ_cf L_cf` | `losses.py` | λ per Table 5 |

## 5. Table → script map (the deliverable)

Every script: `python experiments/<script>.py --config configs/default.yaml [--seeds 3]`,
writes `results/<name>.json` + `results/<name>.tex`, and is runnable **alone**.

| Table (paper) | Paper label | Script | Produces |
|---|---|---|---|
| 3 | `tab:data-splits` | `scripts/build_dataset.py` | split × harmfulness counts |
| 4 | `tab:annotation-agreement` | `experiments/table4_agreement.py` | Cohen's κ — **needs a 2nd annotation pass** (§8 R4) |
| 5 | `tab:hyperparameters` | `experiments/table5_hyperparams.py` | config dump |
| **6** | `tab:main-results` | `experiments/table6_main_results.py` | 12 baselines + ours × Tgt-F1, Harm-F1, Cat-F1, Joint-F1, BERTScore, CF-Faith |
| **7** | `tab:ablation` | `experiments/table7_ablation.py` | 6 variants + full × Harm-F1, Joint-F1, CF-Faith |
| 8 | `tab:classwise-harm` | `experiments/table8_classwise.py` | per-class F1, reuses T4 predictions |
| 9 | `tab:counterfactual-analysis` | `experiments/table9_counterfactual.py` | ΔConf + flip rate for 5 interventions |
| 10 | `tab:lexical-robustness` | `experiments/table10_lexical_robustness.py` | CFR / FPR-H / RS |
| 11 | `tab:cross_dataset` | `experiments/table11_cross_dataset.py` | FHM / MAMI / HarMeme AUROC+Acc |
| 12 | `tab:human-evaluation` | `experiments/table12_human_eval.py` | exports rating sheets + LLM-judge proxy (§8 R5) |
| 13 | `tab:ontology-triples` | `knowledge/build_ontology.py` | representative triples |
| 14 | `tab:ontology-statistics` | `knowledge/build_ontology.py` | 611/14/1287/36 counts |
| 15 | `tab:qualitative_error_analysis` | `experiments/table15_error_analysis.py` | error cases with rules + retrieved concepts |

**Metrics** (`sheharm/metrics.py`): Tgt-F1 = macro-F1 over target concepts; Harm-F1 = macro-F1 (3-way);
Cat-F1 = macro-F1 over **harmful instances only**; Joint-F1 = exact match of (target, harmfulness,
category); BERTScore for rationales; CF-Faith = mean confidence drop under relevant-evidence removal.
All results = mean over **3 seeds**.

## 6. Baselines (Table 4) — 12 required

| Group | Model | Plan | T4 feasibility |
|---|---|---|---|
| text | RoBERTa-base | our heads on text only | trivial |
| multimodal | ViLT, Hate-CLIPper | HF weights + our heads | fine |
| VLM (prompted, fixed schema) | LLaVA-1.5-7B, InternVL, Llama-3.2-Vision-11B, Qwen2.5-VL-7B | 4-bit, fixed output schema, identical label definitions | 4-bit required; Llama-3.2-Vision is **gated** (needs HF licence) |
| knowledge/reasoning | KERMIT, KID-VLM, IntMeme(InstructBLIP), ExplainHM++, SGoT-R1 | clone into `referred_clones/`, adapter per repo | availability risk (§8 R2) |

## 7. Prototype-script deviations (why the given code is not used as-is)

1. BIO/`aspect_head` → replaced by 126-way target classification (C1).
2. `soft_aspect` (attention over OCR tokens) → `t̃` from target-concept embeddings.
3. Rule predicate truths came from raw cosine similarity; the paper estimates them from `z, t̃, k̃`.
4. Greedy rationale decoding → beam search (beam 4).
5. `L_cf` had one intervention + Gaussian noise; the paper defines four relevant interventions and an irrelevant-evidence control.
6. The supplied file is also **syntactically broken** (truncated `SheHarmCAR.__init__` at `f.fuse = …`, corrupted `save_checkpoint`), so it cannot run in any case.
7. `torch.cuda.amp.GradScaler` is deprecated → `torch.amp.GradScaler`.

## 8. Risks / open items

| # | Risk | Handling |
|---|---|---|
| R1 | **Character-Assassination = 2 instances** — a stratified split cannot populate dev/test; 5-class Cat-F1 becomes unstable | report real count after D1; then decide 5-class-with-empty-column vs 4-class |
| R2 | SGoT-R1 (2026) / ExplainHM++ / KID-VLM may have no public code | clone what exists; otherwise reimplement from the paper description and **label it clearly** as a reimplementation in the table |
| R3 | Compute: paper uses A100, we have one T4 16 GB | our model is small (CLIP-B/32 + RoBERTa-base, batch 32) and fits; prompted 7–11B baselines run 4-bit |
| R4 | T2 agreement needs two independent annotators | ship a second-pass script; κ against a disjoint model/human pass, clearly labelled |
| R5 | T10 needs human raters | export randomized rating sheets + an LLM-judge proxy, both labelled |
| R6 | Target label space: 1286 raw strings, 1063 singletons, "woman" = 31% | D3 canonicalization to 126 concepts; macro-F1 over classes present in test |

## 9. Order of work

1. `knowledge/build_ontology.py` → ontology + rules (unblocks retriever/reasoner)
2. `sheharm/` model + losses + metrics
3. `scripts/build_target_inventory.py`, `scripts/build_dataset.py` (after D1/D2 finish)
4. Deterministic battery: shape/gradient/overfit-8-samples/determinism checks — **before any table run**
5. Baselines + `referred_clones/`
6. Experiment scripts T1–T13
7. `requirements.txt`, `README.md`, `walkthrough.md`, git push
8. **Stop.** Ask which table to run in full.