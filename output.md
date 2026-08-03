# Women-Harm Knowledge Ontology — statistics and provenance

All figures below are produced by `python knowledge/build_ontology.py --strict`, which
regenerates the ontology from its seed inventory and **exits non-zero if any count drifts**
from the paper. It is also the first check in the deterministic battery
(`scripts/check_pipeline.py`), so the inventory cannot silently diverge.

---

## Which knowledge graph is used

**None.** The Women-Harm Knowledge Ontology is **purpose-built for this work** — hand-authored
for women-targeted harm in memes, not imported or distilled from any existing resource.

It is **not** derived from ConceptNet, WordNet, Wikidata, ATOMIC, or any other external KG.
This is a deliberate difference from the knowledge-guided baselines we compare against:

| System | Knowledge source |
|---|---|
| KERMIT (Grasso et al., 2024) | ConceptNet, retrieved per meme entity |
| KID-VLM / Just KIDDIN' (Garg et al., 2025) | ConceptNet sub-graphs, infused + distilled |
| **SheHarm-CAR (ours)** | **Purpose-built Women-Harm Knowledge Ontology** |

General-purpose commonsense graphs carry no notion of *women-targeted harm*, of the five harm
categories, or of the contextual exceptions (quotation, condemnation, counter-speech,
awareness) that separate endorsement from criticism. Those distinctions are the entire point
of the contrastive rule layer, so the ontology encodes them directly.

**How it is built.** Flat, reviewable seed lists live in `knowledge/ontology_seed.py`;
`knowledge/build_ontology.py` owns composition, triple generation and count validation.
Concept embeddings are initialised as the mean-pooled RoBERTa-base encoding of each concept's
gloss and are refined during training.

**Artefacts:** `knowledge/ontology.json`, `knowledge/rules.json`.

---

## Table 14 — Statistics of the Women-Harm Knowledge Ontology

| Ontology Component | Count |
|---|---|
| Women-related target concepts | 126 |
| Harm-cue concepts | 438 |
| Harm-category concepts | 5 |
| Context and exception concepts | 42 |
| Relation types | 14 |
| Ontology triples | 1,287 |
| Soft reasoning rules | 36 |

Total concepts: **611**. Verified: every row matches the paper exactly.

---

## Breakdown behind each row

### Women-related target concepts — 126
The controlled inventory the Ontology-Linked Target Predictor classifies over.

| Target type | Count |
|---|---|
| Female-Profession | 30 |
| Female-Relationship | 26 |
| Female-Role | 23 |
| Female-Appearance | 20 |
| Female-Group | 16 |
| Female-Behaviour | 11 |

### Harm-cue concepts — 438
Grouped by the harm category each cue expresses.

| Category | Cues |
|---|---|
| Misogyny | 102 |
| Sexual-Harassment | 97 |
| Violence | 85 |
| Appearance-Attack | 83 |
| Character-Assassination | 71 |

### Harm-category concepts — 5
Sexual-Harassment, Violence, Misogyny, Appearance-Attack, Character-Assassination.

### Context and exception concepts — 42
The concepts that withhold harm; these drive the contrastive exception rules.

| Context type | Count |
|---|---|
| Counter-Speech | 9 |
| Awareness | 8 |
| Quotation | 7 |
| Negation | 6 |
| Empowerment | 6 |
| Non-Targeted | 6 |

### Relation types — 14

| Relation | Domain → Range |
|---|---|
| `is_women_related_role` | target → target type |
| `is_a` | concept → concept type |
| `expresses` | harm cue → harm category |
| `indicates` | harm cue → harmfulness degree |
| `supports` | context → harmfulness |
| `negated_by` | harm category → context |
| `quoted_in` | harm category → context |
| `supported_by` | harm category → evidence |
| `mitigates` | context → harm category |
| `escalates` | harm cue → harm category |
| `co_occurs_with` | harm category → harm category |
| `targets` | harm category → target type |
| `refers_to` | target → modality |
| `evidenced_by` | target type → evidence |

The five relations shown in **Table 1** — `is-a`, `expresses`, `negated-by`, `quoted-in`,
`supported-by` — are all present.

### Ontology triples — 1,287

| Relation | Triples |
|---|---|
| `expresses` | 438 |
| `indicates` | 438 |
| `is_women_related_role` | 126 |
| `mitigates` | 115 |
| `is_a` | 53 |
| `quoted_in` | 35 |
| `negated_by` | 30 |
| `evidenced_by` | 12 |
| `refers_to` | 12 |
| `supported_by` | 10 |
| `supports` | 6 |
| `co_occurs_with` | 5 |
| `targets` | 5 |
| `escalates` | 2 |
| **Total** | **1,287** |

### Soft reasoning rules — 36

| Kind | Count |
|---|---|
| Harm-supporting (R⁺) | 20 |
| Contextual exception (R⁻) | 11 |
| Confidence control | 5 |

Identifiers follow **Table 2** exactly:

| Rule | Content |
|---|---|
| R1–R5 | the five category rules (target ∧ category cue ⇒ category) |
| R6 | harmful cue directly refers to the target ⇒ Explicit-Harm |
| R7 | reading requires image–text association, cultural knowledge, sarcasm or implication ⇒ Implicit-Harm |
| R8 | harmful expression quoted **and** condemned ⇒ Non-Harm |
| R9 | awareness, counter-speech or reporting without endorsement ⇒ Non-Harm |
| R10 | image, OCR text, retrieved knowledge and rules conflict ⇒ reduce the confidence gate |
| R11–R23 | category-specific explicit/implicit variants and multimodal special cases |
| E1–E9, G2–G5 | remaining exceptions and confidence-control rules |

All 20 harm-supporting rules open with the `Targets-Woman` predicate, exactly as every rule
in Table 2 begins with "IF women-related target…". Rule activation is the soft conjunction
ρ_j = Π_l p_jl over 25 predicates whose truth values are estimated from `z`, `ã` and `k̃`.

---

## Reproducing these numbers

```bash
python knowledge/build_ontology.py --strict
```

Prints the built-versus-paper comparison for all seven rows and fails on any mismatch.

## One caveat worth stating

The ontology contains all **126** target concepts. The label space the model is *evaluated*
over is smaller (18 on the current corpus) because concepts attested by fewer than 20
annotated rows back off to their ontology type — coarse entity linking under low support, not
a reduced ontology. `--min-support 0` restores the full 126-way space; measured on this
corpus that gives Tgt-F1 2.3, because 17 test classes hold a single example each.
