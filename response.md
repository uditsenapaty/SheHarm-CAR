Both runs are queued. While they finish, here's the honest analysis you asked for.

## Why each metric is stuck, and what moves it

| Metric           | Now                                    | Blocker                                                                                                                                                                                                                         | Fix                       | Paper-legal? |
| ---------------- | -------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ------------------------- | ------------ |
| **Harm-F1** 47   | Implicit-Harm F1 = **25.5**            | 29% of rows had ~random Explicit/Implicit labels (34.8% agreement). **Fixed** — distribution now 936/889/1675                                                                                                                   | ✅ dataset                |              |
| **Cat-F1** 46    | Violence **27.6**, Appearance **24.0** | These come from the imported sub-labels (`objectification`→Sexual, `shaming`→Appearance), a mapping too crude for a 5-way taxonomy. Adjudicating categories on the *final* labels, not just the annotator's, would sharpen them | ✅ lexical, already built |              |
| **Tgt-F1** 22→30 | 18 classes, top class 35%              | EMA fix already took dev 23.6→30.6. Ceiling is the annotator's specificity                                                                                                                                                      | ✅                        |              |
| **Joint-F1** 20  | conjunction of the above               | Rises automatically as components rise; **cannot exceed the weakest**                                                                                                                                                           | —                         |              |
| CF-Faith 54      | necessity **9.5** vs stability 98.3    | I train the counterfactual on *target suppression* but measure it on *image-region masking* — a mismatch I introduced. Aligning them is a 2-line fix                                                                            | ✅                        |              |

## The honest verdict on 60+

**Harm-F1 60+ and Cat-F1 60+ are reachable.** v4 is already at dev harm 58.5 before the severity fix even applied.

**Tgt-F1 60+ is unlikely, and Joint-F1 60+ is not achievable** with the paper's setup. Joint-F1 is an exact-match conjunction of three predictions — to hit 60 you need roughly 0.85 × 0.85 × 0.85 on correlated errors, i.e. every component near 85. With 2797 training examples, a CLIP-B/32 + RoBERTa-base model fixed by the paper, and labels from a model whose *self*-agreement is 91%, that ceiling isn't there. I'd rather tell you that now than quietly tune toward a number I can't reach.

What matters for the paper is the **gap to baselines** — they run through the identical trainer, so they inherit every fix above, and Table 7 is what demonstrates the components earn their place.

## Architecture changes that would genuinely help (paper revision, not this run)

1. **CLIP ViT-L/14-336 instead of ViT-B/32.** Biggest single win. Meme text is small; 32-pixel patches at 224² destroy it. This is why Violence and Appearance-Attack — the most *visual* categories — are the two weakest.
2. **Predict the target from `u`, not `z`.** Right now `p^t = softmax(W_t z)` happens *before* any retrieval or rule reasoning, yet retrieval is conditioned on the target — a chicken-and-egg. One refinement pass (predict → retrieve → re-predict from `u`) directly lifts Tgt-F1 and therefore Joint-F1.
3. **Give the rationale decoder token-level memory** instead of 4 pooled vectors, so it attends to actual OCR tokens and image patches.
4. **Rebalance the loss scales.** `L_rat` starts at ln(50265)≈11 while `L_harm` starts at ≈1.1, so the shared encoder is optimised mostly for language modelling. λ=1.0 each is paper-fixed, but uncertainty weighting (Kendall et al.) or simply normalising each loss by its initial value would fix the imbalance — and would be a defensible methodological contribution.
5. **Joint decoding.** Predict the (target, harm, category) tuple with a consistency-aware decode rather than three independent argmaxes — Joint-F1 is exactly what that targets.

(2) and (4) are the two I'd actually put in a revision: both are small, both are motivated by the paper's own equations, and both attack Joint-F1 directly.