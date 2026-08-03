"""SheHarm-CAR: target-conditioned neuro-symbolic framework.

Pipeline (diagram SheHarm-CAR_UPTODATE.png):
    1 multimodal encoding + bidirectional cross-modal attention -> z
    2 ontology-linked target prediction (classifier)             -> p^t, a~
    3 women-harm knowledge ontology                              (static)
    4 target-conditioned retrieval, conditioned on a~            -> k~
    5 contrastive symbolic rule reasoning                        -> q^D
    6 confidence-gated neural-symbolic fusion                    -> u
    7 predictions: target, harmfulness, harm category, rationale
    8 consistency + counterfactual faithfulness objectives
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from .. import losses as loss_fn
from ..labels import CATEGORY_LABELS, HARMFULNESS_LABELS
from .encoders import MultimodalEncoder
from .fusion import ConfidenceGatedFusion
from .rationale import RationaleDecoder
from .reasoner import SoftRuleReasoner
from .retriever import OntologyRetriever
from .target_head import TargetIdentifier


@dataclass
class SheHarmOutput:
    total_loss: torch.Tensor | None = None
    losses: dict[str, torch.Tensor] = field(default_factory=dict)
    target_logits: torch.Tensor | None = None
    harm_logits: torch.Tensor | None = None
    category_logits: torch.Tensor | None = None
    rationale_logits: torch.Tensor | None = None
    gamma: torch.Tensor | None = None
    activations: torch.Tensor | None = None
    top_concepts: torch.Tensor | None = None
    symbolic_harm: torch.Tensor | None = None
    symbolic_category: torch.Tensor | None = None
    extras: dict[str, Any] = field(default_factory=dict)


class SheHarmCAR(nn.Module):
    def __init__(
        self,
        target_embeddings: torch.Tensor,
        concept_embeddings: torch.Tensor,
        rules: list[dict],
        predicates: list[str],
        vocab_size: int,
        pad_token_id: int,
        hidden_size: int = 768,
        dropout: float = 0.2,
        top_k: int = 5,
        temperature: float = 0.07,
        num_hard_negatives: int = 10,
        rule_threshold: float = 0.60,
        max_rationale_len: int = 64,
        beam_size: int = 4,
        vision_model: str = "openai/clip-vit-base-patch32",
        text_model: str = "roberta-base",
        freeze_encoders: bool = False,
        cross_modal_layers: int = 2,
        tied_target_head: bool = True,
        # Public benchmarks (Table `tab:cross_dataset`) are binary, and only the objectives
        # they support are optimised: "we retain the multimodal, ontology, rule-reasoning and
        # confidence-gated components and optimize only the supported classification objectives".
        num_harm_classes: int = len(HARMFULNESS_LABELS),
        use_target_loss: bool = True,
        use_category_loss: bool = True,
        use_rationale_loss: bool = True,
        # ablation switches (Table `tab:ablation`)
        use_target_conditioning: bool = True,
        use_ontology_retrieval: bool = True,
        use_exception_rules: bool = True,
        use_confidence_gate: bool = True,
        use_consistency_loss: bool = True,
        use_counterfactual_loss: bool = True,
        # loss weights (Table `tab:hyperparameters`)
        lambda_harm: float = 1.0,
        lambda_cat: float = 1.0,
        lambda_rat: float = 1.0,
        lambda_align: float = 0.1,
        lambda_cons: float = 0.1,
        lambda_cf: float = 0.2,
        lambda_inv: float = 0.5,
        cf_margin: float = 0.10,
    ):
        super().__init__()
        self.encoder = MultimodalEncoder(
            vision_model, text_model, hidden_size, 8, dropout, freeze_encoders, cross_modal_layers
        )
        self.target_head = TargetIdentifier(
            target_embeddings, hidden_size, dropout, tied_classifier=tied_target_head
        )
        self.retriever = OntologyRetriever(
            concept_embeddings, hidden_size, top_k, temperature, num_hard_negatives
        )
        self.reasoner = SoftRuleReasoner(
            rules, predicates, num_harm_classes, len(CATEGORY_LABELS),
            hidden_size, dropout, rule_threshold, use_exception_rules,
        )
        self.fusion = ConfidenceGatedFusion(
            hidden_size, num_harm_classes, len(CATEGORY_LABELS), dropout,
            use_target_conditioning, use_ontology_retrieval, use_confidence_gate,
        )
        self.rationale_decoder = RationaleDecoder(
            vocab_size, hidden_size, 2, 8, dropout, max_rationale_len, pad_token_id
        )

        self.pad_token_id = pad_token_id
        self.beam_size = beam_size
        self.max_rationale_len = max_rationale_len
        self.use_target_conditioning = use_target_conditioning
        self.num_harm_classes = num_harm_classes
        self.use_target_loss = use_target_loss
        self.use_category_loss = use_category_loss
        self.use_rationale_loss = use_rationale_loss
        self.use_consistency_loss = use_consistency_loss
        self.use_counterfactual_loss = use_counterfactual_loss
        self.cf_margin = cf_margin
        self.lambda_inv = lambda_inv
        self.loss_weights = {
            "harm": lambda_harm, "category": lambda_cat, "rationale": lambda_rat,
            "alignment": lambda_align, "consistency": lambda_cons, "counterfactual": lambda_cf,
        }

    # ------------------------------------------------------------------
    # reasoning stack shared by the main pass and every intervention
    # ------------------------------------------------------------------
    def reason(self, z, soft_target, drop_top_concept=False, suppress_strongest_rule=False):
        retrieval = self.retriever(
            z, soft_target if self.use_target_conditioning else torch.zeros_like(soft_target),
            drop_top_concept=drop_top_concept,
        )
        rules = self.reasoner(
            z, soft_target, retrieval["retrieved"], suppress_strongest_rule=suppress_strongest_rule
        )
        fused = self.fusion(
            z, soft_target, retrieval["retrieved"], rules["contrastive"], rules["gate_reduction"]
        )
        return retrieval, rules, fused

    @staticmethod
    def _relevance_masks(relevance: torch.Tensor, keep_first: bool, fraction: float = 0.25):
        """Split evidence into most- and least-relevant portions."""
        count = max(1, int(relevance.size(1) * fraction))
        order = relevance.argsort(dim=-1, descending=True)
        relevant_mask = torch.ones_like(relevance)
        irrelevant_mask = torch.ones_like(relevance)
        relevant_mask.scatter_(1, order[:, :count], 0.0)          # drop the most relevant
        irrelevant_mask.scatter_(1, order[:, -count:], 0.0)        # drop the least relevant
        if keep_first:
            # Never blank the whole sequence: a fully masked text side produces NaN attention.
            relevant_mask[:, 0] = 1.0
            irrelevant_mask[:, 0] = 1.0
        return relevant_mask, irrelevant_mask

    def forward(
        self,
        pixel_values,
        input_ids,
        attention_mask,
        target_labels=None,
        harm_labels=None,
        cat_labels=None,
        rationale_ids=None,
        compute_counterfactuals: bool | None = None,
        **unused,  # e.g. `ocr_text`, which only the ViLT/CLIP baselines consume
    ) -> SheHarmOutput:
        image_tokens, text_tokens = self.encoder.encode_backbones(pixel_values, input_ids, attention_mask)
        encoded = self.encoder.interact(image_tokens, text_tokens, attention_mask)
        z = encoded["z"]

        target_logits, soft_target = self.target_head(z)
        retrieval, rules, fused = self.reason(z, soft_target)

        memory = self.rationale_decoder.build_memory(
            fused["u"], soft_target, retrieval["retrieved"], rules["contrastive"]
        )
        rationale_logits = None
        if rationale_ids is not None and self.use_rationale_loss:
            rationale_logits = self.rationale_decoder(memory, rationale_ids[:, :-1])

        components: dict[str, torch.Tensor] = {}
        if target_labels is not None and self.use_target_loss:
            components["target"] = loss_fn.target_loss(target_logits, target_labels)
        if harm_labels is not None:
            components["alignment"] = self.retriever.alignment_loss(
                retrieval["query"], retrieval["scores"], retrieval["top_index"]
            )
        if harm_labels is not None:
            components["harm"] = loss_fn.harmfulness_loss(fused["harm_logits"], harm_labels)
        if cat_labels is not None and self.use_category_loss:
            components["category"] = loss_fn.category_loss(fused["category_logits"], cat_labels)
        if rationale_ids is not None and rationale_logits is not None:
            components["rationale"] = loss_fn.rationale_loss(
                rationale_logits, rationale_ids[:, 1:], self.pad_token_id
            )
        if self.use_consistency_loss and harm_labels is not None and cat_labels is not None:
            components["consistency"] = loss_fn.consistency_loss(
                fused["harm_logits"], fused["category_logits"],
                rules["symbolic_harm"], rules["symbolic_category"],
                rules["active_mask"], cat_labels,
            )

        run_counterfactuals = self.use_counterfactual_loss if compute_counterfactuals is None else compute_counterfactuals
        if run_counterfactuals and harm_labels is not None:
            # Relevant: suppress the target representation (the paper's null-target intervention).
            _, _, suppressed = self.reason(z, torch.zeros_like(soft_target))
            # Irrelevant: perturb the least relevant image patches, reusing the backbone output.
            _, irrelevant_patches = self._relevance_masks(encoded["patch_relevance"], keep_first=False)
            perturbed = self.encoder.interact(image_tokens, text_tokens, attention_mask, patch_mask=irrelevant_patches)
            _, perturbed_target = self.target_head(perturbed["z"])
            _, _, perturbed_fused = self.reason(perturbed["z"], perturbed_target)
            components["counterfactual"] = loss_fn.counterfactual_loss(
                fused["harm_logits"], suppressed["harm_logits"], perturbed_fused["harm_logits"],
                self.lambda_inv, self.cf_margin,
            )

        if components:
            # L_tgt carries the implicit 1.0 coefficient; when it is disabled (binary
            # benchmarks) the first supported objective takes its place.
            if "target" not in components:
                anchor = "harm" if "harm" in components else next(iter(components))
                weights = {k: v for k, v in self.loss_weights.items() if k != anchor}
                total = loss_fn.total_loss({"target": components[anchor], **components}, weights)
            else:
                total = loss_fn.total_loss(components, self.loss_weights)
        else:
            total = None
        return SheHarmOutput(
            total_loss=total,
            losses=components,
            target_logits=target_logits,
            harm_logits=fused["harm_logits"],
            category_logits=fused["category_logits"],
            rationale_logits=rationale_logits,
            gamma=fused["gamma"],
            activations=rules["activations"],
            top_concepts=retrieval["top_index"],
            symbolic_harm=rules["symbolic_harm"],
            symbolic_category=rules["symbolic_category"],
            extras={
                "u": fused["u"],
                "z": z,
                "soft_target": soft_target,
                "retrieved": retrieval["retrieved"],
                "contrastive": rules["contrastive"],
                "memory": memory,
                "patch_relevance": encoded["patch_relevance"],
                "token_relevance": encoded["token_relevance"],
                "max_activation": rules["max_activation"],
                "predicate_truths": rules["predicate_truths"],
                "image_tokens": image_tokens,
                "text_tokens": text_tokens,
            },
        )

    @torch.no_grad()
    def generate_rationale(self, memory, bos_token_id: int, eos_token_id: int) -> torch.Tensor:
        return self.rationale_decoder.generate(
            memory, bos_token_id, eos_token_id, beam_size=self.beam_size, max_len=self.max_rationale_len
        )

    @torch.no_grad()
    def intervene(self, output: SheHarmOutput, attention_mask, intervention: str):
        """Evidence interventions for the counterfactual analysis table.

        suppress_target | mask_relevant_region | mask_irrelevant_region
        | remove_top_concept | deactivate_strongest_rule
        """
        extras = output.extras
        z, soft_target = extras["z"], extras["soft_target"]
        image_tokens, text_tokens = extras["image_tokens"], extras["text_tokens"]

        if intervention == "suppress_target":
            _, _, fused = self.reason(z, torch.zeros_like(soft_target))
        elif intervention == "remove_top_concept":
            _, _, fused = self.reason(z, soft_target, drop_top_concept=True)
        elif intervention == "deactivate_strongest_rule":
            _, _, fused = self.reason(z, soft_target, suppress_strongest_rule=True)
        elif intervention in ("mask_relevant_region", "mask_irrelevant_region"):
            relevant, irrelevant = self._relevance_masks(extras["patch_relevance"], keep_first=False)
            patch_mask = relevant if intervention == "mask_relevant_region" else irrelevant
            perturbed = self.encoder.interact(image_tokens, text_tokens, attention_mask, patch_mask=patch_mask)
            _, perturbed_target = self.target_head(perturbed["z"])
            _, _, fused = self.reason(perturbed["z"], perturbed_target)
        elif intervention == "mask_relevant_tokens":
            relevant, _ = self._relevance_masks(extras["token_relevance"], keep_first=True)
            perturbed = self.encoder.interact(image_tokens, text_tokens, attention_mask, token_mask=relevant)
            _, perturbed_target = self.target_head(perturbed["z"])
            _, _, fused = self.reason(perturbed["z"], perturbed_target)
        else:
            raise ValueError(f"unknown intervention: {intervention}")

        original = F.softmax(output.harm_logits, dim=-1)
        predicted = original.argmax(dim=-1, keepdim=True)
        intervened = F.softmax(fused["harm_logits"], dim=-1)
        return {
            "confidence_drop": (original.gather(1, predicted) - intervened.gather(1, predicted)).squeeze(1),
            "flipped": intervened.argmax(dim=-1).ne(predicted.squeeze(1)),
            "harm_logits": fused["harm_logits"],
            "category_logits": fused["category_logits"],
        }
