#!/usr/bin/env python3
"""Deterministic battery for SheHarm-CAR — runs in seconds, before any GPU-hour is spent.

Checks, in the cheapest-failing-first order:
    1  ontology and rule inventory load and match the paper counts
    2  concept embeddings build with the right shapes
    3  model instantiates and the forward pass produces finite losses of the right shape
    4  every loss component is present and differentiable
    5  backward pass reaches every trainable module (no silently detached branch)
    6  all six ablation switches instantiate and run
    7  beam-search rationale decoding returns well-formed token ids
    8  all five counterfactual interventions run and produce sane statistics
    9  metrics behave on synthetic perfect / worst-case predictions
   10  two runs with the same seed produce identical logits
   11  the model can overfit eight samples (learning actually happens)
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from sheharm import metrics as metric_lib  # noqa: E402
from sheharm.knowledge import build_embeddings, load_knowledge  # noqa: E402
from sheharm.labels import CATEGORY_LABELS, HARMFULNESS_LABELS, IGNORE_INDEX  # noqa: E402
from sheharm.models import SheHarmCAR  # noqa: E402
from sheharm.trainer import enable_determinism  # noqa: E402

PASSED, FAILED = [], []


def check(name: str):
    def decorator(function):
        def wrapper(*args, **kwargs):
            start = time.time()
            try:
                detail = function(*args, **kwargs)
                PASSED.append(name)
                print(f"  PASS  {name:52s} {time.time()-start:5.1f}s  {detail or ''}")
                return detail
            except Exception as error:  # noqa: BLE001
                FAILED.append((name, error))
                print(f"  FAIL  {name:52s} {time.time()-start:5.1f}s  {type(error).__name__}: {error}")
                raise
        return wrapper
    return decorator


def build_batch(batch_size: int, text_len: int, device, num_targets: int, seed: int = 0):
    generator = torch.Generator().manual_seed(seed)
    return {
        "pixel_values": torch.randn(batch_size, 3, 224, 224, generator=generator).to(device),
        "input_ids": torch.randint(5, 1000, (batch_size, text_len), generator=generator).to(device),
        "attention_mask": torch.ones(batch_size, text_len, dtype=torch.long).to(device),
        "target_labels": torch.randint(0, num_targets, (batch_size,), generator=generator).to(device),
        "harm_labels": torch.randint(0, len(HARMFULNESS_LABELS), (batch_size,), generator=generator).to(device),
        "cat_labels": torch.randint(0, len(CATEGORY_LABELS), (batch_size,), generator=generator).to(device),
        "rationale_ids": torch.randint(5, 1000, (batch_size, 16), generator=generator).to(device),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ontology", type=Path, default=Path("knowledge/ontology.json"))
    parser.add_argument("--rules", type=Path, default=Path("knowledge/rules.json"))
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--cache", type=Path, default=Path("runs/_cache/concept_embeddings.pt"))
    parser.add_argument("--skip-overfit", action="store_true")
    args = parser.parse_args()
    device = torch.device(args.device)
    enable_determinism(0)

    print("\nSheHarm-CAR deterministic battery\n" + "=" * 78)

    knowledge = load_knowledge(args.ontology, args.rules)

    @check("1  ontology + rules load with paper counts")
    def step1():
        counts = {
            "targets": len(knowledge.target_concepts),
            "retrieval": len(knowledge.retrieval_concepts),
            "triples": len(knowledge.triples),
            "rules": len(knowledge.rules),
            "predicates": len(knowledge.predicates),
        }
        assert counts["targets"] == 126, counts
        assert counts["targets"] + counts["retrieval"] == 611, counts
        assert counts["triples"] == 1287, counts
        assert counts["rules"] == 36, counts
        return str(counts)

    step1()

    @check("2  concept embeddings build")
    def step2():
        targets, concepts = build_embeddings(knowledge, device="cpu", cache=args.cache)
        assert targets.shape == (126, 768), targets.shape
        assert concepts.shape == (485, 768), concepts.shape
        assert torch.isfinite(targets).all() and torch.isfinite(concepts).all()
        return f"targets {tuple(targets.shape)} concepts {tuple(concepts.shape)}"

    step2()
    target_embeddings, concept_embeddings = build_embeddings(knowledge, device="cpu", cache=args.cache)

    def make_model(**overrides):
        enable_determinism(0)
        return SheHarmCAR(
            target_embeddings=target_embeddings,
            concept_embeddings=concept_embeddings,
            rules=knowledge.rules,
            predicates=knowledge.predicates,
            vocab_size=50265,
            pad_token_id=1,
            max_rationale_len=16,
            **overrides,
        ).to(device)

    model = make_model()
    batch = build_batch(args.batch_size, 24, device, len(knowledge.target_names))

    @check("3  forward pass shapes and finite losses")
    def step3():
        output = model(**batch)
        assert output.target_logits.shape == (args.batch_size, 126), output.target_logits.shape
        assert output.harm_logits.shape == (args.batch_size, 3)
        assert output.category_logits.shape == (args.batch_size, 5)
        assert output.activations.shape == (args.batch_size, 36)
        assert output.gamma.shape == (args.batch_size,)
        assert torch.isfinite(output.total_loss), output.total_loss
        assert (output.gamma >= 0).all() and (output.gamma <= 1).all()
        assert torch.allclose(output.symbolic_harm.sum(-1), torch.ones(args.batch_size), atol=1e-4)
        return f"loss {output.total_loss.item():.4f} gamma {output.gamma.mean().item():.3f}"

    step3()

    @check("4  every loss component present and finite")
    def step4():
        output = model(**batch)
        expected = {"target", "harm", "category", "rationale", "alignment", "consistency", "counterfactual"}
        missing = expected - set(output.losses)
        assert not missing, f"missing losses: {missing}"
        for name, value in output.losses.items():
            assert torch.isfinite(value), f"{name} is {value}"
        # At initialisation a cross-entropy head should sit at ln(num_classes). These three
        # average over enough classes to be stable, and catch bad initialisation (e.g. a
        # tied lm_head inheriting nn.Embedding's N(0,1) default).
        for name, classes in (("rationale", 50265), ("target", 126), ("alignment", 11)):
            expected = np.log(classes)
            observed = output.losses[name].item()
            assert 0.7 * expected < observed < 1.4 * expected, (
                f"{name} loss {observed:.2f} is far from ln({classes})={expected:.2f}"
            )
        return " ".join(f"{k}={v.item():.3f}" for k, v in sorted(output.losses.items()))

    step4()

    @check("5  backward reaches every trainable module")
    def step5():
        local = make_model()
        output = local(**batch)
        output.total_loss.backward()
        without_grad = [
            name for name, parameter in local.named_parameters()
            if parameter.requires_grad and (parameter.grad is None or parameter.grad.abs().sum() == 0)
        ]
        # The token embedding only sees the ids that actually appear in this tiny batch.
        without_grad = [n for n in without_grad if "token_embedding" not in n and "position_embedding" not in n]
        assert not without_grad, f"no gradient for: {without_grad[:8]}"
        return "all modules received gradient"

    step5()

    @check("6  six ablation switches instantiate and run")
    def step6():
        switches = [
            "use_target_conditioning", "use_ontology_retrieval", "use_exception_rules",
            "use_confidence_gate", "use_consistency_loss", "use_counterfactual_loss",
        ]
        for switch in switches:
            ablated = make_model(**{switch: False})
            output = ablated(**batch)
            assert torch.isfinite(output.total_loss), switch
            if switch == "use_exception_rules":
                assert output.activations.shape[1] == 25, output.activations.shape
            if switch == "use_confidence_gate":
                assert torch.allclose(output.gamma, torch.ones_like(output.gamma))
            if switch == "use_consistency_loss":
                assert "consistency" not in output.losses
            if switch == "use_counterfactual_loss":
                assert "counterfactual" not in output.losses
        return f"{len(switches)} variants ok"

    step6()

    @check("7  beam-search rationale decoding")
    def step7():
        output = model(**batch)
        generated = model.generate_rationale(output.extras["memory"], bos_token_id=0, eos_token_id=2)
        assert generated.shape[0] == args.batch_size, generated.shape
        assert generated.dtype == torch.long
        assert (generated >= 0).all() and (generated < 50265).all()
        assert (generated[:, 0] == 0).all(), "beam search must start from BOS"
        return f"generated {tuple(generated.shape)}"

    step7()

    @check("8  five counterfactual interventions")
    def step8():
        output = model(**batch)
        names = [
            "suppress_target", "mask_relevant_region", "mask_irrelevant_region",
            "remove_top_concept", "deactivate_strongest_rule",
        ]
        summary = []
        for name in names:
            result = model.intervene(output, batch["attention_mask"], name)
            assert torch.isfinite(result["confidence_drop"]).all(), name
            assert result["flipped"].dtype == torch.bool
            summary.append(f"{name.split('_')[0]}={result['confidence_drop'].mean().item():+.3f}")
        return " ".join(summary)

    step8()

    @check("9  metrics on synthetic perfect / worst predictions")
    def step9():
        perfect = {
            "target_true": [0, 1, 2, 3], "target_pred": [0, 1, 2, 3],
            "harm_true": [0, 1, 2, 0], "harm_pred": [0, 1, 2, 0],
            "category_true": [0, 1, IGNORE_INDEX, 2], "category_pred": [0, 1, 0, 2],
        }
        scores = metric_lib.summarize(perfect)
        assert abs(scores["target_f1"] - 100.0) < 1e-6, scores["target_f1"]
        assert abs(scores["harm_f1"] - 100.0) < 1e-6, scores["harm_f1"]
        assert abs(scores["joint"] - 100.0) < 1e-6, scores["joint"]
        wrong = dict(perfect, harm_pred=[1, 2, 0, 1])
        assert metric_lib.summarize(wrong)["harm_f1"] == 0.0
        faith = metric_lib.counterfactual_faithfulness(
            np.array([0.9, 0.8]), np.array([0.3, 0.2]), np.array([1, 0]), np.array([1, 0])
        )
        assert 0 <= faith["cf_faithfulness"] <= 100
        return f"perfect=100.0 worst=0.0 cf={faith['cf_faithfulness']:.1f}"

    step9()

    @check("10 identical seeds produce identical logits")
    def step10():
        first = make_model().eval()
        with torch.no_grad():
            a = first(**batch).harm_logits
        second = make_model().eval()
        with torch.no_grad():
            b = second(**batch).harm_logits
        assert torch.allclose(a, b, atol=1e-6), (a - b).abs().max().item()
        return "bit-identical"

    step10()

    @check("12 baselines forward, backward, and intervene")
    def step12():
        from sheharm.baselines import EncoderBaseline, KnowledgeBaseline

        shared = dict(num_targets=126, vocab_size=50265, pad_token_id=1, max_rationale_len=16)
        checked = []
        for kind in ("roberta_text", "hate_clipper"):
            enable_determinism(0)
            baseline = EncoderBaseline(kind, **shared).to(device)
            output = baseline(**batch)
            assert torch.isfinite(output.total_loss), kind
            assert output.harm_logits.shape == (args.batch_size, 3)
            output.total_loss.backward()
            for intervention in ("mask_relevant_region", "mask_irrelevant_region"):
                result = baseline.intervene(output, batch["attention_mask"], intervention)
                assert torch.isfinite(result["confidence_drop"]).all(), (kind, intervention)
            checked.append(kind)
        enable_determinism(0)
        kermit = KnowledgeBaseline("kermit", concept_embeddings=concept_embeddings, **shared).to(device)
        output = kermit(**batch)
        assert torch.isfinite(output.total_loss)
        checked.append("kermit")
        return " ".join(checked)

    step12()

    @check("13 binary cross-dataset configuration")
    def step13():
        binary = make_model(num_harm_classes=2, use_target_loss=False,
                            use_category_loss=False, use_rationale_loss=False,
                            use_consistency_loss=False, use_counterfactual_loss=False)
        small = dict(batch)
        small["harm_labels"] = torch.randint(0, 2, (args.batch_size,))
        output = binary(pixel_values=small["pixel_values"], input_ids=small["input_ids"],
                        attention_mask=small["attention_mask"], harm_labels=small["harm_labels"])
        assert output.harm_logits.shape == (args.batch_size, 2), output.harm_logits.shape
        assert torch.isfinite(output.total_loss), output.total_loss
        assert set(output.losses) == {"harm", "alignment"}, sorted(output.losses)
        output.total_loss.backward()
        return f"2-way head, losses {sorted(output.losses)}, loss {output.total_loss.item():.3f}"

    step13()

    if not args.skip_overfit:
        @check("11 overfits eight samples (learning happens)")
        def step11():
            local = make_model()
            small = build_batch(8, 24, device, len(knowledge.target_names), seed=7)
            optimizer = torch.optim.AdamW(local.parameters(), lr=3e-4)
            first_loss = None
            for step in range(30):
                optimizer.zero_grad(set_to_none=True)
                output = local(**small)
                output.total_loss.backward()
                torch.nn.utils.clip_grad_norm_(local.parameters(), 1.0)
                optimizer.step()
                if step == 0:
                    first_loss = output.total_loss.item()
            local.eval()
            with torch.no_grad():
                final = local(**small)
            accuracy = (final.harm_logits.argmax(-1) == small["harm_labels"]).float().mean().item()
            assert final.total_loss.item() < first_loss, f"{first_loss:.3f} -> {final.total_loss.item():.3f}"
            assert accuracy >= 0.75, f"harm accuracy only {accuracy:.2f}"
            return f"loss {first_loss:.3f} -> {final.total_loss.item():.3f}, harm acc {accuracy:.2f}"

        step11()

    print("=" * 78)
    print(f"{len(PASSED)} passed, {len(FAILED)} failed")
    return 1 if FAILED else 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception:  # noqa: BLE001
        print("=" * 78)
        print(f"{len(PASSED)} passed, {len(FAILED)} failed — battery aborted at first failure")
        sys.exit(1)
