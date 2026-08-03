#!/usr/bin/env python3
"""Strict, resumable multimodal annotation for WOMEN MABSA memes.

Uses Qwen2.5-VL-7B-Instruct in 4-bit mode so it can run on a single 16 GB T4.
Only schema-valid records are written to annotations.csv.  Images that cannot be
parsed after retries are written to annotation_failures.jsonl for human review.
"""

from __future__ import annotations

import argparse
import csv
import gc
import json
import os
import re
import sys
import tempfile
import time
from pathlib import Path
from typing import Any


FIELDNAMES = [
    "filename",
    "women-related target",
    "harm_type",
    "harm_category",
    "rationale",
]
HARM_TYPES = {"Explicit-Harm", "Implicit-Harm", "Non-Harmful"}
HARM_CATEGORIES = {
    "Misogyny",
    "Sexual-Harassment",
    "Violence",
    "Appearance-Attack",
    "Character-Assassination",
    "NULL",
}
IMAGE_RE = re.compile(r"^img(\d+)\.(?:jpe?g|png)$", re.IGNORECASE)


SYSTEM_PROMPT = """You are a meticulous human-equivalent annotator for the WOMEN MABSA meme dataset.
Inspect the image itself carefully, including all visible text, before deciding. Follow EVERY rule exactly.

TASK
Return exactly one JSON object describing this single image. No Markdown, no explanation outside JSON.

REQUIRED JSON KEYS (all strings):
women-related target, harm_type, harm_category, rationale

DATASET RULES
1. The women-related target is the woman, female role, relationship, profession, appearance, behaviour, or women-associated entity that the interpretation targets. It may be visual, text-only, or jointly inferred. Use a short, concrete target phrase, never a long sentence. Examples: "woman passenger", "women shoppers", "female driver", "woman's appearance".
2. harm_type MUST be exactly one of: Explicit-Harm, Implicit-Harm, Non-Harmful.
3. harm_category MUST be exactly one of: Misogyny, Sexual-Harassment, Violence, Appearance-Attack, Character-Assassination, NULL.
4. Use NULL if and only if harm_type is Non-Harmful. A harmful record must use a non-NULL category.
5. Explicit-Harm is direct abuse, violence, or sexual assault. Implicit-Harm is stereotyping, degradation, or mocking women. Non-Harmful does not target women for abuse/harassment; neutral examples must still be same-entity hard negatives, not generic gendered humour.
6. rationale MUST be one concise line grounded in the actual visual/textual evidence. Do not invent details. Do not leave any field empty.

CALIBRATION EXAMPLE
Image: a cartoon says "New seatbelt design: 45% less car accidents!!" and obscures the woman passenger's view with the belt.
Output: {"women-related target":"woman passenger","harm_type":"Implicit-Harm","harm_category":"Misogyny","rationale":"The meme promotes a misogynistic stereotype by implying women passengers are responsible for car accidents."}
"""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--images-dir", type=Path, default=Path("dataset/images"))
    parser.add_argument("--output", type=Path, default=Path("dataset/annotations_raw.csv"))
    parser.add_argument("--start", type=int, required=False, help="First numeric img ID, inclusive")
    parser.add_argument("--end", type=int, required=False, help="Last numeric img ID, inclusive")
    parser.add_argument("--model", default="Qwen/Qwen2.5-VL-7B-Instruct")
    parser.add_argument("--batch-size", type=int, default=5, help="Checkpoint group size; 1-5 (default: 5)")
    parser.add_argument("--max-new-tokens", type=int, default=160)
    parser.add_argument("--retries", type=int, default=2, help="Extra repair attempts per image")
    parser.add_argument("--max-pixels", type=int, default=768 * 28 * 28, help="Visual-token budget per image; caps Qwen dynamic resolution so large memes cannot exhaust a 16 GB GPU")
    parser.add_argument("--min-pixels", type=int, default=256 * 28 * 28)
    parser.add_argument("--overwrite", action="store_true", help="Replace existing annotations in the selected ID range")
    parser.add_argument("--dry-run", action="store_true", help="List selected images without loading the model")
    parser.add_argument("--self-test", action="store_true", help="Run parser/schema tests without model or GPU")
    return parser.parse_args()


def numeric_id(filename: str) -> int | None:
    match = IMAGE_RE.match(Path(filename).name)
    return int(match.group(1)) if match else None


def selected_images(images_dir: Path, start: int | None, end: int | None) -> list[Path]:
    if not images_dir.is_dir():
        raise FileNotFoundError(f"Images directory does not exist: {images_dir}")
    result: list[tuple[int, Path]] = []
    for path in images_dir.iterdir():
        image_id = numeric_id(path.name)
        if image_id is not None and path.is_file() and (start is None or image_id >= start) and (end is None or image_id <= end):
            result.append((image_id, path))
    result.sort(key=lambda item: (item[0], item[1].suffix.lower()))
    duplicate_ids = {i for i, _ in result if sum(other_i == i for other_i, _ in result) > 1}
    if duplicate_ids:
        shown = ", ".join(f"img{i:05d}" for i in sorted(duplicate_ids)[:10])
        raise ValueError(f"Multiple image files use the same numeric ID: {shown}. Keep exactly one extension per ID.")
    return [path for _, path in result]



def load_csv(path: Path) -> dict[str, dict[str, str]]:
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames != FIELDNAMES:
            raise ValueError(f"{path} header must be exactly: {','.join(FIELDNAMES)}")
        return {row["filename"]: {key: (row.get(key) or "").strip() for key in FIELDNAMES} for row in reader}


def write_csv_atomic(path: Path, rows: dict[str, dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    ordered = sorted(rows.values(), key=lambda row: (numeric_id(row["filename"]) is None, numeric_id(row["filename"]) or 0, row["filename"]))
    with tempfile.NamedTemporaryFile("w", encoding="utf-8", newline="", delete=False, dir=path.parent, suffix=".csv") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDNAMES, lineterminator="\n")
        writer.writeheader()
        writer.writerows(ordered)
        temp_name = handle.name
    Path(temp_name).replace(path)


def extract_json(text: str) -> dict[str, Any]:
    cleaned = text.strip()
    cleaned = re.sub(r"^```(?:json)?\s*|\s*```$", "", cleaned, flags=re.IGNORECASE).strip()
    start, end = cleaned.find("{"), cleaned.rfind("}")
    if start < 0 or end < start:
        raise ValueError("response contains no JSON object")
    value = json.loads(cleaned[start : end + 1])
    if not isinstance(value, dict):
        raise ValueError("JSON response is not an object")
    return value


def validate_annotation(value: dict[str, Any], filename: str) -> dict[str, str]:
    expected = {"women-related target", "harm_type", "harm_category", "rationale"}
    if set(value) != expected:
        raise ValueError(f"JSON keys must be exactly {sorted(expected)}, got {sorted(value)}")
    record = {key: str(value[key]).strip() for key in expected}
    if any(not field for field in record.values()):
        raise ValueError("empty field")
    if record["harm_type"] not in HARM_TYPES:
        raise ValueError(f"invalid harm_type: {record['harm_type']!r}")
    if record["harm_category"] not in HARM_CATEGORIES:
        raise ValueError(f"invalid harm_category: {record['harm_category']!r}")
    if (record["harm_type"] == "Non-Harmful") != (record["harm_category"] == "NULL"):
        raise ValueError("NULL must be used if and only if harm_type is Non-Harmful")
    if "\n" in record["rationale"] or "\r" in record["rationale"]:
        raise ValueError("rationale must be one line")
    if len(record["women-related target"]) > 80:
        raise ValueError("women-related target is too long; use a short concrete target phrase")
    if len(record["rationale"]) > 500:
        raise ValueError("rationale is too long")
    return {"filename": filename, **record}


def load_model(model_name: str, min_pixels: int = 256 * 28 * 28, max_pixels: int = 768 * 28 * 28):
    try:
        import torch
        from transformers import AutoProcessor, BitsAndBytesConfig, Qwen2_5_VLForConditionalGeneration
    except ImportError as exc:
        raise RuntimeError("Missing dependency. Run: pip install -r requirements.txt") from exc
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA GPU not found. This script requires a CUDA-capable T4 or similar GPU.")
    if torch.cuda.get_device_properties(0).total_memory < 12 * 1024**3:
        raise RuntimeError("At least 12 GB GPU memory is required for the 4-bit 7B model.")
    quantization = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.float16,
    )
    token = os.getenv("HF_TOKEN") or os.getenv("HUGGINGFACE_HUB_TOKEN")
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        model_name,
        quantization_config=quantization,
        torch_dtype=torch.float16,
        device_map={"": 0},
        token=token,
        low_cpu_mem_usage=True,
    ).eval()
    processor = AutoProcessor.from_pretrained(model_name, token=token, min_pixels=min_pixels, max_pixels=max_pixels)
    # Decoder-only generation must pad on the left, or every sequence shorter than the
    # longest one in the batch continues from a pad position instead of its own last token.
    processor.tokenizer.padding_side = "left"
    return model, processor, torch


def generate_many(model: Any, processor: Any, torch: Any, image_paths: list[Path], max_new_tokens: int) -> list[str]:
    """Generate independent answers in one padded GPU batch.

    Each conversation contains the complete policy and only one image.  Padding
    makes this faster than five separate launches without mixing image context.
    """
    from qwen_vl_utils import process_vision_info

    conversations = [
        [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": [{"type": "image", "image": str(image_path.resolve())}, {"type": "text", "text": "Annotate this image now using the required JSON schema."}]},
        ]
        for image_path in image_paths
    ]
    texts = [processor.apply_chat_template(conversation, tokenize=False, add_generation_prompt=True) for conversation in conversations]
    image_inputs, video_inputs = process_vision_info(conversations)
    inputs = processor(text=texts, images=image_inputs, videos=video_inputs, padding=True, return_tensors="pt")
    inputs = {key: value.to(model.device) for key, value in inputs.items()}
    with torch.inference_mode():
        generated = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False, use_cache=True)
    answer_tokens = generated[:, inputs["input_ids"].shape[1] :]
    return processor.batch_decode(answer_tokens, skip_special_tokens=True, clean_up_tokenization_spaces=False)


def repair_hint(error: str) -> str:
    """Name the exact mistake, because a generic retry reproduces it verbatim under greedy decoding."""
    if "invalid harm_type" in error:
        return (
            " You put a harm_category value inside harm_type. harm_type is ONLY the severity"
            " (Explicit-Harm, Implicit-Harm, or Non-Harmful); the category name you chose belongs"
            " in harm_category. If the meme targets women, harm_type must be Explicit-Harm or"
            " Implicit-Harm and harm_category must NOT be NULL."
        )
    if "NULL must be used" in error:
        return " NULL is allowed in harm_category if and only if harm_type is Non-Harmful; otherwise pick a real category."
    if "keys must be" in error or "no JSON" in error:
        return ' Return exactly one JSON object with exactly these four keys: "women-related target", "harm_type", "harm_category", "rationale".'
    return ""


def generate_repair(model: Any, processor: Any, torch: Any, image_path: Path, max_new_tokens: int, error: str, sample: bool = False) -> str:
    """Repair one invalid result without throwing away the original evidence."""
    from qwen_vl_utils import process_vision_info

    instruction = f"Annotate this image using the required JSON schema. A prior answer failed validation: {error}.{repair_hint(error)} Return corrected JSON only."
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": [{"type": "image", "image": str(image_path.resolve())}, {"type": "text", "text": instruction}]},
    ]
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    image_inputs, video_inputs = process_vision_info(messages)
    inputs = processor(text=[text], images=image_inputs, videos=video_inputs, padding=True, return_tensors="pt")
    inputs = {key: value.to(model.device) for key, value in inputs.items()}
    sampling = {"do_sample": True, "temperature": 0.7, "top_p": 0.9} if sample else {"do_sample": False}
    with torch.inference_mode():
        generated = model.generate(**inputs, max_new_tokens=max_new_tokens, use_cache=True, **sampling)
    answer_tokens = generated[:, inputs["input_ids"].shape[1] :]
    return processor.batch_decode(answer_tokens, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]


def append_failure(path: Path, image: Path, output: str, error: str) -> None:
    payload = {"filename": image.name, "error": error, "last_model_output": output, "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S%z")}
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, ensure_ascii=False) + "\n")


def run_self_test() -> None:
    assert numeric_id("img00001.jpg") == 1
    assert numeric_id("img02441.PNG") == 2441
    assert numeric_id("not-an-image.jpg") is None
    good = {"women-related target": "woman passenger", "harm_type": "Implicit-Harm", "harm_category": "Misogyny", "rationale": "The meme promotes a misogynistic stereotype by implying women passengers are responsible for car accidents."}
    assert validate_annotation(extract_json(json.dumps(good)), "img00001.jpg")["harm_category"] == "Misogyny"
    bad = {**good, "harm_type": "Non-Harmful"}
    try:
        validate_annotation(bad, "img00001.jpg")
    except ValueError:
        pass
    else:
        raise AssertionError("invalid NULL pairing was accepted")
    print("Self-test passed: filename parsing, JSON extraction, and strict taxonomy validation.")


def main() -> int:
    args = parse_args()
    if args.self_test:
        run_self_test()
        return 0
    if args.start is not None and args.end is not None and args.start > args.end:
        raise ValueError("--start must be less than or equal to --end")
    if not 1 <= args.batch_size <= 5:
        raise ValueError("--batch-size must be from 1 to 5")
    images = selected_images(args.images_dir, args.start, args.end)
    rows = load_csv(args.output)
    pending = [image for image in images if args.overwrite or image.name not in rows]
    print(f"Selected {len(images)} images; {len(pending)} need annotation; {len(images) - len(pending)} already exist.")
    if args.dry_run or not pending:
        for image in pending:
            print(image.name)
        return 0
    try:
        from dotenv import load_dotenv
        load_dotenv(".env.local")
    except ImportError as exc:
        raise RuntimeError("Missing python-dotenv. Run: pip install -r requirements.txt") from exc
    model, processor, torch = load_model(args.model, args.min_pixels, args.max_pixels)
    torch.manual_seed(42)
    failures = args.output.with_name("annotation_failures.jsonl")
    for batch_start in range(0, len(pending), args.batch_size):
        group = pending[batch_start : batch_start + args.batch_size]
        try:
            outputs: list[str | None] = generate_many(model, processor, torch, group, args.max_new_tokens)
        except Exception as exc:
            # A bad/oversized image should not prevent the rest of the group from being saved.
            print(f"Batch fallback to individual repair requests: {exc}", file=sys.stderr)
            outputs = [None] * len(group)
            gc.collect()
            torch.cuda.empty_cache()  # release the failed batch before per-image retries
        for offset, image in enumerate(group):
            index = batch_start + offset + 1
            error = "batch generation failed" if outputs[offset] is None else ""
            output = outputs[offset] or ""
            for attempt in range(args.retries + 1):
                try:
                    if attempt:
                        # Greedy decoding repeats its own mistake, so later attempts sample.
                        output = generate_repair(model, processor, torch, image, args.max_new_tokens, error, sample=attempt > 1)
                    rows[image.name] = validate_annotation(extract_json(output), image.name)
                    print(f"[{index}/{len(pending)}] OK {image.name}")
                    break
                except Exception as exc:  # The next attempt asks the model to repair exactly this issue.
                    error = str(exc)
                    if attempt == args.retries:
                        append_failure(failures, image, output, error)
                        print(f"[{index}/{len(pending)}] REVIEW {image.name}: {error}", file=sys.stderr)
        write_csv_atomic(args.output, rows)
        print(f"Checkpointed {args.output}")
        gc.collect()
        torch.cuda.empty_cache()
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        print("Interrupted safely; prior checkpoints remain in annotations.csv.", file=sys.stderr)
        raise SystemExit(130)
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise SystemExit(2)