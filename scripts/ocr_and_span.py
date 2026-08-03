#!/usr/bin/env python3
"""Strict, resumable OCR + women-related target span extraction for SheHarm-Meme.

The women-related target is defined as the *shortest meaningful OCR span*, so the
transcription and the span must come from a single pass: a span produced against a
different transcription cannot be turned into character offsets.

For every image this writes one row of `filename, ocr_text, target_span`, where
`target_span` is guaranteed to be a verbatim substring of `ocr_text` (or empty
when the meme text contains no women-related expression).
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

FIELDNAMES = ["filename", "ocr_text", "target_span"]
IMAGE_RE = re.compile(r"^img(\d+)\.(?:jpe?g|png)$", re.IGNORECASE)

SYSTEM_PROMPT = """You transcribe meme text and locate the women-related target span.

Return exactly one JSON object. No Markdown, no explanation outside JSON.

REQUIRED JSON KEYS (both strings):
ocr_text, target_span

RULES
1. ocr_text is a verbatim transcription of EVERY readable text element in the image,
   in natural reading order (top to bottom, left to right), joined into ONE single line
   with spaces. Keep the original wording, spelling, casing, and punctuation.
   Do not translate, do not summarise, do not add commentary, do not describe the picture.
   Include caption bars, speech bubbles, overlaid text, and legible signs.
   Exclude platform watermarks and handles only when they are not part of the joke.
   Never emit HTML or Markdown: no <img> tags, no <image> tokens, no invented URLs.
   Never repeat the transcription twice.
   If the image contains no readable text at all, use an empty string.
2. target_span is the SHORTEST contiguous substring of ocr_text that refers to the
   women-related target given by the user. It must be copied CHARACTER FOR CHARACTER
   from ocr_text so it can be located by exact string search.
3. The span names the woman, female role, relationship, profession, appearance,
   behaviour, or women-associated entity being targeted. Prefer the head noun phrase,
   for example "my wife", "female drivers", "her makeup", "girls".
4. If ocr_text contains no expression referring to that target, use an empty string
   for target_span. Never invent words that are absent from ocr_text.

EXAMPLE
Text in image: "NEW SEATBELT DESIGN" / "45% less car accidents!!"
Target: woman passenger
Output: {"ocr_text":"NEW SEATBELT DESIGN 45% less car accidents!!","target_span":""}
"""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--images-dir", type=Path, default=Path("dataset/images"))
    parser.add_argument("--annotations", type=Path, default=Path("dataset/annotations_raw.csv"))
    parser.add_argument("--output", type=Path, default=Path("dataset/ocr.csv"))
    parser.add_argument("--model", default="Qwen/Qwen2.5-VL-7B-Instruct")
    parser.add_argument("--start", type=int, required=False)
    parser.add_argument("--end", type=int, required=False)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--max-new-tokens", type=int, default=320,
                        help="Bounds the straggler: the batch decodes until its LONGEST member stops")
    parser.add_argument("--retries", type=int, default=1,
                        help="Only genuinely malformed output is retried now")
    parser.add_argument("--max-pixels", type=int, default=1024 * 28 * 28, help="Higher than annotation: small meme text needs resolution")
    parser.add_argument("--min-pixels", type=int, default=256 * 28 * 28)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--self-test", action="store_true")
    return parser.parse_args()


def numeric_id(filename: str) -> int | None:
    match = IMAGE_RE.match(Path(filename).name)
    return int(match.group(1)) if match else None


MARKUP_RE = re.compile(r"<img\b[^>]*>|</?image>|!\[[^\]]*\]\([^)]*\)", re.IGNORECASE)


def strip_hallucinated_markup(value: str) -> str:
    """Remove HTML/markdown the model invents instead of transcribing.

    Observed: `<img src="https://i.redd.it/...">` with a fabricated URL, and bare `<image>`
    tokens. These are not text in the meme, and they poison both the OCR channel and the
    span search.
    """
    return MARKUP_RE.sub(" ", value)


def collapse_repetition(value: str) -> str:
    """Undo degenerate whole-output duplication (the model restating its transcription)."""
    text = value.strip()
    half = len(text) // 2
    if half > 20 and text[:half].strip() == text[half:].strip():
        return text[:half].strip()
    return text


def normalize_text(value: str) -> str:
    """One line, collapsed whitespace: offsets must survive a round trip through CSV."""
    cleaned = strip_hallucinated_markup(str(value).replace("\r", " ").replace("\n", " "))
    return collapse_repetition(re.sub(r"\s+", " ", cleaned).strip())


def load_csv(path: Path, fieldnames: list[str]) -> dict[str, dict[str, str]]:
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames != fieldnames:
            raise ValueError(f"{path} header must be exactly: {','.join(fieldnames)}")
        return {row["filename"]: {key: (row.get(key) or "") for key in fieldnames} for row in reader}


def write_csv_atomic(path: Path, rows: dict[str, dict[str, str]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    ordered = sorted(rows.values(), key=lambda row: (numeric_id(row["filename"]) is None, numeric_id(row["filename"]) or 0, row["filename"]))
    with tempfile.NamedTemporaryFile("w", encoding="utf-8", newline="", delete=False, dir=path.parent, suffix=".csv") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        writer.writerows(ordered)
        temp_name = handle.name
    Path(temp_name).replace(path)


def extract_json(text: str) -> dict[str, Any]:
    cleaned = re.sub(r"^```(?:json)?\s*|\s*```$", "", text.strip(), flags=re.IGNORECASE).strip()
    start, end = cleaned.find("{"), cleaned.rfind("}")
    if start < 0 or end < start:
        raise ValueError("response contains no JSON object")
    value = json.loads(cleaned[start : end + 1])
    if not isinstance(value, dict):
        raise ValueError("JSON response is not an object")
    return value


def snap_span(ocr_text: str, span: str) -> str:
    """Recover a span the model paraphrased in casing or spacing.

    Most rejected spans are semantically right but differ from `ocr_text` by case, by
    surrounding punctuation, or by whitespace. Re-prompting to fix that costs a full
    unbatched generation, so the span is snapped to the real substring instead.
    Returns "" when no honest alignment exists — never invents text.
    """
    if not span or span in ocr_text:
        return span
    stripped = span.strip(" \t\"'.,:;!?-–—()[]")
    if stripped and stripped in ocr_text:
        return stripped
    lowered_text, lowered_span = ocr_text.lower(), stripped.lower()
    position = lowered_text.find(lowered_span)
    if position >= 0:                                   # case-only difference
        return ocr_text[position : position + len(stripped)]
    # whitespace-only difference: match on collapsed text, map back to original offsets
    collapsed, offsets = [], []
    for index, character in enumerate(ocr_text):
        if character.isspace():
            if collapsed and collapsed[-1] == " ":
                continue
            collapsed.append(" ")
        else:
            collapsed.append(character.lower())
        offsets.append(index)
    joined = "".join(collapsed)
    needle = " ".join(lowered_span.split())
    position = joined.find(needle)
    if position >= 0:
        start = offsets[position]
        end = offsets[min(position + len(needle) - 1, len(offsets) - 1)] + 1
        return ocr_text[start:end]
    return ""


def validate_record(value: dict[str, Any], filename: str) -> dict[str, str]:
    expected = {"ocr_text", "target_span"}
    if set(value) != expected:
        raise ValueError(f"JSON keys must be exactly {sorted(expected)}, got {sorted(value)}")
    ocr_text = normalize_text(value["ocr_text"])
    target_span = normalize_text(value["target_span"])
    if len(ocr_text) > 2000:
        raise ValueError("ocr_text is too long; transcribe text only, never describe the image")
    # An unalignable span is expected, not an error: many targets are expressed only
    # visually. Measured on the first 794 images, 109 of 131 rejected spans had no honest
    # alignment, so re-prompting for them only ever burned three unbatched generations
    # before falling back to the empty span anyway.
    target_span = snap_span(ocr_text, target_span)
    if len(target_span) > 120:
        target_span = ""
    return {"filename": filename, "ocr_text": ocr_text, "target_span": target_span}


def load_model(model_name: str, min_pixels: int, max_pixels: int):
    try:
        import torch
        from transformers import AutoProcessor, BitsAndBytesConfig, Qwen2_5_VLForConditionalGeneration
    except ImportError as exc:
        raise RuntimeError("Missing dependency. Run: pip install -r requirements.txt") from exc
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA GPU not found. This script requires a CUDA-capable T4 or similar GPU.")
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


def build_conversation(image_path: Path, target: str, hint: str = "") -> list[dict[str, Any]]:
    """With no annotated target yet, transcribe only — the span can be filled in later."""
    if target:
        instruction = (
            f'Transcribe this meme and locate the span for the women-related target: "{target}". '
            f"Return the JSON object only.{hint}"
        )
    else:
        instruction = (
            "Transcribe this meme. No target has been annotated yet, so return an empty "
            f'target_span. Return the JSON object only.{hint}'
        )
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": [{"type": "image", "image": str(image_path.resolve())}, {"type": "text", "text": instruction}]},
    ]


def generate(model: Any, processor: Any, torch: Any, conversations: list[list[dict[str, Any]]], max_new_tokens: int, sample: bool = False) -> list[str]:
    from qwen_vl_utils import process_vision_info

    texts = [processor.apply_chat_template(conversation, tokenize=False, add_generation_prompt=True) for conversation in conversations]
    image_inputs, video_inputs = process_vision_info(conversations)
    inputs = processor(text=texts, images=image_inputs, videos=video_inputs, padding=True, return_tensors="pt")
    inputs = {key: value.to(model.device) for key, value in inputs.items()}
    sampling = {"do_sample": True, "temperature": 0.7, "top_p": 0.9} if sample else {"do_sample": False}
    with torch.inference_mode():
        generated = model.generate(**inputs, max_new_tokens=max_new_tokens, use_cache=True, **sampling)
    answer_tokens = generated[:, inputs["input_ids"].shape[1] :]
    return processor.batch_decode(answer_tokens, skip_special_tokens=True, clean_up_tokenization_spaces=False)


def repair_hint(error: str) -> str:
    if "no JSON" in error or "Expecting" in error or "Unterminated" in error:
        return " Your previous answer was cut off. Keep the transcription complete but concise, and close the JSON object."
    if "keys must be" in error or "no JSON" in error:
        return ' Return exactly one JSON object with exactly these two keys: "ocr_text", "target_span".'
    if "too long" in error:
        return " Transcribe only the readable text; never describe the picture."
    return ""


def append_failure(path: Path, image: Path, output: str, error: str) -> None:
    payload = {"filename": image.name, "error": error, "last_model_output": output, "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S%z")}
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, ensure_ascii=False) + "\n")


def run_self_test() -> None:
    good = {"ocr_text": "MY WIFE  said\nno", "target_span": "MY WIFE"}
    record = validate_record(extract_json(json.dumps(good)), "img00001.jpg")
    assert record["ocr_text"] == "MY WIFE said no", record
    assert record["target_span"] == "MY WIFE"
    assert record["target_span"] in record["ocr_text"]
    empty = validate_record({"ocr_text": "", "target_span": ""}, "img00002.png")
    assert empty["ocr_text"] == "" and empty["target_span"] == ""
    hallucinated = validate_record({"ocr_text": "hello world", "target_span": "my wife"}, "img00003.jpg")
    assert hallucinated["target_span"] == "", hallucinated
    # normalize_text collapses whitespace first, so snapping only has to fix casing here.
    snapped = validate_record({"ocr_text": "MY  WIFE said no", "target_span": "my wife"}, "img00004.jpg")
    assert snapped["target_span"] == "MY WIFE", snapped
    assert snapped["target_span"] in snapped["ocr_text"]
    print("Self-test passed: normalization, span snapping, hallucination rejection, empty-span handling.")


def main() -> int:
    args = parse_args()
    if args.self_test:
        run_self_test()
        return 0

    # Transcription needs no annotation; only the span does. Images without a label yet are
    # still transcribed, so OCR can run to completion ahead of annotation.
    targets = {}
    if args.annotations.exists():
        targets = {row["filename"]: row["women-related target"]
                   for row in csv.DictReader(args.annotations.open(encoding="utf-8-sig"))}
    images = []
    for path in sorted(args.images_dir.iterdir(), key=lambda p: (numeric_id(p.name) or 0, p.name)):
        image_id = numeric_id(path.name)
        if image_id is None or not path.is_file():
            continue
        if (args.start is None or image_id >= args.start) and (args.end is None or image_id <= args.end):
            images.append(path)

    rows = load_csv(args.output, FIELDNAMES)
    pending = [image for image in images if args.overwrite or image.name not in rows]
    unlabelled = sum(1 for image in pending if not targets.get(image.name))
    print(f"Selected {len(images)} images; {len(pending)} need OCR; {len(images) - len(pending)} already done; "
          f"{unlabelled} of the pending have no annotation yet (transcription only, empty span).")
    if args.dry_run or not pending:
        return 0

    model, processor, torch = load_model(args.model, args.min_pixels, args.max_pixels)
    torch.manual_seed(42)
    failures = args.output.with_name("ocr_failures.jsonl")
    started = time.time()

    for batch_start in range(0, len(pending), args.batch_size):
        group = pending[batch_start : batch_start + args.batch_size]
        try:
            outputs: list[str | None] = generate(model, processor, torch, [build_conversation(image, targets.get(image.name, "")) for image in group], args.max_new_tokens)
        except Exception as exc:
            print(f"Batch fallback to individual requests: {exc}", file=sys.stderr)
            outputs = [None] * len(group)
            gc.collect()
            torch.cuda.empty_cache()

        for offset, image in enumerate(group):
            index = batch_start + offset + 1
            error = "batch generation failed" if outputs[offset] is None else ""
            output = outputs[offset] or ""
            for attempt in range(args.retries + 2):
                try:
                    if attempt:
                        truncated = "{" in output and "}" not in output
                        budget = args.max_new_tokens * (3 if truncated else 1)
                        output = generate(
                            model, processor, torch,
                            [build_conversation(image, targets.get(image.name, ""), repair_hint(error))],
                            budget, sample=attempt > 1 and not truncated,
                        )[0]
                    rows[image.name] = validate_record(extract_json(output), image.name)
                    break
                except Exception as exc:
                    error = str(exc)
                    if attempt == args.retries + 1:
                        # Keep the transcription even when only the span failed: OCR text is still usable.
                        try:
                            salvaged = extract_json(output)
                            rows[image.name] = validate_record({"ocr_text": salvaged.get("ocr_text", ""), "target_span": ""}, image.name)
                            error += " (salvaged ocr_text, empty span)"
                        except Exception:
                            rows[image.name] = {"filename": image.name, "ocr_text": "", "target_span": ""}
                            error += " (empty row written)"
                        append_failure(failures, image, output, error)
                        print(f"[{index}/{len(pending)}] REVIEW {image.name}: {error}", file=sys.stderr)

        write_csv_atomic(args.output, rows, FIELDNAMES)
        done = min(batch_start + args.batch_size, len(pending))
        rate = done / max(time.time() - started, 1e-6)
        print(f"[{done}/{len(pending)}] checkpointed | {rate*3600:.0f} img/h | eta {(len(pending)-done)/max(rate,1e-9)/60:.1f} min", flush=True)
        gc.collect()
        torch.cuda.empty_cache()
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        print("Interrupted safely; prior checkpoints remain in the output CSV.", file=sys.stderr)
        raise SystemExit(130)
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise SystemExit(2)
