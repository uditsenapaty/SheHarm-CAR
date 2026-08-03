#!/usr/bin/env python3
"""Fetch every model referred to in the paper: PDFs to referred_papers/, code to referred_clones/.

Clones are shallow and their .git directory is removed, so the code lives inside our repository
as ordinary files and never fights our own git history.

Papers are located through the arXiv title-search API rather than hard-coded IDs, so a wrong
guess degrades to "manual" in the manifest instead of silently downloading the wrong PDF.
"""

from __future__ import annotations

import argparse
import json
import re
import shutil
import subprocess
import sys
import time
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
from pathlib import Path

UA = {"User-Agent": "SheHarm-CAR/1.0 (research reproduction)"}

# key, role, exact paper title, repo (None -> no public code known)
REFERRED = [
    ("roberta", "baseline: text-only encoder", "RoBERTa: A Robustly Optimized BERT Pretraining Approach", None),
    ("vilt", "baseline: multimodal encoder", "ViLT: Vision-and-Language Transformer Without Convolution or Region Supervision", "https://github.com/dandelin/ViLT"),
    ("hate_clipper", "baseline: multimodal encoder", "Hate-CLIPper: Multimodal Hateful Meme Classification based on Cross-modal Interaction of CLIP Features", "https://github.com/gokulkarthik/hateclipper"),
    ("llava", "baseline: VLM", "Visual Instruction Tuning", "https://github.com/haotian-liu/LLaVA"),
    ("internvl", "baseline: VLM", "InternVL: Scaling up Vision Foundation Models and Aligning for Generic Visual-Linguistic Tasks", "https://github.com/OpenGVLab/InternVL"),
    ("llama32_vision", "baseline: VLM", "The Llama 3 Herd of Models", "https://github.com/meta-llama/llama-models"),
    ("qwen25_vl", "baseline: VLM + our annotator", "Qwen2.5-VL Technical Report", "https://github.com/QwenLM/Qwen2.5-VL"),
    ("kermit", "baseline: knowledge-guided", "KERMIT: Knowledge-EmpoweRed Model In harmful meme deTection", "https://github.com/valeriolagatta/KERMIT_MemeDetection"),
    ("kid_vlm", "baseline: knowledge-guided", "Just KIDDIN: Knowledge Infusion and Distillation for Detection of INdecent Memes", "https://github.com/SWAN-AI/KID-VLM"),
    ("intmeme", "baseline: interpretation-guided", "Demystifying Hateful Content: Leveraging Large Multimodal Models for Hateful Meme Detection with Explainable Decisions", "https://github.com/Social-AI-Studio/IntMeme"),
    ("explainhm", "baseline: explainable", "Towards Explainable Harmful Meme Detection through Multimodal Debate between Large Language Models", "https://github.com/HKBUNLP/ExplainHM-WWW2024"),
    ("sgot_r1", "baseline: reasoning-guided", "SGoT-R1: Social Graph of Thought Reasoning-Enhanced Multimodal Large Language Model for Harmful Meme Detection", None),
    ("fhm", "benchmark", "The Hateful Memes Challenge: Detecting Hate Speech in Multimodal Memes", None),
    ("mami", "benchmark", "SemEval-2022 Task 5: Multimedia Automatic Misogyny Identification", None),
    ("harmeme", "benchmark", "Detecting Harmful Memes and Their Targets", None),
]


# Papers that are not on arXiv but have a stable open PDF elsewhere.
DIRECT_PDF = {
    "mami": "https://aclanthology.org/2022.semeval-1.74.pdf",
    "harmeme": "https://aclanthology.org/2021.findings-emnlp.4.pdf",
    "kid_vlm": "https://aclanthology.org/2025.findings-acl.1184.pdf",
}


def arxiv_pdf_url(title: str, timeout: int = 30) -> str | None:
    """Resolve a paper title to its arXiv PDF, refusing loose matches."""
    query = urllib.parse.urlencode({"search_query": f'ti:"{title}"', "max_results": "3"})
    url = f"http://export.arxiv.org/api/query?{query}"
    try:
        with urllib.request.urlopen(urllib.request.Request(url, headers=UA), timeout=timeout) as response:
            feed = ET.fromstring(response.read())
    except Exception:
        return None
    ns = {"a": "http://www.w3.org/2005/Atom"}
    wanted = re.sub(r"[^a-z0-9]", "", title.lower())
    for entry in feed.findall("a:entry", ns):
        found = entry.findtext("a:title", default="", namespaces=ns)
        if re.sub(r"[^a-z0-9]", "", found.lower())[:60] != wanted[:60]:
            continue
        for link in entry.findall("a:link", ns):
            if link.get("title") == "pdf":
                return link.get("href")
        identifier = entry.findtext("a:id", default="", namespaces=ns)
        if identifier:
            return identifier.replace("/abs/", "/pdf/")
    return None


def download(url: str, destination: Path, timeout: int = 120) -> bool:
    try:
        with urllib.request.urlopen(urllib.request.Request(url, headers=UA), timeout=timeout) as response:
            payload = response.read()
        if not payload.startswith(b"%PDF"):
            return False
        destination.write_bytes(payload)
        return True
    except Exception:
        return False


def clone(repo: str, destination: Path, timeout: int = 900) -> tuple[bool, str]:
    """Shallow-clone, then strip git metadata so the code becomes plain files in our repo."""
    if destination.exists():
        shutil.rmtree(destination)
    try:
        result = subprocess.run(
            ["git", "clone", "--depth", "1", "--quiet", repo, str(destination)],
            capture_output=True, text=True, timeout=timeout,
        )
    except subprocess.TimeoutExpired:
        return False, "timeout"
    if result.returncode != 0:
        return False, (result.stderr or "clone failed").strip().splitlines()[-1][:120]
    for meta in (".git", ".github", ".gitmodules"):
        target = destination / meta
        if target.is_dir():
            shutil.rmtree(target)
        elif target.exists():
            target.unlink()
    size = sum(f.stat().st_size for f in destination.rglob("*") if f.is_file())
    return True, f"{size/1e6:.1f} MB"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--papers-dir", type=Path, default=Path("referred_papers"))
    parser.add_argument("--clones-dir", type=Path, default=Path("referred_clones"))
    parser.add_argument("--only", nargs="*", help="Restrict to these keys")
    parser.add_argument("--skip-clones", action="store_true")
    args = parser.parse_args()

    args.papers_dir.mkdir(parents=True, exist_ok=True)
    args.clones_dir.mkdir(parents=True, exist_ok=True)
    report = []

    for key, role, title, repo in REFERRED:
        if args.only and key not in args.only:
            continue
        entry = {"key": key, "role": role, "title": title, "repo": repo}
        pdf = args.papers_dir / f"{key}.pdf"

        if pdf.exists() and pdf.stat().st_size > 10_000:
            entry["paper"] = "cached"
        else:
            url = arxiv_pdf_url(title) or DIRECT_PDF.get(key)
            entry["paper"] = "downloaded" if url and download(url, pdf) else "MANUAL (no open PDF found)"
            entry["paper_url"] = url
            time.sleep(3.0)  # arXiv API asks for one request every few seconds
        print(f"[paper] {key:15s} {entry['paper']}", flush=True)

        if repo and not args.skip_clones:
            ok, detail = clone(repo, args.clones_dir / key)
            entry["code"] = detail if ok else f"FAILED: {detail}"
        elif repo:
            entry["code"] = "skipped"
        else:
            entry["code"] = "no public code found"
        print(f"[code ] {key:15s} {entry['code']}", flush=True)
        report.append(entry)

    (args.clones_dir / "MANIFEST.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    lines = ["# Referred models\n", "Code is vendored without git metadata so it does not interfere with this repository.\n",
             "| Key | Role | Paper | Code |", "|---|---|---|---|"]
    for e in report:
        lines.append(f"| `{e['key']}` | {e['role']} | {e['paper']} | {e['code']} |")
    (args.clones_dir / "MANIFEST.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"\nWrote {args.clones_dir/'MANIFEST.md'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
