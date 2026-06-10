"""
scripts/build_cfr_index.py — Download CFR sections and build the embedding index.

Downloads CFR Parts 11, 50, 58, 211 (Title 21) and Part 46 (Title 45)
from the eCFR versioner API, parses section text, embeds with
all-MiniLM-L6-v2, and writes:

    data/cfr_index/embeddings.npy   — (n_sections, 384) float32
    data/cfr_index/sections.jsonl   — one JSON object per section

Usage
-----
    python3 scripts/build_cfr_index.py             # full index
    python3 scripts/build_cfr_index.py --smoke     # Part 11 only + test query
    python3 scripts/build_cfr_index.py --date 2024-01-01

The index directory is gitignored; re-run this script after a fresh clone.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import requests

_REPO_ROOT = Path(__file__).parent.parent
_INDEX_DIR = _REPO_ROOT / "data" / "cfr_index"
_MODEL_NAME = "all-MiniLM-L6-v2"

# eCFR versioner endpoint templates
_ECFR_T21 = "https://www.ecfr.gov/api/versioner/v1/full/{date}/title-21.xml"
_ECFR_T45 = "https://www.ecfr.gov/api/versioner/v1/full/{date}/title-45.xml"

# (title_url_template, part_number)
_TARGETS = [
    (_ECFR_T21, "11"),
    (_ECFR_T21, "50"),
    (_ECFR_T21, "58"),
    (_ECFR_T21, "211"),
    (_ECFR_T45, "46"),
]

_SMOKE_TARGETS = [(_ECFR_T21, "11")]


def _fetch_xml(url: str, retries: int = 3) -> str:
    for attempt in range(retries):
        try:
            r = requests.get(url, timeout=120)
            r.raise_for_status()
            return r.text
        except requests.RequestException as exc:
            if attempt < retries - 1:
                wait = 2 ** (attempt + 1)
                print(f"  [retry {attempt+1}/{retries}] {exc} — waiting {wait}s")
                time.sleep(wait)
            else:
                raise


def _extract_sections(xml_text: str, part: str) -> list[dict[str, str]]:
    """
    Parse eCFR XML, extract <DIV8> (section-level) elements whose N attribute
    starts with `part.`, returning a list of {part, section_id, title, text}.
    """
    # Pull out section blocks. eCFR uses DIV8 for sections.
    section_pat = re.compile(
        r'<DIV8[^>]+N="(' + re.escape(part) + r'\.[^"]+)"[^>]*>(.*?)</DIV8>',
        re.DOTALL,
    )
    # Fallback: SECTION elements
    fallback_pat = re.compile(
        r'<SECTION>(.*?)</SECTION>',
        re.DOTALL,
    )

    sections: list[dict[str, str]] = []

    for m in section_pat.finditer(xml_text):
        section_id = m.group(1).strip()
        body = m.group(2)
        # Extract HEAD (title)
        head_m = re.search(r'<HEAD>(.*?)</HEAD>', body, re.DOTALL)
        title = re.sub(r'<[^>]+>', ' ', head_m.group(1)).strip() if head_m else section_id
        # Strip XML tags for text
        text = re.sub(r'<[^>]+>', ' ', body)
        text = re.sub(r'\s+', ' ', text).strip()
        if len(text) < 30:
            continue
        sections.append({"part": part, "section_id": section_id, "title": title, "text": text})

    if not sections:
        # eCFR may use a different structure; try paragraph-level fallback
        print(f"  [warn] DIV8 extraction found 0 sections for Part {part}; trying SECTION fallback")
        for m in fallback_pat.finditer(xml_text):
            body = m.group(1)
            sectno_m = re.search(r'<SECTNO>(.*?)</SECTNO>', body, re.DOTALL)
            if not sectno_m:
                continue
            section_id = re.sub(r'<[^>]+>', '', sectno_m.group(1)).strip()
            if not section_id.startswith(part + "."):
                continue
            subject_m = re.search(r'<SUBJECT>(.*?)</SUBJECT>', body, re.DOTALL)
            title = re.sub(r'<[^>]+>', ' ', subject_m.group(1)).strip() if subject_m else section_id
            text = re.sub(r'<[^>]+>', ' ', body)
            text = re.sub(r'\s+', ' ', text).strip()
            if len(text) < 30:
                continue
            sections.append({"part": part, "section_id": section_id, "title": title, "text": text})

    return sections


def _build_index(targets: list[tuple[str, str]], date: str, output_dir: Path) -> None:
    from sentence_transformers import SentenceTransformer

    output_dir.mkdir(parents=True, exist_ok=True)
    model = SentenceTransformer(_MODEL_NAME)

    all_sections: list[dict[str, str]] = []
    seen_urls: dict[str, str] = {}

    for url_template, part in targets:
        url = url_template.format(date=date)
        if url not in seen_urls:
            print(f"Fetching {url} ...")
            xml = _fetch_xml(url)
            seen_urls[url] = xml
        else:
            xml = seen_urls[url]

        secs = _extract_sections(xml, part)
        print(f"  Part {part}: {len(secs)} sections extracted")
        all_sections.extend(secs)

    if not all_sections:
        print("ERROR: no sections extracted — check eCFR XML structure", file=sys.stderr)
        sys.exit(1)

    print(f"Embedding {len(all_sections)} sections with {_MODEL_NAME} ...")
    texts = [f"{s['section_id']} {s['title']} {s['text']}" for s in all_sections]
    embeddings = model.encode(texts, normalize_embeddings=True, show_progress_bar=True)
    embeddings = embeddings.astype(np.float32)

    emb_path = output_dir / "embeddings.npy"
    sec_path = output_dir / "sections.jsonl"

    np.save(str(emb_path), embeddings)
    with sec_path.open("w", encoding="utf-8") as f:
        for s in all_sections:
            f.write(json.dumps(s) + "\n")

    print(f"Index written to {output_dir}/")
    print(f"  embeddings.npy: {embeddings.shape}")
    print(f"  sections.jsonl: {len(all_sections)} lines")


def _smoke_test(output_dir: Path) -> None:
    sys.path.insert(0, str(_REPO_ROOT))
    from harness.cfr_store import CFRStore

    store = CFRStore(output_dir)
    queries = [
        "audit trail for electronic records modification",
        "informed consent documentation requirements",
        "good laboratory practice archive access",
    ]
    for q in queries:
        results = store.retrieve(q, k=3)
        print(f"\nQuery: {q!r}")
        for r in results:
            print(f"  [{r.section_id}] {r.title[:60]}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Build CFR embedding index for RetrievalAgent.")
    parser.add_argument("--smoke", action="store_true", help="Part 11 only + test queries")
    parser.add_argument("--date", default="2024-01-01", help="eCFR snapshot date (YYYY-MM-DD)")
    parser.add_argument("--output-dir", default=str(_INDEX_DIR), help="Directory to write index")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    targets = _SMOKE_TARGETS if args.smoke else _TARGETS

    _build_index(targets, args.date, output_dir)

    if args.smoke:
        print("\n--- Smoke-test retrieval ---")
        _smoke_test(output_dir)


if __name__ == "__main__":
    main()
