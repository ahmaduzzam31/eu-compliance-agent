"""Read the PDFs page by page, remove repeated headers and footers, and split into chunks.

Each chunk keeps its document, page number and (for the EU laws) the Article or Annex
it belongs to, so every answer can cite exactly where it came from.
"""

import csv
import re

import pymupdf
from langchain_text_splitters import RecursiveCharacterTextSplitter

from src import config

# Lines that repeat on every page and carry no meaning (checked by hand on sample pages).
NOISE_PATTERNS = [re.compile(p) for p in [
    r"^EN$",                                        # language marker (EUR-Lex)
    r"^OJ L, \d{1,2}\.\d{1,2}\.\d{4}$",             # AI Act footer date
    r"^\d{1,3}/\d{1,3}$",                           # AI Act page counter, e.g. 60/144
    r"^ELI: http\S+$",                              # AI Act permanent link
    r"^\d{1,2}\.\d{1,2}\.\d{4}$",                   # GDPR header date
    r"^L \d+/\d+$",                                 # GDPR page reference, e.g. L 119/40
    r"^Official Journal of the European Union$",
    r"^Cross[- ]Industry Guidance on (Outsourcing|Operational Resilience)(\s+\w+ \d{4})?$",
    r"^Central Bank of Ireland$",
    r"^Page \d+$",
    r"^Back to [“\"]Contents[”\"]$",
]]

# A line that is only "Article 12" or "ANNEX III" starts a new section in the EU laws.
SECTION_RE = re.compile(r"^(Article \d+|ANNEX [IVXLC]+)$")
EU_LAWS = {"ai_act", "gdpr"}
MIN_CHUNK_CHARS = 40  # drop near-empty fragments such as cover-page leftovers


def clean_page(text: str) -> str:
    lines = [line.strip() for line in text.splitlines()]
    kept = [l for l in lines if l and not any(p.match(l) for p in NOISE_PATTERNS)]
    return "\n".join(kept)


def load_titles() -> dict:
    with open(config.SOURCES_CSV, encoding="utf-8") as f:
        return {row["doc_id"]: row["title"] for row in csv.DictReader(f)}


def build_chunks() -> list[dict]:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=config.CHUNK_SIZE,
        chunk_overlap=config.CHUNK_OVERLAP,
        add_start_index=True,
    )
    titles = load_titles()
    chunks = []

    for doc_id, title in titles.items():
        section = "Recitals" if doc_id in EU_LAWS else ""
        with pymupdf.open(config.RAW_DIR / f"{doc_id}.pdf") as pdf:
            for page_index, page in enumerate(pdf):
                page_no = page_index + 1  # matches the printed page number in all four PDFs
                text = clean_page(page.get_text())
                if not text:
                    continue

                # Where each section heading starts on this page.
                headings, offset = [], 0
                for line in text.split("\n"):
                    if doc_id in EU_LAWS and SECTION_RE.match(line):
                        headings.append((offset, line))
                    offset += len(line) + 1

                for i, piece in enumerate(splitter.create_documents([text])):
                    start = piece.metadata["start_index"]
                    end = start + len(piece.page_content)
                    # The section in force where the chunk starts (a heading in the first
                    # 50 characters counts as the start), plus every section that begins
                    # inside the chunk. Short Articles can sit wholly inside one chunk.
                    for pos, name in headings:
                        if pos <= start + 50:
                            section = name
                    labels = [section] if section else []
                    labels += [name for pos, name in headings if start + 50 < pos < end]
                    body = piece.page_content.strip()
                    if len(body) < MIN_CHUNK_CHARS:
                        continue
                    chunks.append({
                        "chunk_id": f"{doc_id}-p{page_no:03d}-c{i:02d}",
                        "doc_id": doc_id,
                        "doc_label": config.DOC_LABELS[doc_id],
                        "doc_title": title,
                        "page": page_no,
                        "section": "; ".join(labels),
                        "text": body,
                    })
                # The section in force at the end of this page carries over to the next page.
                if headings:
                    section = headings[-1][1]
    return chunks
