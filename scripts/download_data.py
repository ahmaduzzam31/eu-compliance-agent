"""Download the four official source documents and record where they came from.

Run from the project folder:
    .venv/bin/python scripts/download_data.py

Writes:
    data/raw/<doc_id>.pdf   the PDFs (not committed to git, see .gitignore)
    data/sources.csv        one row per document: link, licence, date, size, pages, SHA-256
"""

import csv
import hashlib
from datetime import datetime, timezone
from pathlib import Path

import pymupdf
import requests

ROOT = Path(__file__).resolve().parent.parent
RAW_DIR = ROOT / "data" / "raw"
SOURCES_CSV = ROOT / "data" / "sources.csv"

EUR_LEX_LICENCE = "EUR-Lex reuse policy, Commission Decision 2011/833/EU (reuse allowed with acknowledgement)"
CBI_LICENCE = "Central Bank of Ireland Re-use of Public Sector Information licence (https://www.centralbank.ie/fns/re-use-of-public-sector-information)"

DOCUMENTS = [
    {
        "doc_id": "ai_act",
        "title": "Regulation (EU) 2024/1689 (Artificial Intelligence Act)",
        "publisher": "EUR-Lex, Official Journal of the European Union",
        "version": "OJ L, 12.7.2024",
        "url": "https://eur-lex.europa.eu/legal-content/EN/TXT/PDF/?uri=OJ:L_202401689",
        "licence": EUR_LEX_LICENCE,
    },
    {
        "doc_id": "gdpr",
        "title": "Regulation (EU) 2016/679 (General Data Protection Regulation)",
        "publisher": "EUR-Lex, Official Journal of the European Union",
        "version": "OJ L 119, 4.5.2016",
        "url": "https://eur-lex.europa.eu/legal-content/EN/TXT/PDF/?uri=CELEX:32016R0679",
        "licence": EUR_LEX_LICENCE,
    },
    {
        "doc_id": "cbi_outsourcing",
        "title": "Cross-Industry Guidance on Outsourcing",
        "publisher": "Central Bank of Ireland",
        "version": "December 2021",
        "url": "https://www.centralbank.ie/docs/default-source/publications/consultation-papers/cp138/cross-industry-guidance-on-outsourcing.pdf",
        "licence": CBI_LICENCE,
    },
    {
        "doc_id": "cbi_op_resilience",
        "title": "Cross Industry Guidance on Operational Resilience",
        "publisher": "Central Bank of Ireland",
        "version": "July 2025",
        "url": "https://www.centralbank.ie/docs/default-source/publications/consultation-papers/cp140/cross-industry-guidance-on-operational-resilience.pdf?sfvrsn=bd29921d_9",
        "licence": CBI_LICENCE,
    },
]

HEADERS = {"User-Agent": "eu-compliance-agent/0.1 (portfolio research project)"}


def download(doc: dict) -> dict:
    """Download one PDF, check it really is a PDF, and return its record."""
    response = requests.get(doc["url"], headers=HEADERS, timeout=120)
    response.raise_for_status()
    content = response.content
    if not content.startswith(b"%PDF"):
        raise ValueError(f"{doc['doc_id']}: the download is not a PDF (the site may have blocked the request)")

    path = RAW_DIR / f"{doc['doc_id']}.pdf"
    path.write_bytes(content)
    with pymupdf.open(path) as pdf:
        pages = pdf.page_count

    return {
        **doc,
        "file": str(path.relative_to(ROOT)),
        "downloaded_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "bytes": len(content),
        "pages": pages,
        "sha256": hashlib.sha256(content).hexdigest(),
    }


def main() -> None:
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    records = []
    for doc in DOCUMENTS:
        record = download(doc)
        print(f"OK  {record['doc_id']:<18} {record['pages']:>4} pages  {record['bytes']:>9,} bytes")
        records.append(record)

    fields = ["doc_id", "title", "publisher", "version", "url", "licence",
              "file", "downloaded_utc", "bytes", "pages", "sha256"]
    with SOURCES_CSV.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(records)
    print(f"Saved {SOURCES_CSV.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
