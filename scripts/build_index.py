"""Split the PDFs and build both search indexes.

Run from the project folder (Ollama must be running):
    .venv/bin/python scripts/build_index.py

Writes:
    data/index/chunks.jsonl     every chunk with its metadata (used by BM25 keyword search)
    chroma_db/                  vector database (used by vector search)
    results/index_stats.json    chunk counts and build time
"""

import json
import shutil
import sys
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from langchain_chroma import Chroma  # noqa: E402
from langchain_core.documents import Document  # noqa: E402

from src import config  # noqa: E402
from src.indexing import build_chunks  # noqa: E402
from src.models import get_embeddings  # noqa: E402

EMBED_BATCH = 64


def main() -> None:
    t0 = time.perf_counter()
    chunks = build_chunks()
    split_seconds = time.perf_counter() - t0

    config.INDEX_DIR.mkdir(parents=True, exist_ok=True)
    with config.CHUNKS_FILE.open("w", encoding="utf-8") as f:
        for c in chunks:
            f.write(json.dumps(c, ensure_ascii=False) + "\n")

    # Rebuild the vector database from scratch so it always matches chunks.jsonl.
    if config.CHROMA_DIR.exists():
        shutil.rmtree(config.CHROMA_DIR)
    docs = [Document(page_content=c["text"],
                     metadata={k: v for k, v in c.items() if k != "text"}) for c in chunks]
    t1 = time.perf_counter()
    store = Chroma(
        embedding_function=get_embeddings(),
        collection_name=config.CHROMA_COLLECTION,
        persist_directory=str(config.CHROMA_DIR),
        collection_metadata={"hnsw:space": "cosine"},
    )
    # Send in batches: one request with all chunks makes the Ollama worker give up.
    for i in range(0, len(docs), EMBED_BATCH):
        store.add_documents(docs[i:i + EMBED_BATCH], ids=[c["chunk_id"] for c in chunks[i:i + EMBED_BATCH]])
    embed_seconds = time.perf_counter() - t1

    lengths = [len(c["text"]) for c in chunks]
    stats = {
        "built_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "embedding_model": config.EMBED_MODEL,
        "chunk_size": config.CHUNK_SIZE,
        "chunk_overlap": config.CHUNK_OVERLAP,
        "total_chunks": len(chunks),
        "chunks_per_doc": dict(Counter(c["doc_id"] for c in chunks)),
        "chunk_chars_min": min(lengths),
        "chunk_chars_mean": round(sum(lengths) / len(lengths), 1),
        "chunk_chars_max": max(lengths),
        "split_seconds": round(split_seconds, 2),
        "embed_and_store_seconds": round(embed_seconds, 2),
    }
    config.RESULTS_DIR.mkdir(exist_ok=True)
    (config.RESULTS_DIR / "index_stats.json").write_text(json.dumps(stats, indent=2) + "\n")
    print(json.dumps(stats, indent=2))


if __name__ == "__main__":
    main()
