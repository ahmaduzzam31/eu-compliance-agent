"""The two search methods, shared by the baseline and the agent.

- vector_search: finds chunks with similar meaning (nomic-embed-text + Chroma).
- bm25_search:   finds chunks with the same words, e.g. "Article 9" or "DPIA" (BM25).

Both accept doc_ids to search only some documents, which the agent uses for routing.
"""

import json
import re
from functools import lru_cache

from langchain_chroma import Chroma
from langchain_core.documents import Document
from rank_bm25 import BM25Okapi
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS

from src import config
from src.models import get_embeddings

TOKEN_RE = re.compile(r"\w+")


def tokenize(text: str) -> list[str]:
    """Lower-case words without common stop words such as "the", "to", "have"."""
    return [t for t in TOKEN_RE.findall(text.lower()) if t not in ENGLISH_STOP_WORDS]


@lru_cache(maxsize=1)
def load_chunks() -> list[dict]:
    with config.CHUNKS_FILE.open(encoding="utf-8") as f:
        return [json.loads(line) for line in f]


def to_document(chunk: dict) -> Document:
    return Document(page_content=chunk["text"],
                    metadata={k: v for k, v in chunk.items() if k != "text"})


@lru_cache(maxsize=1)
def get_vector_store() -> Chroma:
    return Chroma(
        collection_name=config.CHROMA_COLLECTION,
        embedding_function=get_embeddings(),
        persist_directory=str(config.CHROMA_DIR),
    )


@lru_cache(maxsize=1)
def get_bm25() -> BM25Okapi:
    return BM25Okapi([tokenize(c["text"]) for c in load_chunks()])


def vector_search(query: str, k: int = config.TOP_K, doc_ids: list[str] | None = None) -> list[Document]:
    where = {"doc_id": {"$in": doc_ids}} if doc_ids else None
    return get_vector_store().similarity_search(query, k=k, filter=where)


def bm25_search(query: str, k: int = config.TOP_K, doc_ids: list[str] | None = None) -> list[Document]:
    chunks = load_chunks()
    scores = get_bm25().get_scores(tokenize(query))
    order = sorted(range(len(chunks)), key=lambda i: scores[i], reverse=True)
    hits = []
    for i in order:
        if scores[i] <= 0:
            break
        if doc_ids and chunks[i]["doc_id"] not in doc_ids:
            continue
        hits.append(to_document(chunks[i]))
        if len(hits) == k:
            break
    return hits
