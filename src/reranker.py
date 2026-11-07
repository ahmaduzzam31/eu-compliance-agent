"""Rerank candidate passages with a cross-encoder.

Vector search compares two separate number lists (question vs passage). A cross-encoder
reads the question and the passage together, so it judges fit more accurately. It is
too slow to run on every chunk, so it only re-orders the ~20 candidates from hybrid search.
"""

from functools import lru_cache

import torch
from langchain_core.documents import Document
from sentence_transformers import CrossEncoder

RERANKER_MODEL = "BAAI/bge-reranker-base"


@lru_cache(maxsize=1)
def get_reranker() -> CrossEncoder:
    device = "mps" if torch.backends.mps.is_available() else "cpu"  # Apple GPU if present
    return CrossEncoder(RERANKER_MODEL, device=device, max_length=512)


def rerank(question: str, docs: list[Document], top_n: int) -> list[Document]:
    if not docs:
        return []
    scores = get_reranker().predict([(question, d.page_content) for d in docs])
    ranked = sorted(zip(docs, scores), key=lambda pair: float(pair[1]), reverse=True)
    out = []
    for d, s in ranked[:top_n]:
        d.metadata["rerank_score"] = round(float(s), 4)
        out.append(d)
    return out
