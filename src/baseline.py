"""Version 1: basic RAG.

question -> vector search (top 5 chunks) -> chat model answers with citations.
No rewriting, no keyword search, no reranking, no checking. This is the yardstick.
"""

import time

from langchain_core.messages import HumanMessage, SystemMessage

from src import config
from src.models import get_llm
from src.prompts import (ANSWER_SYSTEM, ANSWER_USER, doc_summary, extract_citations,
                         format_passages, is_refusal)
from src.retrieval import vector_search


def answer(question: str) -> dict:
    t0 = time.perf_counter()
    docs = vector_search(question, k=config.TOP_K)
    t_search = time.perf_counter() - t0

    messages = [SystemMessage(ANSWER_SYSTEM),
                HumanMessage(ANSWER_USER.format(passages=format_passages(docs), question=question))]
    reply = get_llm().invoke(messages).content.strip()
    refused = is_refusal(reply)
    total = time.perf_counter() - t0

    return {
        "system": "baseline",
        "question": question,
        "answer": reply,
        "refused": refused,
        "citations": [] if refused else extract_citations(reply, docs),
        "contexts": [d.page_content for d in docs],
        "retrieved": [doc_summary(d) for d in docs],
        "steps": [
            {"step": "Vector search", "detail": f"Top {len(docs)} chunks by meaning from all documents",
             "seconds": round(t_search, 2)},
            {"step": "Answer", "detail": "Refused" if refused else "Answered from passages",
             "seconds": round(total - t_search, 2)},
        ],
        "seconds": round(total, 2),
    }
