"""Answer instructions and citation handling shared by the baseline and the agent.

Both systems use exactly the same final-answer prompt, so any difference in scores
comes from how passages are found and checked, not from different wording.
"""

import re

from langchain_core.documents import Document

REFUSAL = "I don't know based on the provided documents."

ANSWER_SYSTEM = f"""You are a careful assistant for EU and Irish financial regulation.
Answer the question using ONLY the numbered passages provided.
Rules:
- After each sentence, add the number of the passage that supports it, like [1] or [2][3].
- Do not use any knowledge that is not in the passages.
- If the passages do not contain the answer, reply with exactly this sentence and nothing else:
{REFUSAL}
- Keep the answer under 150 words and use plain English."""

ANSWER_USER = """Passages:
{passages}

Question: {question}

Answer:"""


def source_label(meta: dict) -> str:
    """Human-readable citation, e.g. 'GDPR, Article 17, page 43'."""
    parts = [meta["doc_label"]]
    if meta.get("section"):
        parts.append(meta["section"])
    parts.append(f"page {meta['page']}")
    return ", ".join(parts)


def format_passages(docs: list[Document]) -> str:
    return "\n\n".join(f"[{i}] {source_label(d.metadata)}:\n{d.page_content}"
                       for i, d in enumerate(docs, start=1))


CITATION_RE = re.compile(r"\[(\d+(?:\s*,\s*\d+)*)\]")


def extract_citations(answer: str, docs: list[Document]) -> list[dict]:
    """Turn [1], [2][3] or [2, 3] in the answer into document, section and page.

    Numbers that do not match a retrieved passage are ignored, so a citation can
    never point to text the model was not shown.
    """
    seen = []
    for group in CITATION_RE.findall(answer):
        for n in (int(x) for x in group.split(",")):
            if 1 <= n <= len(docs) and n not in seen:
                seen.append(n)
    out = []
    for n in seen:
        m = docs[n - 1].metadata
        out.append({"n": n, "label": source_label(m), "doc_label": m["doc_label"],
                    "section": m.get("section", ""), "page": m["page"], "chunk_id": m["chunk_id"]})
    return out


def is_refusal(answer: str) -> bool:
    text = answer.lower().replace("’", "'")
    return "i don't know based on the provided documents" in text or text.strip().startswith("i don't know")


def doc_summary(d: Document) -> dict:
    """Small record of a retrieved passage for display and saving."""
    m = d.metadata
    return {"label": source_label(m), "chunk_id": m["chunk_id"], "doc_id": m["doc_id"],
            "page": m["page"], "text": d.page_content}
