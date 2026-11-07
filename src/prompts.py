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


# ---------- Agent-only prompts (the baseline does not rewrite, route or grade) ----------

LIBRARY = """The library has four documents:
- ai_act: EU Artificial Intelligence Act, Regulation (EU) 2024/1689. Rules for AI systems: prohibited
  practices, high-risk AI systems, transparency duties, general-purpose AI models, penalties.
- gdpr: General Data Protection Regulation, Regulation (EU) 2016/679. Personal data, lawful processing,
  consent, data subject rights, controllers and processors, DPIAs, data breaches, fines.
- cbi_outsourcing: Central Bank of Ireland Cross-Industry Guidance on Outsourcing (December 2021).
  Outsourcing by regulated financial firms: critical or important functions, service providers,
  intragroup and cloud outsourcing, sub-outsourcing, outsourcing registers, exit strategies.
- cbi_op_resilience: Central Bank of Ireland Cross Industry Guidance on Operational Resilience (July 2025).
  Important business services, impact tolerances, mapping, scenario testing, incident response, DORA."""

REWRITE_SYSTEM = f"""You prepare search queries for a regulation library. Do not answer the question.
{LIBRARY}

Tasks:
1. query: rewrite the question as one clear, standalone search query using the formal wording these
   documents would use (for example "natural persons" instead of "people", "data subject" instead of "customer").
   Keep the original meaning.
2. keywords: 3 to 8 exact words or short phrases likely to appear in the relevant passage
   (for example "Article 17", "erasure", "impact tolerance").
3. documents: the document ids most likely to contain the answer. Pick one or more.
   If you are not sure, include all four."""

RETRY_NOTE = """A previous search with the query "{previous}" found no passage that answers the question.
Write a different query: use other wording or broader terms. Search all four documents."""

GRADE_SYSTEM = """You check whether a passage from a regulation helps answer a question.
Say relevant = true only if the passage states facts that are needed to answer the question.
Say relevant = false if it is only on the same general topic but does not help answer it."""

GRADE_USER = """Question: {question}

Passage ({label}):
{passage}"""


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
