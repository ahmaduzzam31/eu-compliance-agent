"""Version 2: agentic RAG with LangGraph.

    plan_search -> retrieve -> grade --(some relevant)--> answer
                     ^                --(none, retry left)--> retry -> plan_search
                     |                --(none, no retry)----> refuse
plan_search: rewrite the question in the documents' wording and pick which documents to search.
retrieve:    hybrid search (vector + BM25, merged with RRF), then rerank to the best 5.
grade:       the chat model checks each passage: does it really help answer the question?
answer:      same answer prompt as the baseline, using only the passages that passed.
"""

import time
from functools import lru_cache
from typing import Literal, TypedDict

from langchain_core.documents import Document
from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.graph import END, START, StateGraph
from pydantic import BaseModel, Field

from src import config
from src.models import get_llm
from src.prompts import (ANSWER_SYSTEM, ANSWER_USER, GRADE_SYSTEM, GRADE_USER, REFUSAL,
                         RETRY_NOTE, REWRITE_SYSTEM, doc_summary, extract_citations,
                         format_passages, is_refusal, source_label)
from src.reranker import rerank
from src.retrieval import hybrid_search

ALL_DOCS = list(config.DOC_LABELS)
MAX_RETRIES = 1        # at most one extra search, so the agent can never loop
CANDIDATES_EACH = 10   # passages taken from each search before reranking

DocId = Literal["ai_act", "gdpr", "cbi_outsourcing", "cbi_op_resilience"]


class SearchPlan(BaseModel):
    query: str = Field(description="Standalone search query in the documents' formal wording")
    keywords: list[str] = Field(description="3 to 8 exact words or phrases likely in the passage")
    documents: list[DocId] = Field(description="Ids of the documents to search")


class Grade(BaseModel):
    relevant: bool = Field(description="True only if the passage helps answer the question")


class AgentState(TypedDict, total=False):
    question: str
    query: str
    keywords: list[str]
    doc_ids: list[str]
    attempt: int
    passages: list[Document]
    grades: list[bool]
    answer: str
    refused: bool
    steps: list[dict]


def _add_step(state: AgentState, name: str, detail: str, t0: float) -> list[dict]:
    return state.get("steps", []) + [
        {"step": name, "detail": detail, "seconds": round(time.perf_counter() - t0, 2)}]


def plan_search(state: AgentState) -> dict:
    t0 = time.perf_counter()
    attempt = state.get("attempt", 0)
    system = REWRITE_SYSTEM
    if attempt:
        system += "\n\n" + RETRY_NOTE.format(previous=state["query"])
    note = ""
    try:
        plan = get_llm().with_structured_output(SearchPlan).invoke(
            [SystemMessage(system), HumanMessage(f"Question: {state['question']}")])
        query = plan.query.strip() or state["question"]
        keywords = [k.strip() for k in plan.keywords if k.strip()][:8]
        doc_ids = list(dict.fromkeys(plan.documents))
    except Exception as e:  # small models sometimes return unreadable output
        query, keywords, doc_ids = state["question"], [], []
        note = f" | Planner output unreadable ({type(e).__name__}), used the original question"
    if attempt or not doc_ids:
        doc_ids = ALL_DOCS
    detail = (f'Query: "{query}" | Keywords: {", ".join(keywords) or "none"} | '
              f'Search: {", ".join(config.DOC_LABELS[d] for d in doc_ids)}{note}')
    name = "Rewrite & route" if not attempt else "Retry: new wording, all documents"
    return {"query": query, "keywords": keywords, "doc_ids": doc_ids,
            "steps": _add_step(state, name, detail, t0)}


def retrieve(state: AgentState) -> dict:
    t0 = time.perf_counter()
    candidates = hybrid_search(state["query"], state["keywords"], CANDIDATES_EACH, state["doc_ids"])
    steps = _add_step(state, "Hybrid search",
                      f"{len(candidates)} unique passages from vector + keyword search", t0)
    t1 = time.perf_counter()
    top = rerank(state["query"], candidates, config.TOP_K)
    steps = _add_step({"steps": steps}, "Rerank",
                      "Kept best 5: " + "; ".join(source_label(d.metadata) for d in top), t1)
    return {"passages": top, "steps": steps}


def grade(state: AgentState) -> dict:
    t0 = time.perf_counter()
    grader = get_llm().with_structured_output(Grade)
    grades, unreadable = [], 0
    for d in state["passages"]:
        try:
            g = grader.invoke([SystemMessage(GRADE_SYSTEM), HumanMessage(GRADE_USER.format(
                question=state["question"], label=source_label(d.metadata), passage=d.page_content))])
            grades.append(bool(g.relevant))
        except Exception:  # keep the passage rather than lose evidence, and say so
            grades.append(True)
            unreadable += 1
    detail = f"{sum(grades)} of {len(grades)} passages relevant"
    if unreadable:
        detail += f" ({unreadable} grade(s) unreadable, kept)"
    return {"grades": grades, "steps": _add_step(state, "Grade passages", detail, t0)}


def after_grade(state: AgentState) -> str:
    if any(state["grades"]):
        return "answer"
    return "retry" if state.get("attempt", 0) < MAX_RETRIES else "refuse"


def retry(state: AgentState) -> dict:
    return {"attempt": state.get("attempt", 0) + 1}


def relevant_passages(state: AgentState) -> list[Document]:
    return [d for d, ok in zip(state["passages"], state["grades"]) if ok]


def answer_node(state: AgentState) -> dict:
    t0 = time.perf_counter()
    docs = relevant_passages(state)
    reply = get_llm().invoke([
        SystemMessage(ANSWER_SYSTEM),
        HumanMessage(ANSWER_USER.format(passages=format_passages(docs), question=state["question"])),
    ]).content.strip()
    refused = is_refusal(reply)
    return {"answer": reply, "refused": refused,
            "steps": _add_step(state, "Answer", "Refused" if refused else
                               f"Answered from {len(docs)} relevant passage(s)", t0)}


def refuse(state: AgentState) -> dict:
    t0 = time.perf_counter()
    return {"answer": REFUSAL, "refused": True,
            "steps": _add_step(state, "Refuse",
                               f"No relevant passage after {state.get('attempt', 0) + 1} searches", t0)}


@lru_cache(maxsize=1)
def build_graph():
    g = StateGraph(AgentState)
    g.add_node("plan_search", plan_search)
    g.add_node("retrieve", retrieve)
    g.add_node("grade", grade)
    g.add_node("retry", retry)
    g.add_node("answer", answer_node)
    g.add_node("refuse", refuse)
    g.add_edge(START, "plan_search")
    g.add_edge("plan_search", "retrieve")
    g.add_edge("retrieve", "grade")
    g.add_conditional_edges("grade", after_grade, {"answer": "answer", "retry": "retry", "refuse": "refuse"})
    g.add_edge("retry", "plan_search")
    g.add_edge("answer", END)
    g.add_edge("refuse", END)
    return g.compile()


def answer(question: str) -> dict:
    t0 = time.perf_counter()
    s = build_graph().invoke({"question": question, "attempt": 0, "steps": []})
    used = relevant_passages(s) if not s["refused"] else []
    retrieved = []
    for d, ok in zip(s["passages"], s["grades"]):
        retrieved.append({**doc_summary(d), "relevant": ok, "rerank_score": d.metadata.get("rerank_score")})
    return {
        "system": "agent",
        "question": question,
        "answer": s["answer"],
        "refused": s["refused"],
        "citations": [] if s["refused"] else extract_citations(s["answer"], used),
        # Passages given to the answer model; if it refused, the last passages it searched.
        "contexts": [d.page_content for d in (used or s["passages"])],
        "retrieved": retrieved,
        "search_query": s["query"],
        "searches": s.get("attempt", 0) + 1,
        "steps": s["steps"],
        "seconds": round(time.perf_counter() - t0, 2),
    }
