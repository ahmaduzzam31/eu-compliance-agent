"""Streamlit chat app for the EU compliance assistant.

Run from the project folder (Ollama must be running):
    .venv/bin/streamlit run app.py
"""

import sys
from pathlib import Path

import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parent))

from src import config  # noqa: E402

st.set_page_config(page_title="EU Compliance Assistant", page_icon="⚖️", layout="wide")

EXAMPLES = [
    "Do I have to tell people that they are talking to an AI system?",
    "What is the right to erasure under GDPR?",
    "What must an outsourcing register contain under the Central Bank guidance?",
    "What is an impact tolerance?",
]
MODES = ["Agentic RAG", "Baseline RAG", "Compare both"]


@st.cache_resource(show_spinner="Loading search indexes and reranker (first start only)...")
def load_systems() -> dict:
    """Load everything once per app start, not once per question."""
    from src import agent, baseline
    from src.reranker import get_reranker
    from src.retrieval import get_bm25, get_vector_store

    get_vector_store()
    get_bm25()
    get_reranker()
    return {"Baseline RAG": baseline.answer, "Agentic RAG": agent.answer}


def render_result(r: dict) -> None:
    if r["refused"]:
        st.warning(r["answer"])
    else:
        st.markdown(r["answer"])
    st.caption(f"⏱ {r['seconds']} s")

    texts = {p["chunk_id"]: p["text"] for p in r["retrieved"]}
    if r["citations"]:
        with st.expander(f"📄 Sources ({len(r['citations'])})", expanded=True):
            for c in r["citations"]:
                st.markdown(f"**[{c['n']}] {c['label']}**")
                quote = texts.get(c["chunk_id"], "")
                st.caption(quote[:500] + ("…" if len(quote) > 500 else ""))
    elif not r["refused"]:
        st.caption("No sources cited.")

    with st.expander(f"🧭 Steps taken ({len(r['steps'])})", expanded=r["system"] == "agent"):
        for i, s in enumerate(r["steps"], start=1):
            st.markdown(f"**{i}. {s['step']}** · {s['seconds']} s")
            for part in s["detail"].split(" | "):
                st.markdown(f"- {part}")

    with st.expander(f"🔎 All retrieved passages ({len(r['retrieved'])})"):
        for p in r["retrieved"]:
            mark = {True: "✅ relevant", False: "❌ not relevant"}.get(p.get("relevant"), "")
            st.markdown(f"**{p['label']}** {mark}")
            st.caption(p["text"][:300] + ("…" if len(p["text"]) > 300 else ""))


def render_turn(turn: dict) -> None:
    with st.chat_message("user"):
        st.markdown(turn["question"])
    with st.chat_message("assistant", avatar="⚖️"):
        results = turn["results"]
        if len(results) == 1:
            render_result(next(iter(results.values())))
        else:
            cols = st.columns(len(results))
            for col, (name, r) in zip(cols, results.items()):
                with col:
                    st.subheader(name)
                    render_result(r)


# ---------- Sidebar ----------
with st.sidebar:
    st.title("⚖️ EU Compliance Assistant")
    mode = st.radio("Version", MODES, index=0)
    st.markdown("**Documents**")
    for label in config.DOC_LABELS.values():
        st.markdown(f"- {label}")
    st.markdown("**Models (all local, free)**")
    st.markdown(f"- Chat: `{config.LLM_MODEL}` via {config.LLM_PROVIDER}\n"
                f"- Embeddings: `{config.EMBED_MODEL}`\n"
                f"- Reranker: `BAAI/bge-reranker-base` (agent only)")
    if st.button("Clear chat"):
        st.session_state.history = []
    st.caption("Portfolio project. Answers can be wrong and are not legal advice. "
               "Always check the cited page in the official document.")

# ---------- Main ----------
st.header("Ask about the EU AI Act, GDPR or Central Bank of Ireland guidance")
systems = load_systems()
st.session_state.setdefault("history", [])

cols = st.columns(len(EXAMPLES))
clicked = None
for col, q in zip(cols, EXAMPLES):
    if col.button(q, use_container_width=True):
        clicked = q

for turn in st.session_state.history:
    render_turn(turn)

question = st.chat_input("Type a question...") or clicked
if question:
    names = ["Agentic RAG", "Baseline RAG"] if mode == "Compare both" else [mode]
    results = {}
    with st.spinner("Thinking... (the agent takes about 10 to 30 seconds)"):
        for name in names:
            results[name] = systems[name](question)
    turn = {"question": question, "results": results}
    st.session_state.history.append(turn)
    render_turn(turn)
