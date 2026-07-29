"""Streamlit chat UI for the Research Paper Answer Bot (stretch goal 2).

Chat with memory, source cards showing paper title + page + score, and live switches
for embedding model, retrieval strategy, k and Corrective RAG.
"""

from __future__ import annotations

import sys
import time
import uuid
from pathlib import Path

import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from rag_bot.config import settings
from rag_bot.embeddings.registry import EMBEDDING_MODELS
from rag_bot.generation.chain import build_pipeline
from rag_bot.generation.llm import list_models
from rag_bot.ingest.chunking import CHUNKER_NAMES
from rag_bot.memory import ConversationalRAG, SessionStore
from rag_bot.retrieval.factory import DESCRIPTIONS, STRATEGIES
from rag_bot.store.chroma_store import get_store, store_size

st.set_page_config(page_title="Research Paper Answer Bot", page_icon="📄", layout="wide")

SAMPLE_QUESTIONS = [
    "What is scaled dot-product attention and why is it scaled?",
    "What are the two pre-training objectives used by BERT?",
    "In LoRA, what does the rank r control?",
    "How many experts does Mixtral activate per token?",
    "Why is FlashAttention faster despite computing exact attention?",
]


@st.cache_resource(show_spinner="Loading models and index…")
def load_bot(embed_key: str, chunker: str, strategy: str, k: int, crag: bool) -> ConversationalRAG:
    pipeline = build_pipeline(strategy, embed_key, chunker, crag=crag, k=k)
    return ConversationalRAG(pipeline, SessionStore())


@st.cache_data(ttl=60, show_spinner=False)
def server_models() -> list[str]:
    try:
        return list_models()
    except Exception:
        return []


@st.cache_data(show_spinner=False)
def collection_size(embed_key: str, chunker: str) -> int:
    try:
        return store_size(get_store(embed_key, chunker))
    except Exception:
        return 0


def render_sources(citations) -> None:
    if not citations:
        return
    st.caption("Top supporting passages")
    columns = st.columns(len(citations))
    for column, citation in zip(columns, citations, strict=False):
        with column, st.container(border=True):
            where = citation.url if citation.source_type == "web" else f"page {citation.page}"
            badge = "🌐 web" if citation.source_type == "web" else "📄 paper"
            st.markdown(f"**[{citation.index}] {citation.title}**")
            st.caption(f"{badge} · {where} · score {citation.score:.3f}")
            with st.expander("passage"):
                st.write(citation.snippet + "…")


# ---------------------------------------------------------------- sidebar
with st.sidebar:
    st.title("⚙️ Configuration")

    embed_key = st.selectbox(
        "Embedding model",
        sorted(EMBEDDING_MODELS),
        index=sorted(EMBEDDING_MODELS).index(settings.embed_model),
        help="Indexes must be built for the selected model (see `make index`).",
    )
    chunker = st.selectbox(
        "Chunking", list(CHUNKER_NAMES), index=list(CHUNKER_NAMES).index(settings.chunker)
    )
    strategy = st.selectbox(
        "Retrieval strategy",
        list(STRATEGIES),
        index=list(STRATEGIES).index(settings.retrieval_strategy),
    )
    st.caption(DESCRIPTIONS[strategy])
    top_k = st.slider("Passages cited (k)", 1, 8, settings.top_k)
    crag = st.toggle(
        "Corrective RAG",
        value=settings.crag_enabled,
        help="Grade retrieved passages; fall back to a free web search when they are weak.",
    )

    st.divider()
    indexed = collection_size(embed_key, chunker)
    st.metric("Chunks indexed", f"{indexed:,}")
    if indexed == 0:
        st.error(
            f"No index for `{embed_key}` + `{chunker}`.\n\nRun:\n"
            f"`python scripts/build_index.py --embed {embed_key} --chunker {chunker}`"
        )

    models = server_models()
    st.caption(f"LLM server: `{settings.lmstudio_base_url}`")
    st.caption(f"Model: `{settings.llm_model}`" + ("" if models else " — server unreachable"))

    st.divider()
    if st.button("🗑️ Clear conversation", use_container_width=True):
        st.session_state.messages = []
        st.session_state.session_id = str(uuid.uuid4())[:8]
        st.rerun()
    st.caption(f"Session `{st.session_state.get('session_id', 'new')}`")

# ---------------------------------------------------------------- main
st.title("📄 Research Paper Answer Bot")
st.caption(
    "RAG over 15 seminal GenAI papers — grounded answers with paper title and page citations. "
    "Fully local, open-source models."
)

if "messages" not in st.session_state:
    st.session_state.messages = []
if "session_id" not in st.session_state:
    st.session_state.session_id = str(uuid.uuid4())[:8]

if not st.session_state.messages:
    st.write("**Try one of these:**")
    for column, question in zip(st.columns(len(SAMPLE_QUESTIONS)), SAMPLE_QUESTIONS, strict=True):
        if column.button(question, use_container_width=True):
            st.session_state.pending = question
            st.rerun()

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if message.get("citations"):
            render_sources(message["citations"])
        if message.get("meta"):
            st.caption(message["meta"])

prompt = st.chat_input("Ask about the papers…") or st.session_state.pop("pending", None)

if prompt:
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        bot = load_bot(embed_key, chunker, strategy, top_k, crag)
        started = time.perf_counter()
        with st.spinner("Retrieving and generating…"):
            result = bot.ask(prompt, session_id=st.session_state.session_id, k=top_k)
        st.markdown(result.answer)
        render_sources(result.citations)

        meta = f"{result.strategy} · {embed_key} · {time.perf_counter() - started:.1f}s"
        if result.rewritten_query:
            meta += f" · rewritten: “{result.rewritten_query}”"
        if result.used_web_search:
            meta += " · 🌐 web fallback used"
        st.caption(meta)

    st.session_state.messages.append(
        {
            "role": "assistant",
            "content": result.answer,
            "citations": result.citations,
            "meta": meta,
        }
    )
