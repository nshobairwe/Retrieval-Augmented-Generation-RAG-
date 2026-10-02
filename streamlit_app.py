import os
import sys

import streamlit as st

ROOT = os.path.dirname(os.path.abspath(__file__))
SRC_DIR = os.path.join(ROOT, "src")
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

from ytrag.search import RAGSearch


st.set_page_config(page_title="Job Assistant", page_icon="💼", layout="wide")

st.markdown(
    """
    <style>
      .stApp { background: linear-gradient(180deg, #f5f8ff 0%, #eef7f3 100%); }
      .block-container { max-width: 1100px; padding-top: 2.3rem; padding-bottom: 7rem; }
      h1 { color: #12263f; font-size: 2.9rem !important; letter-spacing: 0 !important; }
      .subtitle { color: #425972; font-size: 1.18rem; margin: -0.6rem 0 2rem; }
      [data-testid="stChatMessage"] { border-radius: 12px; padding: 1rem 1.2rem; }
      [data-testid="stChatMessage"] p, [data-testid="stChatMessage"] li {
          font-size: 1.12rem !important; line-height: 1.7 !important;
      }
      [data-testid="stChatInput"] textarea { font-size: 1.08rem !important; }
      .source-label { color: #2c5d4d; font-size: 1rem; font-weight: 700; margin-top: 1rem; }
      .source-item { color: #425972; font-size: 1rem; line-height: 1.55; }
      [data-testid="stSidebar"] { background: #12263f; }
      [data-testid="stSidebar"] * { color: #f5f8ff; }
      [data-testid="stSidebar"] a { color: #d9ecff; }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_resource(show_spinner="Loading Job Assistant...")
def get_rag() -> RAGSearch:
    return RAGSearch()


def build_memory_prompt(current_prompt: str, history: list[dict], limit: int = 6) -> str:
    memory = ""
    recent = history[-limit:]
    for entry in recent:
        if entry["role"] == "user":
            memory += f"User: {entry['content']}\n"
        elif entry["role"] == "assistant":
            memory += f"Assistant: {entry['content']}\n"
    if memory:
        return f"Conversation memory:\n{memory}\nCurrent user question: {current_prompt}"
    return current_prompt


def show_sources(sources: list[str]) -> None:
    if not sources:
        return
    st.markdown('<div class="source-label">Sources</div>', unsafe_allow_html=True)
    for source in sources:
        st.markdown(f'<div class="source-item">{source}</div>', unsafe_allow_html=True)


if "messages" not in st.session_state:
    st.session_state.messages = []

with st.sidebar:
    st.header("Job Assistant")
    st.write("A production-ready assistant for job-related questions using the loaded vacancy documents.")
    st.markdown("---")
    if st.button("Clear conversation", key="clear_chat", use_container_width=True):
        st.session_state.messages = []
        st.rerun()

st.title("Job Assistant")
col_title, col_clear = st.columns([4, 1])
with col_clear:
    if st.button("Clear", key="header_clear", use_container_width=True):
        st.session_state.messages = []
        st.rerun()
with col_title:
    st.markdown(
        '<p class="subtitle">A conversational job support assistant that keeps context across your questions and uses the relevant job documents to answer clearly.</p>',
        unsafe_allow_html=True,
    )

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if message["role"] == "assistant":
            show_sources(message.get("sources", []))

if prompt := st.chat_input("Ask about jobs, qualifications, duties, or employers..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    rag = get_rag()
    memory_prompt = build_memory_prompt(prompt, st.session_state.messages)

    response_chunks = []
    with st.chat_message("assistant"):
        text_placeholder = st.empty()
        for chunk in rag.stream_answer(memory_prompt, top_k=3):
            response_chunks.append(chunk)
            text_placeholder.markdown("".join(response_chunks))
        show_sources(rag.sources_for_last_answer())

    answer = "".join(response_chunks)
    sources = rag.sources_for_last_answer()
    st.session_state.messages.append(
        {"role": "assistant", "content": answer, "sources": sources}
    )
