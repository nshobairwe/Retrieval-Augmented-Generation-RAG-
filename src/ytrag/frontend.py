from __future__ import annotations

from typing import Any, Dict, List

import streamlit as st

from ytrag.search import RAGSearch


@st.cache_resource
def get_rag_search() -> RAGSearch:
    return RAGSearch()


def format_result(answer: str, sources: List[Dict[str, Any]], confidence: float) -> Dict[str, Any]:
    """Return a UI-friendly result structure with large readable defaults."""
    safe_answer = answer.strip() if answer else "No answer available."
    return {
        "answer": safe_answer,
        "sources": sources,
        "source_count": len(sources),
        "confidence": float(confidence),
        "font_scale": 1.4,
    }


def render_ui() -> None:
    """Render the Streamlit search interface."""
    st.set_page_config(page_title="YTRAG Search", page_icon="📚", layout="wide")

    st.markdown(
        """
        <style>
            html, body, [class*="stApp"] {
                font-size: 1.2rem;
            }
            .stApp {
                background: linear-gradient(135deg, #f5f7ff 0%, #eef7ff 100%);
            }
            h1, h2, h3 {
                font-size: 2.1rem !important;
                line-height: 1.3;
            }
            .stTextInput > div > div > input {
                font-size: 1.1rem;
                padding: 0.8rem 0.9rem;
            }
            .stButton > button {
                font-size: 1rem;
                padding: 0.7rem 1.1rem;
            }
            .stAlert, .stMarkdown, .stDataFrame {
                font-size: 1.1rem !important;
            }
        </style>
        """,
        unsafe_allow_html=True,
    )

    st.title("YTRAG: Retrieval-Augmented Generation")
    st.caption("Search the project documents and retrieve context-rich answers.")

    with st.sidebar:
        st.header("About")
        st.markdown(
            """
            Use this app to search the dataset and pull the most relevant passages from stored documents.
            The system shows the retrieved context and a concise answer.
            """
        )
        st.markdown("**Large readable interface:** enabled")

    query = st.text_input(
        "Ask a question",
        value="What are the duties and responsibilities for a data scientist?",
        placeholder="Type your query here...",
    )

    if st.button("Search documents", use_container_width=True):
        if not query.strip():
            st.warning("Please enter a question before searching.")
            return

        try:
            rag_search = get_rag_search()
        except Exception as exc:  # pragma: no cover - UI failure path
            st.error(f"Unable to initialize the search engine: {exc}")
            return

        with st.spinner("Searching the document store..."):
            answer = rag_search.search_and_summarize(query, top_k=3)
            results = rag_search.vectorstore.query(query, top_k=3)

        sources = []
        for item in results:
            metadata = item.get("metadata") or {}
            text = metadata.get("text", "")
            if not text:
                continue
            score = item.get("distance", 0.0)
            confidence = round(max(0.0, 1.0 - min(score, 1.0)), 3)
            sources.append(
                {
                    "source": metadata.get("source_file", "Document"),
                    "score": confidence,
                    "preview": text[:350],
                }
            )

        result = format_result(answer, sources, max((src["score"] for src in sources), default=0.0))

        st.subheader("Answer")
        st.markdown(f"<div style='font-size: {result['font_scale']}rem; line-height: 1.7;'> {result['answer']} </div>", unsafe_allow_html=True)

        st.caption(f"Relevant sources: {result['source_count']} | Confidence: {result['confidence']:.2f}")

        st.subheader("Retrieved passages")
        if not sources:
            st.info("No matching passages were found.")
        else:
            for index, source in enumerate(sources, start=1):
                st.markdown(f"### {index}. {source['source']}")
                st.markdown(f"**Relevance:** {source['score']:.3f}")
                st.write(source["preview"])
                st.markdown("---")


def main() -> None:
    render_ui()
