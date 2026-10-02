import os
from collections.abc import Iterator

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_ollama import ChatOllama
from ytrag.vectorstore import FaissVectorStore
from ytrag.data_loader import load_all_documents


class RAGSearch:
    def __init__(
        self,
        persist_dir: str = "faiss_store",
        embedding_model: str = "all-MiniLM-L6-v2",
        llm_model: str | None = None,
        ollama_url: str = "http://localhost:11434",
    ):
        self.vectorstore = FaissVectorStore(persist_dir, embedding_model)
        self.last_sources: list[str] = []
        self.llm_model = llm_model or os.getenv("OLLAMA_MODEL", "qwen2.5:3b")
        self.ollama_url = ollama_url

        faiss_path = os.path.join(persist_dir, "faiss.index")
        meta_path = os.path.join(persist_dir, "metadata.pkl")
        if not (os.path.exists(faiss_path) and os.path.exists(meta_path)):
            docs = load_all_documents("data")
            self.vectorstore.build_from_documents(docs)
        else:
            self.vectorstore.load()
            needs_source_rebuild = not all(
                "source_file" in metadata for metadata in self.vectorstore.metadata
            )
            if needs_source_rebuild:
                print("[INFO] Rebuilding the vector store to add PDF source citations.")
                docs = load_all_documents("data")
                self.vectorstore.build_from_documents(docs)

        self.llm = self._load_llm()

    def _load_llm(self):
        candidates = [self.llm_model, "qwen2.5:3b", "llama3.2:latest", "qwen3:4b"]
        seen = set()
        for model_name in candidates:
            if not model_name or model_name in seen:
                continue
            seen.add(model_name)
            try:
                llm = ChatOllama(
                    model=model_name,
                    base_url=self.ollama_url,
                    temperature=0.3,
                    num_ctx=4096,
                )
                print(f"[INFO] Ollama LLM initialized: {model_name} ({self.ollama_url})")
                return llm
            except Exception as error:
                print(f"[WARN] Could not load Ollama model '{model_name}': {error}")

        print("[WARN] No compatible Ollama model could be loaded. Using retrieval snippets only.")
        return None

    def _context_for(self, query: str, top_k: int) -> tuple[str, list[str]]:
        results = self.vectorstore.query(query, top_k=top_k)
        relevant_results = []
        for result in results:
            metadata = result.get("metadata")
            if not metadata:
                continue
            distance = result.get("distance")
            try:
                numeric_distance = float(distance)
            except (TypeError, ValueError):
                continue
            if numeric_distance <= 1.2:
                relevant_results.append(result)

        texts = [r["metadata"].get("text", "") for r in relevant_results if r.get("metadata")]
        sources = []
        for result in relevant_results:
            metadata = result.get("metadata") or {}
            source_file = metadata.get("source_file")
            if not source_file:
                continue
            employer = metadata.get("employer", "Employer not identified")
            page = metadata.get("page")
            citation = f"{employer} | {source_file}"
            if isinstance(page, int):
                citation += f", page {page + 1}"
            if citation not in sources:
                sources.append(citation)
        return "\n\n---\n\n".join(texts), sources

    @staticmethod
    def _messages(query: str, context: str) -> list:
        return [
            SystemMessage(
                content=(
                    "You are a warm, professional assistant answering questions using only the "
                    "supplied documents. Speak naturally and clearly. If the requested information "
                    "is not present in the available documents, say so politely and briefly. "
                    "Do not guess, invent facts, or answer outside the provided context. "
                    "When the information is missing, respond with a respectful explanation and offer "
                    "a helpful next step or related topic the user can ask about."
                )
            ),
            HumanMessage(content=f"Question: {query}\n\nDocument context:\n{context}"),
        ]

    @staticmethod
    def _not_found_response(query: str) -> str:
        return (
            "I’m sorry, but I couldn’t find information related to your question in the documents "
            "currently available to me. Please provide more context or ask about a topic covered in the "
            f"uploaded materials. If you’d like, I can help you with a related question from the same documents."
        )

    def stream_answer(self, query: str, top_k: int = 5) -> Iterator[str]:
        """Yield a conversational answer as the LLM produces it, or a retrieval-based fallback."""
        context, self.last_sources = self._context_for(query, top_k)
        if not context:
            yield self._not_found_response(query)
            return

        if self.llm is None:
            yield (
                "I could not load a local language model in this environment, so I am returning the "
                "most relevant retrieved passages instead.\n\n"
                + context[:2000]
            )
            return

        try:
            for chunk in self.llm.stream(self._messages(query, context)):
                text = chunk.content
                if isinstance(text, str) and text:
                    yield text
        except Exception as error:
            fallback = (
                "I’m sorry, but the local model could not complete the answer in this environment. "
                "Here are the most relevant passages from the available documents that may help:\n\n"
                + context[:2000]
            )
            yield fallback

    def sources_for_last_answer(self) -> list[str]:
        """Return the PDF employer, filename, and page used for the latest answer."""
        return self.last_sources.copy()

    def search_and_summarize(self, query: str, top_k: int = 5) -> str:
        """Return a complete answer for callers that do not need streaming."""
        return "".join(self.stream_answer(query, top_k=top_k))


if __name__ == "__main__":
    rag_search = RAGSearch()
    query = input("Enter your question: ").strip()
    if query:
        summary = rag_search.search_and_summarize(query, top_k=3)
        print("Summary:", summary)
