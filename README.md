# YTRAG: Learning Retrieval-Augmented Generation (RAG)

This project is a practical classroom introduction to the **data-ingestion and retrieval** parts of a Retrieval-Augmented Generation (RAG) system.

Instead of asking an AI model to rely only on its training data, a RAG system first searches a collection of your documents and uses the most relevant passages as context for an answer. In this project, you will learn how that searchable document collection is built.

## What You Will Learn

By completing the notebooks, you should be able to:

- Explain what a LangChain `Document` is and why metadata matters.
- Load `.txt` and `.pdf` files into a consistent document structure.
- Split long documents into smaller chunks suitable for retrieval.
- Turn text chunks into numerical embeddings with `sentence-transformers`.
- Store embeddings and metadata in a persistent ChromaDB vector database.
- Retrieve the chunks most similar to a user's question.

## How the RAG Pipeline Works

```text
Source files (.txt and .pdf)
            |
            v
      Document loaders
            |
            v
     Text chunking / splitting
            |
            v
  Embedding model (MiniLM)
            |
            v
 ChromaDB vector store (saved on disk)
            |
            v
 Similarity search for a question
```

The project currently focuses on **retrieval**. A complete RAG application would add an LLM after retrieval to generate an answer from the returned chunks.

## Project Structure

```text
YTRAG/
|-- data/
|   |-- pdf/                 # Example PDF documents to load
|   |-- text_files/          # Example plain-text learning material
|   `-- vector_store/        # Persistent ChromaDB files created by the notebook
|-- notebook/
|   |-- document.ipynb       # Lesson 1: documents and document loaders
|   `-- pdf_loader.ipynb     # Lesson 2: PDF ingestion, embeddings, storage, retrieval
|-- src/ytrag/               # Python package starter files
|-- requirements.txt         # Dependencies for pip users
|-- pyproject.toml           # Project metadata and dependencies for uv users
`-- README.md
```

## Prerequisites

- Python 3.11 or newer
- A working internet connection the first time you run the embedding model. The `all-MiniLM-L6-v2` model is downloaded from Hugging Face and then cached locally.
- Jupyter Notebook or VS Code with the Jupyter extension

## Installation

Open PowerShell in the project folder and create a virtual environment:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

Or, if you use [uv](https://docs.astral.sh/uv/), install the project dependencies with:

```powershell
uv sync
```

Then register the environment as a Jupyter kernel:

```powershell
python -m ipykernel install --user --name ytrag --display-name "Python (ytrag)"
```

## Classroom Lessons

### Lesson 1: Understanding Documents and Loaders

Open `notebook/document.ipynb`.

This notebook introduces the LangChain `Document` object, which contains:

- `page_content`: the actual text to search.
- `metadata`: useful details such as the source file, author, page, or file type.

Work through these activities in order:

1. Create and inspect a simple `Document`.
2. Create example text files about Python and machine learning.
3. Use `TextLoader` to load one text file.
4. Use `DirectoryLoader` to load every text file in `data/text_files`.
5. Use `PyMuPDFLoader` to load every PDF in `data/pdf`.

### Lesson 2: From PDFs to Searchable Knowledge

Open `notebook/pdf_loader.ipynb`.

This notebook builds the retrieval pipeline:

1. Find PDF files recursively and load their pages.
2. Add source metadata to every loaded page.
3. Split pages into overlapping chunks with `RecursiveCharacterTextSplitter`.
4. Generate embeddings using `all-MiniLM-L6-v2`.
5. Save the chunks, embeddings, and metadata in ChromaDB.
6. Ask a question and retrieve the top matching chunks.

Run notebook cells from top to bottom. The first model download can take a little longer than later runs.

## Try Your Own Question

At the end of `pdf_loader.ipynb`, change the query and rerun the retrieval cell:

```python
retrieved_documents = rag_retriever.retrieve(
    "What skills are required for artificial intelligence jobs?",
    top_k=3,
)
retrieved_documents
```

Try questions such as:

- `What does the application letter say about the applicant's experience?`
- `Which qualifications are mentioned in the Ajira documents?`
- `What information appears in the PDF files about employment?`

Read the returned `content`, `metadata`, `similarity_score`, and `distance`. The retrieval result is evidence for an eventual AI answer, not the final generated answer itself.

## Student Exercises

1. Add a new text file to `data/text_files` about a topic you know well, then load it with `DirectoryLoader`.
2. Add a PDF to `data/pdf`, rerun the ingestion cells, and confirm that its filename appears in the metadata.
3. Compare chunk sizes of `500`, `1000`, and `1500`. How do the retrieved results change?
4. Change `chunk_overlap` from `200` to `0` and explain what information could be lost at chunk boundaries.
5. Change `top_k` from `3` to `5`. Which extra results are useful, and which are less relevant?
6. Use a `score_threshold` such as `0.35` and observe how it filters weak matches.
7. Add an LLM step that receives the retrieved chunks and answers only from that context.

## Important Concepts

| Concept | Meaning in this project |
|---|---|
| Document | Text plus metadata in a standard structure. |
| Chunk | A smaller piece of a document used for search. |
| Embedding | A list of numbers representing the meaning of text. |
| Vector store | A database designed to store and search embeddings. |
| Similarity search | Finding chunks whose embeddings are closest to a query embedding. |
| Metadata | Extra information that identifies where a chunk came from. |

## Resetting the Vector Store

ChromaDB persists the stored chunks in `data/vector_store`. Each time you run the add-documents cell, it adds new entries to the existing collection. During experimentation, this can create duplicates.

To start fresh, close any notebook using the store, delete the contents of `data/vector_store`, and rerun the ingestion notebook. Do not remove the directory unless you intend to recreate it.

## Troubleshooting

**`ModuleNotFoundError`**

Confirm that your virtual environment is active, install the requirements again, and select the `Python (ytrag)` kernel in Jupyter.

**The embedding model will not load**

Check your internet connection for the first run. After the model has downloaded successfully, it is normally read from the local Hugging Face cache.

**No documents are retrieved**

Make sure the ingestion and `add_documents` cells ran successfully. Also check that the requested `top_k` is not greater than the number of stored chunks.

**Results are duplicated**

Clear `data/vector_store` and run the pipeline once from the beginning.

## Teaching Notes

- Start with `document.ipynb` before moving to `pdf_loader.ipynb`.
- Encourage students to inspect values after every step rather than treating the notebook as a script to run blindly.
- Use the metadata in retrieval results to discuss trust, traceability, and citations in AI systems.
- Remind students that retrieval quality depends on the source material, chunking strategy, embedding model, and query wording.

## Dependencies Used

- [LangChain](https://python.langchain.com/) for document loaders and text splitting
- [PyPDF / PyMuPDF](https://pymupdf.readthedocs.io/) for PDF extraction
- [Sentence Transformers](https://www.sbert.net/) for embeddings
- [ChromaDB](https://www.trychroma.com/) for vector storage and retrieval
- [FAISS](https://faiss.ai/) for vector-search tooling

Happy learning. Build the pipeline slowly, inspect each output, and let the data tell you what the system is actually doing.
