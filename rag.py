"""
rag.py — Phase 4: RAG (Retrieval-Augmented Generation) module
=============================================================
Updated for LangChain 1.x:
  - HuggingFaceEmbeddings moved to langchain_huggingface
  - Chroma moved to langchain_chroma
  - RetrievalQA replaced with modern LCEL chain (retriever | prompt | llm)
  - PromptTemplate moved to langchain_core.prompts
"""

import os
from pathlib import Path
from dotenv import load_dotenv

# ── LangChain imports ─────────────────────────────────────────────────────────

from langchain_ollama import ChatOllama
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

# HuggingFaceEmbeddings moved from langchain_community to langchain_huggingface
from langchain_huggingface import HuggingFaceEmbeddings

# Chroma moved from langchain_community to langchain_chroma
from langchain_chroma import Chroma

# Modern LCEL imports replacing RetrievalQA
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

load_dotenv()

MODEL         = os.getenv("MODEL", "phi3")
BASE_URL      = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
DOCS_DIR      = os.getenv("DOCS_DIR", "./docs")
CHROMA_DIR    = os.getenv("CHROMA_DIR", "./chroma_db")
CHUNK_SIZE    = int(os.getenv("CHUNK_SIZE", "500"))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "50"))
TOP_K         = int(os.getenv("TOP_K", "3"))
TEMPERATURE   = float(os.getenv("TEMPERATURE", "0.7"))


# ── Shared objects ────────────────────────────────────────────────────────────

llm = ChatOllama(
    model=MODEL,
    base_url=BASE_URL,
    temperature=TEMPERATURE,
)

embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2",
    model_kwargs={"device": "cpu"},
    encode_kwargs={"normalize_embeddings": True},
)

splitter = RecursiveCharacterTextSplitter(
    chunk_size=CHUNK_SIZE,
    chunk_overlap=CHUNK_OVERLAP,
    separators=["\n\n", "\n", ". ", " ", ""],
)

RAG_PROMPT = PromptTemplate(
    input_variables=["context", "question"],
    template="""You are a helpful assistant. Answer the question using ONLY
the context provided below. If the answer is not in the context, say
"I couldn't find that in the provided documents."

Context:
{context}

Question: {question}

Answer:"""
)


# ── Phase 1: Indexing pipeline ────────────────────────────────────────────────

def load_documents(docs_dir: str) -> list:
    docs = []
    docs_path = Path(docs_dir)

    if not docs_path.exists():
        docs_path.mkdir(parents=True)
        return docs

    for pdf_path in docs_path.glob("*.pdf"):
        try:
            loader = PyPDFLoader(str(pdf_path))
            docs.extend(loader.load())
            print(f"  Loaded PDF: {pdf_path.name}")
        except Exception as e:
            print(f"  Could not load {pdf_path.name}: {e}")

    for ext in ["*.txt", "*.md"]:
        for txt_path in docs_path.glob(ext):
            try:
                loader = TextLoader(str(txt_path), encoding="utf-8")
                docs.extend(loader.load())
                print(f"  Loaded text: {txt_path.name}")
            except Exception as e:
                print(f"  Could not load {txt_path.name}: {e}")

    return docs


def index_documents(docs_dir: str = DOCS_DIR) -> tuple[int, int]:
    print(f"\nIndexing documents from: {docs_dir}")

    raw_docs = load_documents(docs_dir)
    if not raw_docs:
        print("  No documents found.")
        return 0, 0

    print(f"  Loaded {len(raw_docs)} document pages/files")

    chunks = splitter.split_documents(raw_docs)
    print(f"  Split into {len(chunks)} chunks")

    Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=CHROMA_DIR,
        collection_name="local_docs",
    )

    print(f"  Stored {len(chunks)} chunks in ChromaDB at {CHROMA_DIR}/")
    return len(raw_docs), len(chunks)


# ── Phase 2: Query pipeline ───────────────────────────────────────────────────

def load_vectorstore() -> Chroma | None:
    if not Path(CHROMA_DIR).exists():
        return None
    try:
        return Chroma(
            persist_directory=CHROMA_DIR,
            embedding_function=embeddings,
            collection_name="local_docs",
        )
    except Exception:
        return None


def _format_docs(docs) -> str:
    """Combine retrieved chunks into a single context string."""
    return "\n\n".join(doc.page_content for doc in docs)


def rag_answer(question: str) -> tuple[str, list]:
    """
    Answer a question using RAG.
    Returns (answer_text, source_chunks_list).
    """
    vectorstore = load_vectorstore()
    if vectorstore is None:
        return (
            "No documents indexed yet. "
            "Upload files to the docs/ folder and click 'Index Documents'.",
            [],
        )

    retriever = vectorstore.as_retriever(
        search_type="similarity",
        search_kwargs={"k": TOP_K},
    )

    # Retrieve source docs separately so we can return them
    source_docs = retriever.invoke(question)

    # Build LCEL chain: format context → fill prompt → LLM → parse string
    chain = (
        {"context": lambda _: _format_docs(source_docs), "question": RunnablePassthrough()}
        | RAG_PROMPT
        | llm
        | StrOutputParser()
    )

    try:
        answer = chain.invoke(question)
        return answer, source_docs
    except Exception as e:
        return f"Error during retrieval: {e}", []


def get_index_stats() -> dict:
    vs = load_vectorstore()
    if vs is None:
        return {"indexed": False, "chunks": 0}
    try:
        count = vs._collection.count()
        return {"indexed": True, "chunks": count}
    except Exception:
        return {"indexed": False, "chunks": 0}


def clear_index():
    import shutil
    if Path(CHROMA_DIR).exists():
        shutil.rmtree(CHROMA_DIR)