"""
app.py — Phase 4: Streamlit UI with RAG mode toggle
=====================================================
What's new from Phase 3
------------------------
1. Mode toggle in sidebar: "Chat" (Phase 3 LLMChain) vs "RAG" (RetrievalQA)
2. Document uploader: drag-and-drop files → saved to docs/ → indexed
3. Index stats: shows how many chunks are stored in ChromaDB
4. Source display: after a RAG answer, shows which document chunks were used

Learning goals
--------------
  - See how two completely separate chains (LLMChain vs RetrievalQA) share
    the same UI without either knowing about the other
  - Understand st.file_uploader() for accepting user files
  - See how "source documents" prove where the answer came from

Run:
  streamlit run app.py
"""

import streamlit as st
import os
from pathlib import Path
from chat import chat_once, clear_memory, get_memory_messages, get_model
from rag import index_documents, rag_answer, get_index_stats, clear_index

DOCS_DIR = os.getenv("DOCS_DIR", "./docs")

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Local Chatbot · Phase 4",
    page_icon="🤖",
    layout="centered",
)

st.markdown("""
<style>
.block-container { padding-top: 2rem; }
.app-header { display:flex; align-items:center; gap:10px; margin-bottom:1.5rem;
              padding-bottom:.75rem; border-bottom:1px solid rgba(128,128,128,.2); }
.model-badge { font-size:11px; font-family:monospace; background:rgba(128,128,128,.15);
               padding:2px 8px; border-radius:99px; opacity:.7; }
.phase-badge { font-size:11px; background:#E1F5EE; color:#085041;
               padding:2px 8px; border-radius:99px; }
.source-box { border:1px solid rgba(128,128,128,.2); border-radius:8px;
              padding:10px 14px; margin-top:6px; font-size:12px;
              color: var(--secondary-text-color); }
</style>
""", unsafe_allow_html=True)

st.markdown(f"""
<div class="app-header">
  <span style="font-size:1.4rem;font-weight:600;">Local Chatbot</span>
  <span class="model-badge">{get_model()}</span>
  <span class="phase-badge">Phase 4 · RAG</span>
</div>
""", unsafe_allow_html=True)

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### Mode")

    # This toggle is the key new addition — switches between the two chains
    mode = st.radio(
        "Answer using:",
        ["💬 Chat (general)", "📄 RAG (your documents)"],
        help=("Chat uses the model's own knowledge.\n"
              "RAG answers from documents you upload."),
    )
    rag_mode = mode.startswith("📄")

    st.markdown("---")
    st.markdown("### Documents")

    # Show current index status
    stats = get_index_stats()
    if stats["indexed"]:
        st.success(f"{stats['chunks']} chunks indexed")
    else:
        st.warning("No documents indexed yet")

    # File uploader — saves files to docs/ folder
    uploaded = st.file_uploader(
        "Upload PDF or TXT files",
        type=["pdf", "txt", "md"],
        accept_multiple_files=True,
    )
    if uploaded:
        Path(DOCS_DIR).mkdir(exist_ok=True)
        for f in uploaded:
            dest = Path(DOCS_DIR) / f.name
            dest.write_bytes(f.read())
        st.caption(f"Saved {len(uploaded)} file(s) to docs/")

    # Index button — runs the full indexing pipeline
    col1, col2 = st.columns(2)
    with col1:
        if st.button("Index docs", use_container_width=True):
            with st.spinner("Indexing…"):
                n_docs, n_chunks = index_documents()
            if n_chunks:
                st.success(f"{n_chunks} chunks from {n_docs} pages")
            else:
                st.error("No documents found in docs/ folder")
            st.rerun()
    with col2:
        if st.button("Clear index", use_container_width=True):
            clear_index()
            st.rerun()

    st.markdown("---")

    # Chat memory controls (only relevant in Chat mode)
    st.markdown("### Chat memory")
    msgs = get_memory_messages()
    st.metric("Messages", len(msgs))
    if st.button("Clear conversation", use_container_width=True):
        clear_memory()
        st.rerun()

    st.markdown("---")
    st.markdown("### How Phase 4 works")
    st.markdown("""
**Indexing** (runs once):
docs → chunks → embeddings → ChromaDB

**RAG query** (every question):
question → embed → retrieve top-k chunks → LLM answers from context

**Chat mode** still uses Phase 3's LLMChain with memory.
    """)

# ── Session state ─────────────────────────────────────────────────────────────
# We keep a separate display history for RAG mode because RetrievalQA doesn't
# use ConversationBufferMemory — it's stateless by default.
if "rag_history" not in st.session_state:
    st.session_state.rag_history = []  # list of {"role", "content", "sources"}

# ── Render conversation ───────────────────────────────────────────────────────
if rag_mode:
    # RAG mode: render from session_state rag_history
    for msg in st.session_state.rag_history:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            # Show source documents below assistant replies
            if msg["role"] == "assistant" and msg.get("sources"):
                with st.expander(f"Sources ({len(msg['sources'])} chunks used)"):
                    for i, doc in enumerate(msg["sources"], 1):
                        src = doc.metadata.get("source", "unknown")
                        page = doc.metadata.get("page", "")
                        label = f"{Path(src).name}" + (f" · page {page+1}" if page != "" else "")
                        st.caption(f"**Chunk {i} — {label}**")
                        st.markdown(f'<div class="source-box">{doc.page_content[:300]}…</div>',
                                    unsafe_allow_html=True)
else:
    # Chat mode: render from LangChain memory
    for msg in get_memory_messages():
        role = "user" if "Human" in type(msg).__name__ else "assistant"
        with st.chat_message(role):
            st.markdown(msg.content)

# ── Handle new input ──────────────────────────────────────────────────────────
if prompt := st.chat_input(
    "Ask about your documents…" if rag_mode else "Ask me anything…"
):
    with st.chat_message("user"):
        st.markdown(prompt)

    if rag_mode:
        # ── RAG mode ──────────────────────────────────────────────────────────
        # rag_answer() returns (answer_string, [source_Document, ...])
        # It is NOT a generator — RetrievalQA returns the full answer at once.
        st.session_state.rag_history.append({"role": "user", "content": prompt, "sources": []})

        with st.chat_message("assistant"):
            with st.spinner("Searching documents…"):
                answer, sources = rag_answer(prompt)
            st.markdown(answer)

            # Show source chunks in an expander so the user can verify
            if sources:
                with st.expander(f"Sources ({len(sources)} chunks used)"):
                    for i, doc in enumerate(sources, 1):
                        src = doc.metadata.get("source", "unknown")
                        page = doc.metadata.get("page", "")
                        label = f"{Path(src).name}" + (f" · page {page+1}" if page != "" else "")
                        st.caption(f"**Chunk {i} — {label}**")
                        st.markdown(f'<div class="source-box">{doc.page_content[:300]}…</div>',
                                    unsafe_allow_html=True)

        st.session_state.rag_history.append({
            "role": "assistant",
            "content": answer,
            "sources": sources,
        })

    else:
        # ── Chat mode (Phase 3 unchanged) ─────────────────────────────────────
        with st.chat_message("assistant"):
            try:
                st.write_stream(chat_once(prompt))
            except Exception as e:
                st.error(f"Error: {e}\n\nIs Ollama running? Try: `ollama serve`")

    st.rerun()