"""
app.py — Phase 2: Streamlit chat UI
-------------------------------------
Learning goals:
  - Understand st.session_state: how Streamlit persists data across reruns
  - Use st.chat_message() and st.chat_input() — Streamlit's built-in chat widgets
  - Use st.write_stream() to render streamed tokens live in the browser
  - See how the same stream_response() generator powers both terminal and web UI

Run:
  streamlit run app.py
"""

import streamlit as st
from chat import stream_response, get_model

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Local Chatbot",
    page_icon="🤖",
    layout="centered",
)

# ── Custom styling ────────────────────────────────────────────────────────────
# Streamlit renders inside an iframe — we inject CSS to tighten up the default
# look and brand it as our own.
st.markdown("""
<style>
/* Shrink the default top padding */
.block-container { padding-top: 2rem; }

/* Style the chat input bar */
.stChatInput textarea {
    border-radius: 12px;
    font-size: 15px;
}

/* Subtle header */
.app-header {
    display: flex;
    align-items: center;
    gap: 10px;
    margin-bottom: 1.5rem;
    padding-bottom: 0.75rem;
    border-bottom: 1px solid rgba(128,128,128,0.2);
}
.model-badge {
    font-size: 11px;
    font-family: monospace;
    background: rgba(128,128,128,0.15);
    padding: 2px 8px;
    border-radius: 99px;
    color: inherit;
    opacity: 0.7;
}
</style>
""", unsafe_allow_html=True)

# ── Header ────────────────────────────────────────────────────────────────────
st.markdown(f"""
<div class="app-header">
    <span style="font-size:1.4rem; font-weight:600;">Local Chatbot</span>
    <span class="model-badge">{get_model()}</span>
</div>
""", unsafe_allow_html=True)

# ── Session state: conversation history ───────────────────────────────────────
# KEY CONCEPT: Streamlit reruns the entire script top-to-bottom on every
# interaction. st.session_state persists data across those reruns —
# without it, history would reset to [] on every message.
if "history" not in st.session_state:
    st.session_state.history = []   # list of {"role":..., "content":...}

# Sidebar: controls
with st.sidebar:
    st.markdown("### Settings")
    temperature = st.slider("Temperature", 0.0, 1.0, 0.7, 0.05,
                            help="Higher = more creative. Lower = more focused.")
    st.markdown("---")
    if st.button("Clear conversation", use_container_width=True):
        st.session_state.history = []
        st.rerun()

    st.markdown("---")
    st.markdown("### How it works")
    st.markdown("""
**session_state** persists the message list across Streamlit reruns.

**stream_response()** is a Python generator that yields tokens — `st.write_stream()` consumes them live.

**chat.py** handles all LLM logic; this file is pure UI.
    """)

    st.markdown("---")
    st.markdown("#### Message count")
    st.metric("Messages", len(st.session_state.history))

# ── Render existing conversation ──────────────────────────────────────────────
# st.chat_message() creates a styled bubble with a role avatar.
# We replay every message from history so the UI matches the state.
for msg in st.session_state.history:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# ── Handle new input ──────────────────────────────────────────────────────────
# st.chat_input() renders the sticky bottom input bar.
# It returns the submitted text (or None if no submission yet).
if prompt := st.chat_input("Ask me anything…"):

    # 1. Show and store the user message immediately
    st.session_state.history.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # 2. Stream the assistant response
    # st.write_stream() accepts any iterable of strings and renders
    # each token as it arrives — same generator, different renderer vs terminal.
    with st.chat_message("assistant"):
        try:
            response = st.write_stream(
                stream_response(st.session_state.history)
            )
        except Exception as e:
            response = f"Error: {e}\n\nIs Ollama running? Try: `ollama serve`"
            st.error(response)

    # 3. Persist the complete assistant reply to history
    st.session_state.history.append({"role": "assistant", "content": response})