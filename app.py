"""
app.py — Phase 3: Streamlit UI (minimal changes from Phase 2)
=============================================================
What changed from Phase 2
--------------------------
1. Imports: now calls chat_once() and clear_memory() from chat.py
   instead of stream_response() with a history list.

2. History display: we fetch LangChain's memory messages for rendering
   instead of maintaining our own session_state list.

3. Clear button: calls clear_memory() to wipe LangChain's internal state.

Everything else — page config, sidebar, chat_input — is identical.
This demonstrates that separating UI from LLM logic pays off immediately:
a big internal change (Phase 2 → Phase 3) barely touched this file.

Run:
  streamlit run app.py
"""

import streamlit as st
from chat import chat_once, clear_memory, get_memory_messages, get_model
# remove chat_once, clear_memory, get_memory_messages from import
from chat import build_chain

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Local Chatbot · Phase 3",
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
</style>
""", unsafe_allow_html=True)

st.markdown(f"""
<div class="app-header">
  <span style="font-size:1.4rem;font-weight:600;">Local Chatbot</span>
  <span class="model-badge">{get_model()}</span>
  <span class="phase-badge">Phase 3 · LangChain</span>
</div>
""", unsafe_allow_html=True)

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### Settings")
    st.slider("Temperature", 0.0, 1.0, 0.7, 0.05,
              help="Higher = more creative. Restart the app to apply changes.")

    st.markdown("---")
    if st.button("Clear conversation", use_container_width=True):
        clear_memory()          # ← calls LangChain memory.clear()
        st.rerun()

    st.markdown("---")
    st.markdown("### What's new in Phase 3")
    st.markdown("""
**LangChain** manages conversation history automatically via `ConversationBufferMemory`.

**ChatPromptTemplate** formats the system prompt + history + user message into a structured prompt.

**LLMChain** wires memory + template + model into one callable unit.

You no longer maintain a `history` list manually.
    """)

    # Show raw memory contents — great for learning!
    st.markdown("---")
    msgs = get_memory_messages()
    st.markdown(f"### Memory ({len(msgs)} messages)")
    if msgs:
        for m in msgs:
            role = type(m).__name__.replace("Message", "")
            st.caption(f"**{role}:** {str(m.content)[:80]}…" if len(str(m.content)) > 80 else f"**{role}:** {m.content}")
    else:
        st.caption("Empty — start a conversation!")

# ── Render conversation from LangChain memory ─────────────────────────────────
# Instead of session_state history, we read directly from LangChain's memory.
# LangChain stores pairs: HumanMessage, AIMessage, HumanMessage, AIMessage...
msgs = get_memory_messages()
for msg in msgs:
    role = "user" if "Human" in type(msg).__name__ else "assistant"
    with st.chat_message(role):
        st.markdown(msg.content)


# Persist chain across Streamlit reruns
if "chain" not in st.session_state:
    from chat import build_chain
    st.session_state.chain = build_chain()

chain = st.session_state.chain



# ── Handle new input ──────────────────────────────────────────────────────────
if prompt := st.chat_input("Ask me anything…"):

    # Show the user's message immediately
    with st.chat_message("user"):
        st.markdown(prompt)

    # Stream the assistant reply
    # chat_once() is a generator that both calls the chain AND saves to memory
    with st.chat_message("assistant"):
        try:
            # st.write_stream consumes any generator of strings
            st.write_stream(chat_once(prompt))
        except Exception as e:
            st.error(f"Error: {e}\n\nIs Ollama running? Try: `ollama serve`")

    # No manual history append needed — LangChain memory handles it
    st.rerun()  # refresh sidebar message count