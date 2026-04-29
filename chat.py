"""
chat.py — Phase 3: LangChain-powered LLM module
================================================
What changed from Phase 2
--------------------------
Phase 2 used the raw OpenAI Python client and a plain list for history.
Phase 3 replaces those with LangChain building blocks:

  OLD                          NEW
  ─────────────────────────    ───────────────────────────────────
  OpenAI(base_url=...)         ChatOllama(model=..., base_url=...)
  list[dict] history           ConversationBufferMemory
  manually format messages     ChatPromptTemplate
  manual history append        LLMChain (does it automatically)

The public interface stays the same:
  stream_response(history) → yields string tokens
  get_model()              → returns model name string

So app.py barely needs to change.

New concepts explained inline below — read the comments as you go.
"""

import os
from dotenv import load_dotenv

# ── LangChain imports ─────────────────────────────────────────────────────────

# ChatOllama: a LangChain chat model that talks to your local Ollama server.
# It's a drop-in replacement for ChatOpenAI — same interface, local model.
from langchain_ollama import ChatOllama

# ChatPromptTemplate: define the shape of your prompt with named slots.
# MessagesPlaceholder: a slot that gets filled with a list of chat messages
#   (this is how {chat_history} gets injected into the prompt).
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

# ConversationBufferMemory: stores every user+assistant exchange.
# return_messages=True means it stores them as proper Message objects,
# not a plain text string — required when using MessagesPlaceholder.
from langchain.memory import ConversationBufferMemory

# RunnableWithMessageHistory is the modern LangChain way to attach memory
# to a chain. We use the older LLMChain approach here because it's simpler
# to understand for beginners — but the concept is identical.
from langchain.chains import LLMChain

load_dotenv()

MODEL    = os.getenv("MODEL", "phi3")
BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
SYSTEM   = os.getenv("SYSTEM_PROMPT", "You are a helpful assistant. Be concise and clear.")


# ── 1. The model ──────────────────────────────────────────────────────────────
# ChatOllama connects to your local Ollama server.
# streaming=True means invoke() streams tokens back instead of waiting
# for the full response — we'll consume them via a callback below.
llm = ChatOllama(
    model=MODEL,
    base_url=BASE_URL,
    temperature=float(os.getenv("TEMPERATURE", "0.7")),
    streaming=True,
)


# ── 2. The prompt template ────────────────────────────────────────────────────
# A ChatPromptTemplate is built from a list of message tuples.
# Each tuple is (role, content) where content can contain {slot} placeholders.
#
# The three parts here:
#   ("system", SYSTEM)         — sets the model's persona/behaviour
#   MessagesPlaceholder(...)   — where ConversationBufferMemory injects history
#   ("human", "{input}")       — the user's current message
#
# When the chain runs, it fills in {chat_history} and {input} automatically.
prompt = ChatPromptTemplate.from_messages([
    ("system", SYSTEM),
    MessagesPlaceholder(variable_name="chat_history"),  # ← memory goes here
    ("human", "{input}"),
])


# ── 3. The memory ─────────────────────────────────────────────────────────────
# ConversationBufferMemory stores every turn (user + assistant).
#
# memory_key="chat_history"  — the slot name in the prompt template above
# return_messages=True       — store as Message objects, not a text blob
#                              (required for MessagesPlaceholder)
memory = ConversationBufferMemory(
    memory_key="chat_history",
    return_messages=True,
)


# ── 4. The chain ──────────────────────────────────────────────────────────────
# LLMChain wires together: prompt → llm, and hooks memory in automatically.
# When you call chain.invoke({"input": "hello"}), it:
#   1. Loads history from memory
#   2. Fills the prompt template (system + history + user message)
#   3. Calls the LLM
#   4. Saves the reply back to memory
#   5. Returns the response
chain = LLMChain(
    llm=llm,
    prompt=prompt,
    memory=memory,
    verbose=False,   # set True to print every prompt to the terminal — very
)                    # educational when learning! try: VERBOSE=true in .env


# ── Public API (same shape as Phase 2) ───────────────────────────────────────

def stream_response(history: list[dict]):
    """
    Generator — yields string tokens as the model produces them.

    IMPORTANT: We ignore the `history` parameter here because LangChain's
    ConversationBufferMemory manages history internally. The parameter stays
    in the signature so app.py doesn't need to change.

    How streaming works with LangChain:
      LLMChain.invoke() waits for the full response before returning.
      To stream, we use the __astream__ / stream() method on the chain,
      which yields AIMessageChunk objects one token at a time.
    """
    # chain.stream() is LangChain's streaming interface.
    # Each chunk has a .text or .get("text") field with the token.
    for chunk in chain.stream({"input": "__get_last__"}):
        # chain.stream returns dict chunks: {"text": "token", ...}
        token = chunk.get("text", "")
        if token:
            yield token


def chat_once(user_message: str):
    """
    Send a message and stream the reply. Returns the full response string.
    This is the main function used by both app.py and the terminal CLI.
    """
    full_reply = ""
    for chunk in chain.stream({"input": user_message}):
        token = chunk.get("text", "")
        full_reply += token
        yield token


def get_memory_messages() -> list:
    """Return raw memory contents — useful for the /history debug command."""
    return memory.chat_memory.messages


def clear_memory():
    """Wipe the conversation history."""
    memory.clear()


def get_model() -> str:
    return MODEL    

    # replace the module-level memory and chain with this:
def build_chain():
    memory = ConversationBufferMemory(
        memory_key="chat_history",
        return_messages=True,
    )
    chain = LLMChain(
        llm=llm,
        prompt=prompt,
        memory=memory,
        verbose=False,
    )
    return chain
