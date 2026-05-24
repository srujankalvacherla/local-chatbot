"""
chat.py — Updated for LangChain 1.x + LangGraph persistence
=============================================================
Replaced RunnableWithMessageHistory (deprecated) with LangGraph's
built-in MemorySaver for conversation history.
"""

import os
from dotenv import load_dotenv
from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage, AIMessage
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import START, MessagesState, StateGraph

load_dotenv()

MODEL    = os.getenv("MODEL", "phi3")
BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
SYSTEM   = os.getenv("SYSTEM_PROMPT", "You are a helpful assistant. Be concise and clear.")

THREAD_ID = "default"

llm = ChatOllama(
    model=MODEL,
    base_url=BASE_URL,
    temperature=float(os.getenv("TEMPERATURE", "0.7")),
    streaming=True,
)

prompt = ChatPromptTemplate.from_messages([
    ("system", SYSTEM),
    MessagesPlaceholder(variable_name="messages"),
])

# ── LangGraph setup ───────────────────────────────────────────────────────────

def call_model(state: MessagesState):
    chain = prompt | llm
    response = chain.invoke(state)
    return {"messages": response}

memory = MemorySaver()
graph = StateGraph(state_schema=MessagesState)
graph.add_edge(START, "model")
graph.add_node("model", call_model)
app = graph.compile(checkpointer=memory)

_config = {"configurable": {"thread_id": THREAD_ID}}


# ── Public API (unchanged for app.py and terminal.py) ────────────────────────

def chat_once(user_message: str):
    """Generator — yields tokens. Saves to memory automatically."""
    input_messages = [HumanMessage(user_message)]
    for chunk, _ in app.stream(
        {"messages": input_messages},
        config=_config,
        stream_mode="messages",
    ):
        token = chunk.content if hasattr(chunk, "content") else str(chunk)
        if token:
            yield token


def get_memory_messages() -> list:
    """Return full conversation history as LangChain message objects."""
    state = app.get_state(_config)
    return state.values.get("messages", []) if state else []


def clear_memory():
    """Clear conversation by resetting to a fresh thread."""
    global _config
    import uuid
    _config = {"configurable": {"thread_id": str(uuid.uuid4())}}


def get_model() -> str:
    return MODEL