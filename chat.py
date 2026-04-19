"""
chat.py — LLM communication module (no UI)
------------------------------------------
Learning goals:
  - Separate business logic from presentation layer
  - Use a generator function to yield streamed tokens
    so ANY frontend (terminal or Streamlit) can consume them

Both app.py (Streamlit) and the terminal loop import from here.
"""

import os
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

MODEL         = os.getenv("MODEL", "phi3")
BASE_URL      = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434/v1")
SYSTEM_PROMPT = os.getenv("SYSTEM_PROMPT", "You are a helpful assistant.")

# Single shared client — Ollama's OpenAI-compatible endpoint
client = OpenAI(base_url=BASE_URL, api_key="ollama")


def build_messages(history: list[dict]) -> list[dict]:
    """
    Prepend the system prompt to the conversation history.
    history is a list of {"role": "user"/"assistant", "content": "..."}
    """
    return [{"role": "system", "content": SYSTEM_PROMPT}] + history


def stream_response(history: list[dict]):
    """
    Generator — yields string tokens one at a time as the model produces them.

    Usage:
        for token in stream_response(history):
            print(token, end="", flush=True)

    Why a generator?
      Streamlit's st.write_stream() and the terminal loop both need
      to consume tokens incrementally. A generator lets each caller
      decide how to render them without duplicating the API call logic.
    """
    stream = client.chat.completions.create(
        model=MODEL,
        messages=build_messages(history),
        stream=True,
        temperature=float(os.getenv("TEMPERATURE", "0.7")),
    )
    for chunk in stream:
        token = chunk.choices[0].delta.content or ""
        if token:
            yield token


def get_model() -> str:
    return MODEL