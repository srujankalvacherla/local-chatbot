"""
Phase 1: Local chatbot — terminal interface via Ollama
-------------------------------------------------------
Learning goals:
  - Call a local LLM using the OpenAI-compatible API that Ollama exposes
  - Maintain a conversation history (list of role/content dicts)
  - Stream tokens so responses feel live
  - Understand how system prompts work

Run:
  python chat.py
"""

import os
from dotenv import load_dotenv
from openai import OpenAI
from rich.console import Console
from rich.markdown import Markdown
from rich.rule import Rule

# ── Load config from .env ─────────────────────────────────────────────────────
load_dotenv()

MODEL          = os.getenv("MODEL", "phi3")
BASE_URL       = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434/v1")
SYSTEM_PROMPT  = os.getenv("SYSTEM_PROMPT", "You are a helpful assistant.")

# ── Set up client ─────────────────────────────────────────────────────────────
# Ollama exposes an OpenAI-compatible HTTP API, so we point the OpenAI client
# at localhost instead of api.openai.com. api_key is required by the client
# library but ignored by Ollama — "ollama" is just a placeholder.
client  = OpenAI(base_url=BASE_URL, api_key="ollama")
console = Console()


# ── Conversation history ──────────────────────────────────────────────────────
# This is the core of multi-turn chat. Every message — user and assistant —
# is appended here and sent back to the model on the next turn so it has
# full context of the conversation.
history: list[dict] = [
    {"role": "system", "content": SYSTEM_PROMPT}
]


def chat(user_message: str) -> str:
    """
    Send a user message, stream the response, and return the full reply.
    
    Key concept: we append the user message BEFORE calling the API,
    then append the assistant reply AFTER we've collected the full stream.
    """
    history.append({"role": "user", "content": user_message})

    # Stream the response — tokens arrive one by one as the model generates them
    stream = client.chat.completions.create(
        model=MODEL,
        messages=history,
        stream=True,          # <-- this is what makes it feel "live"
        temperature=0.7,      # 0 = deterministic, 1 = creative, 0.7 is a good default
    )

    # Collect streamed tokens into a single string while printing each chunk
    console.print("\n[bold cyan]Assistant:[/bold cyan] ", end="")
    full_reply = ""
    for chunk in stream:
        delta = chunk.choices[0].delta.content or ""
        print(delta, end="", flush=True)
        full_reply += delta

    print()  # newline after streaming finishes

    # Add the complete reply to history so the model remembers it next turn
    history.append({"role": "assistant", "content": full_reply})
    return full_reply


def print_welcome():
    console.print(Rule(style="dim"))
    console.print(f"[bold]Local Chatbot — Phase 1[/bold]  [dim]model: {MODEL}[/dim]")
    console.print("[dim]Commands:  /history  /clear  /quit[/dim]")
    console.print(Rule(style="dim"))


def print_history():
    """Print the raw conversation history — useful for learning."""
    console.print(Rule("conversation history", style="dim"))
    for i, msg in enumerate(history):
        role  = msg["role"].upper()
        color = {"system": "yellow", "user": "green", "assistant": "cyan"}.get(msg["role"], "white")
        console.print(f"[{color}][{i}] {role}[/{color}]  {msg['content'][:200]}")
    console.print(Rule(style="dim"))


def main():
    print_welcome()

    while True:
        try:
            user_input = console.input("\n[bold green]You:[/bold green] ").strip()
        except (KeyboardInterrupt, EOFError):
            console.print("\n[dim]Bye![/dim]")
            break

        if not user_input:
            continue

        # Built-in commands
        if user_input == "/quit":
            console.print("[dim]Bye![/dim]")
            break
        elif user_input == "/clear":
            history.clear()
            history.append({"role": "system", "content": SYSTEM_PROMPT})
            console.print("[dim]History cleared.[/dim]")
            continue
        elif user_input == "/history":
            print_history()
            continue

        # Send message to the model
        try:
            chat(user_input)
        except Exception as e:
            console.print(f"\n[red]Error:[/red] {e}")
            console.print("[dim]Is Ollama running? Try: ollama serve[/dim]")


if __name__ == "__main__":
    main()