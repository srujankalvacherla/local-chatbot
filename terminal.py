"""
terminal.py — Phase 3: terminal interface (optional, for learning)
==================================================================
Same experience as Phase 1's chat.py but now powered by LangChain.
Run this side-by-side with app.py to see both UIs sharing the same
chat.py module.

Run:
  python terminal.py
"""

import os
from dotenv import load_dotenv
from rich.console import Console
from rich.rule import Rule
from chat import chat_once, clear_memory, get_memory_messages, get_model

load_dotenv()
console = Console()


def print_welcome():
    console.print(Rule(style="dim"))
    console.print(f"[bold]Local Chatbot · Phase 3 · LangChain[/bold]  [dim]model: {get_model()}[/dim]")
    console.print("[dim]Commands: /history  /clear  /quit[/dim]")
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

        if user_input == "/quit":
            console.print("[dim]Bye![/dim]")
            break
        elif user_input == "/clear":
            clear_memory()
            console.print("[dim]Memory cleared.[/dim]")
            continue
        elif user_input == "/history":
            console.print(Rule("LangChain memory contents", style="dim"))
            for i, msg in enumerate(get_memory_messages()):
                role = type(msg).__name__.replace("Message", "")
                color = "green" if role == "Human" else "cyan"
                console.print(f"[{color}][{i}] {role}[/{color}]  {msg.content[:200]}")
            console.print(Rule(style="dim"))
            continue

        try:
            console.print("\n[bold cyan]Assistant:[/bold cyan] ", end="")
            for token in chat_once(user_input):
                print(token, end="", flush=True)
            print()
        except Exception as e:
            console.print(f"\n[red]Error:[/red] {e}")
            console.print("[dim]Is Ollama running? Try: ollama serve[/dim]")


if __name__ == "__main__":
    main()