"""
CLI interface for Commander AI
"""

from prompt_toolkit import prompt
from models.llm import TASK_END_TOKEN
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.prompt import Prompt
from rich.spinner import Spinner
from rich.align import Align
from rich.text import Text
from rich import box

import os
import glob
import shutil

console = Console()

def main_menu():
    console.clear()
    menu_text = ("\n[bold cyan]Commander AI Main Menu[/bold cyan]\n\n"
                 "[bold green]1.[/bold green] New chat/session\n"
                 "[bold green]2.[/bold green] List chats\n"
                 "[bold green]3.[/bold green] Load previous chat\n"
                 "[bold green]4.[/bold green] Delete a chat\n"
                 "[bold green]5.[/bold green] Delete all cache/history\n"
                 "[bold red]0.[/bold red] Exit\n")
    menu_panel = Panel(
        Align.center(Text.from_markup(menu_text, justify="center"), vertical="middle"),
        title="[bold magenta]Welcome to Commander AI[/bold magenta]",
        border_style="bright_blue",
        box=box.ROUNDED,
        padding=(1, 4)
    )
    console.print(menu_panel)
    return Prompt.ask("[bold yellow]Select an option[/bold yellow]", choices=["1","2","3","4","5","6","0"], default="1")

def start_cli(agent):
    while True:
        choice = main_menu()
        if choice == '1':
            console.clear()
            console.print(Panel("[bold green]New chat started![/bold green]", border_style="green"))
            run_agent_cli(agent, new_session=True)
        elif choice == '2':
            console.clear()
            list_chats()
        elif choice == '3':
            console.clear()
            load_chat()
        elif choice == '4':
            console.clear()
            delete_chat()
        elif choice == '5':
            console.clear()
            delete_all_cache()
        elif choice == '0':
            console.clear()
            console.print(Panel("[bold red]Goodbye![/bold red]", border_style="red"))
            break
        else:
            console.print(Panel("[bold yellow]Invalid option. Try again.[/bold yellow]", border_style="yellow"))

def run_agent_cli(agent, new_session=False):
    console.print(Panel("[bold magenta]Type 'exit' to return to menu.[/bold magenta]", border_style="magenta"))
    chat_history = []
    session_file = None
    import os, datetime
    chat_dir = os.path.join(os.path.dirname(__file__), '../cache/chats')
    os.makedirs(chat_dir, exist_ok=True)
    session_file = os.path.join(chat_dir, f'session_{datetime.datetime.now().strftime("%Y%m%d_%H%M%S")}.json')
    while True:
        user_input = Prompt.ask("[bold blue]You[/bold blue]")
        if user_input.strip().lower() == 'exit':
            break
        # Step 1: Planning
        model = "OpenAI GPT-4" if agent.llm.use_api and agent.llm.api_key else "Ollama (local LLM)"
        with console.status(f"[bold yellow]Planning next action using {model}...[/bold yellow]", spinner="dots"):
            plan = agent.llm.plan(user_input, chat_history)
        message = plan.get('message')
        if message:
            console.print(f"[bold cyan]AI:[/bold cyan] {message}")
            chat_history.append({"role": "assistant", "content": message})
        tool = plan.get('tool', '?')
        desc = plan.get('ui', {}).get('description', '')
        # Show tool execution or AI response
        if tool == 'inquiry':
            result = agent.execute_plan(plan, user_input)
            if isinstance(result, dict) and result.get('__type') == 'inquiry':
                console.print(f"[bold cyan]AI:[/bold cyan] {result['text']}")
                user_response = Prompt.ask("[bold blue]Your response[/bold blue]")
                chat_history.append({"role": "assistant", "content": result['text']})
                chat_history.append({"role": "user", "content": user_response})
                plan = agent.llm.plan(user_response)
                result = agent.execute_plan(plan, user_response)
                chat_history.append({"role": "llm_plan", "content": str(plan)})
                chat_history.append({"role": "tool", "content": str(result)})
                continue
            else:
                console.print(f"[bold cyan]AI:[/bold cyan] {result}")
        elif tool != 'none' and tool != '?':
            display = desc or f"Executing {tool}"
            console.print(f"[bold yellow]{display}[/bold yellow]")
            with console.status(f"[bold green]{display}[/bold green]", spinner="bouncingBar"):
                result = agent.execute_plan(plan, user_input)
            console.print(f"[bold green]Done:[/bold green] [bold cyan]{tool}[/bold cyan]")
        else:
            with console.status("[bold green]Processing...[/bold green]", spinner="bouncingBar"):
                result = agent.execute_plan(plan, user_input)
        # Step 3: Agentic follow-up (if needed)
        steps = 0
        max_steps = 20
        while agent._should_continue(plan, result) and steps < max_steps:
            steps += 1
            with console.status(f"[bold cyan]Reasoning next step...[/bold cyan]", spinner="bouncingBall"):
                followup_prompt = agent._agentic_followup_prompt(user_input, plan, result)
                followup_plan = agent._robust_parse_plan(agent.llm._plan_with_api(followup_prompt, chat_history))
                chat_history.append({"role": "llm_followup_plan", "content": str(followup_plan)})
            
            followup_message = followup_plan.get('message')
            if followup_message:
                console.print(f"[bold cyan]AI:[/bold cyan] {followup_message}")
                chat_history.append({"role": "assistant", "content": followup_message})
            followup_tool = followup_plan.get('tool', '?')
            followup_desc = followup_plan.get('ui', {}).get('description', '')
            if followup_tool == 'inquiry':
                followup_result = agent.execute_plan(followup_plan, user_input)
                if isinstance(followup_result, dict) and followup_result.get('__type') == 'inquiry':
                    console.print(f"[bold cyan]AI:[/bold cyan] {followup_result['text']}")
                    user_response = Prompt.ask("[bold blue]Your response[/bold blue]")
                    chat_history.append({"role": "assistant", "content": followup_result['text']})
                    chat_history.append({"role": "user", "content": user_response})
                    plan = agent.llm.plan(user_response)
                    result = agent.execute_plan(plan, user_response)
                    chat_history.append({"role": "llm_plan", "content": str(plan)})
                    chat_history.append({"role": "tool", "content": str(result)})
                    if plan.get('message'):
                        console.print(f"[bold cyan]AI:[/bold cyan] {plan['message']}")
                        chat_history.append({"role": "assistant", "content": plan['message']})
                    # Update the plan and result for the outer loop
                    plan = followup_plan
                    result = followup_result
                else:
                    console.print(f"[bold cyan]AI:[/bold cyan] {followup_result}")
            elif followup_tool != 'none' and followup_tool != '?':
                display = followup_desc or f"Executing {followup_tool}"
                console.print(f"[bold yellow]{display}[/bold yellow]")
                with console.status(f"[bold green]{display}[/bold green]", spinner="bouncingBar"):
                    followup_result = agent.execute_plan(followup_plan, user_input)
                chat_history.append({"role": "tool", "content": str(followup_result)})
                console.print(f"[bold green]Done:[/bold green] [bold cyan]{followup_tool}[/bold cyan]")
            else:
                with console.status("[bold green]Processing...[/bold green]", spinner="bouncingBar"):
                    followup_result = agent.execute_plan(followup_plan, user_input)
                chat_history.append({"role": "tool", "content": str(followup_result)})
            
            plan, result = followup_plan, followup_result
        
        # Final message is already printed when received
        # Save chat history after each turn
        chat_history.append({"role": "user", "content": user_input})
        chat_history.append({"role": "llm_plan", "content": str(plan)})
        chat_history.append({"role": "tool", "content": str(result)})
        # Actively save after each turn
        if session_file:
            import json, os
            with open(session_file, 'w') as f:
                json.dump(chat_history, f, indent=2)
                f.flush()
                os.fsync(f.fileno())

def list_chats():
    chat_dir = os.path.join(os.path.dirname(__file__), '../cache/chats')
    if not os.path.exists(chat_dir):
        console.print(Panel("[bold yellow]No chats found.[/bold yellow]", border_style="yellow"))
        return
    chats = sorted(glob.glob(os.path.join(chat_dir, 'session_*.json')))
    if not chats:
        console.print(Panel("[bold yellow]No chats found.[/bold yellow]", border_style="yellow"))
        return
    table = Table(title="Available Chats", box=box.SIMPLE, border_style="cyan")
    table.add_column("#", style="bold green", width=4)
    table.add_column("Session File", style="white")
    for i, chat in enumerate(chats):
        table.add_row(str(i+1), os.path.basename(chat))
    console.print(table)

def load_chat():
    chat_dir = os.path.join(os.path.dirname(__file__), '../cache/chats')
    chats = sorted(glob.glob(os.path.join(chat_dir, 'session_*.json')))
    if not chats:
        console.print(Panel("[bold yellow]No chats to load.[/bold yellow]", border_style="yellow"))
        return
    list_chats()
    idx = Prompt.ask("[bold blue]Enter chat number to view[/bold blue]")
    try:
        idx = int(idx) - 1
        if 0 <= idx < len(chats):
            with open(chats[idx], 'r') as f:
                import json
                history = json.load(f)
            chat_panel = Panel.fit("\n".join([
                f"[bold magenta]{entry.get('role','agent').capitalize()}[/bold magenta]: [white]{clean_output(entry.get('content',''))}[/white]"
                for entry in history
            ]), title="[bold cyan]Chat History[/bold cyan]", border_style="cyan", padding=(1,2))
            console.print(chat_panel)
        else:
            console.print(Panel("[bold red]Invalid chat number.[/bold red]", border_style="red"))
    except Exception:
        console.print(Panel("[bold red]Invalid input.[/bold red]", border_style="red"))

def delete_chat():
    chat_dir = os.path.join(os.path.dirname(__file__), '../cache/chats')
    chats = sorted(glob.glob(os.path.join(chat_dir, 'session_*.json')))
    if not chats:
        console.print(Panel("[bold yellow]No chats to delete.[/bold yellow]", border_style="yellow"))
        return
    list_chats()
    idx = Prompt.ask("[bold red]Enter chat number to delete[/bold red]")
    try:
        idx = int(idx) - 1
        if 0 <= idx < len(chats):
            os.remove(chats[idx])
            console.print(Panel("[bold green]Chat deleted.[/bold green]", border_style="green"))
        else:
            console.print(Panel("[bold red]Invalid chat number.[/bold red]", border_style="red"))
    except Exception:
        console.print(Panel("[bold red]Invalid input.[/bold red]", border_style="red"))

def delete_all_cache():
    cache_dir = os.path.join(os.path.dirname(__file__), '../cache')
    confirm = Prompt.ask("[bold red]Are you sure you want to delete ALL cache and chat history? (y/n)[/bold red]", choices=["y","n"], default="n")
    if confirm == 'y':
        try:
            shutil.rmtree(cache_dir)
            os.makedirs(os.path.join(cache_dir, 'chats'), exist_ok=True)
            console.print(Panel("[bold green]All cache and chat history deleted.[/bold green]", border_style="green"))
        except Exception as e:
            console.print(Panel(f"[bold red]Error deleting cache: {e}[/bold red]", border_style="red"))
    else:
        console.print(Panel("[bold yellow]Cancelled.[/bold yellow]", border_style="yellow"))

def clean_output(text):
    import re
    # Remove debug, bracketed, and prompt lines
    if not text:
        return ''
    # Remove lines like [DEBUG] ... or [Agent follow-up] ...
    lines = text.splitlines()
    lines = [l for l in lines if not l.strip().startswith('[')]
    # Remove prompt echoes
    lines = [l for l in lines if not l.strip().lower().startswith('prompt:')]
    # Remove empty lines
    lines = [l for l in lines if l.strip()]
    # Remove task end token
    lines = [l.replace(TASK_END_TOKEN, '').strip() for l in lines]
    # Remove excessive whitespace
    return '\n'.join(lines)
