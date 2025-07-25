"""
Commander AI - Entrypoint

- Loads config
- Initializes core agent
- Starts CLI
"""
from agent_core.agent import Agent
from ui.cli import start_cli

if __name__ == "__main__":
    agent = Agent()
    start_cli(agent)
