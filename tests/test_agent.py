import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from app.config import settings
settings.hf_cache_dir = "/lustre/isaac24/scratch/dpelaia/hf_cache/"

from app.agent.core import Agent
from app.utils.server_cli import show_interactive_server_commands

def main():
    show_interactive_server_commands()

    # agent = Agent(model_id="Qwen/Qwen2.5-Coder-32B-Instruct")

    # prompt = """
    # Take an image and close the microscope. The servers are already running and connected.
    # """

    # response = agent.chat(prompt)

if __name__ == "__main__":
    main()