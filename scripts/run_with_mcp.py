from smolagents import MCPClient
import numpy as np
import json
import base64
import matplotlib.pyplot as plt
import sys
import os
from atomonous.config import settings
from atomonous.agent.core import Agent
from dotenv import load_dotenv

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from atomonous.config import settings
from atomonous.agent.core import Agent

load_dotenv()

agent = Agent.from_api_key(
    model_id="openai/google/gemma-4-31b-it",  # or any Novita model slug
    api_base="https://api.novita.ai/v3/openai",
    api_key=os.environ.get("NOVITA_API_KEY")
)

# Connect to the default MCP client configured in settings
agent.connect_mcp_client()

print("Tools loaded from MCP:")
for tool in agent.tools:
    print(f"- {tool.name}")

prompt = input("Enter a command: ")
response = agent.chat(prompt)
