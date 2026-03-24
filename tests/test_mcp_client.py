from smolagents import MCPClient
import numpy as np
import json
import base64
import matplotlib.pyplot as plt
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

def basic_connection_test():
    try:
        mcp_client = MCPClient({"url": "http://localhost:8000/mcp", "transport": "streamable-http"})
        tools = mcp_client.get_tools()
        for tool in tools:
            print(tool.name)
            print(tool.inputs)
            print()

        print("6th")
        print(tools[8].name)
        get_scanned_image = tools[8]
        data_str = get_scanned_image.__call__()
        print(type(data_str))
        # data = json.dumps(data)

        dictionary_data = json.loads(data_str)
        print(type(dictionary_data))

        meta = json.loads(dictionary_data["metadata"])
        raw_bytes = dictionary_data["payload"]
        encoded_data = raw_bytes.encode('utf-8')
        decoded_bytes = base64.b64decode(encoded_data)
        image = np.frombuffer(decoded_bytes, dtype=meta["dtype"]).reshape(meta["shape"])

        print(image.shape)

        # Display the array as an image
        plt.imshow(image, cmap="gray")
        plt.axis('off') # Optional: removes axis labels and ticks

        # To save the figure as an image file
        plt.savefig('haadf.png')

        get_spectrum = tools[9]
        res = get_spectrum.__call__(detector_name="eds")
        print(res)

        dictionary_data = json.loads(res)
        meta = json.loads(dictionary_data["metadata"])
        raw_bytes = dictionary_data["payload"]
        encoded_data = raw_bytes.encode('utf-8')
        decoded = base64.b64decode(encoded_data).decode()
        spectrum = json.loads(decoded)

        elements = list(spectrum.keys())
        values = list(spectrum.values())

        plt.figure()
        plt.bar(elements, values)

        plt.xlabel("Elements")
        plt.ylabel("Intensity (relative)")
        plt.title("EDS Elemental Composition")

        plt.savefig('spectrum.png')
    finally:
        mcp_client.disconnect()

from app.config import settings
from app.agent.core import Agent
from dotenv import load_dotenv

load_dotenv()

agent = Agent.from_api_key(
    model_id="openai/meta-llama/llama-3.3-70b-instruct",  # or any Novita model slug
    api_base="https://api.novita.ai/v3/openai",
    api_key=os.environ.get("NOVITA_API_KEY")
)

print("Tools loaded from MCP:")
for tool in agent.agent.tools.values():
    print(f"- {tool.name}")

prompt = input("Enter a command: ")
response = agent.chat(prompt)
