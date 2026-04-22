import sys
import warnings
import re
from pathlib import Path
from typing import Optional, Union, Any, Generator, Self
from datetime import datetime
import json
from PIL import Image

# Ensure project root is on sys.path for reliable imports
_PROJECT_ROOT = None
for _parent in Path(__file__).resolve().parents:
    if (_parent / "pyproject.toml").exists():
        _PROJECT_ROOT = _parent
        break
if _PROJECT_ROOT is None:
    _PROJECT_ROOT = Path.cwd()
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import torch
import numpy as np
import yaml
from smolagents import CodeAgent, TransformersModel, ActionStep, Model, LiteLLMModel, MCPClient, Tool
import litellm

from atomonous.utils.helpers import get_total_ram_gb
from atomonous.utils.memory import SessionMemory
from atomonous.agent.streamed_run import StreamedRun
from atomonous.agent.supervised_executor import SupervisedExecutor
from atomonous.config import settings
from atomonous.data.factory import ConverterFactory

class Agent:
    def __init__(self, model: Model, session_name: str = "", data_factory: Optional[ConverterFactory] = None):
        self.model = model
        self.mcp_clients = []

        # Initialize session memory
        self.memory = SessionMemory(
            artifacts_base_dir=settings.artifacts_dir,
            session_name=session_name
        )

        if data_factory is None:
            self.data_factory = ConverterFactory(register_default=True)
        else:
            self.data_factory = data_factory

        self.agent = CodeAgent(
            tools=[], 
            model=self.model,
            max_steps=settings.agent_max_steps,
            step_callbacks={ActionStep : self._process_step},
            executor=SupervisedExecutor(
                data_factory=self.data_factory,
                additional_authorized_imports=[
                    "atomonous.config.*", "numpy", "time", "os", "json", "yaml"
                ]
            ),
            instructions="""
            You are an expert scientific AI assistant powered by the Atomonous framework. 
            You can interact with the current context and instrumentation dynamically through the available MCP tools.

            Guidelines:
            1. Reliability and truthfulness are mandatory:
               - Never claim success unless tool outputs explicitly confirm success.
               - If tool output contains failure indicators, treat the task as failed.
               - For image capture tasks, verify an output artifact/path is returned.            
            """,
            stream_outputs=True
        )

        self._setup_executor_context()
    
    @property
    def tools(self) -> list[Tool]:
        return list(self.agent.tools.values())

    def disconnect_mcp_clients(self):
        """Disconnects all initialized MCP clients."""
        for client in self.mcp_clients:
            try:
                client.disconnect()
            except Exception:
                pass
        self.mcp_clients.clear()

    def __del__(self):
        self.disconnect_mcp_clients()

    def connect_mcp_client(self, server_parameters: dict[str, Any] | None = None, adapter_kwargs: Optional[dict] = None, structured_output: bool = False):
        """
        Connects an MCP client to the specified server parameters and adds its tools to the CodeAgent.
        `server_parameters` can be a dictionary mapping (e.g. {"url": "...", "transport": "streamable-http"})
        or a list of such dictionaries for connecting multiple servers.
        If `server_parameters` is None, it will default to connecting to the server specified in the `settings.mcp_url`.
        """
        if server_parameters is None:
            server_parameters = {"url": settings.mcp_url, "transport": "streamable-http"}

        try:
            client = MCPClient(
                server_parameters=server_parameters,
                adapter_kwargs=adapter_kwargs,
                structured_output=structured_output
            )
            self.mcp_clients.append(client)
            
            # Add all the fetched tools
            for tool in client.get_tools():
                if tool.name not in self.agent.tools.keys():
                    self.agent.tools[tool.name] = tool
                else:
                    warnings.warn(f"Tool name conflict: '{tool.name}' already exists in the agent's tools. Skipping this tool from MCP client.")
        except ModuleNotFoundError:
            warnings.warn("Failed to initialize MCPClient. Ensure `smolagents[mcp]` is installed.")
            
    @classmethod
    def from_model_id(cls, model_id: str = "Auto", session_name: str = "", data_factory: Optional[ConverterFactory] = None) -> Self:
        model_size_b = 0
        try:
            size_match = re.search(r'(\d+\.?\d*)B', model_id)
            if size_match:
                model_size_b = float(size_match.group(1))
        except:
            pass
        
        if model_size_b < 3:
            max_tokens = 512
            temperature = 0.4
            top_p = 0.85
            rep_penalty = 1.15
        elif model_size_b <= 14:
            max_tokens = 1024
            temperature = 0.6
            top_p = 0.9
            rep_penalty = 1.1
        else:
            max_tokens = 1536
            temperature = 0.7
            top_p = 0.95
            rep_penalty = 1.05

        gen_params = {
            "max_new_tokens": max_tokens,
            "temperature": temperature,
            "top_p": top_p,
            "repetition_penalty": rep_penalty,
        }

        model = TransformersModel(
            model_id=model_id,
            max_new_tokens=max_tokens,
            device_map="mps" if torch.backends.mps.is_available() else "auto",
            torch_dtype=torch.bfloat16,
            trust_remote_code=True,
            temperature=temperature,
            top_p=top_p,
            repetition_penalty=rep_penalty,
            model_kwargs={
                "low_cpu_mem_usage": True,
                "use_cache": True,
                "load_in_4bit": True,
            }
        )
        instance = cls(model=model, session_name=session_name, data_factory=data_factory)
        instance.gen_params = gen_params
        return instance

    @classmethod
    def from_api_key(cls, model_id: str, api_base: str, api_key: str, session_name: str = "", data_factory: Optional[ConverterFactory] = None) -> Self:
        model = LiteLLMModel(model_id=model_id, api_base=api_base, api_key=api_key)
        model.flatten_messages_as_text = not litellm.supports_vision(model_id)
        return cls(model=model, session_name=session_name, data_factory=data_factory)

    def _setup_executor_context(self):
        try:
            context = {
                "np": np,
                "settings": settings,
                "yaml": yaml,
            }
            if self.data_factory:
                context["data_factory"] = self.data_factory
            self.agent.python_executor.send_variables(context)
        except Exception:
            pass

    def chat(self, query: str, stream: bool = False) -> str | Generator:
        """Queries the LLM. If stream is True, returns a Generator."""
        sr = StreamedRun(lambda: self.agent.run(query, stream=True))
        if stream == True:
            return sr.stream()
        return str(sr.final().output)

    def _process_step(self, step: ActionStep, agent: CodeAgent):
        if self.data_factory is None: return

        # Prune older images from memory to keep context lean
        for prev_step in agent.memory.steps:
            if isinstance(prev_step, ActionStep) and prev_step.step_number <= step.step_number - 2:
                prev_step.observations_images = None

        # Get tool-intercepted artifacts
        has_new_images = False
        if hasattr(agent.python_executor, "intercepted_artifacts"):
            for artifact in agent.python_executor.intercepted_artifacts:
                # flatten_messages_as_text is a flag that indicates whether the model supports vision
                if isinstance(artifact, Image.Image) and not agent.model.flatten_messages_as_text:
                    step.observations_images = (step.observations_images or []) + [artifact]
                    has_new_images = True
            
            agent.python_executor.intercepted_artifacts = []

        if has_new_images:
            step.observations = (step.observations or "") + "\n\n[System: Large-scale data processed into vision artifacts.]"

        full_output: str = agent.python_executor.last_output.logs if hasattr(agent.python_executor, "last_output") else ""

        step_data = {
            "step_number": step.step_number,
            "model_output": step.model_output,
            "observations": step.observations,
            "action_output": str(step.action_output) if step.action_output else None,
            "code_action": step.code_action,
            "full_output": full_output,
        }
        
        step_file = self.memory.session_dir / f"step_{step.step_number}.json"
        with open(step_file, "w") as f:
            json.dump(step_data, f, indent=2)