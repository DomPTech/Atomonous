import sys
from pathlib import Path

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
from smolagents import CodeAgent, TransformersModel, DuckDuckGoSearchTool
from smolagents.models import ChatMessageStreamDelta
from app.tools.microscopy import TOOLS, MicroscopeServer
from app.utils.helpers import get_total_ram_gb
from app.agent.supervised_executor import SupervisedExecutor
try:
    import pyTEMlib.probe_tools as pt
except ImportError:
    pt = None
import numpy as np
from app.tools import microscopy
from app.config import settings

class MicroscopeClientProxy:
    """Proxy to forward calls to the active global CLIENT instance."""
    def __getattr__(self, name):
        if microscopy.CLIENT is None:
             raise RuntimeError("Microscope client is not connected. Please call 'connect_client()' first.")
        return getattr(microscopy.CLIENT, name)

class Agent:
    def __init__(self, model_id: str = "Auto"):
        ram_gb = get_total_ram_gb()
        load_in_8bit = False
        low_cpu_mem_usage = True

        # Auto-select model based on available RAM
        if model_id == "Auto" or not model_id:
            if ram_gb < 16:
                model_id = "Qwen/Qwen2.5-0.5B-Instruct"
                # bitsandbytes 8-bit isn't stable on MPS yet
                load_in_8bit = False if torch.backends.mps.is_available() else True
            elif ram_gb > 48:
                # 14B fits comfortably under 50GB (approx 28GB in FP16)
                model_id = "Qwen/Qwen2.5-14B-Instruct" 
            else:
                model_id = "Qwen/Qwen2.5-7B-Instruct" # ~15GB

        if ram_gb > 16:
            low_cpu_mem_usage = False

        self.model = TransformersModel(
            model_id=model_id,
            max_new_tokens=1024,
            device_map="mps" if torch.backends.mps.is_available() else "auto",
            torch_dtype=torch.bfloat16,
            trust_remote_code=True,
            model_kwargs={
                "low_cpu_mem_usage": low_cpu_mem_usage,
                "use_cache": True,
                "load_in_8bit": load_in_8bit,
            }
        )
        # Full tool suite for the microscopy agent
        self.agent = CodeAgent(
            tools=TOOLS, 
            model=self.model, 
            executor=SupervisedExecutor(additional_authorized_imports=[
                "app.tools.microscopy", "app.config.*", 
                "numpy", "time", "os", "scipy", "matplotlib", "skimage"
            ]),
            instructions="""
            You are an expert microscopy AI assistant. 
            You can control the microscope by starting the server, connecting the client, and then using tools to:
            - Adjust magnification and capture images.
            - Move the stage and check status.
            - Control the electron beam (blank/unblank, place beam).
            - Calibrate and set screen current.
            - And more.

            You can also design structuered Experiments using submit_experiment.
            
            Default context & assumptions (use these unless the user specifies otherwise):
            - Always start servers and connect to the client when asked to do anything on the microscope.
            - If no server list is provided, start ALL servers: MicroscopeServer.Central, MicroscopeServer.AS, MicroscopeServer.Ceos.
            - After starting servers, wait 1 second, then connect the client using settings.
            - Use mode='mock' unless the user explicitly requests real hardware.
            
            Guidelines:
            1. Use 'settings' for configuration:
               - Use 'settings.server_host' and 'settings.server_port' for connections.
               - Use 'settings.autoscript_path' if needed for server startup.
            2. Reliability:
               - Always wait at least 1 second (`time.sleep(1)`) after starting servers before attempting to connect the client.
               - Use mode='mock' for simulations unless 'real' is explicitly requested.
            3. Housekeeping:
               - Always call 'close_microscope()' when the task is finished.
            4. Decide whether or not to construct structured Experiments or just execute tools quickly.
            
            Available servers: MicroscopeServer.Central, MicroscopeServer.AS, MicroscopeServer.Ceos.

            'settings' and the 'MicroscopeServer' Enum are pre-imported and available for use in your code execution environment.
            """,
            stream_outputs=True
        )

        # Inject agent instance for workflow execution
        microscopy.AGENT_INSTANCE = self.agent

        # Preload common classes into the Python executor context
        try:
            self.agent.python_executor.send_variables({
                "MicroscopeServer": MicroscopeServer,
                "tem": MicroscopeClientProxy(),
                "pt": pt,
                "np": np,
                "settings": settings,
            })
        except Exception:
            # Non-fatal: some executors may not support variable injection
            pass

    def chat(self, query: str) -> str:
        """
        Process user input and return a response.
        """
        import os
        import re

        def _generate_until_success(prompt_text: str) -> str:
            current_prompt = prompt_text
            while True:
                print("\\nAgent is working on the workflow...")
                output = str(self.agent.run(current_prompt)).strip()
                print(f"\\nAgent Output:\\n{output}\\n")
                
                # Extract path ending in .yaml avoiding surrounding punctuation
                match = re.search(r'([/\\w\\.-]+\\.yaml)', output)
                if match:
                    path = match.group(1).strip()
                    if os.path.exists(path):
                        return path
                    else:
                        print(f"Warning: Discovered path '{path}' does not exist on disk. Forcing retry.")
                        current_prompt = f"The path you provided ({path}) does not exist. Did you successfully run `design_workflow`? Please try again and provide the correct absolute path."
                else:
                    print("Warning: Could not extract a valid .yaml path from output. Forcing retry.")
                    current_prompt = "You did not provide a valid .yaml file path. You MUST use the `design_workflow` tool and output the resulting absolute path."

        init_prompt = (
            f"Please design a workflow for the following experimental task:\\n{query}\\n\\n"
            "You MUST use the `design_workflow` tool to define and save this workflow. Provide the absolute path of the saved yaml file as your final answer."
        )
        
        parsed_yaml_path = _generate_until_success(init_prompt)
        
        while True:
            print(f"\\nProposed Workflow YAML: {parsed_yaml_path}")
            print("Options:")
            print("1. Accept workflow and execute")
            print("2. Modify workflow")
            print("3. Reject workflow and stop")
            choice = input("Enter your choice (1/2/3): ").strip()
            
            if choice == '1':
                break
            elif choice == '2':
                mod_query = input("Enter your modifications: ")
                mod_prompt = f"Please modify the previously designed workflow as follows: {mod_query}\\nUse `design_workflow` to save it and return the updated absolute path."
                parsed_yaml_path = _generate_until_success(mod_prompt)
            elif choice == '3':
                return "Execution canceled by user."
            else:
                print("Invalid choice.")
                
        # Execute workflow
        from app.tools.workflow_framework import WorkflowTemplate, WorkflowExecutor
        from app.tools.microscopy import NODE_REGISTRY
        import yaml
        
        try:
            with open(parsed_yaml_path, 'r') as f:
                parsed_template_yaml = yaml.safe_load(f)
            template = WorkflowTemplate(**parsed_template_yaml)
            executor = WorkflowExecutor(template, NODE_REGISTRY)
            
            print(f"\\n--- Initiating Workflow: {template.name} ---\\n")
            
            # Use context to pass the LLM in so CodeNodes can wake the Agent up.
            final_state = executor.run(context={"agent": self.agent})
            
            print("\\nAgent is generating a summary of the execution...")
            summary_prompt = (
                f"The workflow '{template.name}' has finished executing.\\n"
                f"State Data: {final_state.data}\\n"
                f"History: {final_state.history}\\n"
                f"Errors: {final_state.errors}\\n"
                f"Metrics: {final_state.metrics}\\n"
                "Please provide a brief, user-friendly summary of what was accomplished."
            )
            summary = str(self.agent.run(summary_prompt)).strip()
            
            return f"Workflow {template.name} execution finished.\\n\\nSummary:\\n{summary}\\n\\nHistory: {final_state.history}\\nErrors: {final_state.errors}\\nMetrics: {final_state.metrics}"
        except Exception as e:
            return f"Failed to execute workflow: {e}"

    def stream_chat(self, query: str):
        """
        Stream user input processing as a sequence of events.
        Yields dicts with keys: type ("delta"|"final") and content.
        This wraps the chat logic into simplified delta streams for web interfaces, though it does not yet support interactive input gracefully.
        """
        yield {"type": "delta", "content": "Running interactive workflow loop (Not fully supported in pure streams yet)\\n"}
        res = self.chat(query)
        yield {"type": "final", "content": res}