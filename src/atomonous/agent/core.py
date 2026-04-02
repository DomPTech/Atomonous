import sys
from pathlib import Path
from typing import Callable, Optional, Union, Any, Dict, List
import yaml
import numpy as np
from datetime import datetime
import json
import base64

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
from smolagents import CodeAgent, TransformersModel, ActionStep, Model, LiteLLMModel, MCPClient
from smolagents.models import ChatMessageStreamDelta, ChatMessage

from atomonous.tools import workflows
from atomonous.tools.workflows import get_workflow_tools, get_default_registry
from atomonous.tools.workflow_framework import WorkflowTemplate, WorkflowExecutor
from atomonous.utils.helpers import get_total_ram_gb
from atomonous.utils.memory import SessionMemory
from atomonous.agent.supervised_executor import SupervisedExecutor
from atomonous.config import settings


class WorkflowCreated(Exception):
    """Raised when a workflow is successfully created and should trigger approval flow."""
    pass


def _safe_to_string(value: object) -> str:
    try:
        return str(value)
    except Exception as exc:
        return f"<unprintable {type(value).__name__}: {exc}>"

def _should_generate_workflow(query: str) -> bool:
    """
    Returns True for complex, multi-step experimental requests, false otherwise.
    """
    import re

    q = (query or "").strip().lower()
    if not q:
        return False

    # Explicit user preference
    if any(phrase in q for phrase in [
        "no workflow",
        "without workflow",
        "don't use workflow",
        "do not use workflow",
        "skip workflow",
    ]):
        return False

    if "workflow" in q:
        return True

    # Strong indicators of larger experiment
    complex_markers = [
        "experiment",
        "protocol",
        "multi-step",
        "multistep",
        "optimiz",
        "calibration",
        "sweep",
        "for each",
        "iterate",
        "loop",
        "parameter scan",
        "across values",
        "until",
    ]
    if any(marker in q for marker in complex_markers):
        return True

    # Sequential language
    sequence_markers = [" then ", " followed by ", " after ", " before ", " while "]
    sequence_hits = sum(1 for marker in sequence_markers if marker in f" {q} ")
    if sequence_hits >= 1:
        return True

    # If several distinct action verbs appear, treat as complex
    action_verbs = [
        "start", "connect", "set", "adjust", "move", "place", "blank", "unblank",
        "capture", "acquire", "calibrate", "tune", "measure", "analyze", "optimize",
    ]
    action_count = sum(1 for verb in action_verbs if re.search(rf"\b{re.escape(verb)}\b", q))
    if action_count >= 3:
        return True

    return False



class Agent:
    def __init__(self, model: Model, session_name: str = ""):
        self.model = model

        # Initialize session memory
        self.memory = SessionMemory(
            artifacts_base_dir=settings.artifacts_dir,
            session_name=session_name
        )

        # Initialize MCP Client
        self.mcp_client = MCPClient({"url": settings.mcp_url, "transport": "streamable-http"}, structured_output=False)
        mcp_tools = [self._wrap_mcp_tool(t) for t in self.mcp_client.get_tools()]

        # Full tool suite combining external MCP tools and native workflow tools
        workflow_tools = get_workflow_tools(self)
        all_tools = mcp_tools + workflow_tools

        self.agent = CodeAgent(
            tools=all_tools, 
            model=self.model,
            max_steps=settings.agent_max_steps,  # Configurable limit to prevent infinite loops
            executor=SupervisedExecutor(additional_authorized_imports=[
                "atomonous.tools.workflows", "atomonous.config.*", 
                "numpy", "time", "os", "scipy", "json", "yaml"
            ]),
            step_callbacks={ActionStep : self._handle_agent_step},
            instructions="""
            You are an expert scientific AI assistant powered by the Atomonous framework. 
            You can interact with the current context and instrumentation dynamically through the available MCP tools.

            Guidelines:
            1. Use 'settings' for configuration:
               - Use 'settings.server_host' and 'settings.server_port' for connections.
            2. Reliability and truthfulness are mandatory:
                    - Never claim success unless tool outputs explicitly confirm success.
                    - If any tool output contains failure indicators (e.g. "error", "failed", "could not", "exception", "timeout"), treat the task as failed.
                    - When a step fails, report the failure clearly and include the failing step + output.
                    - Prefer recovery attempts (fix config / start missing services / retry once) before giving up.
                    - For image capture tasks, verify an output artifact/path is returned before claiming completion.
            3. **Iterative Tasks & Loops (CRITICAL)**: When designing workflows with `design_workflow`, you MUST NOT create individual nodes for tasks that belong in a loop. Instead, use a single `CodeNode` to handle the entire iteration.
               - **WRONG**: Creating `node_current_10`, `node_current_20`, `node_current_30` etc.
               - **RIGHT**: Creating a single `CodeNode` with description: "Loop over beam current values [10, 20, 30]... for each value, set current, tune, and acquire."
               - This keeps the workflow graph clean and allows the Agent to use higher-level logic (like Python loops) during execution.
                4. **CodeNode execution mode (CRITICAL)**: If a prompt says you are executing an already-designed workflow step/task node, do not design a new workflow and do not suggest node structures. Execute the requested operations directly using tools and/or python_executor.
            
            'settings' is pre-imported and available for use in your code execution environment.
            """,
            stream_outputs=True
        )

        # Track workflow approval state
        self.workflow_approval_pending = False
        self.detected_workflow_path = None

        # Preload common classes into the Python executor context
        self._setup_executor_context()
    
    @classmethod
    def from_model_id(cls, model_id: str = "Auto", session_name: str = "") -> "Agent":
        # Extract model size from model_id to configure parameters appropriately
        # Larger models need higher temperature and more tokens for proper tool usage
        model_size_b = 0
        try:
            import re
            size_match = re.search(r'(\d+\.?\d*)B', model_id)
            if size_match:
                model_size_b = float(size_match.group(1))
        except:
            pass
        
        # Small models (<3B): Conservative settings
        # Medium models (3-14B): Balanced 
        # Large models (>14B): More freedom for complex reasoning and tool use
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

        # Store generation parameters for reuse
        gen_params = {
            "max_new_tokens": max_tokens,
            "temperature": temperature,
            "top_p": top_p,
            "repetition_penalty": rep_penalty,
        }

        low_cpu_mem_usage = get_total_ram_gb() < 32
        
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
                "low_cpu_mem_usage": low_cpu_mem_usage,
                "use_cache": True,
                "load_in_4bit": True,
            }
        )
        instance = cls(model=model, session_name=session_name)
        instance.gen_params = gen_params
        return instance

    @classmethod
    def from_api_key(cls, model_id: str, api_base: str, api_key: str, session_name: str = "") -> "Agent":
        model = LiteLLMModel(
            model_id=model_id,
            api_base=api_base,
            api_key=api_key
        )
        return cls(model=model, session_name=session_name)

    def _setup_executor_context(self):
        """
        Injects common variables/classes/modules into the Python executor context.
        """
        try:
            self.agent.python_executor.send_variables({
                "np": np,
                "settings": settings,
                "yaml": yaml,
            })
        except Exception:
            # Non-fatal: some executors may not support variable injection
            pass

    def run_subagent(self, prompt: str, allowed_tools: Optional[list[str]] = None, disallowed_tools: Optional[list[str]] = None) -> str:
        """
        Run a focused subagent task with optional tool filtering.
        Uses fresh agent with filtered tools for execution.
        
        Args:
            prompt: The task prompt for the agent.
            allowed_tools: If provided, only these tools are available (whitelist).
                          If None, all tools are available unless disallowed_tools is set.
            disallowed_tools: Tools to explicitly remove (blacklist).
                             Only used if allowed_tools is None.
        
        Returns:
            The agent's response as a string.
        """
        # Determine which tools to use
        all_tools = [self._wrap_mcp_tool(t) for t in self.mcp_client.get_tools()] + get_workflow_tools(self)
        if allowed_tools is not None:
            filtered_tools = [t for t in all_tools if getattr(t, "name", None) in allowed_tools]
        elif disallowed_tools is not None:
            disallowed_set = set(disallowed_tools)
            filtered_tools = [t for t in all_tools if getattr(t, "name", None) not in disallowed_set]
        else:
            filtered_tools = all_tools
        
        subagent = CodeAgent(
            tools=filtered_tools,
            model=self.model,
            max_steps=settings.agent_max_steps,  # Configurable limit to prevent infinite loops
            executor=SupervisedExecutor(additional_authorized_imports=[
                "atomonous.tools.workflows", "atomonous.config.*",
                "numpy", "time", "os", "scipy", "json", "yaml"
            ]),
            instructions=self.agent.instructions,
            stream_outputs=True
        )
        
        self._setup_executor_context()
        
        return str(subagent.run(prompt)).strip()

    def _wrap_mcp_tool(self, tool: Any) -> Any:
        """
        Wraps an MCP tool to intercept large base64 outputs and save them as artifacts.
        This prevents the LLM from seeing massive base64 strings and keeps the context clean.
        """
        original_forward = tool.forward

        def wrapped_forward(*args, **kwargs):
            result = original_forward(*args, **kwargs)
            
            # Identify large payloads returned as JSON strings
            if isinstance(result, str) and result.strip().startswith('{'):
                try:
                    if '"payload":' not in result or '"metadata":' not in result:
                        return result

                    data = json.loads(result)
                    payload = data.get("payload")
                    metadata_raw = data.get("metadata")
                    
                    if payload and metadata_raw and isinstance(payload, str) and len(payload) > 500:
                        meta = json.loads(metadata_raw)
                        decoded_bytes = base64.b64decode(payload.encode('utf-8'))
                        
                        timestamp = datetime.now().strftime("%H%M%S")
                        
                        file_name = f"data_{timestamp}.bin"
                        file_path = self.memory.session_dir / file_name
                        with open(file_path, 'wb') as f:
                            f.write(decoded_bytes)
                            
                        return (
                            f"SUCCESS: Large payload intercepted and saved to: {file_path}.\\n"
                            f"Metadata: {metadata_raw}"
                        )
                except Exception as e:
                    print(f"[Warning] Failed to wrap tool output: {e}")
            
            return result

        tool.forward = wrapped_forward
        return tool

    def _handle_agent_step(self, memory_step, agent) -> None:
        """
        Callback invoked after each agent step to check for workflow creation.
        If a workflow was just designed, captures it and triggers approval flow.
        """
        if not isinstance(memory_step, ActionStep):
            return
        
        if not self.workflow_approval_pending:
            return
        
        # Check if design_workflow was called in the code action
        if memory_step.code_action and "design_workflow(" in memory_step.code_action:
            # design_workflow was called in this step, check if it completed successfully
            try:
                workflow_path = getattr(self, "last_created_workflow", None)
                if workflow_path and Path(workflow_path).exists() and workflow_path != self.detected_workflow_path:
                    self.detected_workflow_path = workflow_path
                    # Signal that we should halt and request approval
                    raise WorkflowCreated()
            except WorkflowCreated:
                raise
            except Exception:
                pass

    @staticmethod
    def _emit(emit: Optional[Callable[[str], None]], message: str) -> None:
        if emit and message:
            emit(message)

    def _run_direct_task(self, query: str) -> str:
        direct_prompt = (
            "Execute the user's request using tools when needed.\\n"
            "Critical reliability rules:\\n"
            "- Never claim success if any step output indicates error/failure/could-not/exception/timeout.\\n"
            "- If a step fails, attempt one reasonable recovery and retry.\\n"
            "- Return a concise final result that is either:\\n"
            "  * SUCCESS: <what was done and concrete evidence>\\n"
            "  * FAILED: <what failed, where, and why>\\n"
            f"User request: {query}"
        )
        return str(self.agent.run(direct_prompt)).strip()

    def _generate_workflow_until_success(
        self,
        prompt_text: str,
        max_attempts: int = 3,
        emit: Optional[Callable[[str], None]] = None,
    ) -> tuple[Optional[str], str]:
        """Run the agent to design a workflow and detect the created YAML path via step callback."""
        last_output = ""
        base_prompt = prompt_text
        retry_instruction = (
            "You did not create a workflow using the `design_workflow` tool. "
            "Please call `design_workflow(name, yaml_content)` and then return the absolute path of the saved YAML file."
        )

        # Enable workflow approval detection via step callback
        self.workflow_approval_pending = True
        self.detected_workflow_path = None
        
        # Capture baseline to avoid accepting pre-existing workflows from previous requests
        baseline_workflow_path = None
        try:
            baseline_workflow_path = getattr(self, "last_created_workflow", None)
        except Exception:
            pass

        for attempt in range(max_attempts):
            self._emit(emit, "\nAgent is working on the workflow...\n")
            try:
                last_output = str(self.agent.run(prompt_text)).strip()
            except WorkflowCreated:
                # Workflow was detected by step callback
                if self.detected_workflow_path:
                    self._emit(emit, f"\nâ Workflow created at: {self.detected_workflow_path}\n")
                    self.workflow_approval_pending = False
                    return self.detected_workflow_path, last_output
            except Exception as e:
                # Other exception, log and continue retry loop
                last_output = str(e)
                self._emit(emit, f"\nAgent error: {e}\n")
            else:
                self._emit(emit, f"\nAgent Output:\n{last_output}\n")

            # Also check if workflow was created (fallback for agents that complete without signal)
            try:
                yaml_path = getattr(self, "last_created_workflow", None)
                if (yaml_path and 
                    Path(yaml_path).exists() and 
                    yaml_path != self.detected_workflow_path and
                    yaml_path != baseline_workflow_path):  # Only accept newly created workflows
                    self.detected_workflow_path = yaml_path
                    self.workflow_approval_pending = False
                    return yaml_path, last_output
            except Exception:
                pass

            if attempt < max_attempts - 1:
                prompt_text = f"{base_prompt}\n\n{retry_instruction}"

        self.workflow_approval_pending = False
        return None, last_output

    def _execute_workflow(self, parsed_yaml_path: str, emit: Optional[Callable[[str], None]] = None) -> str:
        try:
            with open(parsed_yaml_path, 'r') as f:
                parsed_template_yaml = yaml.safe_load(f)
            template = WorkflowTemplate(**parsed_template_yaml)
            executor = WorkflowExecutor(template, get_default_registry())

            try:
                png_path = Path(parsed_yaml_path).parent / (Path(parsed_yaml_path).stem + ".png")
                png_path_str = str(png_path) if png_path.exists() else None
                self.memory.save_workflow(parsed_yaml_path, png_path_str)
            except Exception as e:
                self._emit(emit, f"[Agent] Warning: Failed to save workflow files: {e}\n")

            self._emit(emit, f"\n--- Initiating Workflow: {template.name} ---\n\n")
            final_state = executor.run(context={"agent": self})

            try:
                self.memory.save_execution_steps(
                    history=final_state.history,
                    errors=final_state.errors,
                    metrics=final_state.metrics,
                    summary=""
                )
            except Exception as e:
                self._emit(emit, f"[Agent] Warning: Failed to save execution steps: {e}\n")

            self._emit(emit, "\nAgent is generating a summary of the execution...\n")
            summary_prompt = (
                f"The workflow '{template.name}' has finished executing.\\n"
                f"State Data: {final_state.data}\\n"
                f"History: {final_state.history}\\n"
                f"Errors: {final_state.errors}\\n"
                f"Metrics: {final_state.metrics}\\n"
                "Please provide a user-friendly summary of what was accomplished."
            )

            full_summary = ""
            for item in self.generate(summary_prompt):
                if isinstance(item, tuple) and item[0] == "final":
                    full_summary = item[1]
                elif isinstance(item, str):
                    full_summary += item
                    self._emit(emit, item)

            history_text = _safe_to_string(final_state.history)
            errors_text = _safe_to_string(final_state.errors)
            metrics_text = _safe_to_string(final_state.metrics)
            return (
                f"Workflow {template.name} execution finished.\\n\\n"
                f"Summary:\\n{full_summary}\\n\\n"
            )
        except Exception as e:
            return f"Failed to execute workflow: {e}"

    def _run_chat(self, query: str, emit: Optional[Callable[[str], None]] = None, interactive_approval: bool = True) -> str:
        if not _should_generate_workflow(query):
            self._emit(emit, "\nHandling request as a direct task (no workflow generation).\n")
            try:
                return self._run_direct_task(query)
            except Exception as e:
                return f"Failed to execute task: {e}"

        init_prompt = (
            f"Please design a workflow for the following experimental task:\\n{query}\\n\\n"
            "You MUST use the `design_workflow` tool to define and save this workflow. Provide the absolute path of the saved yaml file as your final answer."
        )
        parsed_yaml_path, last_output = self._generate_workflow_until_success(init_prompt, emit=emit)

        if parsed_yaml_path is None:
            if last_output and "Workflow" in last_output:
                return last_output
            return f"Failed to design workflow after multiple attempts. Last agent output:\n{last_output}"

        if interactive_approval:
            while True:
                self._emit(emit, f"\nProposed Workflow YAML: {parsed_yaml_path}\n")
                self._emit(emit, "Options:\n")
                self._emit(emit, "1. Accept workflow and execute\n")
                self._emit(emit, "2. Modify workflow\n")
                self._emit(emit, "3. Reject workflow and stop\n")
                choice = input("Enter your choice (1/2/3): ").strip()

                if choice == '1':
                    break
                if choice == '2':
                    mod_query = input("Enter your modifications: ")
                    mod_prompt = (
                        f"Please modify the previously designed workflow as follows: {mod_query}\\n"
                        "Use `design_workflow` to save it and return the updated absolute path."
                    )
                    parsed_yaml_path, _ = self._generate_workflow_until_success(mod_prompt, emit=emit)
                    if parsed_yaml_path is None:
                        return "Failed to modify workflow after multiple attempts."
                    continue
                if choice == '3':
                    return "Execution canceled by user."
                self._emit(emit, "Invalid choice.\n")
        else:
            self._emit(emit, f"\nProposed Workflow YAML: {parsed_yaml_path}\n")
            self._emit(emit, "Auto-accepting workflow in non-interactive stream mode.\n")

        return self._execute_workflow(parsed_yaml_path, emit=emit)

    def chat(self, query: str) -> str:
        """
        Process user input and return a response.
        """
        return self._run_chat(
            query,
            emit=lambda message: print(message, end="", flush=True),
            interactive_approval=True,
        )

    def stream_chat(self, query: str):
        """
        Stream user input processing as a sequence of events.
        Yields dicts with keys: type ("delta"|"final") and content.
        Uses the same internal execution path as chat(), but auto-accepts workflow
        approval in non-interactive contexts.
        """
        deltas: list[str] = []
        res = self._run_chat(
            query,
            emit=deltas.append,
            interactive_approval=False,
        )
        for delta in deltas:
            yield {"type": "delta", "content": delta}
        yield {"type": "final", "content": res}
    
    def generate(self, query: str):
        """
        Streams response to a query using the underlying smolagents model API.
        Finally yields a tuple ("final", complete_response).
        """
        
        try:
            message = ChatMessage(role="user", content=query)
            gen_kwargs = dict(getattr(self, "gen_params", {}))

            full_text = ""
            stream_fn = getattr(self.model, "generate_stream", None)

            # Prefer model-native streaming when supported.
            if callable(stream_fn):
                for delta in stream_fn(messages=[message], **gen_kwargs):
                    chunk = _extract_generated_text(delta)
                    if not chunk:
                        continue
                    full_text += chunk
                    if chunk.strip():
                        yield chunk

                yield ("final", full_text.strip())
                return

            # Fallback for models without generate_stream.
            output = self.model.generate(messages=[message], **gen_kwargs)
            full_text = _extract_generated_text(output)
            if full_text.strip():
                yield full_text
            yield ("final", full_text.strip())
            
        except Exception as e:
            error_msg = f"[Warning] Failed to generate response: {e}"
            yield ("error", error_msg)

def _extract_generated_text(output: Union[str, ChatMessage, ChatMessageStreamDelta, dict, object]) -> str:
    """
    Extract plain text from model.generate() output.
    """
    if output is None:
        return ""
    
    if isinstance(output, str):
        return output.strip()
    
    if isinstance(output, ChatMessage):
        content = output.content
        if isinstance(content, str):
            return content.strip()
        # If content is a list of dicts
        if isinstance(content, list):
            chunks = []
            for item in content:
                if isinstance(item, dict) and "text" in item:
                    chunks.append(item["text"])
                elif isinstance(item, str):
                    chunks.append(item)
            return "\n".join(chunks).strip()
        return str(content).strip()

    if isinstance(output, ChatMessageStreamDelta):
        if output.content is None:
            return ""
        if isinstance(output.content, str):
            return output.content
        return str(output.content)

    if isinstance(output, dict):
        content = output.get("content", "")
        if isinstance(content, str):
            return content
        if content is None:
            return ""
        return str(content)
    
    return str(output).strip()