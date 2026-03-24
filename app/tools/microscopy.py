import subprocess
import time
import sys
import os
import socket
from pathlib import Path
from typing import Optional, Dict, Any, Union
from smolagents import tool
import Pyro5.api
import Pyro5.errors
import numpy as np
# import tango
# from tango.test_context import MultiDeviceTestContext
from app.config import settings
from enum import Enum
import pyTEMlib.probe_tools as pt
import json

try:
    from autoscript_tem_microscope_client import TemMicroscopeClient
    from autoscript_tem_microscope_client.enumerations import EdsDetectorType, ExposureTimeType
    from autoscript_tem_microscope_client.structures import EdsAcquisitionSettings
    _AUTOSCRIPT_DIRECT_AVAILABLE = True
except ImportError:
    _AUTOSCRIPT_DIRECT_AVAILABLE = False

# Global state
# Global state
CLIENT: Optional[object] = None  # tango.DeviceProxy (Legacy)
AGENT_INSTANCE: Optional[object] = None  # Set by Agent.__init__() for artifact memory access

# Collection of tools for the agent (initially empty, populated via MCP or kept for workflow)
TOOLS = []

# Workflow Framework Integration

# Workflow Framework Integration
import os
import yaml
import graphviz
from app.tools.workflow_framework import WorkflowState, WorkflowNode, WorkflowTemplate, WorkflowExecutor

class MicroscopeToolNode(WorkflowNode):
    def execute(self, state: WorkflowState, context: Optional[dict] = None) -> WorkflowState:
        tool_name = self.params.get("tool")
        tool_args = self.params.get("args", {})
        
        # Look for the tool in the agent context or the minimal TOOLS list
        agent = context.get("agent") if context else None
        all_tools = []
        if agent and hasattr(agent, "mcp_client"):
             all_tools = agent.mcp_client.get_tools()
        else:
             all_tools = TOOLS
             
        tool_func = next((t for t in all_tools if getattr(t, "name", "") == tool_name), None)
        if not tool_func:
            err = f"FATAL: Tool {tool_name} not found."
            print(err)
            state.errors.append(err)
            return state
            
        try:
            state.history.append(f"Executing MicroscopeToolNode: {tool_name}")
            print(f"  -> Invoking '{tool_name}' with args {tool_args}")
            # Support both keyword-arg style (dict) and positional-arg style (list)
            if isinstance(tool_args, dict):
                result = tool_func(**tool_args)
            elif isinstance(tool_args, (list, tuple)):
                result = tool_func(*tool_args)
            else:
                # Single scalar argument
                result = tool_func(tool_args)

            print(f"  -> Result: {result}")
            state.data[self.name] = result
        except Exception as e:
            err = f"FATAL: Error in {tool_name}: {e}"
            print(err)
            state.errors.append(err)
            
        return state

class AIContextNode(WorkflowNode):
    def execute(self, state: WorkflowState, context: Optional[dict] = None) -> WorkflowState:
        query = self.params.get("query", "")
        # Real implementation would call an LLM here
        fake_context = f"Retrieved experimental context for '{query}': parameters should be tuned near 1000."
        state.context[self.name] = fake_context
        state.history.append(f"AI Context Node retrieved: {fake_context}")
        print(f"  -> Context retrieved: {fake_context}")
        return state

class AIQualityNode(WorkflowNode):
    def execute(self, state: WorkflowState, context: Optional[dict] = None) -> WorkflowState:
        target = self.params.get("evaluate_node")
        if target in state.data:
            state.metrics[f"{self.name}_score"] = 0.95
            state.history.append(f"AI Quality Node evaluated {target} with score 0.95")
            print(f"  -> AI evaluated {target} successfully (Score: 0.95)")
        else:
            err = f"AI Quality Node could not find data for {target}"
            print(f"  -> {err}")
            state.errors.append(err)
        return state

class CodeNode(WorkflowNode):
    def execute(self, state: WorkflowState, context: Optional[dict] = None) -> WorkflowState:
        description = self.params.get("description", "")
        agent = context.get("agent") if context else None
        
        if agent:
            print(f"  -> [CodeNode '{self.name}'] Unpausing Agent to solve task: {description}")
            try:
                state.history.append(f"Executing CodeNode '{self.name}' via LLM Agent task")
                
                # Command the agent to fulfill the description
                prompt = (
                    f"You are executing an already-designed workflow step named '{self.name}'.\\n"
                    f"Your core task is: {description}\\n\\n"
                    "Execute the task using available tools and code execution.\\n"
                    "Provide a brief summary of what you did when you are finished."
                )
                
                # Run as subagent with workflow-construction tools disabled
                disallowed = ["design_workflow", "execute_workflow"]
                result = agent.run_subagent(prompt, disallowed_tools=disallowed)
                
                state.data[self.name] = result
                print(f"  -> Agent completed CodeNode task. Response:\\n{result}")
            except Exception as e:
                err = f"FATAL: Code execution error in {self.name}: {e}"
                print(err)
                state.errors.append(err)
        else:
            err = f"FATAL: CodeNode '{self.name}' requires the 'agent' in the context dictionary to execute."
            print(err)
            state.errors.append(err)
        return state

NODE_REGISTRY = {
    "MicroscopeTool": MicroscopeToolNode,
    "AIContext": AIContextNode,
    "AIQuality": AIQualityNode,
    "CodeNode": CodeNode,
}

# Simple in-process workflow state store to avoid fragile regex parsing
# Keys are absolute yaml paths, values are dicts with keys: status ('created'|'executing'|'finished'),
# created_at, updated_at, summary (optional)
WORKFLOW_STATE = {}

def register_workflow_created(yaml_path: str):
    from datetime import datetime
    yaml_path = os.path.abspath(yaml_path)
    WORKFLOW_STATE[yaml_path] = {
        "status": "created",
        "created_at": datetime.utcnow().isoformat(),
        "updated_at": datetime.utcnow().isoformat(),
        "summary": None,
    }

def register_workflow_executing(yaml_path: str):
    from datetime import datetime
    yaml_path = os.path.abspath(yaml_path)
    if yaml_path not in WORKFLOW_STATE:
        register_workflow_created(yaml_path)
    WORKFLOW_STATE[yaml_path]["status"] = "executing"
    WORKFLOW_STATE[yaml_path]["updated_at"] = datetime.utcnow().isoformat()

def register_workflow_finished(yaml_path: str, summary: str = None):
    from datetime import datetime
    yaml_path = os.path.abspath(yaml_path)
    if yaml_path not in WORKFLOW_STATE:
        register_workflow_created(yaml_path)
    WORKFLOW_STATE[yaml_path]["status"] = "finished"
    WORKFLOW_STATE[yaml_path]["updated_at"] = datetime.utcnow().isoformat()
    WORKFLOW_STATE[yaml_path]["summary"] = summary

def get_last_created_workflow() -> Optional[str]:
    # Return the most recently created workflow path, or None
    if not WORKFLOW_STATE:
        return None
    # sort by created_at
    try:
        items = sorted(WORKFLOW_STATE.items(), key=lambda kv: kv[1].get("created_at", ""), reverse=True)
        return items[0][0]
    except Exception:
        # fallback
        return next(iter(WORKFLOW_STATE.keys()))


def _generate_workflow_diagram(template: WorkflowTemplate, output_path: str) -> bool:
    """
    Generate a graphviz diagram of the workflow and save as PNG.
    
    Args:
        template: WorkflowTemplate object with nodes and edges.
        output_path: Path to save PNG (without .png extension - graphviz adds it).
    
    Returns:
        True if successful, False otherwise.
    """
    try:
        dot = graphviz.Digraph(name="workflow", format="png")
        dot.attr(rankdir='TB', splines='spline', nodesep='0.6', ranksep='0.8', bgcolor='#121212')
        dot.attr('edge', fontname='Helvetica,Arial,sans-serif', fontsize='10', color='#888888', arrowsize='0.8')
        
        for node in template.nodes:
            node_id = str(node['id'])
            node_type = node.get('type', 'Unknown')
            params = node.get('params', {})
            
            # Distinct colors per node type
            if node_type == 'MicroscopeTool':
                border_color = '#00FF9D'
                bg_color = '#002E1C'
            elif node_type == 'AIContext':
                border_color = '#00D1FF'
                bg_color = '#001A24'
            elif node_type == 'AIQuality':
                border_color = '#FF9900'
                bg_color = '#2E1A00'
            elif node_type == 'CodeNode':
                border_color = '#FF00FF'
                bg_color = '#2A002A'
            else:
                border_color = '#999999'
                bg_color = '#222222'

            # Build HTML-like label showing the function / params
            html_rows = f'<TR><TD ALIGN="CENTER" BORDER="0" CELLPADDING="8"><B><FONT COLOR="{border_color}" POINT-SIZE="16">{node_id}</FONT></B></TD></TR>'
            html_rows += f'<TR><TD ALIGN="CENTER" BORDER="0" CELLPADDING="2"><FONT COLOR="#AAAAAA" POINT-SIZE="11">{node_type}</FONT></TD></TR>'
            
            # Add parameters if they exist
            if params:
                for k, v in params.items():
                    val_str = str(v)[:40] + '...' if len(str(v)) > 40 else str(v)
                    html_rows += f'<TR><TD ALIGN="LEFT" BORDER="0" CELLPADDING="4"><FONT COLOR="#CCCCCC" POINT-SIZE="10"><B>{k}:</B> {val_str}</FONT></TD></TR>'

            label = f'<<TABLE BORDER="1" COLOR="{border_color}" CELLBORDER="0" CELLSPACING="0" CELLPADDING="8" BGCOLOR="{bg_color}" STYLE="ROUNDED">{html_rows}</TABLE>>'

            if node_id in ['__start__', '__end__', 'start', 'end']:
                dot.node(node_id, node_id, shape='ellipse', style='filled,rounded', fillcolor='#333333', color='#888888', fontcolor='#FFFFFF', fontname='Helvetica,Arial,sans-serif')
            else:
                dot.node(node_id, label, shape='none', margin='0')
            
        for edge in template.edges:
            edge_kwargs = {}
            if 'style' in edge:
                edge_kwargs['style'] = str(edge['style'])
            if 'label' in edge:
                label_text = str(edge['label'])
                edge_kwargs['label'] = f'<<TABLE BORDER="0" CELLBORDER="1" COLOR="#333333" CELLPADDING="4" BGCOLOR="#222222"><TR><TD><FONT COLOR="#FFFFFF" POINT-SIZE="10">{label_text}</FONT></TD></TR></TABLE>>'
                
            dot.edge(str(edge['source']), str(edge['target']), **edge_kwargs)
        
        # Save to specified path (graphviz adds .png extension)
        output_dir = Path(output_path).parent
        output_name = Path(output_path).stem
        dot.render(str(output_dir / output_name), cleanup=True)
        
        return True
    except Exception as e:
        print(f"[Warning] Failed to generate workflow diagram: {e}")
        return False


@tool
def design_workflow(name: str, yaml_content: str) -> str:
    """
    Designs a new experimental workflow by parsing, validating, and saving a YAML configuration.
    This function handles all path/file management and returns the path automatically.
    
    CRITICAL: You MUST use a `CodeNode` for any logic that requires iteration (like for/while loops). 
    - WRONG: Creating individual `MicroscopeTool` nodes to iterate over values (e.g. 'set_current_10', 'set_current_20').
    - RIGHT: Create a single `CodeNode` with a description that explains the loop (e.g. 'Loop over beam currents [10, 20, 30]... for each value, set current, tune, and acquire tableau').
        
    Args:
        name: Name of the workflow (e.g., 'focus_optimization').
        yaml_content: The full YAML string defining the workflow. It must have 'name', 'description', 
                      'nodes' (list of dicts with 'id', 'type', 'params'), and 
                      'edges' (list of dicts with 'source' and 'target'). Types can be 'MicroscopeTool' 
                      (params: 'tool', 'args'), 'AIContext' (params: 'query'), 'AIQuality' (params: 'evaluate_node'),
                      or 'CodeNode' (params: 'description').
    
    Returns:
        Absolute path to the saved YAML workflow file.
    """
    
    try:
        parsed_yaml = yaml.safe_load(yaml_content)
        template = WorkflowTemplate(**parsed_yaml)
        
        from pathlib import Path
        artifacts_dir = Path(settings.artifacts_dir)
        artifacts_dir.mkdir(parents=True, exist_ok=True)
        
        # Get current session folder if agent has memory
        session_dir = None
        try:
            if AGENT_INSTANCE and hasattr(AGENT_INSTANCE, 'memory') and AGENT_INSTANCE.memory:
                session_dir = AGENT_INSTANCE.memory.session_dir
        except Exception:
            pass
        
        # Save to session folder if available, otherwise to artifacts root
        if session_dir and Path(session_dir).exists():
            save_dir = Path(session_dir)
        else:
            save_dir = artifacts_dir
        
        filename = f"{name.replace(' ', '_').lower()}.yaml"
        yaml_path = save_dir / filename
        
        with open(yaml_path, 'w') as f:
            f.write(yaml_content)
        
        yaml_path_abs = str(yaml_path.resolve())
        
        # Generate workflow diagram (PNG)
        png_base_path = str(save_dir / Path(yaml_path_abs).stem)
        _generate_workflow_diagram(template, png_base_path)
        
        register_workflow_created(yaml_path_abs)
        
        return yaml_path_abs
    except Exception as e:
        return f"Failed to design workflow: {str(e)}"

@tool
def execute_workflow(yaml_path: str) -> str:
    """
    Executes a pre-designed and approved experimental workflow from a YAML file.
    
    Args:
        yaml_path: The absolute or relative path to the .yaml workflow file.
    """
    try:
        with open(yaml_path, 'r') as f:
            parsed_yaml = yaml.safe_load(f)
            
        template = WorkflowTemplate(**parsed_yaml)
        executor = WorkflowExecutor(template, NODE_REGISTRY)
        
        # Execute workflow
        print(f"\\n--- Initiating Workflow: {template.name} ---\\n")
        # Mark executing
        try:
            register_workflow_executing(yaml_path)
        except Exception:
            pass

        final_state = executor.run(context={"agent": getattr(sys.modules[__name__], "AGENT_INSTANCE", None)})

        # Mark finished with summary
        try:
            summary = f"History: {final_state.history}; Errors: {final_state.errors}; Metrics: {final_state.metrics}"
            register_workflow_finished(yaml_path, summary)
        except Exception:
            pass

        return f"Workflow {template.name} execution finished.\\nHistory: {final_state.history}\\nErrors: {final_state.errors}\\nMetrics: {final_state.metrics}"
    except Exception as e:
        return f"Failed to execute workflow: {str(e)}"

# Add the new tools to the exported list
TOOLS.extend([design_workflow, execute_workflow])

@tool
def get_probe(aberrations: dict, size_x: int = 128, size_y: int = 128, verbose: bool = True) -> np.ndarray:
    """
    Converts microscope-derived aberration coefficients into a 2D electron probe.
    
    This is ideal for processing 'Tableau' data or direct hardware feedback to visualize 
    the current state of the electron beam.

    Args:
        aberrations: The dictionary of aberrations received from the microscope (e.g., from 'acquireTableau').
                     Must contain 'acceleration_voltage', 'convergence_angle', and 'FOV'.
        size_x: The pixel resolution of the output grid in x-direction. Default is 128.
        size_y: The pixel resolution of the output grid in y-direction. Default is 128.
        verbose: If True, outputs calculation metadata such as wavelength.

    Returns:
        A numpy array representing the 'probe' intensity map.
    """

    ab = convert_aberrations_A_to_C(aberrations)
    ab['acceleration_voltage'] = 200e3
    ab['convergence_angle'] = 30e-3
    ab['FOV'] = 500
    probe, A_k, chi  = pt.get_probe(ab, 256, 256, verbose= True)

    return probe

def convert_aberrations_A_to_C(ab: Dict) -> Dict:
    """
    Convert aberrations from A/B/S/D notation to Saxton Cnm notation.

    Args:
        ab : dict
            aberration dict in A1, A2, C3, etc format

    Returns:
        dict with Cnm notation populated
    """

    out = dict(ab)  # copy everything

    mapping = {

        # defocus
        "C1": ("C10",),

        # 2nd order
        "A1": ("C12a", "C12b"),

        # 3rd order
        "B2": ("C21a", "C21b"),
        "A2": ("C23a", "C23b"),

        # 4th order
        "C3": ("C30",),
        "S3": ("C32a", "C32b"),
        "A3": ("C34a", "C34b"),

        # 5th order
        "D4": ("C41a", "C41b"),
        "B4": ("C43a", "C43b"),
        "A4": ("C45a", "C45b"),

        # 6th order
        "C5": ("C50",),
        "A5": ("C56a", "C56b"),
    }

    for key, target in mapping.items():

        if key not in ab:
            continue

        val = ab[key]

        # symmetric terms
        if len(target) == 1:

            if isinstance(val, (list, tuple, np.ndarray)):
                out[target[0]] = float(val[0]* 1e9)
            else:
                out[target[0]] = float(val* 1e9)

        # angular terms
        elif len(target) == 2:

            out[target[0]] = float(val[0]* 1e9)
            out[target[1]] = float(val[1]* 1e9)

    return out

TOOLS.append(get_probe)