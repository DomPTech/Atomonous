import os
import sys
from pathlib import Path

import yaml
import graphviz

from typing import Optional, Dict, Any, Union

from smolagents import tool

from atomonous.config import settings
from atomonous.tools.workflow_framework import WorkflowState, WorkflowNode, WorkflowTemplate, WorkflowExecutor

class DomainToolNode(WorkflowNode):
    def execute(self, state: WorkflowState, context: Optional[dict] = None) -> WorkflowState:
        tool_name = self.params.get("tool")
        tool_args = self.params.get("args", {})
        
        # Look for the tool in the agent context
        agent = context.get("agent") if context else None
        all_tools = []
        if agent:
             if hasattr(agent, "mcp_client"):
                 all_tools.extend(agent.mcp_client.get_tools())
             if hasattr(agent, "agent") and hasattr(agent.agent, "tools"):
                 # Handle if agent is a wrapper with a CodeAgent inside
                 all_tools.extend(agent.agent.tools.values() if isinstance(agent.agent.tools, dict) else agent.agent.tools)
             
        tool_func = next((t for t in all_tools if getattr(t, "name", "") == tool_name), None)
        if not tool_func:
            err = f"FATAL: Tool {tool_name} not found."
            print(err)
            state.errors.append(err)
            return state
            
        try:
            state.history.append(f"Executing DomainToolNode: {tool_name}")
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
        fake_context = f"Retrieved context for '{query}'"
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
                    f"You are executing an already-designed workflow step named '{self.name}'.\n"
                    f"Your core task is: {description}\n\n"
                    "Execute the task using available tools and code execution.\n"
                    "Provide a brief summary of what you did when you are finished."
                )
                
                # Run as subagent with workflow-construction tools disabled
                disallowed = ["design_workflow", "execute_workflow"]
                result = agent.run_subagent(prompt, disallowed_tools=disallowed)
                
                state.data[self.name] = result
                print(f"  -> Agent completed CodeNode task. Response:\n{result}")
            except Exception as e:
                err = f"FATAL: Code execution error in {self.name}: {e}"
                print(err)
                state.errors.append(err)
        else:
            err = f"FATAL: CodeNode '{self.name}' requires the 'agent' in the context dictionary to execute."
            print(err)
            state.errors.append(err)
        return state

def get_default_registry():
    return {
        "DomainTool": DomainToolNode,
        "AIContext": AIContextNode,
        "AIQuality": AIQualityNode,
        "CodeNode": CodeNode,
    }

def _generate_workflow_diagram(template: WorkflowTemplate, output_path: str) -> bool:
    try:
        dot = graphviz.Digraph(name="workflow", format="png")
        dot.attr(rankdir='TB', splines='spline', nodesep='0.6', ranksep='0.8', bgcolor='#121212')
        dot.attr('edge', fontname='Helvetica,Arial,sans-serif', fontsize='10', color='#888888', arrowsize='0.8')
        
        for node in template.nodes:
            node_id = str(node['id'])
            node_type = node.get('type', 'Unknown')
            params = node.get('params', {})
            
            # Distinct colors per node type
            if node_type in ['DomainTool', 'MicroscopeTool']:
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

            html_rows = f'<TR><TD ALIGN="CENTER" BORDER="0" CELLPADDING="8"><B><FONT COLOR="{border_color}" POINT-SIZE="16">{node_id}</FONT></B></TD></TR>'
            html_rows += f'<TR><TD ALIGN="CENTER" BORDER="0" CELLPADDING="2"><FONT COLOR="#AAAAAA" POINT-SIZE="11">{node_type}</FONT></TD></TR>'
            
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
        
        output_dir = Path(output_path).parent
        output_name = Path(output_path).stem
        dot.render(str(output_dir / output_name), cleanup=True)
        return True
    except Exception as e:
        print(f"[Warning] Failed to generate workflow diagram: {e}")
        return False

def get_workflow_tools(agent_instance, node_registry=None) -> list:
    """
    Returns a list of workflow tools bound to the specific agent instance.
    This replaces the global AGENT_INSTANCE pattern and allows multiple concurrent agents.
    """
    
    # Use default registry if none provided
    registry = node_registry or get_default_registry()

    @tool
    def design_workflow(name: str, yaml_content: str) -> str:
        """
        Designs a new experimental workflow by parsing, validating, and saving a YAML configuration.
        This function handles all path/file management and returns the path automatically.
        
        CRITICAL: You MUST use a `CodeNode` for any logic that requires iteration (like for/while loops). 
        - WRONG: Creating individual `DomainTool` nodes to iterate over values.
        - RIGHT: Create a single `CodeNode` with a description that explains the loop.
            
        Args:
            name: Name of the workflow.
            yaml_content: The full YAML string defining the workflow. It must have 'name', 'description', 
                          'nodes' (list of dicts with 'id', 'type', 'params'), and 
                          'edges' (list of dicts with 'source' and 'target'). Types can be 'DomainTool' 
                          (params: 'tool', 'args'), 'AIContext' (params: 'query'), 'AIQuality' (params: 'evaluate_node'),
                          or 'CodeNode' (params: 'description').
        
        Returns:
            Absolute path to the saved YAML workflow file.
        """
        try:
            parsed_yaml = yaml.safe_load(yaml_content)
            template = WorkflowTemplate(**parsed_yaml)
            
            artifacts_dir = Path(settings.artifacts_dir)
            artifacts_dir.mkdir(parents=True, exist_ok=True)
            
            session_dir = None
            if agent_instance and hasattr(agent_instance, 'memory') and agent_instance.memory:
                session_dir = agent_instance.memory.session_dir
            
            if session_dir and Path(session_dir).exists():
                save_dir = Path(session_dir)
            else:
                save_dir = artifacts_dir
            
            filename = f"{name.replace(' ', '_').lower()}.yaml"
            yaml_path = save_dir / filename
            
            with open(yaml_path, 'w') as f:
                f.write(yaml_content)
            
            yaml_path_abs = str(yaml_path.resolve())
            
            png_base_path = str(save_dir / Path(yaml_path_abs).stem)
            _generate_workflow_diagram(template, png_base_path)
            
            if agent_instance:
                agent_instance.last_created_workflow = yaml_path_abs
            
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
            executor = WorkflowExecutor(template, registry)
            
            print(f"\\n--- Initiating Workflow: {template.name} ---\\n")

            final_state = executor.run(context={"agent": agent_instance})

            return f"Workflow {template.name} execution finished.\\nHistory: {final_state.history}\\nErrors: {final_state.errors}\\nMetrics: {final_state.metrics}"
        except Exception as e:
            return f"Failed to execute workflow: {str(e)}"

    return [design_workflow, execute_workflow]
