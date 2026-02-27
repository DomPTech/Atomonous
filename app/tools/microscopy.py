import subprocess
import time
import sys
import os
import socket
from typing import Optional, Dict, Any, Union
from smolagents import tool
import Pyro5.api
import Pyro5.errors
import numpy as np
from app.config import settings
from enum import Enum
import pyTEMlib.probe_tools as pt
import json

# Global state for the client and server process
CLIENT: Optional[object] = None # asyncroscopy.clients.notebook_client.NotebookClient
SERVER_PROCESSES: Dict[str, subprocess.Popen] = {}

def _wait_for_port(host: str, port: int, timeout: float = 10.0) -> bool:
    """Wait for a port to become available (server listening)."""
    start_time = time.time()
    while time.time() - start_time < timeout:
        try:
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.settimeout(1)
            sock.connect((host, port))
            sock.close()
            return True
        except (socket.error, ConnectionRefusedError):
            time.sleep(0.2)
        except Exception:
            time.sleep(0.2)
    return False

# Define the microscope servers and their twins
class MicroscopeServer(Enum):
    Central = {
        "server": "asyncroscopy.servers.protocols.central_server",
        "port": 9000
    }
    AS = {
        "server": "asyncroscopy.servers.AS_server",
        "twin": "asyncroscopy.servers.AS_server_twin",
        "port": 9001
    }
    Ceos = {
        "server": "asyncroscopy.servers.Ceos_server",
        "twin": "asyncroscopy.servers.Ceos_server_twin",
        "port": 9003
    }

@tool
def start_server(mode: str = "mock", servers: Optional[list[Union[str, MicroscopeServer]]] = None) -> str:
    """
    Starts the microscope servers (Twisted architecture).
    
    Args:
        mode: "mock" for testing/simulation (uses twin servers), "real" for actual hardware.
        servers: List of server modules to start. Can be Enum constants or strings.
            Example: ["MicroscopeServer.Central", "AS"]. Available options:
            - MicroscopeServer.Central: The main control server (Port 9000).
            - MicroscopeServer.AS: The AS server or its twin (Port 9001).
            - MicroscopeServer.Ceos: The Ceos server or its twin (Port 9003).
            Defaults to starting all three [Central, AS, Ceos] if None.
    """

    global SERVER_PROCESSES
    if servers is None:
        servers = [MicroscopeServer.Central, MicroscopeServer.AS, MicroscopeServer.Ceos]
    else:
        parsed_servers = []
        for s in servers:
            if isinstance(s, MicroscopeServer):
                parsed_servers.append(s)
            elif isinstance(s, str):
                name = s.split('.')[-1] if '.' in s else s
                try:
                    parsed_servers.append(MicroscopeServer[name])
                except KeyError:
                    return f"Invalid server name: {s}. Valid options: {[e.name for e in MicroscopeServer]}"
            else:
                return f"Invalid type for server: {type(s)}. Must be string or MicroscopeServer enum."
        servers = parsed_servers

    base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    repo_path = os.path.join(base_dir, "external", "asyncroscopy")
    
    env = os.environ.copy()
    env["PYTHONPATH"] = f"{repo_path}{os.pathsep}{env.get('PYTHONPATH', '')}"
    env["AUTOSCRIPT_PATH"] = settings.autoscript_path

    started = []
    ports_to_wait = []
    
    try:
        for server in servers:
            server_name = server.value.get("server")
            module = server_name
            port = server.value.get("port")
            if mode == "mock":
                module = server.value.get("twin", server_name)
            
            # Check if this specific module is already tracked and running
            if module in SERVER_PROCESSES and SERVER_PROCESSES[module].poll() is None:
                print(f"Server {module} already running (tracked).")
                started.append(f"{module} (already running)")
                continue

            # Check if something is already listening on the port (might be an orphaned process)
            if _wait_for_port("localhost", port, timeout=0.2):
                print(f"Server port {port} already listening. Assuming it's the correct server.")
                started.append(f"{module} (already listening)")
                continue

            cmd = [sys.executable, "-m", module, str(port)]

            print(f"Starting server: {module} on port {port}")
            proc = subprocess.Popen(
                cmd,
                cwd=base_dir,
                env=env,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL
            )
            SERVER_PROCESSES[module] = proc
            started.append(module)
            ports_to_wait.append(port)
        
        if not started:
            return "All requested servers are already running."

        # Wait for newly started servers to be ready
        if ports_to_wait:
            print(f"Waiting for servers on ports {ports_to_wait} to be ready...")
            for port in ports_to_wait:
                if not _wait_for_port("localhost", port, timeout=10.0):
                    return f"Failed to start server on port {port} - timeout waiting for it to listen"
        
        return f"Servers status: {', '.join(started)} in {mode} mode."

    except Exception as e:
        return f"Failed to start servers: {e}"

@tool
def connect_client(host: Optional[str] = None, port: Optional[int] = None) -> str:
    """
    Connects the client to the central server and sets up routing.
    
    Args:
        host: Central server host (defaults to settings.server_host).
        port: Central server port (defaults to settings.server_port).
    """
    global CLIENT
    from asyncroscopy.clients.notebook_client import NotebookClient

    # Use settings defaults if not provided
    host = host or settings.server_host
    port = port or settings.server_port

    # Safety delay to ensure servers are ready
    time.sleep(1)
    
    routing_table = {
        "Central": ("localhost", MicroscopeServer.Central.value.get("port")),
        "AS": ("localhost", MicroscopeServer.AS.value.get("port")),
        "Ceos": ("localhost", MicroscopeServer.Ceos.value.get("port"))
    }

    try:
        CLIENT = NotebookClient.connect(host=host, port=port)
        if not CLIENT:
            return "Failed to connect to central server."
        
        # Configure routing on the central server
        resp = CLIENT.send_command("Central", "set_routing_table", routing_table)
        if isinstance(resp, str) and "ERROR" in resp:
            return f"Failed to set routing table: {resp}"
        
        # Initialize AS server
        as_resp = CLIENT.send_command("AS", "connect_AS", {
            "host": settings.instrument_host, 
            "port": settings.instrument_port
        })
        if isinstance(as_resp, str) and "ERROR" in as_resp:
            return f"Failed to reach AS server: {as_resp}. Did you start all servers?"
        
        return f"Connected successfully. Routing: {resp}, AS: {as_resp}"
    except Exception as e:
        CLIENT = None
        return f"Connection error: {e}"

@tool
def adjust_magnification(amount: float, destination: str = "AS") -> str:
    """
    Adjusts the microscope magnification level.
    
    Args:
        amount: The magnification level to set.
        destination: The server to send the command to (default 'AS').
    """
    global CLIENT
    if not CLIENT:
        return "Error: Client not connected."
    
    try:
        # AS server might not have 'set_microscope_status' directly, needs investigation of command list
        # Based on AS_server_AtomBlastTwin, we might need a different command
        resp = CLIENT.send_command(destination, "set_magnification", {"value": amount})
        return f"Magnification command sent to {destination}: {resp}"
    except Exception as e:
        return f"Error adjusting magnification: {e}"

@tool
def capture_image(detector: str = "Ceta", destination: str = "AS") -> str:
    """
    Captures an image and saves it.
    
    Args:
        detector: The detector to use.
        destination: The server to send the command to (default 'AS').
    """
    global CLIENT
    if not CLIENT:
        return "Error: Client not connected."
        
    try:
        # Twisted servers return numpy arrays wrapped in package_message
        print(f"[TOOLS DEBUG] Requesting image from {destination}...")
        img = CLIENT.send_command(destination, "get_scanned_image", {
            "scanning_detector": detector,
            "size": 512,
            "dwell_time": 2e-6
        })
        print(f"[TOOLS DEBUG] Received response of type: {type(img)}")
        
        if img is None:
            return "Failed to capture image (None returned)."
            
        if isinstance(img, str):
            return f"Failed to capture image. Error from server: {img}"

        output_path = f"/tmp/microscope_capture_{int(time.time())}.npy"
        np.save(output_path, img)
        return f"Image captured from {destination} and saved to {output_path} (Shape: {img.shape})"
    except Exception as e:
        return f"Error capturing image: {e}"

@tool
def close_microscope() -> str:
    """
    Safely closes the microscope connection and stops the servers.
    """
    global SERVER_PROCESSES, CLIENT
    resp = "Microscope closed."
    
    CLIENT = None
    
    for module, proc in SERVER_PROCESSES.items():
        proc.terminate()
        resp += f" {module} stopped."
    SERVER_PROCESSES.clear()
        
    return resp

@tool
def get_stage_position(destination: str = "AS") -> str:
    """
    Get the current stage position (x, y, z).
    
    Args:
        destination: The server to send the command to (default 'AS').
    """
    global CLIENT
    if not CLIENT:
        return "Error: Client not connected."
    
    try:
        pos = CLIENT.send_command(destination, "get_stage")
        return f"Stage Position from {destination}: {pos}"
    except Exception as e:
        return f"Error getting stage position: {e}"

@tool
def calibrate_screen_current(destination: str = "AS") -> str:
    """
    Calibrates the gun lens values to screen current.
    Start with screen current at ~100 pA. Screen must be inserted.
    
    Args:
        destination: The server to send the command to (default 'AS').
    """
    global CLIENT
    if not CLIENT:
        return "Error: Client not connected."
    
    try:
        resp = CLIENT.send_command(destination, "calibrate_screen_current")
        return f"Screen current calibration: {resp}"
    except Exception as e:
        return f"Error calibrating screen current: {e}"

@tool
def set_beam_current(current_pa: float, destination: str = "AS") -> str:
    """
    Sets the screen current (via gun lens). Must have screen current calibrated first.
    
    Args:
        current_pa: The target current in picoamperes (pA).
        destination: The server to send the command to (default 'AS').
    """
    global CLIENT
    if not CLIENT:
        return "Error: Client not connected."
    
    try:
        resp = CLIENT.send_command(destination, "set_beam_current", {"current": current_pa})
        return f"Set current response: {resp}"

    except Exception as e:
        return f"Error setting current: {e}"

@tool
def place_beam(x: float, y: float, destination: str = "AS") -> str:
    """
    Sets the resting beam position.
    
    Args:
        x: Normalized X position [0:1].
        y: Normalized Y position [0:1].
        destination: The server to send the command to (default 'AS').
    """
    global CLIENT
    if not CLIENT:
        return "Error: Client not connected."
    
    try:
        resp = CLIENT.send_command(destination, "place_beam", {"x": x, "y": y})
        return f"Beam move response: {resp}"
    except Exception as e:
        return f"Error placing beam: {e}"

@tool
def blank_beam(destination: str = "AS") -> str:
    """
    Blanks the electron beam.
    
    Args:
        destination: The server to send the command to (default 'AS').
    """
    global CLIENT
    if not CLIENT:
        return "Error: Client not connected."
    
    try:
        resp = CLIENT.send_command(destination, "blank_beam")
        return f"Blank beam response: {resp}"
    except Exception as e:
        return f"Error blanking beam: {e}"

@tool
def unblank_beam(duration: Optional[float] = None, destination: str = "AS") -> str:
    """
    Unblanks the electron beam.
    
    Args:
        duration: Optional dwell time in seconds. If provided, the beam will auto-blank after this time.
        destination: The server to send the command to (default 'AS').
    """
    global CLIENT
    if not CLIENT:
        return "Error: Client not connected."
    
    args = {}
    if duration is not None:
        args["duration"] = duration
    
    try:
        resp = CLIENT.send_command(destination, "unblank_beam", args)
        return f"Unblank beam response: {resp}"
    except Exception as e:
        return f"Error unblanking beam: {e}"

@tool
def get_microscope_status(destination: str = "AS") -> str:
    """
    Returns the current status of the microscope server.
    
    Args:
        destination: The server to query (default 'AS').
    """
    global CLIENT
    if not CLIENT:
        return "Error: Client not connected."
    
    try:
        return CLIENT.send_command(destination, "get_status")
    except Exception as e:
        return f"Error getting status: {e}"

@tool
def get_microscope_state(destination: str = "AS") -> Dict[str, Any]:
    """
    Returns the full state of the microscope as a dictionary of variables.
    Use this for validating constraints or checking specific values.
    
    Args:
        destination: The server to query (default 'AS').
    """
    global CLIENT
    if not CLIENT:
        return {"error": "Client not connected."}
    
    try:
        state = CLIENT.send_command(destination, "get_state")
        if isinstance(state, dict):
            return state
        # Fallback for older servers that don't have get_state
        return {"status": CLIENT.send_command(destination, "get_status")}
    except Exception as e:
        return {"error": str(e)}

@tool
def set_column_valve(state: str, destination: str = "AS") -> str:
    """
    Sets the state of the column valve.
    
    Args:
        state: "open" or "closed".
        destination: The server to send the command to (default 'AS').
    """
    global CLIENT
    if not CLIENT:
        return "Error: Client not connected."
    
    try:
        resp = CLIENT.send_command(destination, "set_microscope_status", {"parameter": "column_valve", "value": state})
        return f"Column valve command sent to {destination}: {resp}"
    except Exception as e:
        return f"Error setting column valve: {e}"

@tool
def set_optics_mode(mode: str, destination: str = "AS") -> str:
    """
    Sets the optical mode (TEM or STEM).
    
    Args:
        mode: "TEM" or "STEM".
        destination: The server to send the command to (default 'AS').
    """
    global CLIENT
    if not CLIENT:
        return "Error: Client not connected."
    
    try:
        resp = CLIENT.send_command(destination, "set_microscope_status", {"parameter": "optics_mode", "value": mode})
        return f"Optics mode command sent to {destination}: {resp}"
    except Exception as e:
        return f"Error setting optics mode: {e}"

@tool
def discover_commands(destination: str = "AS") -> str:
    """
    Discovers available commands on a microscope server. 
    None of these commands are to be used directly, only for display purposes.
    
    Args:
        destination: The server to query (default 'AS').
    """
    global CLIENT
    if not CLIENT:
        return "Error: Client not connected."
    
    try:
        cmds = CLIENT.send_command(destination, "discover_commands")
        return str(cmds)
    except Exception as e:
        return f"Error discovering commands: {e}"

@tool
def get_ceos_info() -> str:
    """
    Gets information from the CEOS server.
    """
    global CLIENT
    if not CLIENT:
        return "Error: Client not connected."
    
    try:
        return CLIENT.send_command("Ceos", "getInfo")
    except Exception as e:
        return f"Error getting CEOS info: {e}"

@tool
def tune_C1A1(destination: str = "AS") -> str:
    """
    Tunes the C1 and A1 aberrations.
    
    Args:
        destination: The server to send the command to (default 'AS').
    """
    global CLIENT
    if not CLIENT:
        return "Error: Client not connected."
    
    try:
        return CLIENT.send_command(destination, "tune_C1A1")
    except Exception as e:
        return f"Error tuning C1A1: {e}"

@tool
def acquire_tableau(tab_type: str = "Fast", angle: float = 18.0) -> Any:
    """
    Acquires a tableau from the CEOS server.
    
    Args:
        tab_type: Type of tableau (e.g., 'Fast').
        angle: Angle for the tableau.
    """
    global CLIENT
    if not CLIENT:
        return "Error: Client not connected."
    
    try:
        tableau_data = CLIENT.send_command("Ceos", "acquireTableau", {"tabType": tab_type, "angle": angle})
        return json.dumps(tableau_data)
    except Exception as e:
        return f"Error acquiring tableau: {e}"

@tool
def get_atom_count(destination: str = "AS") -> str:
    """
    Returns the current atom count monitored by the server.
    
    Args:
        destination: The server to query (default 'AS').
    """
    global CLIENT
    if not CLIENT:
        return "Error: Client not connected."
    
    try:
        return CLIENT.send_command(destination, "get_atom_count")
    except Exception as e:
        return f"Error getting atom count: {e}"

# Collection of all tools for the agent
TOOLS = [
    adjust_magnification,
    capture_image,
    close_microscope,
    start_server,
    connect_client,
    get_stage_position,
    calibrate_screen_current,
    set_beam_current,
    place_beam,
    blank_beam,
    unblank_beam,
    get_microscope_status,
    get_microscope_state,
    set_column_valve,
    set_optics_mode,
    discover_commands,
    get_ceos_info,
    tune_C1A1,
    acquire_tableau,
    get_atom_count,
]

# Workflow Framework Integration
import os
import yaml
import graphviz
from app.tools.workflow_framework import WorkflowState, WorkflowNode, WorkflowTemplate, WorkflowExecutor

class MicroscopeToolNode(WorkflowNode):
    def execute(self, state: WorkflowState, context: Optional[dict] = None) -> WorkflowState:
        tool_name = self.params.get("tool")
        tool_args = self.params.get("args", {})
        
        tool_func = next((t for t in TOOLS if getattr(t, "name", "") == tool_name), None)
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
                # Inject state into agent's python executor so it can interact with the workflow
                if hasattr(agent, "python_executor"):
                    try:
                        agent.python_executor.send_variables({"state": state})
                    except Exception:
                        pass
                
                # Command the agent to fulfill the description
                prompt = (
                    f"You are executing a Workflow task node named '{self.name}'.\\n"
                    f"Your core task is: {description}\\n\\n"
                    "The current workflow `state` object (type WorkflowState) has been injected into your python_executor local variables.\\n"
                    "Read `state.data` or perform standard tool calls to satisfy the request.\\n"
                    "If you determine new data, you can assign it like `state.data['new_key'] = val`.\\n"
                    "Provide a brief summary of what you did when you are finished."
                )
                
                # Unpause the agent!
                result = agent.run(prompt)
                
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


@tool
def design_workflow(name: str, yaml_content: str) -> str:
    """
    Designs a new experimental workflow by parsing a YAML configuration,
    validating it, and saving it as an executable .yaml file and a .png diagram.
    The AI should output the YAML content as a string.
    
    CRITICAL: You MUST use a `CodeNode` for any logic that requires iteration (like for/while loops). 
    - WRONG: Creating individual `MicroscopeTool` nodes to iterate over values (e.g. 'set_current_10', 'set_current_20').
    - RIGHT: Create a single `CodeNode` with a description that explains the loop (e.g. 'Loop over beam currents [10, 20, 30]... for each value, set current, tune, and acquire tableau').
    
    Args:
        name: Name of the workflow file (e.g., 'focus_optimization'). It will be saved as app/workflows/{name}.yaml
        yaml_content: The full YAML string defining the workflow. It must have 'name', 'description', 
                      'nodes' (list of dicts with 'id', 'type', 'params'), and 
                      'edges' (list of dicts with 'source' and 'target'). Types can be 'MicroscopeTool' 
                      (params: 'tool', 'args'), 'AIContext' (params: 'query'), 'AIQuality' (params: 'evaluate_node'),
                      or 'CodeNode' (params: 'description').
    """
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    workflows_dir = os.path.join(base_dir, "workflows")
    os.makedirs(workflows_dir, exist_ok=True)
               
    yaml_path = os.path.join(workflows_dir, f"{name}.yaml")
    png_path = os.path.join(workflows_dir, name) # Graphviz auto appends .png if told to format
    
    try:
        parsed_yaml = yaml.safe_load(yaml_content)
        # Validate through Pydantic
        template = WorkflowTemplate(**parsed_yaml)
        
        # Save yaml
        with open(yaml_path, 'w') as f:
            f.write(yaml_content)

        # Register state so external callers (LLM bridge) can detect the created workflow
        try:
            register_workflow_created(yaml_path)
        except Exception:
            pass

        # Generate Graphviz chart
        dot = graphviz.Digraph(comment=template.name)
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
            
        # Render
        dot.render(png_path, format='png', cleanup=True)
        
        return f"Successfully designed workflow! Saved YAML to {yaml_path} and diagram to {png_path}.png. Please ask the user to approve the workflow before calling execute_workflow."
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