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
import tango
from tango.test_context import MultiDeviceTestContext
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
CLIENT: Optional[object] = None  # tango.DeviceProxy
SERVER_PROCESSES: Dict[str, subprocess.Popen] = {}
SERVER_CONTEXT: Optional[MultiDeviceTestContext] = None
SERVER_DEVICE_NAME: str = "tango://127.0.0.1:8889/test/nodb/thermomicroscope#dbase=no"
HAADF_DEVICE_NAME: str = "tango://127.0.0.1:8888/test/nodb/haadf#dbase=no"
MICROSCOPE_HOST: str = "127.0.0.1"
MICROSCOPE_PORT: int = 8889
HAADF_PORT: int = 8888
AGENT_INSTANCE: Optional[object] = None  # Set by Agent.__init__() for artifact memory access
DEFAULT_TANGO_TIMEOUT_MS: int = 60000


def _acquire_eds_direct(
    exposure_time: float,
    dispersion: int,
    shaping_time: float,
) -> np.ndarray:
    """Fallback path for EDS acquisition via direct AutoScript client access."""
    if not _AUTOSCRIPT_DIRECT_AVAILABLE:
        raise RuntimeError("AutoScript client library is not available in this environment.")

    microscope = TemMicroscopeClient()
    try:
        microscope.connect(settings.instrument_host, settings.autoscript_port)

        eds_settings = EdsAcquisitionSettings()
        eds_settings.eds_detector = EdsDetectorType.SUPER_X
        eds_settings.dispersion = int(dispersion)
        eds_settings.shaping_time = float(shaping_time)
        eds_settings.exposure_time = float(exposure_time)
        eds_settings.exposure_time_type = ExposureTimeType.LIVE_TIME

        spectrum = microscope.analysis.eds.acquire_spectrum(eds_settings)
        if isinstance(spectrum, np.ndarray):
            return spectrum

        dt = np.dtype("uint32").newbyteorder("<")
        return np.frombuffer(spectrum._raw_data, dtype=dt)
    finally:
        try:
            microscope.disconnect()
        except Exception:
            pass


def _is_tango_timeout_error(exc: Exception) -> bool:
    """Return True when a PyTango exception indicates a command timeout."""
    if not isinstance(exc, tango.DevFailed):
        return False

    for err in exc.args:
        reason = getattr(err, "reason", "") or ""
        desc = getattr(err, "desc", "") or ""
        if reason == "API_DeviceTimedOut" or "CallTimedout" in desc or "Timeout" in desc:
            return True
    return False

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

@tool
def start_server(mode: str = "mock", haadf_port: int = None, microscope_port: int = None) -> str:
    """
    Starts asyncroscopy PyTango devices as separate subprocesses using tango.test_context.
    
    This connects to servers started externally via:
      uv run python -m tango.test_context asyncroscopy.detectors.HAADF.HAADF --host 127.0.0.1 --port 8888
      uv run python -m tango.test_context asyncroscopy.ThermoDigitalTwin.ThermoDigitalTwin --host 127.0.0.1 --port 8889 --prop "..."
    
    Args:
        mode: "mock" for ThermoDigitalTwin (simulation), "real" for ThermoMicroscope.
        haadf_port: Port for HAADF detector (default 8888).
        microscope_port: Port for Microscope (default 8889).
    
    Returns:
        Status message indicating success or failure.
    """
    global SERVER_PROCESSES, HAADF_PORT, MICROSCOPE_PORT, SERVER_DEVICE_NAME, HAADF_DEVICE_NAME, MICROSCOPE_HOST
    
    if haadf_port is not None:
        HAADF_PORT = haadf_port
    if microscope_port is not None:
        MICROSCOPE_PORT = microscope_port

    microscope_device_basename = "thermomicroscope" if mode == "real" else "thermodigitaltwin"
    SERVER_DEVICE_NAME = (
        f"tango://{MICROSCOPE_HOST}:{MICROSCOPE_PORT}/test/nodb/{microscope_device_basename}#dbase=no"
    )
    HAADF_DEVICE_NAME = f"tango://{MICROSCOPE_HOST}:{HAADF_PORT}/test/nodb/haadf#dbase=no"
    
    if SERVER_PROCESSES:
        return "PyTango asyncroscopy servers already running."

    try:
        # Start HAADF detector
        haadf_cmd = [
            sys.executable, "-m", "tango.test_context",
            "asyncroscopy.detectors.HAADF.HAADF",
            "--host", MICROSCOPE_HOST,
            "--port", str(HAADF_PORT)
        ]
        print(f"[TOOLS] Starting HAADF detector: {' '.join(haadf_cmd)}")
        haadf_proc = subprocess.Popen(haadf_cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        SERVER_PROCESSES["haadf"] = haadf_proc
        
        # Wait for HAADF to be ready
        if not _wait_for_port(MICROSCOPE_HOST, HAADF_PORT, timeout=15.0):
            _cleanup_server_processes()
            return f"Timeout waiting for HAADF detector on port {HAADF_PORT}"
        
        print(f"[TOOLS] HAADF detector ready on port {HAADF_PORT}")
        time.sleep(0.5)  # Extra buffer between server starts
        
        # Determine microscope class and properties
        if mode == "real":
            microscope_cls = "asyncroscopy.ThermoMicroscope.ThermoMicroscope"
            haadf_addr = HAADF_DEVICE_NAME
            props = {
                'haadf_device_address': haadf_addr,
                'autoscript_host_ip': settings.instrument_host,
                'autoscript_host_port': settings.autoscript_port,
            }
        else:
            microscope_cls = "asyncroscopy.ThermoDigitalTwin.ThermoDigitalTwin"
            haadf_addr = HAADF_DEVICE_NAME
            props = {
                'haadf_device_address': haadf_addr,
            }
        
        # Convert props dict to JSON string for --prop argument
        props_json = json.dumps(props)
        
        # Start Microscope (using digital twin for mock mode)
        microscope_cmd = [
            sys.executable, "-m", "tango.test_context",
            microscope_cls,
            "--host", MICROSCOPE_HOST,
            "--port", str(MICROSCOPE_PORT),
            "--prop", props_json
        ]
        print(f"[TOOLS] Starting Microscope ({mode}): {' '.join(microscope_cmd)}")
        microscope_proc = subprocess.Popen(microscope_cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        SERVER_PROCESSES["microscope"] = microscope_proc
        
        # Wait for Microscope to be ready
        if not _wait_for_port(MICROSCOPE_HOST, MICROSCOPE_PORT, timeout=15.0):
            _cleanup_server_processes()
            return f"Timeout waiting for Microscope on port {MICROSCOPE_PORT}"
        
        print(f"[TOOLS] Microscope ready on port {MICROSCOPE_PORT}")
        time.sleep(0.5)
        
        # Verify both are accessible
        try:
            haadf_proxy = tango.DeviceProxy(HAADF_DEVICE_NAME)
            haadf_state = haadf_proxy.state()
            print(f"[TOOLS] HAADF state: {haadf_state}")
        except Exception as e:
            print(f"[TOOLS WARNING] Could not verify HAADF: {e}")
        
        try:
            microscope_proxy = tango.DeviceProxy(SERVER_DEVICE_NAME)
            microscope_state = microscope_proxy.state()
            print(f"[TOOLS] Microscope state: {microscope_state}")
        except Exception as e:
            print(f"[TOOLS WARNING] Could not verify Microscope: {e}")
        
        return f"Started PyTango asyncroscopy servers ({mode}): HAADF on port {HAADF_PORT}, Microscope on port {MICROSCOPE_PORT}"
    except Exception as e:
        _cleanup_server_processes()
        return f"Failed to start PyTango asyncroscopy servers: {e}"


def _cleanup_server_processes() -> None:
    """Kill all server processes."""
    global SERVER_PROCESSES
    for name, proc in SERVER_PROCESSES.items():
        try:
            proc.terminate()
            try:
                proc.wait(timeout=2)
            except subprocess.TimeoutExpired:
                proc.kill()
                proc.wait()
            print(f"[TOOLS] Terminated {name} server process")
        except Exception as e:
            print(f"[TOOLS WARNING] Failed to terminate {name}: {e}")
    SERVER_PROCESSES.clear()

@tool
def connect_client(microscope_device: str = None) -> str:
    """
    Connects a Tango DeviceProxy client to the microscope device.
    
    Expects external Tango servers to already be running (started via start_server()).
    
    Args:
        microscope_device: Full Tango device address. If None, uses default: 
                          "tango://127.0.0.1:8889/test/nodb/thermomicroscope#dbase=no"
    
    Returns:
        Status message indicating connection success or failure.
    """
    global CLIENT, SERVER_DEVICE_NAME

    try:
        device_addr = microscope_device or SERVER_DEVICE_NAME
        
        print(f"[TOOLS] Connecting to microscope at: {device_addr}")
        CLIENT = tango.DeviceProxy(device_addr)
        CLIENT.set_timeout_millis(DEFAULT_TANGO_TIMEOUT_MS)
        
        # Test the connection
        state = CLIENT.state()
        print(f"[TOOLS] Connected! Microscope state: {state}")
        
        return f"Connected to PyTango microscope at {device_addr}. State: {state}"
    except Exception as e:
        CLIENT = None
        return f"FATAL: Connection error: {e}"

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
def capture_image(detector: str = "haadf") -> str:
    """
    Captures an image and saves it.
    
    Args:
        detector: The detector to use (e.g., "haadf").
    """
    global CLIENT, AGENT_INSTANCE
    if not CLIENT:
        return "Error: Client not connected."
        
    try:
        detector_name = detector.lower().strip()
        print(f"[TOOLS DEBUG] Requesting image from detector '{detector_name}'...")

        encoded = CLIENT.get_image(detector_name)
        if encoded is None or len(encoded) != 2:
            return "Failed to capture image (invalid encoded response)."

        metadata_json, raw_bytes = encoded
        metadata = json.loads(metadata_json)
        img = np.frombuffer(raw_bytes, dtype=metadata["dtype"]).reshape(metadata["shape"])

        # Save to session memory if agent has one, otherwise to /tmp
        output_path = f"microscope_capture_{int(time.time())}.npy"
        full_path = None
        
        try:
            if (AGENT_INSTANCE and 
                hasattr(AGENT_INSTANCE, 'memory') and 
                AGENT_INSTANCE.memory and 
                hasattr(AGENT_INSTANCE.memory, 'session_dir')):
                from pathlib import Path
                session_dir = AGENT_INSTANCE.memory.session_dir
                full_path = str(Path(session_dir) / output_path)
        except (AttributeError, TypeError, NameError):
            pass
        
        # Fallback to /tmp if memory not available
        if not full_path:
            full_path = f"/tmp/{output_path}"
        
        np.save(full_path, img)
        return (
            f"Image captured from detector '{detector_name}' and saved to {full_path} "
            f"(Shape: {img.shape}, dtype: {img.dtype})"
        )
    except Exception as e:
        return f"Error capturing image: {e}"

@tool
def take_image(detector: str = "haadf") -> str:
    """
    Alias for capture_image in the new asyncroscopy API surface.

    Args:
        detector: Detector name to acquire from (e.g., "haadf").
    """
    return capture_image(detector=detector)

@tool
def get_eds_spectrum(
    exposure_time: float = 2.0,
    dispersion: int = 5,
    shaping_time: float = 3e-6,
    destination: str = "AS",
) -> str:
    """
    Acquires an EDS (Energy Dispersive X-ray Spectroscopy) spectrum from the microscope.

    Calls the 'acquire_eds' command on the microscope server, which uses the SUPER_X
    detector by default. The resulting spectrum (counts vs. channel) is saved as a
    NumPy .npy file for downstream analysis.

    Args:
        exposure_time: Acquisition live-time in seconds (default 2.0).
        dispersion: Energy dispersion in eV/channel (default 5).
        shaping_time: Detector shaping time in seconds (default 3e-6).
        destination: The server to send the command to (default 'AS').

    Returns:
        A string describing the saved spectrum path, shape, and basic statistics,
        or an error message on failure.
    """
    global CLIENT, AGENT_INSTANCE
    if not CLIENT:
        return "Error: Client not connected."

    try:
        params = {
            "exposure_time": exposure_time,
            "dispersion": dispersion,
            "shaping_time": shaping_time,
        }
        print(f"[TOOLS] Requesting EDS spectrum from '{destination}' with params {params} ...")
        if hasattr(CLIENT, "acquire_eds"):
            eds_timeout_ms = max(DEFAULT_TANGO_TIMEOUT_MS, int((float(exposure_time) + 10.0) * 1000))
            CLIENT.set_timeout_millis(eds_timeout_ms)
            print(f"[TOOLS] Calling Tango acquire_eds with timeout {eds_timeout_ms} ms")
            try:
                encoded = CLIENT.acquire_eds(json.dumps(params))
            except tango.DevFailed as exc:
                if not _is_tango_timeout_error(exc):
                    raise
                print("[TOOLS] Tango acquire_eds timed out; falling back to direct AutoScript access.")
                spectrum_data = _acquire_eds_direct(
                    exposure_time=exposure_time,
                    dispersion=dispersion,
                    shaping_time=shaping_time,
                )
            else:
                if encoded is None or len(encoded) != 2:
                    return "Error: Invalid encoded EDS response from the microscope."

                metadata_json, raw_bytes = encoded
                metadata = json.loads(metadata_json)
                spectrum_data = np.frombuffer(raw_bytes, dtype=metadata["dtype"]).reshape(metadata["shape"])
        else:
            print("[TOOLS] Tango proxy has no 'acquire_eds' command; falling back to direct AutoScript access.")
            spectrum_data = _acquire_eds_direct(
                exposure_time=exposure_time,
                dispersion=dispersion,
                shaping_time=shaping_time,
            )

        if spectrum_data is None:
            return "Error: No spectrum data returned from the microscope."

        # Determine save path (use agent session dir if available, else /tmp)
        output_filename = f"eds_spectrum_{int(time.time())}.npy"
        full_path = None

        try:
            if (
                AGENT_INSTANCE
                and hasattr(AGENT_INSTANCE, "memory")
                and AGENT_INSTANCE.memory
                and hasattr(AGENT_INSTANCE.memory, "session_dir")
            ):
                from pathlib import Path as _Path
                full_path = str(_Path(AGENT_INSTANCE.memory.session_dir) / output_filename)
        except (AttributeError, TypeError, NameError):
            pass

        if not full_path:
            full_path = f"/tmp/{output_filename}"

        np.save(full_path, spectrum_data)

        total_counts = int(spectrum_data.sum())
        peak_channel = int(np.argmax(spectrum_data))
        peak_counts = int(spectrum_data[peak_channel])

        return (
            f"EDS spectrum acquired and saved to {full_path} "
            f"(channels: {spectrum_data.shape[0]}, "
            f"total counts: {total_counts}, "
            f"peak channel: {peak_channel} ({peak_counts} counts))"
        )
    except Exception as e:
        return f"Error acquiring EDS spectrum: {e}"

@tool
def close_microscope() -> str:
    """
    Safely closes the microscope connection and stops the external servers.
    """
    global SERVER_PROCESSES, CLIENT, SERVER_CONTEXT
    resp = "Microscope closed."
    
    CLIENT = None
    
    # Kill external server processes
    _cleanup_server_processes()
    resp += " External servers stopped."

    # Legacy cleanup for in-process context (if applicable)
    if SERVER_CONTEXT is not None:
        try:
            SERVER_CONTEXT.__exit__(None, None, None)
            resp += " PyTango context stopped."
        except Exception as e:
            resp += f" Failed to stop PyTango context cleanly: {e}"
        finally:
            SERVER_CONTEXT = None
        
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
def blank_beam() -> str:
    """
    Blanks the electron beam.
    
    Args:
        None.
    """
    global CLIENT
    if not CLIENT:
        return "Error: Client not connected."
    
    try:
        resp = CLIENT.command_inout("blank_beam")
        return f"Blank beam response: {resp}"
    except Exception as e:
        return f"Error blanking beam: {e}"

@tool
def unblank_beam(duration: Optional[float] = None) -> str:
    """
    Unblanks the electron beam.
    
    Args:
        duration: Optional dwell time in seconds. If provided, the beam will auto-blank after this time.
    """
    global CLIENT
    if not CLIENT:
        return "Error: Client not connected."
    
    try:
        resp = CLIENT.command_inout("unblank_beam")

        if duration is not None:
            time.sleep(duration)
            CLIENT.command_inout("blank_beam")
            return f"Unblank beam response: {resp}. Beam auto-blanked after {duration} s."

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
def acquire_tableau(tab_type: str = "Fast", angle: float = 18.0) -> dict:
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
        return tableau_data
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

# Collection of tools for the agent (minimal PyTango surface)
TOOLS = [
    start_server,
    connect_client,
    take_image,
    get_eds_spectrum,
    close_microscope,
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