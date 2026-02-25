import time
import os
import sys
import pytest
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from app.tools.microscopy import *
from app.config import settings

@pytest.fixture(scope="module", autouse=True)
def microscope_setup():
    """Setup and teardown for microscope tests."""
    settings.instrument_host = "localhost"
    settings.instrument_port = 9001
    settings.autoscript_port = 9001
    
    print("\n--- Initializing Microscope Test Environment ---")
    start_res = start_server(mode="mock", servers=[MicroscopeServer.Central, "MicroscopeServer.AS", "Ceos"])
    print(f"Start Server: {start_res}")
    
    time.sleep(1) # Give servers time to bind
    
    conn_res = connect_client(host="localhost")
    print(f"Connect Client: {conn_res}")
    
    yield
    
    print("\n--- Tearing Down Microscope Test Environment ---")
    close_microscope()

def test_get_status():
    print("\n--- Testing: get_microscope_status ---")
    res = get_microscope_status()
    print(f"Result: {res}")
    assert "Microscope is Ready" in res

def test_get_stage_position():
    print("\n--- Testing: get_stage_position ---")
    res = get_stage_position()
    print(f"Result: {res}")
    assert "Stage Position from AS" in res

def test_adjust_magnification():
    print("\n--- Testing: adjust_magnification ---")
    res = adjust_magnification(5000.0)
    print(f"Result: {res}")
    assert "Magnification command sent to AS" in res
    assert "5000.0x" in res

def test_capture_image():
    print("\n--- Testing: capture_image ---")
    res = capture_image(detector="Ceta")
    print(f"Result: {res}")
    assert "Image captured from AS" in res
    assert ".npy" in res
    if "saved to " in res:
        path = res.split("saved to ")[1].split(" (Shape")[0]
        if os.path.exists(path):
            os.remove(path)

def test_calibrate_screen_current():
    print("\n--- Testing: calibrate_screen_current ---")
    res = calibrate_screen_current()
    print(f"Result: {res}")
    assert "Screen current calibration" in res
    assert "calibrated" in res.lower()

def test_set_beam_current():
    print("\n--- Testing: set_beam_current ---")
    res = set_beam_current(150.0)
    print(f"Result: {res}")
    assert "Set current response" in res
    assert "150.0 pA" in res

def test_place_beam():
    print("\n--- Testing: place_beam ---")
    res = place_beam(0.2, 0.8)
    print(f"Result: {res}")
    assert "Beam move response" in res
    assert "0.2" in res
    assert "0.8" in res

def test_blank_beam():
    print("\n--- Testing: blank_beam ---")
    res = blank_beam()
    print(f"Result: {res}")
    assert "Blank beam response" in res
    assert "blanked" in res.lower()

def test_unblank_beam_fixed_duration():
    print("\n--- Testing: unblank_beam (fixed duration) ---")
    res = unblank_beam(duration=0.5)
    print(f"Result: {res}")
    assert "Unblank beam response" in res
    assert "0.5s" in res

def test_unblank_beam_continuous():
    print("\n--- Testing: unblank_beam (continuous) ---")
    res = unblank_beam()
    print(f"Result: {res}")
    assert "Unblank beam response" in res
    assert "unblanked" in res.lower()

def test_get_state():
    print("\n--- Testing: get_microscope_state ---")
    state = get_microscope_state()
    print(f"Result: {state}")
    assert isinstance(state, dict)
    assert "status" in state
    assert "magnification" in state
    assert "beam_blanked" in state
    assert "column_valve_open" in state
    assert "optics_mode" in state

def test_set_column_valve():
    print("\n--- Testing: set_column_valve ---")
    res = set_column_valve("open")
    print(f"Result: {res}")
    assert "Column valve command sent to AS" in res
    assert "open" in res
    
    state = get_microscope_state()
    assert state["column_valve_open"] is True
    
    res = set_column_valve("closed")
    print(f"Result: {res}")
    assert "closed" in res
    state = get_microscope_state()
    assert state["column_valve_open"] is False

def test_set_optics_mode():
    print("\n--- Testing: set_optics_mode ---")
    res = set_optics_mode("STEM")
    print(f"Result: {res}")
    assert "Optics mode command sent to AS" in res
    assert "STEM" in res
    
    state = get_microscope_state()
    assert state["optics_mode"] == "STEM"
    
    res = set_optics_mode("TEM")
    print(f"Result: {res}")
    assert "TEM" in res
    state = get_microscope_state()
    assert state["optics_mode"] == "TEM"

def test_submit_experiment_with_constraints():
    print("\n--- Testing: submit_experiment with constraints ---")
    
    # First set a known state
    adjust_magnification(5000.0)
    
    experiment = {
        "id": "exp_test_001",
        "description": "Test constraints",
        "actions": [
            {"name": "capture_image", "params": {"detector": "Ceta"}}
        ],
        "constraints": [
            {"parameter": "magnification", "min_value": 2000.0}
        ],
        "observables": ["image"],
        "reward": {"metric_type": "image_entropy"}
    }
    
    res = submit_experiment(experiment)
    print(f"Result: {res}")
    assert "completed" in res
    assert "Success: True" in res

def test_submit_experiment_constraint_violation():
    print("\n--- Testing: submit_experiment constraint violation ---")
    
    # Set low magnification
    adjust_magnification(100.0)
    
    experiment = {
        "id": "exp_test_fail",
        "description": "Test constraint failure",
        "actions": [
            {"name": "capture_image", "params": {"detector": "Ceta"}}
        ],
        "constraints": [
            {"parameter": "magnification", "min_value": 5000.0}
        ],
        "observables": ["image"],
        "reward": {"metric_type": "image_entropy"}
    }
    
    res = submit_experiment(experiment)
    print(f"Result: {res}")
    assert "rejected due to constraints" in res

if __name__ == "__main__":
    # Fallback for running without pytest
    pytest.main([__file__])
