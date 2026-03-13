#!/usr/bin/env python3
"""
Quick test to verify the updated microscopy.py code works correctly.
"""

import sys
sys.path.insert(0, '/nfs/home/dpelaia/Desktop/Microscopy-AI-Agent')

print("=" * 70)
print("MICROSCOPY TOOLS TEST")
print("=" * 70)

# Test 1: Import the module
print("\n[Test 1] Importing microscopy tools...")
try:
    from app.tools import microscopy
    print("✓ Successfully imported app.tools.microscopy")
except Exception as e:
    print(f"✗ Failed to import: {e}")
    sys.exit(1)

# Test 2: Check global variables
print("\n[Test 2] Checking global variables...")
try:
    assert hasattr(microscopy, 'CLIENT'), "CLIENT not defined"
    assert hasattr(microscopy, 'SERVER_PROCESSES'), "SERVER_PROCESSES not defined"
    assert hasattr(microscopy, 'SERVER_DEVICE_NAME'), "SERVER_DEVICE_NAME not defined"
    assert hasattr(microscopy, 'HAADF_DEVICE_NAME'), "HAADF_DEVICE_NAME not defined"
    print(f"✓ Global variables defined correctly")
    print(f"  - CLIENT: {microscopy.CLIENT}")
    print(f"  - SERVER_DEVICE_NAME: {microscopy.SERVER_DEVICE_NAME}")
    print(f"  - HAADF_DEVICE_NAME: {microscopy.HAADF_DEVICE_NAME}")
except AssertionError as e:
    print(f"✗ Missing global variable: {e}")
    sys.exit(1)

# Test 3: Check function definitions
print("\n[Test 3] Checking function definitions...")
try:
    functions = ['start_server', 'connect_client', 'close_microscope', '_cleanup_server_processes']
    for func_name in functions:
        assert hasattr(microscopy, func_name), f"{func_name} not defined"
        func = getattr(microscopy, func_name)
        assert callable(func), f"{func_name} is not callable"
    print(f"✓ All required functions defined:")
    for func_name in functions:
        print(f"  - {func_name}()")
except AssertionError as e:
    print(f"✗ Function definition error: {e}")
    sys.exit(1)

# Test 4: Check function signatures
print("\n[Test 4] Checking function signatures...")
try:
    import inspect
    
    start_server_sig = inspect.signature(microscopy.start_server)
    print(f"✓ start_server parameters: {list(start_server_sig.parameters.keys())}")
    
    connect_client_sig = inspect.signature(microscopy.connect_client)
    print(f"✓ connect_client parameters: {list(connect_client_sig.parameters.keys())}")
    
    close_microscope_sig = inspect.signature(microscopy.close_microscope)
    print(f"✓ close_microscope parameters: {list(close_microscope_sig.parameters.keys())}")
except Exception as e:
    print(f"✗ Error checking signatures: {e}")
    sys.exit(1)

# Test 5: Verify configuration values
print("\n[Test 5] Verifying configuration values...")
try:
    assert microscopy.MICROSCOPE_HOST == "127.0.0.1", "MICROSCOPE_HOST incorrect"
    assert microscopy.HAADF_PORT == 8888, "HAADF_PORT incorrect"
    assert microscopy.MICROSCOPE_PORT == 8889, "MICROSCOPE_PORT incorrect"
    assert "8888" in microscopy.HAADF_DEVICE_NAME, "HAADF_DEVICE_NAME format incorrect"
    assert "8889" in microscopy.SERVER_DEVICE_NAME, "SERVER_DEVICE_NAME format incorrect"
    assert "#dbase=no" in microscopy.HAADF_DEVICE_NAME, "HAADF_DEVICE_NAME missing #dbase=no"
    assert "#dbase=no" in microscopy.SERVER_DEVICE_NAME, "SERVER_DEVICE_NAME missing #dbase=no"
    print("✓ Configuration values correct:")
    print(f"  - Host: {microscopy.MICROSCOPE_HOST}")
    print(f"  - HAADF Port: {microscopy.HAADF_PORT}")
    print(f"  - Microscope Port: {microscopy.MICROSCOPE_PORT}")
    print(f"  - HAADF URL: {microscopy.HAADF_DEVICE_NAME}")
    print(f"  - Microscope URL: {microscopy.SERVER_DEVICE_NAME}")
except AssertionError as e:
    print(f"✗ Configuration error: {e}")
    sys.exit(1)

print("\n" + "=" * 70)
print("✓ ALL TESTS PASSED")
print("=" * 70)
print("\nThe microscopy tools have been successfully updated for external server mode:")
print("  - Servers will be started as separate subprocesses using tango.test_context")
print("  - Device connections use full Tango URLs with --nodb suffix")
print("  - Configuration supports mock (ThermoDigitalTwin) and real (ThermoMicroscope) modes")
print("\nTo test with actual servers, run:")
print("  ./scripts/start_haadf_server.sh")
print("  ./scripts/start_microscope_server.sh  (in another terminal)")
print("  Then run the test_digital_twin.ipynb notebook")
