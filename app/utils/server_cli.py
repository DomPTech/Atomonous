import sys
import os

# Ensure the root directory is in sys.path
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

from app.tools.microscopy import start_server, connect_client
from app.config import settings

def show_interactive_server_commands():
    print("=" * 40)
    print("   Microscopy Server Manager CLI   ")
    print("=" * 40)
    
    # Select Mode
    print("\n--- 1. Set Environment Mode ---")
    print("1) mock (Simulation/Twin - default)")
    print("2) real (Actual Hardware)")
    mode_input = input("Enter choice (1 or 2) [1]: ").strip()
    mode = "real" if mode_input == "2" else "mock"
    
    # Configure Network/Ports
    print("\n--- 2. Configuration Overrides ---")
    print("Press Enter to keep current default value.")
    
    as_port = input(f"AutoScript Port [{settings.autoscript_port}]: ").strip()
    if as_port.isdigit():
        settings.autoscript_port = int(as_port)
        
    inst_host = input(f"Instrument Host [{settings.instrument_host}]: ").strip()
    if inst_host:
        settings.instrument_host = inst_host
        
    inst_port = input(f"Instrument Port [{settings.instrument_port}]: ").strip()
    if inst_port.isdigit():
        settings.instrument_port = int(inst_port)

    # Start Servers
    print("\n--- 3. Starting Servers ---")
    
    print(f"Mode: {mode}")
    print("Target Devices: microscope + haadf")
    print(f"AutoScript Port: {settings.autoscript_port}")
    print(f"Instrument Host: {settings.instrument_host}:{settings.instrument_port}")
    print("-" * 40)
    
    # Call the start_server tool from app.tools.microscopy
    try:
        result = start_server(mode=mode)
        print("\n[RESULT]")
        print(result)

        # Verify connectivity and routing
        print("\n--- 4. Verifying Connectivity ---")
        conn_result = connect_client()
        if "Failed" in conn_result:
            print(f"[CONNECTION ERROR] {conn_result}")
        else:
            print(f"[CONNECTION SUCCESS] {conn_result}")
    except Exception as e:
        print("\n[ERROR]")
        print(f"Failed to start servers: {e}")
        
    print("=" * 40)

def main():
    try:
        while True:
            show_interactive_server_commands()
            
            cont = input("\nDo you want to run another command? (y/N): ").strip().lower()
            if cont != 'y':
                print("Exiting.")
                break
    except KeyboardInterrupt:
        print("\nProcess interrupted by user. Exiting.")
        sys.exit(0)

if __name__ == "__main__":
    main()
