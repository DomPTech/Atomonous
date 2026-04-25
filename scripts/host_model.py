import socket
import subprocess
import shutil

from typing import Any

from atomonous import settings
import sys


def prompt_with_default(label: str, default: Any) -> str:
    raw = input(f"{label} [{default}]: ").strip()
    return raw if raw else default

def prompt_int(label: str, default: int) -> int:
    while True:
        raw = input(f"{label} [{default}]: ").strip()
        if not raw:
            return default
        try:
            return int(raw)
        except ValueError:
            print("Please enter a valid integer.")

def prompt_bool(label: str, default: bool = False) -> bool:
    default_hint = "Y/n" if default else "y/N"
    while True:
        raw = input(f"{label} [{default_hint}]: ").strip().lower()
        if not raw:
            return default
        if raw in {"y", "yes"}:
            return True
        if raw in {"n", "no"}:
            return False
        print("Please answer with 'y' or 'n'.")

def get_local_ip():
    try:
        # Create a dummy socket to find the preferred local IP
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("8.8.8.8", 80))
        ip = s.getsockname()[0]
        s.close()
        return ip
    except Exception:
        return "127.0.0.1"

def main():
    cli_path = shutil.which("transformers")
    preflight_cmd = [
        sys.executable,
        "-m",
        "transformers.commands.transformers_cli",
        "serve",
        "--help",
    ]
    preflight_result = subprocess.run(preflight_cmd, capture_output=True, text=True)
    if preflight_result.returncode != 0:
        print("Error: Transformers Serve CLI is unavailable or serving extras are missing.")
        print("Please install them with: pip install 'transformers[serving]'")
        if preflight_result.stderr.strip():
            print("Details:")
            print(preflight_result.stderr.strip())
        sys.exit(1)
    if cli_path is None:
        print("Warning: 'transformers' executable was not found on PATH.")
        print("Using Python module invocation for serving instead.")

    print("\nProvide serve settings (press Enter to keep defaults).\n")
    model = prompt_with_default("Model ID or path", "Qwen/Qwen2.5-7B-Instruct")
    
    hf_cache_dir = prompt_with_default("Hugging Face cache directory", settings.hf_cache_dir)
    settings.hf_cache_dir = hf_cache_dir  # Update global settings with user input
    
    host = prompt_with_default("Host interface", "0.0.0.0")
    port = prompt_int("Port", 8095)
    dtype = prompt_with_default("Dtype", "auto")
    model_timeout = prompt_int("Model timeout in seconds (-1 disables unload)", 300)
    continuous_batching = prompt_bool("Enable continuous batching", default=False)

    local_ip = get_local_ip()
    
    print("\n" + "="*60)
    print("STARTING MODEL SERVER")
    print("="*60)
    print(f"Model: {model}")
    print(f"Host: {host}")
    print(f"Local IP: {local_ip}")
    print(f"Port: {port}")
    print("-" * 60)
    print("\n🔗 TO CONNECT FROM YOUR LOCAL MACHINE:")
    print("Initialize your Agent with the following Python code:\n")
    print("  from atomonous import Agent\n")
    print("  agent = Agent.from_api_key(")
    print(f'      model_id="openai/{model}",')
    print(f'      api_base="http://{local_ip}:{port}/v1",')
    print('      api_key="not-needed"  # local transformers serve endpoint')
    print("  )")
    print("\nNote: If hosting on a remote supercomputer, use an SSH tunnel")
    print(f"and replace '{local_ip}' with 'localhost'.")
    print("="*60 + "\n")

    cmd = [
        sys.executable,
        "-m",
        "transformers.commands.transformers_cli",
        "serve",
        "--host", host,
        "--port", str(port),
        "--dtype", dtype,
        "--model-timeout", str(model_timeout),
        "--force-model", model,
    ]

    if continuous_batching:
        cmd.append("--continuous-batching")

    cmd.extend([
        "--trust-remote-code",
    ])
    
    try:
        subprocess.run(cmd)
    except KeyboardInterrupt:
        print("\nShutting down server...")

if __name__ == "__main__":
    main()
