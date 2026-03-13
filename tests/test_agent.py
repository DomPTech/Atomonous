import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from app.config import settings
settings.hf_cache_dir = "/lustre/isaac24/scratch/dpelaia/hf_cache/"

from app.agent.core import Agent
from app.tools.microscopy import close_microscope, connect_client, start_server


def _require_success(result: str, expected_substrings: tuple[str, ...], step_name: str) -> str:
    if not any(substring in result for substring in expected_substrings):
        raise RuntimeError(f"{step_name} failed: {result}")
    return result

def main():
    try:
        start_result = start_server(mode="real")
        print(start_result)
        _require_success(
            start_result,
            (
                "Started PyTango asyncroscopy servers",
                "PyTango asyncroscopy servers already running",
            ),
            "Server startup",
        )

        connect_result = connect_client()
        print(connect_result)
        _require_success(
            connect_result,
            ("Connected to PyTango microscope",),
            "Microscope connection",
        )

        agent = Agent(model_id="Qwen/Qwen2.5-Coder-32B-Instruct")

        prompt = """
        The microscope servers are already started in real mode and the microscope client is already connected.

        Please perform this sequence exactly:
        1. Acquire one HAADF image.
        2. Acquire one EDS spectrum using reasonable default acquisition settings.
        3. Return a short summary that includes the saved file path for the image and the saved file path for the EDS spectrum, plus any basic acquisition statistics reported by the tools.
        4. Close the microscope when finished.

        Do not say the acquisition succeeded unless the tool output explicitly confirms success and includes saved artifact paths.
        """

        response = agent.chat(prompt)
        print(response)
    finally:
        print(close_microscope())

if __name__ == "__main__":
    main()