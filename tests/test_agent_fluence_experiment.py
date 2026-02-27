import sys
import os
import time

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from app.config import settings
settings.hf_cache_dir = "/lustre/isaac24/scratch/dpelaia/hf_cache/"
from app.agent.core import Agent
from app.tools.microscopy import *

settings.instrument_host = "10.46.217.241"
settings.instrument_port = 9095
settings.autoscript_port = 9091

start_server(mode="mock")
time.sleep(1)
connect_client()
time.sleep(1)

prompt = '''
Please run the following experiment on the real microscope (use real servers):
The AS server is already running, no need to start it. Only start CEOs and Central.

1) Calibrate screen current.
2) For the following beam current values (pA): [10, 30, 50, 80, 100, 200, 500, 1000, 5000, 10000] (just do the first four)
   - Set the beam current on AS using `set_beam_current`.
   - Re-focus using `tune_C1A1`.
   - Acquire the tableau from Ceos with these arguments: {'tabType':"Fast", 'angle':18}.
   - To get the aberrations from the tableau (which is a dictionary):
      - get ['results']['aberrations'].
   - Compute the probe from the returned aberrations and store results in a list.
   - Break after the first iteration for a quick test run.
3) Capture one image from AS (size 512) and save it.

Return a short summary of actions and any saved file paths.
'''

agent = Agent(model_id="Qwen/Qwen2.5-Coder-14B-Instruct")
print(agent.chat(prompt))
