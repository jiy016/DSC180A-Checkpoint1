"""
script.py
Run the experiment with parameters in config.json
"""

import asyncio
from code import run_experiment
from config import CFG

if __name__ == "__main__":
    asyncio.run(run_experiment(k=CFG["k"]))
