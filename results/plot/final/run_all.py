#!/usr/bin/env python3
"""run_all.py -- regenerate every CSV and PDF of the final paper figures (Exp. 1-3)."""
import runpy
from pathlib import Path

HERE = Path(__file__).resolve().parent
for script in ["exp1_latency.py", "exp1_energy.py", "exp1_network.py", "exp2_latency.py", "exp2_energy.py",
               "exp3_latency.py"]:
    print(f"\n=== {script}")
    runpy.run_path(str(HERE / script), run_name="__main__")
