# This script measures cold import time for torchao in two modes:
# - eager: simulates pre-change behavior by importing heavy submodules explicitly
# - lazy: imports only top-level and accesses nothing
# It runs multiple trials in fresh subprocesses for stable results.

import json
import os
import subprocess
import sys
from statistics import mean, stdev

PY = sys.executable
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

EAGER_SNIPPET = r"""
import time, importlib
start=time.time()
import torchao
# simulate old eager behavior: import heavy submodules and attributes
import torchao.quantization  # heavy
# access re-exported attributes
from torchao.quantization import autoquant, quantize_
end=time.time()
print(round(end-start,3))
"""

LAZY_SNIPPET = r"""
import time, importlib
start=time.time()
import torchao
end=time.time()
print(round(end-start,3))
"""


def run_trial(code: str) -> float:
	proc = subprocess.run([PY, "-c", code], capture_output=True, text=True, cwd=REPO_ROOT)
	if proc.returncode != 0:
		raise RuntimeError(proc.stderr)
	return float(proc.stdout.strip().splitlines()[-1])


def run_many(code: str, n: int = 7):
	times = [run_trial(code) for _ in range(n)]
	return {
		"times": times,
		"mean": round(mean(times), 4),
		"stdev": round(stdev(times), 4) if len(times) > 1 else 0.0,
	}


def main():
	res_lazy = run_many(LAZY_SNIPPET)
	res_eager = run_many(EAGER_SNIPPET)
	speedup = None
	if res_eager["mean"] > 0:
		speedup = round(res_eager["mean"] / res_lazy["mean"], 2)
	print(json.dumps({
		"lazy": res_lazy,
		"eager_simulated": res_eager,
		"speedup_x": speedup,
	}, indent=2))


if __name__ == "__main__":
	main()