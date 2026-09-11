#!/usr/bin/env bash
set -euo pipefail

# These files own PT2E accelerator coverage; the remaining PT2E files are
# exercised by the CPU workflow.
pt2e_gpu_tests=(
  test/quantization/pt2e/test_quantize_pt2e_qat_with_gpu.py
  test/quantization/pt2e/test_quantize_pt2e_with_gpu.py
  test/quantization/pt2e/test_learnable_fake_quantize_with_gpu.py
)

pytest "${pt2e_gpu_tests[@]}" --verbose -s --durations=25 "$@"
