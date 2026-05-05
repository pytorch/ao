#!/bin/bash
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD 3-Clause license found in the
# LICENSE file in the root directory of this source tree.

# Run all low-precision attention benchmarks for FA4.
#   Section 1: FA2 BF16 vs FA4 BF16
#   Section 2: FA4 BF16 vs FA4 FP8
# Usage: bash benchmarks/prototype/attention/run_all_benchmarks_fa4.sh

set -euo pipefail

BENCH_DIR="benchmarks/prototype/attention"
SEQ_LENGTHS="1024 2048 4096 8192 16384 32768 65536 131072"

echo "================================================================"
echo "  Low-Precision Attention Benchmarks (FA4 BF16 vs FA4 FP8)"
echo "================================================================"

echo ""
echo "================================================================"
echo "  [1/3] eval_flux_model.py — FA4 vs FA4 FP8, compile, 2048x2048"
echo "================================================================"
python "$BENCH_DIR/eval_flux_model.py" --baseline fa4 --test fa4_fp8 --compile

echo ""
echo "================================================================"
echo "  [2/3] benchmark_sdpa.py — FA4 vs FA4 FP8"
echo "================================================================"
python "$BENCH_DIR/benchmark_sdpa.py" --baseline fa4 --test fa4_fp8

echo ""
echo "================================================================"
echo "  [3/3] eval_llama3_model.py — FA4 vs FA4 FP8, compile"
echo "================================================================"
python "$BENCH_DIR/eval_llama3_model.py" --baseline fa4 --test fa4_fp8 --compile --seq_lengths $SEQ_LENGTHS

echo ""
echo "================================================================"
echo "  All 3 benchmarks complete."
echo "================================================================"
