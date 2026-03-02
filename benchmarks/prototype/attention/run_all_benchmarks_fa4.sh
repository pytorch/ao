#!/bin/bash
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD 3-Clause license found in the
# LICENSE file in the root directory of this source tree.

# Run all low-precision attention benchmarks (FA4 baseline vs FA4 FP8 test).
# Usage: bash benchmarks/prototype/attention/run_all_benchmarks_fa4.sh

set -euo pipefail

BENCH_DIR="benchmarks/prototype/attention"
BASELINE="fa4"
TEST="fa4_fp8"

echo "================================================================"
echo "  Low-Precision Attention Benchmarks ($BASELINE vs $TEST)"
echo "================================================================"

# --------------------------------------------------------------------------
# 1. Single attention layer benchmark
# --------------------------------------------------------------------------
echo ""
echo "================================================================"
echo "  [1/9] benchmark_sdpa.py — Single Attention Layer"
echo "================================================================"
python "$BENCH_DIR/benchmark_sdpa.py" --baseline "$BASELINE" --test "$TEST"

# --------------------------------------------------------------------------
# 2. LLaMA 3 model benchmarks (4 configurations)
# --------------------------------------------------------------------------
echo ""
echo "================================================================"
echo "  [2/9] eval_llama3_model.py — No compile, no fuse_rope_using_torch_compile"
echo "================================================================"
python "$BENCH_DIR/eval_llama3_model.py" --baseline "$BASELINE" --test "$TEST"

echo ""
echo "================================================================"
echo "  [3/9] eval_llama3_model.py — Compile, no fuse_rope_using_torch_compile"
echo "================================================================"
python "$BENCH_DIR/eval_llama3_model.py" --baseline "$BASELINE" --test "$TEST" --compile

echo ""
echo "================================================================"
echo "  [4/9] eval_llama3_model.py — No compile, fuse_rope_using_torch_compile"
echo "================================================================"
python "$BENCH_DIR/eval_llama3_model.py" --baseline "$BASELINE" --test "$TEST" --fuse_rope_using_torch_compile

echo ""
echo "================================================================"
echo "  [5/9] eval_llama3_model.py — Compile, fuse_rope_using_torch_compile"
echo "================================================================"
python "$BENCH_DIR/eval_llama3_model.py" --baseline "$BASELINE" --test "$TEST" --compile --fuse_rope_using_torch_compile

# --------------------------------------------------------------------------
# 3. FLUX model benchmarks (4 configurations)
# --------------------------------------------------------------------------
echo ""
echo "================================================================"
echo "  [6/9] eval_flux_model.py — No compile, no fuse_rope_using_torch_compile"
echo "================================================================"
python "$BENCH_DIR/eval_flux_model.py" --baseline "$BASELINE" --test "$TEST"

echo ""
echo "================================================================"
echo "  [7/9] eval_flux_model.py — Compile, no fuse_rope_using_torch_compile"
echo "================================================================"
python "$BENCH_DIR/eval_flux_model.py" --baseline "$BASELINE" --test "$TEST" --compile

echo ""
echo "================================================================"
echo "  [8/9] eval_flux_model.py — No compile, fuse_rope_using_torch_compile"
echo "================================================================"
python "$BENCH_DIR/eval_flux_model.py" --baseline "$BASELINE" --test "$TEST" --fuse_rope_using_torch_compile

echo ""
echo "================================================================"
echo "  [9/9] eval_flux_model.py — Compile, fuse_rope_using_torch_compile"
echo "================================================================"
python "$BENCH_DIR/eval_flux_model.py" --baseline "$BASELINE" --test "$TEST" --compile --fuse_rope_using_torch_compile

echo ""
echo "================================================================"
echo "  All benchmarks complete."
echo "================================================================"
