# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD 3-Clause license found in the
# LICENSE file in the root directory of this source tree.
#
# Sweeps (BLOCK_SIZE, BLOCK_SIZE_ITER, num_warps) for the colwise FP8 kernel
# on representative DeepSeek-MoE-16B training shapes (MI300X).
#
# Run via subprocess-per-config to isolate GPU context failures.
# Usage:
#   cd ~/ao
#   python benchmarks/prototype/moe_training/fp8_rowwise/bench_colwise_block_configs.py

import itertools
import json
import os
import subprocess
import sys

import torch

# ---------------------------------------------------------------------------
# When called as a worker subprocess (BENCH_CFG env var set), run one config.
# ---------------------------------------------------------------------------
if "BENCH_CFG" in os.environ:
    import triton
    import triton.language as tl
    from triton.testing import do_bench

    from torchao.prototype.moe_training.utils import generate_jagged_offs

    EPS = 1e-12
    FP8_DTYPE_MAP = {
        "float8_e4m3fn": (torch.float8_e4m3fn, tl.float8e4nv),
        "float8_e4m3fnuz": (torch.float8_e4m3fnuz, tl.float8e4b8),
    }

    # Kernel uses same dimension convention as production:
    #   input shape = (K, N) where K=token rows (jagged), N=hidden cols
    #   offsets mark group boundaries along K
    #   scales buffer = N * N_GROUPS
    #   BLOCK_SIZE tiles N (columns), BLOCK_SIZE_ITER tiles K (rows, inner loop)
    @triton.jit
    def _colwise_kernel(
        input_ptr, offsets_ptr, out_ptr, scales_ptr,
        K: tl.int64, N: tl.int64, N_GROUPS: tl.int64,
        str_ir: tl.int64, str_ic: tl.int64,
        str_or: tl.int64, str_oc: tl.int64,
        fp8_min: tl.constexpr, fp8_max: tl.constexpr,
        in_dtype: tl.constexpr, out_dtype: tl.constexpr,
        BLOCK_SIZE: tl.constexpr, BLOCK_SIZE_ITER: tl.constexpr, EPS: tl.constexpr,
    ):
        # block_col_id tiles the N (hidden/col) dimension
        bcol = tl.program_id(0)
        gidx = tl.program_id(1)
        # group boundaries along K (token) dimension
        rs = tl.load(offsets_ptr + gidx - 1, mask=gidx > 0, other=0)
        re = tl.load(offsets_ptr + gidx)
        col_offs = (bcol * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)).to(tl.int64)
        amax = tl.zeros((BLOCK_SIZE,), dtype=in_dtype)
        for r0 in range(rs, re, BLOCK_SIZE_ITER):
            row_offs = (r0 + tl.arange(0, BLOCK_SIZE_ITER)).to(tl.int64)
            offs = row_offs[:, None] * str_ir + col_offs[None, :] * str_ic
            mask = (row_offs[:, None] < re) & (col_offs[None, :] < N)
            d = tl.load(input_ptr + offs, mask=mask, other=0.0).to(in_dtype)
            amax = tl.maximum(amax, tl.max(tl.abs(d), axis=0)).to(in_dtype)
        amax = amax.to(tl.float64)
        s = (fp8_max / tl.clamp(amax, min=EPS, max=float("inf"))).to(tl.float32)
        s = tl.exp2(tl.floor(tl.log2(s)))
        # scales layout: N * N_GROUPS — N cols per group, stride by N between groups
        sc_offs = col_offs + N * gidx
        sc_mask = tl.arange(0, BLOCK_SIZE) < N
        tl.store(scales_ptr + sc_offs, s, mask=sc_mask)
        for r0 in range(rs, re, BLOCK_SIZE_ITER):
            row_offs = (r0 + tl.arange(0, BLOCK_SIZE_ITER)).to(tl.int64)
            offs = row_offs[:, None] * str_ir + col_offs[None, :] * str_ic
            mask = (row_offs[:, None] < re) & (col_offs[None, :] < N)
            d = tl.load(input_ptr + offs, mask=mask, other=0.0).to(in_dtype)
            sd = d * s[None, :]
            fp8d = tl.clamp(sd, min=fp8_min, max=fp8_max).to(out_dtype)
            o_offs = row_offs[:, None] * str_or + col_offs[None, :] * str_oc
            tl.store(out_ptr + o_offs, fp8d, mask=mask)

    cfg = json.loads(os.environ["BENCH_CFG"])
    M, K, n_groups = cfg["M"], cfg["K"], cfg["n_groups"]
    bs, bsi, nw = cfg["bs"], cfg["bsi"], cfg["nw"]
    fp8_dtype_name = cfg["fp8_dtype"]
    fp8_dtype, tl_fp8_dtype = FP8_DTYPE_MAP[fp8_dtype_name]

    device = torch.device("cuda")
    # Production colwise kernel input shape: (K_tokens, N_hidden) row-major
    # K=token rows (jagged dim), N=hidden cols
    # Here in cfg: M=total_tokens=K_triton, K=hidden=N_triton
    K_triton = M   # token dimension (jagged, row-iter)
    N_triton = K   # hidden dimension (column, block-parallel)
    inp = torch.randn(K_triton, N_triton, dtype=torch.bfloat16, device=device)
    offs = generate_jagged_offs(n_groups, K_triton, multiple_of=16)
    fp8_min = torch.finfo(fp8_dtype).min
    fp8_max = torch.finfo(fp8_dtype).max

    def run():
        out = torch.empty_like(inp, dtype=fp8_dtype).as_strided(inp.size(), (1, K_triton))
        sc = torch.empty(N_triton * n_groups, dtype=torch.float32, device=device)
        grid = (triton.cdiv(N_triton, bs), n_groups)
        _colwise_kernel[grid](
            inp, offs, out, sc,
            K_triton, N_triton, n_groups,
            inp.stride(0), inp.stride(1),
            out.stride(0), out.stride(1),
            fp8_min, fp8_max,
            tl.bfloat16, tl_fp8_dtype,
            BLOCK_SIZE=bs, BLOCK_SIZE_ITER=bsi, EPS=EPS,
            num_warps=nw, num_stages=2,
        )
        return out, sc

    # warmup
    for _ in range(3):
        run()
    torch.cuda.synchronize()

    t_us = do_bench(run, return_mode="median") * 1e3
    print(json.dumps({"t_us": t_us}))
    sys.exit(0)


# ---------------------------------------------------------------------------
# Main driver: spawn one subprocess per config.
# ---------------------------------------------------------------------------
BLOCK_SIZES      = [32, 64, 128, 256]
BLOCK_SIZE_ITERS = [32, 64, 128, 256]
NUM_WARPS_LIST   = [4, 8]

SHAPES = [
    dict(M=16640, K=2048, n_groups=64,  label="grad_out  M=16640 K=2048  E=64"),
    dict(M=16640, K=5120, n_groups=64,  label="grad_out  M=16640 K=5120  E=64"),
    dict(M=16640, K=2048, n_groups=128, label="A         M=16640 K=2048  E=128"),
    dict(M=16640, K=5120, n_groups=128, label="A         M=16640 K=5120  E=128"),
]


def run_one(shape, bs, bsi, nw, fp8_dtype_name):
    cfg = {**shape, "bs": bs, "bsi": bsi, "nw": nw, "fp8_dtype": fp8_dtype_name}
    del cfg["label"]
    env = {**os.environ, "BENCH_CFG": json.dumps(cfg)}
    result = subprocess.run(
        [sys.executable, __file__],
        env=env,
        capture_output=True,
        text=True,
        timeout=60,
    )
    if result.returncode != 0:
        return float("inf")
    try:
        return json.loads(result.stdout.strip())["t_us"]
    except Exception:
        return float("inf")


def main():
    gpu = subprocess.check_output(
        ["python", "-c", "import torch; print(torch.cuda.get_device_name())"],
        text=True,
    ).strip()
    hip = subprocess.check_output(
        ["python", "-c", "import torch; print(torch.version.hip)"],
        text=True,
    ).strip()
    print(f"GPU  : {gpu}")
    print(f"ROCm : {hip}")
    print()

    fp8_dtype_name = "float8_e4m3fnuz" if hip != "None" else "float8_e4m3fn"
    combos = list(itertools.product(BLOCK_SIZES, BLOCK_SIZE_ITERS, NUM_WARPS_LIST))
    overall_best = {}

    for shape in SHAPES:
        label = shape["label"]
        print(f"=== {label} ===")

        results = []
        for bs, bsi, nw in combos:
            t_us = run_one(shape, bs, bsi, nw, fp8_dtype_name)
            results.append((t_us, bs, bsi, nw))
            status = f"{t_us:.1f} us" if t_us != float("inf") else "FAIL"
            print(f"  BS={bs:3d} BSI={bsi:3d} warps={nw}  {status}", flush=True)

        results.sort()
        best_t, best_bs, best_bsi, best_nw = results[0]
        overall_best[label] = (best_bs, best_bsi, best_nw, best_t)
        print(
            f"\n  BEST: BLOCK_SIZE={best_bs}, BLOCK_SIZE_ITER={best_bsi}, "
            f"num_warps={best_nw}  →  {best_t:.1f} us\n"
        )

    print("=" * 60)
    print("SUMMARY — best config per shape:")
    for lbl, (bs, bsi, nw, t) in overall_best.items():
        print(f"  {lbl}  →  BS={bs} BSI={bsi} warps={nw}  ({t:.1f} us)")


if __name__ == "__main__":
    main()
