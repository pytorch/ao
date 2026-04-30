# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD 3-Clause license found in the
# LICENSE file in the root directory of this source tree.

"""MXFP8 3D MoE quantization kernel for AMD GPUs (FlyDSL counterpart of
``cutedsl_quantize_3d.py``).

For each expert in an ``(E, N, K)`` tensor, derives one E8M0 scale per
32-element block of N and emits per-expert column-major FP8 E4M3FN data.
Structurally identical to the 2D 32x1 kernel applied per-expert; the only
new piece is an extra grid dimension (``block_idx.z``) and a constant
``expert * N * K`` offset added to every input/output address.

Parallelization:
  Grid:  ``(N // 32, K // (waves_per_block * K_TILE), E)`` workgroups
  Block: ``waves_per_block * 64`` lanes; each lane owns ``VEC`` contiguous
  K-columns of the tile (``K_TILE = 64 * VEC = 256``). Phase 1 issues
  ``buffer_load_dwordx2`` (8 B/lane = 512 B/wave-inst, perfect coalescing).
"""

from __future__ import annotations

import functools
import os
from typing import Tuple

import torch

from .flydsl_utils import (
    AMD_WAVE_SIZE,
    BLOCK_SIZE,
    CHUNKS_PER_BLOCK,
    VEC,
    _flydsl_runtime_available,
    _missing_flydsl_runtime_packages,
)

if _flydsl_runtime_available():
    from .flydsl_utils import current_stream_fast


# Each lane owns VEC contiguous K-cols so Phase-1 issues `buffer_load_dwordx2`
# (8 B/lane = 512 B/wave, perfect coalescing). Mirrors the 32x1 widening from
# 97b17d697. Multi-wave/LDS rationale: see flydsl_quantize_2d_32x1.py.
_K_TILE = AMD_WAVE_SIZE * VEC
_MAX_WAVES_PER_BLOCK = 4


@functools.cache
def _pick_waves_per_block(K: int) -> int:
    """Largest power-of-two ≤ _MAX_WAVES_PER_BLOCK that divides K into K-tiles."""
    for waves in (_MAX_WAVES_PER_BLOCK, 2, 1):
        if K % (waves * _K_TILE) == 0:
            return waves
    raise AssertionError(f"K ({K}) must be divisible by at least {_K_TILE}")


if _flydsl_runtime_available():
    import flydsl.compiler as flyc
    import flydsl.expr as fx
    from flydsl.compiler.kernel_function import CompilationContext
    from flydsl.expr import arith, buffer_ops, gpu, range_constexpr, vector
    from flydsl.expr.arith import ArithValue
    from flydsl.expr.typing import T
    from flydsl.expr.vector import ReductionOp
    from flydsl.runtime.device import get_rocm_arch
    from flydsl.utils.smem_allocator import SmemAllocator, SmemPtr
    from flydsl._mlir import ir

    from .flydsl_utils import (
        floor_scale_and_inv_scale,
        make_fp8_clamp_vectors,
        quantize_pack_chunk_to_i32,
    )

    @functools.cache
    def _compile_quantize_3d(
        input_dtype_name: str,
        scaling_mode: str,
        N: int,
        K: int,
        waves_per_block: int,
        scale_block_k: int = 1,
        blocked_scale_output: bool = False,
    ):
        if scaling_mode != "floor":
            raise NotImplementedError(
                f"Only scaling_mode='floor' is supported (got {scaling_mode!r})"
            )
        if scale_block_k not in (1, BLOCK_SIZE):
            raise NotImplementedError(
                f"scale_block_k must be 1 or {BLOCK_SIZE} (got {scale_block_k})"
            )
        if input_dtype_name == "torch.bfloat16":
            in_dtype = fx.BFloat16
        elif input_dtype_name == "torch.float32":
            in_dtype = fx.Float32
        else:
            raise ValueError(f"Unsupported input dtype: {input_dtype_name}")

        in_elem_bytes = 2 if input_dtype_name == "torch.bfloat16" else 4
        block_threads = AMD_WAVE_SIZE * waves_per_block
        per_wave_tile_bytes = BLOCK_SIZE * _K_TILE * in_elem_bytes
        sym = (
            f"flydsl_mxfp8_3d_{input_dtype_name.replace('.', '_')}"
            f"_w{waves_per_block}_smem"
        )
        alloc = SmemAllocator(None, arch=str(get_rocm_arch()), global_sym_name=sym)
        lds_off = alloc._align(alloc.ptr, 16)
        alloc.ptr = lds_off + per_wave_tile_bytes * waves_per_block

        EXPERT_INPUT_STRIDE = N * K
        EXPERT_OUTPUT_BYTES = N * K
        SCALES_K_DIM = K if scale_block_k == 1 else K // BLOCK_SIZE
        SCALES_PER_EXPERT = (N // BLOCK_SIZE) * SCALES_K_DIM

        # tcgen05-style blocked layout: each per-expert scale matrix is the
        # logical (K, N // 32) view padded to (PSR, PSC) and re-tiled by
        # to_blocked() into 128x4 super-tiles, each laid out as 4 stacked
        # 32x4 blocks (4*32*4 = 512 bytes per super-tile). Match
        # `cutedsl_quantize_3d._store_scale_32x32` and `to_blocked`.
        n_blocks_n = N // BLOCK_SIZE
        PADDED_SCALE_ROWS = ((K + 127) // 128) * 128
        PADDED_SCALE_COLS = ((n_blocks_n + 3) // 4) * 4
        BLOCKED_PER_EXPERT = PADDED_SCALE_ROWS * PADDED_SCALE_COLS
        TILE_ROW_STRIDE = 128 * PADDED_SCALE_COLS

        @flyc.kernel(known_block_size=[block_threads, 1, 1])
        def quantize_3d_kernel(
            x: fx.Tensor,        # (E, N, K) bf16/f32 row-major
            q: fx.Tensor,        # 1D i32 view of (E, N, K) per-expert col-major fp8
            scales: fx.Tensor,   # (E, N // 32, K) uint8 E8M0
        ):
            n_block = fx.block_idx.x
            k_block = fx.block_idx.y
            expert = fx.block_idx.z
            tid = fx.thread_idx.x
            wave_id = tid // fx.Int32(AMD_WAVE_SIZE)
            lane_id = tid % fx.Int32(AMD_WAVE_SIZE)

            x_rsrc = buffer_ops.create_buffer_resource(x)
            q_rsrc = buffer_ops.create_buffer_resource(q)
            s_rsrc = buffer_ops.create_buffer_resource(scales)

            # Wave i owns LDS rows [i*BLOCK_SIZE, (i+1)*BLOCK_SIZE) of one
            # shared (BLOCK_SIZE * waves_per_block, _K_TILE) region.
            lds_full = SmemPtr(
                alloc.get_base(),
                lds_off,
                in_dtype.ir_type,
                shape=(BLOCK_SIZE * waves_per_block, _K_TILE),
            )
            lds_full.get()

            expert_in_off = expert * fx.Int32(EXPERT_INPUT_STRIDE)
            row_base = n_block * fx.Int32(BLOCK_SIZE)
            k_tile_global = k_block * fx.Int32(waves_per_block) + wave_id
            # Each lane covers VEC contiguous K-cols starting at k_lane_base;
            # Phase 1 issues buffer_load_dwordx2 (8 B/lane = 512 B/wave-inst).
            k_lane_base = k_tile_global * fx.Int32(_K_TILE) + lane_id * fx.Int32(VEC)

            # PHASE 1 — cooperative LDS load: each lane reads VEC contiguous
            # bf16 and explodes them into VEC adjacent LDS columns.
            wave_lds_row_off = wave_id * fx.Int32(BLOCK_SIZE)
            for i in range_constexpr(0, BLOCK_SIZE):
                g_off = (
                    expert_in_off
                    + (row_base + fx.Int32(i)) * fx.Int32(K)
                    + k_lane_base
                )
                vec_in = buffer_ops.buffer_load(
                    x_rsrc, g_off, vec_width=VEC, dtype=in_dtype,
                )
                lds_row_idx = ArithValue(
                    wave_lds_row_off + fx.Int32(i)
                ).index_cast(T.index)
                for j in range_constexpr(0, VEC):
                    elem = vector.extract(vec_in, static_position=[j])
                    lds_col_idx = ArithValue(
                        lane_id * fx.Int32(VEC) + fx.Int32(j)
                    ).index_cast(T.index)
                    lds_full.store(elem, [lds_row_idx, lds_col_idx])

            gpu.barrier()

            # PHASES 2/3/4 — for each of the VEC K-columns this lane owns,
            # read M-stride column from LDS in chunks, compute amax/scale,
            # quantize+pack, write 32 fp8 bytes (8 consecutive i32 ⇒
            # dwordx4 store fusion) and a uint8 scale.
            f8_min_v, f8_max_v = make_fp8_clamp_vectors()
            expert_out_byte_off = expert * fx.Int32(EXPERT_OUTPUT_BYTES)

            def _load_chunks_and_amax(k_local: int):
                """LDS-read CHUNKS_PER_BLOCK chunks for this lane's k_local-th
                K-column, returning (list_of_vec_f32_chunks, scalar_amax_f32)."""
                lds_col_idx = ArithValue(
                    lane_id * fx.Int32(VEC) + fx.Int32(k_local)
                ).index_cast(T.index)
                chunks_local = []
                amax_local = fx.Float32(0.0)
                for c in range_constexpr(0, CHUNKS_PER_BLOCK):
                    elems = []
                    for j in range_constexpr(0, VEC):
                        row_lds_idx = ArithValue(
                            wave_lds_row_off + fx.Int32(c * VEC + j)
                        ).index_cast(T.index)
                        elems.append(lds_full.load([row_lds_idx, lds_col_idx]))
                    if input_dtype_name == "torch.bfloat16":
                        vec_bf = vector.from_elements(T.vec(VEC, T.bf16), elems)
                        vec_f32 = arith.extf(T.vec(VEC, T.f32), vec_bf)
                    else:
                        vec_f32 = vector.from_elements(T.vec(VEC, T.f32), elems)
                    chunks_local.append(vec_f32)
                    amax_local = amax_local.maximumf(
                        fx.math.absf(vec_f32).reduce(ReductionOp.MAX)
                    )
                return chunks_local, amax_local

            def _store_klocal(k_local: int, chunks_local, scale_u8, inv_scale):
                """Quantize + pack + store one K-column of FP8 + write its scale."""
                k_col_global = k_lane_base + fx.Int32(k_local)
                col_i32_base = (
                    expert_out_byte_off + k_col_global * fx.Int32(N) + row_base
                ) // fx.Int32(VEC)
                for c in range_constexpr(0, CHUNKS_PER_BLOCK):
                    out = quantize_pack_chunk_to_i32(
                        chunks_local[c], inv_scale, f8_min_v, f8_max_v,
                    )
                    buffer_ops.buffer_store(out, q_rsrc, col_i32_base + fx.Int32(c))

                if blocked_scale_output:
                    # Logical (k_row=k_col_global, n_block) → flat per-expert
                    # offset in the to_blocked() layout. For (32,32) all
                    # 8 lanes × VEC k_local in one K-block hold the same
                    # warp-reduced scale and write distinct k_rows.
                    row_tile = k_col_global // fx.Int32(128)
                    row_in_tile = k_col_global % fx.Int32(128)
                    b32 = row_in_tile // fx.Int32(BLOCK_SIZE)
                    r32 = row_in_tile % fx.Int32(BLOCK_SIZE)
                    col_tile = n_block // fx.Int32(4)
                    col_in4 = n_block % fx.Int32(4)
                    blocked_off = (
                        expert * fx.Int32(BLOCKED_PER_EXPERT)
                        + row_tile * fx.Int32(TILE_ROW_STRIDE)
                        + col_tile * fx.Int32(512)
                        + r32 * fx.Int32(16)
                        + b32 * fx.Int32(4)
                        + col_in4
                    )
                    buffer_ops.buffer_store(scale_u8, s_rsrc, blocked_off)
                else:
                    # (32, 1): each (lane, k_local) writes its own scale.
                    # (32, 32): all 8 lanes × VEC k_local in a K-block hold
                    # the same reduced scale and write the same byte
                    # (k_col_global // 32) — benign race, same value.
                    scale_k_idx = (
                        k_col_global if scale_block_k == 1
                        else k_col_global // fx.Int32(BLOCK_SIZE)
                    )
                    buffer_ops.buffer_store(
                        scale_u8, s_rsrc,
                        expert * fx.Int32(SCALES_PER_EXPERT)
                        + n_block * fx.Int32(SCALES_K_DIM)
                        + scale_k_idx,
                    )

            if scale_block_k == BLOCK_SIZE:
                # (32, 32) mode: one MX block of K = 32 K-positions =
                # 8 lanes × VEC k_local. Fold per-k_local amax in-lane in a
                # single loop (avoids a Python list of symbolic values, which
                # @flyc.kernel's AST rewriter would mis-handle since `for x
                # in range(...)` becomes range_constexpr). Re-read chunks in
                # the second pass to keep at most CHUNKS_PER_BLOCK live.
                block_amax = fx.Float32(0.0)
                for k_local in range_constexpr(0, VEC):
                    _, amax_local = _load_chunks_and_amax(k_local)
                    block_amax = block_amax.maximumf(amax_local)

                # 8 lanes per K-block (lane_id*VEC..lane_id*VEC+VEC-1 with
                # VEC k_local covers 32 K positions). width=8 keeps each
                # group of 8 lanes independent.
                width8 = fx.Int32(8)
                for sh in (4, 2, 1):
                    peer = block_amax.shuffle_xor(fx.Int32(sh), width8)
                    block_amax = block_amax.maximumf(peer)

                scale_u8, inv_scale = floor_scale_and_inv_scale(block_amax)
                for k_local in range_constexpr(0, VEC):
                    chunks_local, _ = _load_chunks_and_amax(k_local)
                    _store_klocal(k_local, chunks_local, scale_u8, inv_scale)
            else:
                # (32, 1) mode: each (lane, k_local) gets its own scale.
                # Single-pass per k_local — same register profile as 32x1.
                for k_local in range_constexpr(0, VEC):
                    chunks_local, amax_local = _load_chunks_and_amax(k_local)
                    scale_u8, inv_scale = floor_scale_and_inv_scale(amax_local)
                    _store_klocal(k_local, chunks_local, scale_u8, inv_scale)

        @flyc.jit
        def launch_quantize(
            x: fx.Tensor,
            q: fx.Tensor,
            scales: fx.Tensor,
            grid_n: fx.Int32,
            grid_k: fx.Int32,
            grid_e: fx.Int32,
            stream: fx.Stream = fx.Stream(None),
        ):
            alloc.finalized = False
            ctx = CompilationContext.get_current()
            with ir.InsertionPoint(ctx.gpu_module_body):
                alloc.finalize()
            quantize_3d_kernel(x, q, scales).launch(
                grid=(grid_n, grid_k, grid_e),
                block=(block_threads, 1, 1),
                stream=stream,
            )

        return launch_quantize

else:
    def _compile_quantize_3d(*_args, **_kwargs):
        missing = _missing_flydsl_runtime_packages()
        raise ImportError(
            f"FlyDSL is not available. Missing package(s): {', '.join(missing)}."
        )


def mxfp8_quantize_flydsl_3d(
    x: torch.Tensor,
    block_size: int = 32,
    scale_block_n: int = 32,
    scale_block_k: int = 1,
    scaling_mode: str = "floor",
    stage_count: int = 2,
    blocked_scale_output: bool = False,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Quantize a 3D MoE (E, N, K) tensor to MXFP8 along N using a FlyDSL kernel.

    AMD counterpart of :func:`mxfp8_quantize_cutedsl_3d`. Output data is
    per-expert column-major (strides ``(N*K, 1, N)``); scales are
    ``(E, N // 32, K)`` for ``scale_block_k=1`` or ``(E, N // 32, K // 32)``
    for ``scale_block_k=32``. When ``blocked_scale_output=True`` the scales
    are emitted in the per-expert flat tcgen05 blocked layout
    ``(E, padded_scale_rows * padded_scale_cols)`` matching
    :func:`torchao.prototype.mx_formats.utils.to_blocked` of the (K, N//32)
    logical scale matrix.

    ``scale_block_n`` and ``stage_count`` are accepted for API parity with
    the cutedsl wrapper. ``scale_block_n`` must equal 32 (the only N-tiling
    the kernel implements); ``stage_count`` (TMA pipeline depth) is ignored
    on AMD. Unsupported values raise :class:`NotImplementedError`.
    """
    if scaling_mode != "floor":
        raise NotImplementedError(
            "mxfp8_quantize_flydsl_3d: "
            f"scaling_mode={scaling_mode!r} is not supported by the FlyDSL "
            "baseline; only 'floor' is implemented."
        )
    if scale_block_n != 32:
        raise NotImplementedError(
            "mxfp8_quantize_flydsl_3d: "
            f"scale_block_n must be 32 (got {scale_block_n})."
        )
    del stage_count
    assert x.dtype in (torch.bfloat16, torch.float32), (
        "Input tensor must be float32 or bfloat16"
    )
    assert x.is_cuda, "Input tensor must be CUDA"
    assert x.is_contiguous(), "Input tensor must be contiguous (row-major)"
    assert block_size == BLOCK_SIZE, f"Only block_size={BLOCK_SIZE} is supported"
    assert scale_block_k in (1, BLOCK_SIZE), (
        f"scale_block_k must be 1 or {BLOCK_SIZE} (got {scale_block_k})"
    )
    E, N, K = x.shape
    assert N % BLOCK_SIZE == 0, "N must be divisible by block_size"
    assert K % _K_TILE == 0, f"K must be divisible by {_K_TILE}"

    # Diagnostic override: FLYDSL_3D_FORCE_WAVES ∈ {1, 2, 4} bypasses the
    # auto-pick to test the LDS-budget / CTAs-per-CU scheduling hypothesis.
    _force = os.environ.get("FLYDSL_3D_FORCE_WAVES")
    if _force is not None:
        waves_per_block = int(_force)
        assert waves_per_block in (1, 2, 4), \
            f"FLYDSL_3D_FORCE_WAVES must be 1, 2, or 4 (got {_force})"
        assert K % (waves_per_block * _K_TILE) == 0, (
            f"FLYDSL_3D_FORCE_WAVES={_force} requires K ({K}) divisible by "
            f"{waves_per_block * _K_TILE}"
        )
    else:
        waves_per_block = _pick_waves_per_block(K)
    k_per_block = waves_per_block * _K_TILE

    n_blocks_n = N // BLOCK_SIZE
    scales_k_dim = K if scale_block_k == 1 else K // BLOCK_SIZE

    # Allocate flat fp8 storage; the kernel writes 32-bit packed fp8 (i32
    # view), and the caller receives a per-expert col-major fp8 alias.
    q_storage = torch.empty(E * N * K, dtype=torch.float8_e4m3fn, device=x.device)
    if blocked_scale_output:
        padded_rows = ((K + 127) // 128) * 128
        padded_cols = ((n_blocks_n + 3) // 4) * 4
        scales_u8 = torch.zeros(
            (E, padded_rows * padded_cols), device=x.device, dtype=torch.uint8,
        )
    else:
        scales_u8 = torch.empty(
            (E, n_blocks_n, scales_k_dim), device=x.device, dtype=torch.uint8,
        )

    launch = _compile_quantize_3d(
        str(x.dtype), scaling_mode, N, K, waves_per_block,
        scale_block_k, blocked_scale_output,
    )
    # Pass `x` raw (not via from_dlpack) so FlyDSL's bare-pointer fast path
    # avoids the per-call DLPack adapter overhead.
    launch(x, q_storage.view(torch.int32), scales_u8,
           n_blocks_n, K // k_per_block, E,
           stream=current_stream_fast(x.device))
    return (
        q_storage.as_strided((E, N, K), (N * K, 1, N)),
        scales_u8.view(torch.float8_e8m0fnu),
    )
