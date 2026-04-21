# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD 3-Clause license found in the
# LICENSE file in the root directory of this source tree.

"""MXFP8 32x1 quantization kernel for AMD GPUs (FlyDSL counterpart of
``cutedsl_quantize_2d_32x1.py``).

For each column of an ``(M, K)`` row-major tensor, derives one E8M0 scale
per 32-element block of M and emits per-column FP8 E4M3FN data. Output is
column-major to match the cutedsl version. FLOOR mode only.

The N-direction quantization needs 32 elements that are stride-K apart in
the row-major input (uncoalesced for per-lane gather), so we cooperatively
stage a ``(32, 64)`` tile through LDS using coalesced row-major loads, then
each lane reads its column out of LDS.

Parallelization:
  Grid:  ``(M // 32, K // 64)`` workgroups
  Block: 64 lanes (1 wave); each lane owns one K-column of the tile
"""

from __future__ import annotations

import functools
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


# K-tile width matches the wave size: one lane per K-column in the tile.
_K_TILE = AMD_WAVE_SIZE

# Multi-wave workgroup. Each workgroup runs N independent waves, each wave
# handling its own K-tile of K_TILE columns. So one workgroup covers
# M_BLOCK rows × (N * K_TILE) columns. Mirrors triton's num_warps strategy —
# multi-wave workgroups let the SIMD scheduler overlap memory latency across
# waves within a single CU.
#
# Picked adaptively at launch time: more waves = more parallelism per
# workgroup, but K must be divisible by WAVES_PER_BLOCK * K_TILE.
_MAX_WAVES_PER_BLOCK = 4


def _pick_waves_per_block(K: int) -> int:
    """Largest power-of-two ≤ _MAX_WAVES_PER_BLOCK such that K % (waves*K_TILE) == 0."""
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
    def _compile_quantize_2d_32x1(
        input_dtype_name: str,
        scaling_mode: str,
        M: int,
        K: int,
        waves_per_block: int,
    ):
        """JIT-compile for given (dtype, mode, M, K).

        M is in the cache key because the col-major output stride is M (in
        element units); K because the row-major input stride is K. Both must
        be Python ints captured in the kernel closure for the compiler to
        constant-fold the address arithmetic.
        """
        if scaling_mode != "floor":
            raise NotImplementedError(
                "FlyDSL MXFP8 32x1 supports scaling_mode='floor' only "
                f"(got {scaling_mode!r}); RCEIL is a planned follow-up."
            )
        if input_dtype_name == "torch.bfloat16":
            in_dtype = fx.BFloat16
        elif input_dtype_name == "torch.float32":
            in_dtype = fx.Float32
        else:
            raise ValueError(f"Unsupported input dtype: {input_dtype_name}")

        in_elem_bytes = 2 if input_dtype_name == "torch.bfloat16" else 4
        block_threads = AMD_WAVE_SIZE * waves_per_block
        # One LDS region per wave (waves operate on independent K-tiles).
        per_wave_tile_bytes = BLOCK_SIZE * _K_TILE * in_elem_bytes
        # Distinct symbol per (dtype, waves) so cache entries don't collide.
        sym = (
            f"flydsl_mxfp8_2d_32x1_{input_dtype_name.replace('.', '_')}"
            f"_w{waves_per_block}_smem"
        )
        alloc = SmemAllocator(None, arch=str(get_rocm_arch()), global_sym_name=sym)
        lds_off = alloc._align(alloc.ptr, 16)
        alloc.ptr = lds_off + per_wave_tile_bytes * waves_per_block

        @flyc.kernel(known_block_size=[block_threads, 1, 1])
        def quantize_2d_32x1_kernel(
            x: fx.Tensor,        # (M, K) bf16/f32 row-major
            q: fx.Tensor,        # 1D i32 view of (M, K) col-major fp8 (stride (1, M))
            scales: fx.Tensor,   # (K, M // 32) uint8 E8M0
        ):
            m_block = fx.block_idx.x
            k_block = fx.block_idx.y       # one block of K_PER_BLOCK columns
            tid = fx.thread_idx.x

            # Wave/lane decomposition within the workgroup.
            wave_id = tid // fx.Int32(AMD_WAVE_SIZE)
            lane_id = tid % fx.Int32(AMD_WAVE_SIZE)

            x_rsrc = buffer_ops.create_buffer_resource(x)
            q_rsrc = buffer_ops.create_buffer_resource(q)
            s_rsrc = buffer_ops.create_buffer_resource(scales)

            # Each wave gets its own LDS region (per_wave_tile_bytes apart).
            # We index LDS using lane_id only (not tid), and pick the right
            # region by adding wave_id * per_wave_tile_bytes to the base
            # offset via a separate SmemPtr per wave constructed at compile
            # time. Since wave_id is dynamic but small (0..3), we use a runtime
            # offset on the byte iterator.
            lds_byte_off_per_wave = ArithValue(
                wave_id * fx.Int32(per_wave_tile_bytes // in_elem_bytes)
            ).index_cast(T.index)
            lds_full = SmemPtr(
                alloc.get_base(),
                lds_off,
                in_dtype.ir_type,
                shape=(BLOCK_SIZE * waves_per_block, _K_TILE),
            )
            lds_full.get()

            tid_idx_lane = ArithValue(lane_id).index_cast(T.index)
            row_base = m_block * fx.Int32(BLOCK_SIZE)
            # Each wave handles K-tile (k_block * WAVES_PER_BLOCK + wave_id).
            k_tile_global = k_block * fx.Int32(waves_per_block) + wave_id
            k_global = k_tile_global * fx.Int32(_K_TILE) + lane_id

            # PHASE 1 — cooperative LDS load (per-wave). Wave i loads its tile
            # rows into LDS rows [i*BLOCK_SIZE, (i+1)*BLOCK_SIZE).
            wave_lds_row_off = wave_id * fx.Int32(BLOCK_SIZE)
            for i in range_constexpr(0, BLOCK_SIZE):
                g_off = (row_base + fx.Int32(i)) * fx.Int32(K) + k_global
                v = buffer_ops.buffer_load(
                    x_rsrc, g_off, vec_width=1, dtype=in_dtype,
                )
                lds_row_idx = ArithValue(
                    wave_lds_row_off + fx.Int32(i)
                ).index_cast(T.index)
                lds_full.store(v, [lds_row_idx, tid_idx_lane])

            gpu.barrier()  # waves write to disjoint regions; barrier kept for safety.

            # PHASE 2 — each lane reads its column from LDS in 4-element chunks.
            chunks = []
            local_amax = fx.Float32(0.0)
            for c in range_constexpr(0, CHUNKS_PER_BLOCK):
                elems = []
                for j in range_constexpr(0, VEC):
                    row_lds_local = c * VEC + j
                    row_lds_idx = ArithValue(
                        wave_lds_row_off + fx.Int32(row_lds_local)
                    ).index_cast(T.index)
                    elems.append(lds_full.load([row_lds_idx, tid_idx_lane]))
                if input_dtype_name == "torch.bfloat16":
                    vec_in = vector.from_elements(T.vec(VEC, T.bf16), elems)
                    vec_f32 = arith.extf(T.vec(VEC, T.f32), vec_in)
                else:
                    vec_f32 = vector.from_elements(T.vec(VEC, T.f32), elems)
                chunks.append(vec_f32)
                local_amax = local_amax.maximumf(
                    fx.math.absf(vec_f32).reduce(ReductionOp.MAX)
                )

            scale_u8, inv_scale = floor_scale_and_inv_scale(local_amax)

            # PHASE 3 — quantize, pack, and write per-column 32 fp8 bytes.
            f8_min_v, f8_max_v = make_fp8_clamp_vectors()
            col_i32_base = (
                k_global * fx.Int32(M) + row_base
            ) // fx.Int32(VEC)
            for c in range_constexpr(0, CHUNKS_PER_BLOCK):
                out = quantize_pack_chunk_to_i32(
                    chunks[c], inv_scale, f8_min_v, f8_max_v,
                )
                buffer_ops.buffer_store(out, q_rsrc, col_i32_base + fx.Int32(c))

            # PHASE 4 — one uint8 scale per (k_global, m_block).
            buffer_ops.buffer_store(
                scale_u8, s_rsrc,
                k_global * fx.Int32(M // BLOCK_SIZE) + m_block,
            )

        @flyc.jit
        def launch_quantize(
            x: fx.Tensor,
            q: fx.Tensor,
            scales: fx.Tensor,
            grid_m: fx.Int32,
            grid_k: fx.Int32,
            stream: fx.Stream = fx.Stream(None),
        ):
            alloc.finalized = False
            ctx = CompilationContext.get_current()
            with ir.InsertionPoint(ctx.gpu_module_body):
                alloc.finalize()
            quantize_2d_32x1_kernel(x, q, scales).launch(
                grid=(grid_m, grid_k, 1),
                block=(block_threads, 1, 1),
                stream=stream,
            )

        return launch_quantize

else:
    def _compile_quantize_2d_32x1(*_args, **_kwargs):
        missing = _missing_flydsl_runtime_packages()
        raise ImportError(
            f"FlyDSL is not available. Missing package(s): {', '.join(missing)}."
        )


def mxfp8_quantize_flydsl_2d_32x1(
    x: torch.Tensor,
    block_size: int = 32,
    scaling_mode: str = "floor",
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Quantize a 2D ``(M, K)`` tensor to MXFP8 (32x1) on AMD via FlyDSL.

    AMD counterpart of :func:`mxfp8_quantize_cutedsl_2d_32x1`. Quantizes along
    M; output data is column-major (stride ``(1, M)``).

    Args:
        x: Input ``(M, K)`` tensor, dtype bfloat16 or float32, row-major.
        block_size: MXFP8 block size along M. Only 32 is supported.
        scaling_mode: ``"floor"`` only in this baseline.

    Returns:
        Tuple ``(q_data, scales)`` where ``q_data`` is column-major
        ``torch.float8_e4m3fn`` of shape ``(M, K)`` (stride ``(1, M)``) and
        ``scales`` is ``(K, M // 32)`` viewed as ``torch.float8_e8m0fnu``.
    """
    assert x.dtype in (torch.bfloat16, torch.float32), (
        f"Input dtype must be bfloat16 or float32, got {x.dtype}"
    )
    assert x.is_cuda, "Input tensor must be on a CUDA/HIP device"
    assert block_size == BLOCK_SIZE, (
        f"Only block_size={BLOCK_SIZE} is supported (got {block_size})"
    )
    assert x.is_contiguous(), "Input must be contiguous (row-major)"
    assert x.ndim == 2, f"Expected 2D input, got shape {tuple(x.shape)}"

    M, K = x.shape
    assert M % BLOCK_SIZE == 0, (
        f"M ({M}) must be divisible by block_size ({BLOCK_SIZE})"
    )
    assert K % _K_TILE == 0, f"K ({K}) must be divisible by {_K_TILE}"
    waves_per_block = _pick_waves_per_block(K)
    k_per_block = waves_per_block * _K_TILE

    # The kernel issues 32-bit packed-fp8 stores, so it needs an int32 view of
    # the output storage. Col-major fp8 (stride (1, M)) can't be `.view`d as
    # int32 directly, so we allocate a flat i32 buffer and alias it back to
    # the col-major fp8 layout for the caller via `set_`.
    q_i32_flat = torch.empty(M * K // 4, dtype=torch.int32, device=x.device)
    scales_u8 = torch.empty((K, M // BLOCK_SIZE), device=x.device, dtype=torch.uint8)

    launch = _compile_quantize_2d_32x1(
        str(x.dtype), scaling_mode, int(M), int(K), waves_per_block,
    )

    import flydsl.compiler as flyc
    x_fly = flyc.from_dlpack(x).mark_layout_dynamic(leading_dim=1, divisibility=2)
    launch(x_fly, q_i32_flat, scales_u8,
           int(M // BLOCK_SIZE), int(K // k_per_block),
           stream=torch.cuda.current_stream())

    q_data = torch.empty(0, dtype=torch.float8_e4m3fn, device=x.device)
    q_data.set_(q_i32_flat.untyped_storage(), 0, (M, K), (1, M))
    return q_data, scales_u8.view(torch.float8_e8m0fnu)
