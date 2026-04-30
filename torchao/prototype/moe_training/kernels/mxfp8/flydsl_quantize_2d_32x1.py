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

The M-direction quantization needs 32 elements that are stride-K apart in
the row-major input (uncoalesced for per-lane gather), so we cooperatively
stage a ``(32, K_TILE)`` tile through LDS using coalesced row-major loads,
then each lane reads its K-columns out of LDS.

Parallelization:
  Grid:  ``(M // 32, K // (waves_per_block * K_TILE))`` workgroups
  Block: ``waves_per_block * 64`` lanes; each lane owns ``VEC`` contiguous
  K-columns of the tile (``K_TILE = 64 * VEC = 256``).
"""

from __future__ import annotations

import functools
from typing import Optional, Tuple

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
# (8 B/lane = 512 B/wave, perfect coalescing).
_K_TILE = AMD_WAVE_SIZE * VEC

# Stack multiple MX blocks per column per WG visit so the writeback fills a
# full 128 B HBM cache line (4 × 32 fp8 = 128 B contiguous per col),
# eliminating the +43% RMW pattern observed at 16384² where output > L2.
# Each wave owns one stacked 32-row MX block, so the per-WG wave count
# equals the M-stack depth. Adaptive: large M gets 4 waves / M_TILE=128
# (multi-wave HBM hide + cache-line-fill writes); smaller test shapes fall
# back through 2-wave / 1-wave configurations.
_MAX_M_BLOCKS_PER_WG = 4
_LDS_BUDGET_BYTES = 65536


@functools.cache
def _pick_layout(M: int, K: int, in_elem_bytes: int) -> Tuple[int, int]:
    """Pick ``(m_blocks_per_wg, waves_per_block)`` for given ``(M, K)``.

    The kernel pairs one wave with each stacked MX block, and LDS is shared
    at the WG level so its budget does not scale with wave count. Pick the
    largest power-of-two stack depth ≤ ``_MAX_M_BLOCKS_PER_WG`` such that
    M is divisible by ``BLOCK_SIZE * stack`` and the WG-level LDS fits in
    the per-CU budget.
    """
    if K % _K_TILE != 0:
        raise AssertionError(f"K ({K}) must be divisible by {_K_TILE}")
    for stack in (4, 2, 1):
        if stack > _MAX_M_BLOCKS_PER_WG:
            continue
        m_tile = BLOCK_SIZE * stack
        if M % m_tile != 0:
            continue
        if m_tile * _K_TILE * in_elem_bytes > _LDS_BUDGET_BYTES:
            continue
        return stack, stack
    raise AssertionError(
        f"M ({M}) must be a multiple of block_size ({BLOCK_SIZE}); "
        f"≤ {_MAX_M_BLOCKS_PER_WG} stacked blocks per WG fit in LDS"
    )


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
        m_blocks_per_wg: int,
    ):
        if scaling_mode != "floor":
            raise NotImplementedError(
                f"Only scaling_mode='floor' is supported (got {scaling_mode!r})"
            )
        if input_dtype_name == "torch.bfloat16":
            in_dtype = fx.BFloat16
        elif input_dtype_name == "torch.float32":
            in_dtype = fx.Float32
        else:
            raise ValueError(f"Unsupported input dtype: {input_dtype_name}")

        # One wave per stacked MX block.
        waves_per_block = m_blocks_per_wg
        m_tile = BLOCK_SIZE * m_blocks_per_wg
        in_elem_bytes = 2 if input_dtype_name == "torch.bfloat16" else 4
        block_threads = AMD_WAVE_SIZE * waves_per_block
        wg_tile_bytes = m_tile * _K_TILE * in_elem_bytes
        sym = (
            f"flydsl_mxfp8_2d_32x1_{input_dtype_name.replace('.', '_')}"
            f"_w{waves_per_block}_smem"
        )
        alloc = SmemAllocator(None, arch=str(get_rocm_arch()), global_sym_name=sym)
        lds_off = alloc._align(alloc.ptr, 16)
        alloc.ptr = lds_off + wg_tile_bytes

        @flyc.kernel(known_block_size=[block_threads, 1, 1])
        def quantize_2d_32x1_kernel(
            x: fx.Tensor,        # (M, K) bf16/f32 row-major
            q: fx.Tensor,        # 1D i32 view of (M, K) col-major fp8 (stride (1, M))
            scales: fx.Tensor,   # (K, M // 32) uint8 E8M0
        ):
            m_block = fx.block_idx.x
            k_block = fx.block_idx.y
            tid = fx.thread_idx.x
            wave_id = tid // fx.Int32(AMD_WAVE_SIZE)
            lane_id = tid % fx.Int32(AMD_WAVE_SIZE)

            x_rsrc = buffer_ops.create_buffer_resource(x)
            q_rsrc = buffer_ops.create_buffer_resource(q)
            s_rsrc = buffer_ops.create_buffer_resource(scales)

            # 4 waves cooperate on one (M_TILE, K_TILE) = (128, 256) tile
            # sharing this 64 KB LDS region; each wave owns a 32-row strip.
            lds_full = SmemPtr(
                alloc.get_base(),
                lds_off,
                in_dtype.ir_type,
                shape=(m_tile, _K_TILE),
            )
            lds_full.get()

            row_base = m_block * fx.Int32(m_tile)
            wave_row_off = wave_id * fx.Int32(BLOCK_SIZE)
            k_lane_base = k_block * fx.Int32(_K_TILE) + lane_id * fx.Int32(VEC)

            # PHASE 1 — each wave loads its own 32-row strip into the
            # shared LDS tile. dwordx2 reads (VEC bf16/lane = 8 B), 32
            # iters/lane → 4 waves × 32 rows = 128 rows total covered.
            for i in range_constexpr(0, BLOCK_SIZE):
                g_off = (
                    row_base + wave_row_off + fx.Int32(i)
                ) * fx.Int32(K) + k_lane_base
                vec_in = buffer_ops.buffer_load(
                    x_rsrc, g_off, vec_width=VEC, dtype=in_dtype,
                )
                lds_row_idx = ArithValue(
                    wave_row_off + fx.Int32(i)
                ).index_cast(T.index)
                for j in range_constexpr(0, VEC):
                    elem = vector.extract(vec_in, static_position=[j])
                    lds_col_idx = ArithValue(
                        lane_id * fx.Int32(VEC) + fx.Int32(j)
                    ).index_cast(T.index)
                    lds_full.store(elem, [lds_row_idx, lds_col_idx])

            gpu.barrier()

            # PHASES 2/3/4 — each wave processes its own 32-row strip
            # (one MX block per column). 4 waves writing concurrently to
            # 4 disjoint 32 B fragments of the same 128 B HBM cache line
            # let the L2 controller coalesce into a single line fill →
            # no RMW.
            f8_min_v, f8_max_v = make_fp8_clamp_vectors()
            m_block_global = (
                m_block * fx.Int32(m_blocks_per_wg) + wave_id
            )
            m_row_global_base = row_base + wave_row_off

            for k_local in range_constexpr(0, VEC):
                lds_col_idx = ArithValue(
                    lane_id * fx.Int32(VEC) + fx.Int32(k_local)
                ).index_cast(T.index)
                k_col_global = k_lane_base + fx.Int32(k_local)

                chunks = []
                local_amax = fx.Float32(0.0)
                for c in range_constexpr(0, CHUNKS_PER_BLOCK):
                    elems = []
                    for j in range_constexpr(0, VEC):
                        row_lds_idx = ArithValue(
                            wave_row_off + fx.Int32(c * VEC + j)
                        ).index_cast(T.index)
                        elems.append(lds_full.load([row_lds_idx, lds_col_idx]))
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

                col_i32_base = (
                    k_col_global * fx.Int32(M) + m_row_global_base
                ) // fx.Int32(VEC)
                for c in range_constexpr(0, CHUNKS_PER_BLOCK):
                    out = quantize_pack_chunk_to_i32(
                        chunks[c], inv_scale, f8_min_v, f8_max_v,
                    )
                    buffer_ops.buffer_store(out, q_rsrc, col_i32_base + fx.Int32(c))

                buffer_ops.buffer_store(
                    scale_u8, s_rsrc,
                    k_col_global * fx.Int32(M // BLOCK_SIZE) + m_block_global,
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
    stage_count: int = 2,
    blocked_scale_output: bool = False,
    offs: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Quantize a 2D (M, K) tensor to MXFP8 along M using a FlyDSL kernel.

    AMD counterpart of :func:`mxfp8_quantize_cutedsl_2d_32x1`. Output data
    is column-major (stride ``(1, M)``); scales are ``(K, M // 32)``.

    ``stage_count`` is accepted for API parity with the cutedsl wrapper
    (TMA pipeline depth) and ignored on AMD. ``blocked_scale_output`` and
    ``offs`` are not yet implemented and raise :class:`NotImplementedError`
    when set; the matching dispatcher in ``quant.py`` enforces the same
    contract one level up.
    """
    if scaling_mode != "floor":
        raise NotImplementedError(
            "mxfp8_quantize_flydsl_2d_32x1: "
            f"scaling_mode={scaling_mode!r} is not supported by the FlyDSL "
            "baseline; only 'floor' is implemented."
        )
    if blocked_scale_output:
        raise NotImplementedError(
            "mxfp8_quantize_flydsl_2d_32x1: blocked_scale_output=True is "
            "tcgen05-specific to SM 10.x and not supported by the FlyDSL baseline."
        )
    if offs is not None:
        raise NotImplementedError(
            "mxfp8_quantize_flydsl_2d_32x1: token-group offs are not yet "
            "supported by the FlyDSL baseline."
        )
    del stage_count
    assert x.dtype in (torch.bfloat16, torch.float32), (
        "Input tensor must be float32 or bfloat16"
    )
    assert x.is_cuda, "Input tensor must be CUDA"
    assert x.is_contiguous(), "Input tensor must be contiguous (row-major)"
    assert block_size == BLOCK_SIZE, f"Only block_size={BLOCK_SIZE} is supported"
    M, K = x.shape

    in_elem_bytes = 2 if x.dtype == torch.bfloat16 else 4
    m_blocks_per_wg, _ = _pick_layout(M, K, in_elem_bytes)
    m_tile = BLOCK_SIZE * m_blocks_per_wg

    # Allocate flat fp8 storage; the kernel writes 32-bit packed fp8 (i32
    # view), and the caller receives a col-major fp8 alias.
    q_storage = torch.empty(M * K, dtype=torch.float8_e4m3fn, device=x.device)
    scales_u8 = torch.empty((K, M // BLOCK_SIZE), device=x.device, dtype=torch.uint8)

    launch = _compile_quantize_2d_32x1(
        str(x.dtype), scaling_mode, M, K, m_blocks_per_wg,
    )
    # Pass `x` raw (not via from_dlpack) so FlyDSL's bare-pointer fast path
    # avoids the per-call DLPack adapter overhead.
    launch(x, q_storage.view(torch.int32), scales_u8,
           M // m_tile, K // _K_TILE,
           stream=current_stream_fast(x.device))
    return (
        q_storage.as_strided((M, K), (1, M)),
        scales_u8.view(torch.float8_e8m0fnu),
    )
