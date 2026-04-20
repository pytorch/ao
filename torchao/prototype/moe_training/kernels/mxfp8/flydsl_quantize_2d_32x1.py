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
        tile_bytes = BLOCK_SIZE * _K_TILE * in_elem_bytes
        # Distinct LDS allocator symbol per cache entry to avoid name collisions.
        sym = f"flydsl_mxfp8_2d_32x1_{input_dtype_name.replace('.', '_')}_smem"
        alloc = SmemAllocator(None, arch=str(get_rocm_arch()), global_sym_name=sym)
        lds_off = alloc._align(alloc.ptr, 16)
        alloc.ptr = lds_off + tile_bytes

        @flyc.kernel(known_block_size=[AMD_WAVE_SIZE, 1, 1])
        def quantize_2d_32x1_kernel(
            x: fx.Tensor,        # (M, K) bf16/f32 row-major
            q: fx.Tensor,        # 1D i32 view of (M, K) col-major fp8 (stride (1, M))
            scales: fx.Tensor,   # (K, M // 32) uint8 E8M0
        ):
            m_block = fx.block_idx.x
            k_tile = fx.block_idx.y
            tid = fx.thread_idx.x

            x_rsrc = buffer_ops.create_buffer_resource(x)
            q_rsrc = buffer_ops.create_buffer_resource(q)
            s_rsrc = buffer_ops.create_buffer_resource(scales)

            # LDS tile holds the input dtype directly; cast to f32 happens in
            # PHASE 2 via vector.from_elements + arith.extf (exact for bf16).
            lds = SmemPtr(alloc.get_base(), lds_off, in_dtype.ir_type,
                          shape=(BLOCK_SIZE, _K_TILE))
            lds.get()

            tid_idx = ArithValue(tid).index_cast(T.index)
            row_base = m_block * fx.Int32(BLOCK_SIZE)
            k_global = k_tile * fx.Int32(_K_TILE) + tid

            # PHASE 1 — cooperative LDS load. 32 wave-loads, one per row of
            # the tile; each is 64 lanes × 1 element = 1 cache line, fully
            # coalesced.
            for i in range_constexpr(0, BLOCK_SIZE):
                g_off = (row_base + fx.Int32(i)) * fx.Int32(K) + k_global
                v = buffer_ops.buffer_load(x_rsrc, g_off, vec_width=1, dtype=in_dtype)
                lds.store(v, [ArithValue(fx.Int32(i)).index_cast(T.index), tid_idx])

            gpu.barrier()  # Elided by the compiler when block = 1 wave.

            # PHASE 2 — each lane reads its column from LDS in 4-element chunks.
            chunks = []
            local_amax = fx.Float32(0.0)
            for c in range_constexpr(0, CHUNKS_PER_BLOCK):
                elems = []
                for j in range_constexpr(0, VEC):
                    row_lds = c * VEC + j
                    elems.append(lds.load([
                        ArithValue(fx.Int32(row_lds)).index_cast(T.index),
                        tid_idx,
                    ]))
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

            # PHASE 3 — quantize, pack, and write per-column 32 fp8 bytes
            # (8 i32 stores starting at byte offset k_global*M + row_base).
            f8_min_v, f8_max_v = make_fp8_clamp_vectors()
            col_i32_base = (k_global * fx.Int32(M) + row_base) // fx.Int32(VEC)
            for c in range_constexpr(0, CHUNKS_PER_BLOCK):
                out = quantize_pack_chunk_to_i32(
                    chunks[c], inv_scale, f8_min_v, f8_max_v,
                )
                buffer_ops.buffer_store(out, q_rsrc, col_i32_base + fx.Int32(c))

            # PHASE 4 — one byte per K-column at scales[k_global, m_block].
            buffer_ops.buffer_store(
                scale_u8, s_rsrc, k_global * fx.Int32(M // BLOCK_SIZE) + m_block,
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
                block=(AMD_WAVE_SIZE, 1, 1),
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

    # The kernel issues 32-bit packed-fp8 stores, so it needs an int32 view of
    # the output storage. Col-major fp8 (stride (1, M)) can't be `.view`d as
    # int32 directly, so we allocate a flat i32 buffer and alias it back to
    # the col-major fp8 layout for the caller via `set_`.
    q_i32_flat = torch.empty(M * K // 4, dtype=torch.int32, device=x.device)
    scales_u8 = torch.empty((K, M // BLOCK_SIZE), device=x.device, dtype=torch.uint8)

    launch = _compile_quantize_2d_32x1(str(x.dtype), scaling_mode, int(M), int(K))

    import flydsl.compiler as flyc
    x_fly = flyc.from_dlpack(x).mark_layout_dynamic(leading_dim=1, divisibility=2)
    launch(x_fly, q_i32_flat, scales_u8,
           int(M // BLOCK_SIZE), int(K // _K_TILE),
           stream=torch.cuda.current_stream())

    q_data = torch.empty(0, dtype=torch.float8_e4m3fn, device=x.device)
    q_data.set_(q_i32_flat.untyped_storage(), 0, (M, K), (1, M))
    return q_data, scales_u8.view(torch.float8_e8m0fnu)
