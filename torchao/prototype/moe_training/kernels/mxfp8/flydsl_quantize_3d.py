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
  Grid:  ``(N // 32, K // 64, E)`` workgroups
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
    def _compile_quantize_3d(
        input_dtype_name: str,
        scaling_mode: str,
        N: int,
        K: int,
    ):
        """JIT-compile for given (dtype, mode, N, K). E is launch-only."""
        if scaling_mode != "floor":
            raise NotImplementedError(
                "FlyDSL MXFP8 3D supports scaling_mode='floor' only "
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
        sym = f"flydsl_mxfp8_3d_{input_dtype_name.replace('.', '_')}_smem"
        alloc = SmemAllocator(None, arch=str(get_rocm_arch()), global_sym_name=sym)
        lds_off = alloc._align(alloc.ptr, 16)
        alloc.ptr = lds_off + tile_bytes

        EXPERT_INPUT_STRIDE = N * K          # row-major (E, N, K) input
        EXPERT_OUTPUT_BYTES = N * K          # per-expert col-major fp8 slab
        SCALES_PER_EXPERT = (N // BLOCK_SIZE) * K

        @flyc.kernel(known_block_size=[AMD_WAVE_SIZE, 1, 1])
        def quantize_3d_kernel(
            x: fx.Tensor,        # (E, N, K) bf16/f32 row-major
            q: fx.Tensor,        # 1D i32 view of (E, N, K) per-expert col-major fp8
            scales: fx.Tensor,   # (E, N // 32, K) uint8 E8M0
        ):
            n_block = fx.block_idx.x
            k_tile = fx.block_idx.y
            expert = fx.block_idx.z
            tid = fx.thread_idx.x

            x_rsrc = buffer_ops.create_buffer_resource(x)
            q_rsrc = buffer_ops.create_buffer_resource(q)
            s_rsrc = buffer_ops.create_buffer_resource(scales)

            lds = SmemPtr(alloc.get_base(), lds_off, in_dtype.ir_type,
                          shape=(BLOCK_SIZE, _K_TILE))
            lds.get()

            tid_idx = ArithValue(tid).index_cast(T.index)
            expert_in_off = expert * fx.Int32(EXPERT_INPUT_STRIDE)
            row_base = n_block * fx.Int32(BLOCK_SIZE)
            k_global = k_tile * fx.Int32(_K_TILE) + tid

            # PHASE 1 — cooperative LDS load (32 wave-loads, each 1 cache line).
            for i in range_constexpr(0, BLOCK_SIZE):
                g_off = (
                    expert_in_off
                    + (row_base + fx.Int32(i)) * fx.Int32(K)
                    + k_global
                )
                v = buffer_ops.buffer_load(x_rsrc, g_off, vec_width=1, dtype=in_dtype)
                lds.store(v, [ArithValue(fx.Int32(i)).index_cast(T.index), tid_idx])

            gpu.barrier()

            # PHASE 2 — per-lane column read + amax accumulation.
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

            # PHASE 3 — quantize, pack, write per-expert col-major.
            f8_min_v, f8_max_v = make_fp8_clamp_vectors()
            expert_out_byte_off = expert * fx.Int32(EXPERT_OUTPUT_BYTES)
            col_i32_base = (
                expert_out_byte_off + k_global * fx.Int32(N) + row_base
            ) // fx.Int32(VEC)
            for c in range_constexpr(0, CHUNKS_PER_BLOCK):
                out = quantize_pack_chunk_to_i32(
                    chunks[c], inv_scale, f8_min_v, f8_max_v,
                )
                buffer_ops.buffer_store(out, q_rsrc, col_i32_base + fx.Int32(c))

            # PHASE 4 — uint8 scale at scales[expert, n_block, k_global].
            buffer_ops.buffer_store(
                scale_u8, s_rsrc,
                expert * fx.Int32(SCALES_PER_EXPERT)
                + n_block * fx.Int32(K)
                + k_global,
            )

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
                block=(AMD_WAVE_SIZE, 1, 1),
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
    scaling_mode: str = "floor",
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Quantize a 3D MoE ``(E, N, K)`` tensor to MXFP8 on AMD via FlyDSL.

    AMD counterpart of :func:`mxfp8_quantize_cutedsl_3d`. Quantizes along N;
    output data is per-expert column-major (strides ``(N*K, 1, N)``).

    Args:
        x: Input ``(E, N, K)`` tensor, dtype bfloat16 or float32, row-major.
        block_size: MXFP8 block size along N. Only 32 is supported.
        scaling_mode: ``"floor"`` only in this baseline.

    Returns:
        Tuple ``(q_data, scales)`` where ``q_data`` is per-expert col-major
        ``torch.float8_e4m3fn`` of shape ``(E, N, K)`` (strides ``(N*K, 1, N)``)
        and ``scales`` is ``(E, N // 32, K)`` viewed as ``torch.float8_e8m0fnu``.
    """
    assert x.dtype in (torch.bfloat16, torch.float32), (
        f"Input dtype must be bfloat16 or float32, got {x.dtype}"
    )
    assert x.is_cuda, "Input tensor must be on a CUDA/HIP device"
    assert block_size == BLOCK_SIZE, (
        f"Only block_size={BLOCK_SIZE} is supported (got {block_size})"
    )
    assert x.is_contiguous(), "Input must be contiguous (row-major)"
    assert x.ndim == 3, f"Expected 3D input, got shape {tuple(x.shape)}"

    E, N, K = x.shape
    assert N % BLOCK_SIZE == 0, (
        f"N ({N}) must be divisible by block_size ({BLOCK_SIZE})"
    )
    assert K % _K_TILE == 0, f"K ({K}) must be divisible by {_K_TILE}"

    q_i32_flat = torch.empty(E * N * K // 4, dtype=torch.int32, device=x.device)
    scales_u8 = torch.empty(
        (E, N // BLOCK_SIZE, K), device=x.device, dtype=torch.uint8,
    )

    launch = _compile_quantize_3d(str(x.dtype), scaling_mode, int(N), int(K))

    import flydsl.compiler as flyc
    x_fly = flyc.from_dlpack(x).mark_layout_dynamic(leading_dim=2, divisibility=2)
    launch(x_fly, q_i32_flat, scales_u8,
           int(N // BLOCK_SIZE), int(K // _K_TILE), int(E),
           stream=torch.cuda.current_stream())

    q_data = torch.empty(0, dtype=torch.float8_e4m3fn, device=x.device)
    q_data.set_(q_i32_flat.untyped_storage(), 0, (E, N, K), (N * K, 1, N))
    return q_data, scales_u8.view(torch.float8_e8m0fnu)
