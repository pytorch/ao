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
  Grid:  ``(N // 32, K // (waves_per_block * 64), E)`` workgroups
  Block: ``waves_per_block * 64`` lanes; each lane owns one K-column of the tile.
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


# See flydsl_quantize_2d_32x1.py for multi-wave/LDS rationale. Same per-expert.
_K_TILE = AMD_WAVE_SIZE
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
        SCALES_PER_EXPERT = (N // BLOCK_SIZE) * K

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

            tid_idx_lane = ArithValue(lane_id).index_cast(T.index)
            expert_in_off = expert * fx.Int32(EXPERT_INPUT_STRIDE)
            row_base = n_block * fx.Int32(BLOCK_SIZE)
            k_tile_global = k_block * fx.Int32(waves_per_block) + wave_id
            k_global = k_tile_global * fx.Int32(_K_TILE) + lane_id

            # PHASE 1 — cooperative LDS load: each lane writes one bf16 to
            # its column position in the wave's LDS row strip.
            wave_lds_row_off = wave_id * fx.Int32(BLOCK_SIZE)
            for i in range_constexpr(0, BLOCK_SIZE):
                g_off = (
                    expert_in_off
                    + (row_base + fx.Int32(i)) * fx.Int32(K)
                    + k_global
                )
                v = buffer_ops.buffer_load(
                    x_rsrc, g_off, vec_width=1, dtype=in_dtype,
                )
                lds_row_idx = ArithValue(
                    wave_lds_row_off + fx.Int32(i)
                ).index_cast(T.index)
                lds_full.store(v, [lds_row_idx, tid_idx_lane])

            gpu.barrier()

            # PHASES 2/3/4 — read M-stride column from LDS in chunks, compute
            # amax/scale, quantize+pack, write 32 fp8 bytes (8 consecutive i32
            # ⇒ dwordx4 store fusion) and one uint8 scale.
            chunks = []
            local_amax = fx.Float32(0.0)
            for c in range_constexpr(0, CHUNKS_PER_BLOCK):
                elems = []
                for j in range_constexpr(0, VEC):
                    row_lds_idx = ArithValue(
                        wave_lds_row_off + fx.Int32(c * VEC + j)
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
    scaling_mode: str = "floor",
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Quantize a 3D MoE (E, N, K) tensor to MXFP8 along N using a FlyDSL kernel.

    AMD counterpart of :func:`mxfp8_quantize_cutedsl_3d`. Output data is
    per-expert column-major (strides ``(N*K, 1, N)``); scales are
    ``(E, N // 32, K)``.
    """
    assert x.dtype in (torch.bfloat16, torch.float32), (
        "Input tensor must be float32 or bfloat16"
    )
    assert x.is_cuda, "Input tensor must be CUDA"
    assert x.is_contiguous(), "Input tensor must be contiguous (row-major)"
    assert block_size == BLOCK_SIZE, f"Only block_size={BLOCK_SIZE} is supported"
    E, N, K = x.shape
    assert N % BLOCK_SIZE == 0, "N must be divisible by block_size"
    assert K % _K_TILE == 0, f"K must be divisible by {_K_TILE}"

    waves_per_block = _pick_waves_per_block(K)
    k_per_block = waves_per_block * _K_TILE

    # Allocate flat fp8 storage; the kernel writes 32-bit packed fp8 (i32
    # view), and the caller receives a per-expert col-major fp8 alias.
    q_storage = torch.empty(E * N * K, dtype=torch.float8_e4m3fn, device=x.device)
    scales_u8 = torch.empty(
        (E, N // BLOCK_SIZE, K), device=x.device, dtype=torch.uint8,
    )

    launch = _compile_quantize_3d(
        str(x.dtype), scaling_mode, N, K, waves_per_block,
    )
    # Pass `x` raw (not via from_dlpack) so FlyDSL's bare-pointer fast path
    # avoids the per-call DLPack adapter overhead.
    launch(x, q_storage.view(torch.int32), scales_u8,
           N // BLOCK_SIZE, K // k_per_block, E,
           stream=torch.cuda.current_stream())
    return (
        q_storage.as_strided((E, N, K), (N * K, 1, N)),
        scales_u8.view(torch.float8_e8m0fnu),
    )
