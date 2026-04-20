# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD 3-Clause license found in the
# LICENSE file in the root directory of this source tree.

"""MXFP8 3D MoE quantization kernel implemented with FlyDSL (AMD GPUs).

AMD counterpart to ``cutedsl_quantize_3d.py``. For each expert in an
``(E, N, K)`` tensor, derives one E8M0 scale per 32-element block of N
and emits per-expert column-major FP8 E4M3FN data.

This is the 3D analog of ``flydsl_quantize_2d_32x1.py``: the per-expert
slab is a row-major ``(N, K)`` matrix, quantization is along N (with
stride K in memory), and the per-expert output is col-major ``(N, K)``.
The expert dimension shows up as an extra grid axis plus a constant
expert-offset added to every input/output address.

Initial scope of this baseline (CDNA3+ / RDNA4+):
    * FLOOR scaling mode only (RCEIL deferred — no AMD analog to PTX
      ``cvt.rp.satfinite.ue8m0x2.f32``).
    * Input dtype: bf16 or f32, row-major ``(E, N, K)``.
    * Output:
        - ``q_data``: shape ``(E, N, K)`` with strides ``(N*K, 1, N)`` —
          per-expert col-major ``torch.float8_e4m3fn``.
        - ``scales``: shape ``(E, N // 32, K)`` viewed as
          ``torch.float8_e8m0fnu`` (uint8 container with E8M0 semantics).
    * Requires ``N % 32 == 0`` and ``K % 64 == 0``.
    * No ``blocked_scale_output`` yet (planned follow-up).

Parallelization:
    Grid:  (N // 32, K // 64, E)   — one workgroup per (N-block, K-tile, expert)
    Block: 64 threads (1 wave)     — each lane owns one K-column of the tile

Per workgroup:
    Same 4-phase structure as the 2D 32x1 kernel:
      PHASE 1 — cooperative LDS load of a (32 N-rows × 64 K-cols) tile
      PHASE 2 — per-lane column read + Step-5 compute (amax, scale, inv_scale)
      PHASE 3 — quantize/clamp/pack -> col-major output store
      PHASE 4 — uint8 scale store
"""

from __future__ import annotations

import functools
from typing import Tuple

import torch

from .flydsl_utils import (
    F8_MAX,
    _flydsl_runtime_available,
    _missing_flydsl_runtime_packages,
)


# Tile constants (same as 2D 32x1 — quantize block is 32 elements along N,
# wave-aligned tile of 64 K-cols).
_BLOCK_THREADS = 64
_N_BLOCK = 32
_K_TILE = _BLOCK_THREADS
_VEC = 4
_CHUNKS_PER_BLOCK = _N_BLOCK // _VEC          # 8
_E8M0_BIAS = 127


if _flydsl_runtime_available():
    import flydsl.compiler as flyc
    import flydsl.expr as fx
    from flydsl.compiler.kernel_function import CompilationContext
    from flydsl.expr import arith, buffer_ops, gpu, range_constexpr, rocdl, vector
    from flydsl.expr.arith import ArithValue
    from flydsl.expr.typing import T
    from flydsl.expr.vector import ReductionOp
    from flydsl.runtime.device import get_rocm_arch
    from flydsl.utils.smem_allocator import SmemAllocator, SmemPtr
    from flydsl._mlir import ir

    @functools.cache
    def _compile_mxfp8_quantize_flydsl_3d(
        input_dtype_name: str,
        scaling_mode: str,
        N: int,
        K: int,
    ):
        """JIT-compile the FlyDSL 3D kernel for a given (dtype, mode, N, K).

        N and K are part of the cache key because per-expert stride = N * K
        (output bytes) and per-expert input stride = N * K (input elements);
        we want both as Python int constants captured in the kernel closure.
        E is launch-only — it doesn't affect any per-workgroup work.
        """
        if scaling_mode != "floor":
            raise NotImplementedError(
                "FlyDSL MXFP8 quantize_3d supports scaling_mode='floor' only "
                f"(got {scaling_mode!r}); RCEIL is a planned follow-up."
            )
        if input_dtype_name == "torch.bfloat16":
            in_dtype = fx.BFloat16
        elif input_dtype_name == "torch.float32":
            in_dtype = fx.Float32
        else:
            raise ValueError(f"Unsupported input dtype: {input_dtype_name}")

        # Per-workgroup LDS: a single (N_BLOCK × K_TILE) tile in input dtype.
        in_elem_bytes = 2 if input_dtype_name == "torch.bfloat16" else 4
        tile_bytes = _N_BLOCK * _K_TILE * in_elem_bytes
        arch_str = str(get_rocm_arch())
        sym = f"flydsl_mxfp8_3d_{input_dtype_name.replace('.', '_')}_smem"
        alloc = SmemAllocator(None, arch=arch_str, global_sym_name=sym)
        lds_off = alloc._align(alloc.ptr, 16)
        alloc.ptr = lds_off + tile_bytes

        # Per-expert strides as Python int constants.
        EXPERT_INPUT_STRIDE = N * K          # row-major (E, N, K) input
        EXPERT_OUTPUT_BYTES = N * K          # per-expert col-major slab (fp8 = 1 byte/elem)
        EXPERT_OUTPUT_I32 = EXPERT_OUTPUT_BYTES // _VEC
        SCALES_PER_EXPERT = (N // _N_BLOCK) * K   # (N//32) × K scales per expert

        @flyc.kernel(known_block_size=[_BLOCK_THREADS, 1, 1])
        def quantize_3d_kernel(
            x: fx.Tensor,        # (E, N, K) bf16 or f32, row-major
            q: fx.Tensor,        # 1D i32 view of (E, N, K) per-expert col-major fp8
            scales: fx.Tensor,   # (E, N // 32, K) uint8 E8M0
        ):
            # block_idx.x = n_block (which 32-row chunk of N)
            # block_idx.y = k_tile  (which 64-col chunk of K)
            # block_idx.z = expert  (which expert this WG handles)
            n_block = fx.block_idx.x
            k_tile = fx.block_idx.y
            expert = fx.block_idx.z
            tid = fx.thread_idx.x

            x_rsrc = buffer_ops.create_buffer_resource(x)
            q_rsrc = buffer_ops.create_buffer_resource(q)
            s_rsrc = buffer_ops.create_buffer_resource(scales)

            # LDS tile: shape (N_BLOCK, K_TILE), input dtype, row-major.
            base = alloc.get_base()
            lds = SmemPtr(base, lds_off, in_dtype.ir_type, shape=(_N_BLOCK, _K_TILE))
            lds.get()

            tid_idx = ArithValue(tid).index_cast(T.index)

            # Per-expert input offset (in elements).
            expert_in_off = expert * fx.Int32(EXPERT_INPUT_STRIDE)

            # Per-workgroup row/col bases within the per-expert slab.
            n_row_base = n_block * fx.Int32(_N_BLOCK)            # n_block * 32
            k_global = k_tile * fx.Int32(_K_TILE) + tid          # k_tile * 64 + tid

            # ---------- PHASE 1: cooperative LDS load (row-by-row, coalesced) ----------
            # 32 wave-loads, each = 64 lanes × 1 element = 1 cache line.
            for i in range_constexpr(0, _N_BLOCK):
                row_idx = n_row_base + fx.Int32(i)
                # Element offset: expert*N*K + row_idx*K + k_global
                g_off = expert_in_off + row_idx * fx.Int32(K) + k_global
                v_in = buffer_ops.buffer_load(x_rsrc, g_off, vec_width=1, dtype=in_dtype)
                i_idx = ArithValue(fx.Int32(i)).index_cast(T.index)
                lds.store(v_in, [i_idx, tid_idx])

            gpu.barrier()  # may be elided (1 wave per workgroup)

            # ---------- PHASE 2: per-lane column read + compute ----------
            chunks = []
            local_amax = fx.Float32(0.0)
            for c in range_constexpr(0, _CHUNKS_PER_BLOCK):
                elems = []
                for j in range_constexpr(0, _VEC):
                    row_lds = c * _VEC + j
                    row_idx_lds = ArithValue(fx.Int32(row_lds)).index_cast(T.index)
                    elems.append(lds.load([row_idx_lds, tid_idx]))
                if input_dtype_name == "torch.bfloat16":
                    vec_in = vector.from_elements(T.vec(4, T.bf16), elems)
                    vec_f32 = arith.extf(T.vec(4, T.f32), vec_in)
                else:
                    vec_f32 = vector.from_elements(T.vec(4, T.f32), elems)
                chunks.append(vec_f32)
                chunk_amax = fx.math.absf(vec_f32).reduce(ReductionOp.MAX)
                local_amax = local_amax.maximumf(chunk_amax)

            # ---------- SCALE derivation (FLOOR) ----------
            bits = ArithValue(local_amax).bitcast(T.i32)
            exp_biased = (bits.shrui(fx.Int32(23))) & fx.Int32(0xFF)
            E_amax = exp_biased - fx.Int32(127)
            scale_unb = E_amax - fx.Int32(8)
            scale_unb = arith.maxsi(arith.unwrap(scale_unb), arith.unwrap(fx.Int32(-127)))
            scale_unb = arith.minsi(scale_unb, arith.unwrap(fx.Int32(128)))
            scale_biased = scale_unb + fx.Int32(_E8M0_BIAS)
            scale_u8 = arith.trunci(T.i8, arith.unwrap(scale_biased))

            neg_unb = fx.Int32(0) - scale_unb
            neg_unb_f = arith.sitofp(T.f32, arith.unwrap(neg_unb))
            inv_scale = fx.math.exp2(neg_unb_f)

            # ---------- PHASE 3: quantize, clamp, pack, col-major store ----------
            # Per-expert col-major (N, K) layout: element [n, k] in expert e
            # at byte offset = expert*(N*K) + k*N + n.
            # Lane k_global owns column k_global; its 32 fp8 bytes are at
            #   expert*(N*K) + k_global*N + n_row_base + 0..31.
            f32x4_ty = T.vec(4, T.f32)
            f8_max_v = vector.broadcast(f32x4_ty, arith.unwrap(fx.Float32(F8_MAX)))
            f8_min_v = vector.broadcast(f32x4_ty, arith.unwrap(fx.Float32(-F8_MAX)))

            expert_out_byte_off = expert * fx.Int32(EXPERT_OUTPUT_BYTES)
            col_byte_base = expert_out_byte_off + k_global * fx.Int32(N) + n_row_base
            col_i32_base = col_byte_base // fx.Int32(_VEC)

            for c in range_constexpr(0, _CHUNKS_PER_BLOCK):
                qv = chunks[c] * inv_scale
                qv = arith.maximumf(qv, f8_min_v)
                qv = arith.minimumf(qv, f8_max_v)
                qv0 = vector.extract(qv, static_position=[0], dynamic_position=[])
                qv1 = vector.extract(qv, static_position=[1], dynamic_position=[])
                qv2 = vector.extract(qv, static_position=[2], dynamic_position=[])
                qv3 = vector.extract(qv, static_position=[3], dynamic_position=[])
                out = fx.Int32(0)
                out = rocdl.cvt_pk_fp8_f32(
                    res=T.i32, src_a=qv0, src_b=qv1, old=arith.unwrap(out), word_sel=False,
                )
                out = rocdl.cvt_pk_fp8_f32(
                    res=T.i32, src_a=qv2, src_b=qv3, old=out, word_sel=True,
                )
                buffer_ops.buffer_store(out, q_rsrc, col_i32_base + fx.Int32(c))

            # ---------- PHASE 4: scale store ----------
            # scales shape (E, N // 32, K) row-major.
            # Element [expert, n_block, k_global] at offset
            #   expert * (N//32) * K  +  n_block * K  +  k_global.
            scale_off = (
                expert * fx.Int32(SCALES_PER_EXPERT)
                + n_block * fx.Int32(K)
                + k_global
            )
            buffer_ops.buffer_store(scale_u8, s_rsrc, scale_off)

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
                block=(_BLOCK_THREADS, 1, 1),
                stream=stream,
            )

        return launch_quantize

else:
    def _compile_mxfp8_quantize_flydsl_3d(*_args, **_kwargs):
        missing = _missing_flydsl_runtime_packages()
        raise ImportError(
            "FlyDSL is not available. Missing package(s): "
            f"{', '.join(missing)}."
        )


def mxfp8_quantize_flydsl_3d(
    x: torch.Tensor,
    block_size: int = 32,
    scaling_mode: str = "floor",
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Quantize a 3D MoE tensor to MXFP8 using a FlyDSL kernel on AMD GPUs.

    AMD counterpart of ``mxfp8_quantize_cutedsl_3d``. Quantizes along the
    N dimension; output data is per-expert column-major.

    Args:
        x: Input tensor, shape ``(E, N, K)``, dtype ``bfloat16`` or
            ``float32``, row-major contiguous, on a HIP/CUDA device.
        block_size: MXFP8 block size along N. Only ``32`` is supported.
        scaling_mode: ``"floor"`` only in this baseline.

    Returns:
        Tuple ``(q_data, scales)``:
            * ``q_data``: shape ``(E, N, K)`` ``torch.float8_e4m3fn``,
              strides ``(N*K, 1, N)`` (per-expert col-major).
            * ``scales``: shape ``(E, N // 32, K)`` viewed as
              ``torch.float8_e8m0fnu``.
    """
    assert x.dtype in (torch.bfloat16, torch.float32), (
        f"Input dtype must be bfloat16 or float32, got {x.dtype}"
    )
    assert x.is_cuda, "Input tensor must be on a CUDA/HIP device"
    assert block_size == _N_BLOCK, (
        f"Only block_size={_N_BLOCK} is supported (got {block_size})"
    )
    assert x.is_contiguous(), "Input must be contiguous (row-major)"
    assert x.ndim == 3, f"Expected 3D input, got shape {tuple(x.shape)}"

    E, N, K = x.shape
    assert N % _N_BLOCK == 0, (
        f"N ({N}) must be divisible by block_size ({_N_BLOCK})"
    )
    assert K % _K_TILE == 0, (
        f"K ({K}) must be divisible by {_K_TILE} in this baseline kernel."
    )

    N_BLOCKS = N // _N_BLOCK

    # Allocate the kernel's output as a flat int32 buffer (matches the 32x1
    # idiom — kernel emits 32-bit packed-fp8 stores). Then alias its storage
    # as the per-expert col-major fp8 tensor for the caller.
    q_i32_flat = torch.empty(E * N * K // 4, dtype=torch.int32, device=x.device)
    # Scales: (E, N // 32, K) row-major.
    scales_u8 = torch.empty((E, N_BLOCKS, K), device=x.device, dtype=torch.uint8)

    launch = _compile_mxfp8_quantize_flydsl_3d(
        str(x.dtype), scaling_mode, int(N), int(K),
    )

    import flydsl.compiler as flyc

    # Row-major (E, N, K): leading_dim is K = dim 2, divisible by 2 (bf16) or 4 (f32).
    x_fly = flyc.from_dlpack(x).mark_layout_dynamic(leading_dim=2, divisibility=2)

    grid_n = N // _N_BLOCK
    grid_k = K // _K_TILE
    grid_e = E
    stream = torch.cuda.current_stream()
    launch(x_fly, q_i32_flat, scales_u8,
           int(grid_n), int(grid_k), int(grid_e), stream=stream)

    # Alias i32 buffer as per-expert col-major (E, N, K) fp8 — strides (N*K, 1, N).
    q_data = torch.empty(0, dtype=torch.float8_e4m3fn, device=x.device)
    q_data.set_(q_i32_flat.untyped_storage(), 0, (E, N, K), (N * K, 1, N))

    return q_data, scales_u8.view(torch.float8_e8m0fnu)
