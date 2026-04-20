# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD 3-Clause license found in the
# LICENSE file in the root directory of this source tree.

"""MXFP8 2D 32x1 quantization kernel implemented with FlyDSL (AMD GPUs).

AMD counterpart to ``cutedsl_quantize_2d_32x1.py``. For each column of an
``(M, K)`` tensor, derives one E8M0 scale per 32-element block of M and
emits column-major FP8 E4M3FN data.

This is the dim-0 / M-direction transpose-quantize counterpart to
``flydsl_quantize_2d_1x32.py`` (which quantizes along K). Because the
input is row-major but the quant block runs along M (= 32 elements with
stride K in memory), we stage a (32, 64) tile through LDS to convert the
strided per-lane access into coalesced wave-level loads.

Initial scope of this baseline (CDNA3+ / RDNA4+):
    * FLOOR scaling mode only.
    * Input dtype: bf16 or f32, row-major ``(M, K)``.
    * Output: column-major ``(M, K)`` ``torch.float8_e4m3fn`` (stride ``(1, M)``)
      and ``(K, M // 32)`` E8M0 scales viewed as ``torch.float8_e8m0fnu``.
    * Requires ``M % 32 == 0`` and ``K % 64 == 0``.
    * No ``offs`` / token-group support yet.

Parallelization:
    Grid:  (M // 32, K // 64)   — one workgroup per (M-block, K-tile)
    Block: 64 threads (1 wave) — each lane owns one K-column of the tile

Per workgroup:
    PHASE 1 — Cooperative LDS load:
        32 wave-loads, one per row of the tile. Each wave-load is 64 lanes
        × 1 bf16 = 1 cache line, perfectly coalesced.

    PHASE 2 — Per-lane column read + compute:
        Each lane reads its 32 bf16 column from LDS into VGPRs, casts to
        f32, computes intra-vector amax, derives FLOOR-mode E8M0 scale +
        inv_scale, quantizes/clamps/packs 32 f32 -> 8 i32 of 4 fp8 bytes.

    PHASE 3 — Col-major output store:
        Output is (M, K) col-major: lane k's column is contiguous in memory,
        so 8 packed-i32 stores per lane covers 32 fp8 bytes.

    PHASE 4 — Scale store:
        One uint8 scale per K-column of the tile (stored at scales[k_global, m_block]).
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


# Tile constants.
_BLOCK_THREADS = 64                      # = AMD wave size
_M_BLOCK = 32                            # MXFP8 quant block in M direction
_K_TILE = _BLOCK_THREADS                 # one lane per K-col → K_TILE = 64
_VEC = 4                                 # 4 fp8 = 1 i32 (cvt_pk packing)
_CHUNKS_PER_BLOCK = _M_BLOCK // _VEC     # 8
_E8M0_BIAS = 127
_TILE_BYTES_BF16 = _M_BLOCK * _K_TILE * 2   # 4096


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
    def _compile_mxfp8_quantize_flydsl_2d_32x1(
        input_dtype_name: str,
        scaling_mode: str,
        M: int,
        K: int,
    ):
        """JIT-compile the FlyDSL 32x1 kernel for a given (dtype, mode, M, K).

        M is part of the cache key because the col-major output stride is M
        (in element units), and we want it as a Python int constant captured
        in the kernel closure. K is captured for the row-major input stride.
        """
        if scaling_mode != "floor":
            raise NotImplementedError(
                "FlyDSL MXFP8 quantize_2d_32x1 supports scaling_mode='floor' only "
                f"(got {scaling_mode!r}); RCEIL is a planned follow-up."
            )
        if input_dtype_name == "torch.bfloat16":
            in_dtype = fx.BFloat16
        elif input_dtype_name == "torch.float32":
            in_dtype = fx.Float32
        else:
            raise ValueError(f"Unsupported input dtype: {input_dtype_name}")

        # LDS allocation: stage tile as bf16 (no cast on PHASE 1) — mirrors
        # the verified LDS Step 4 path. The cast to f32 happens in PHASE 2
        # via `arith.extf` on a vec<4 x bf16> (element-wise, exact).
        # For f32 input, LDS holds f32 (4 bytes/elem); we still cap at 4*tile.
        in_elem_bytes = 2 if input_dtype_name == "torch.bfloat16" else 4
        tile_bytes = _M_BLOCK * _K_TILE * in_elem_bytes
        arch_str = str(get_rocm_arch())
        # Keep allocators distinct per cache entry to avoid symbol collisions
        # when multiple kernel variants are JITed in the same process.
        sym = f"flydsl_mxfp8_2d_32x1_{input_dtype_name.replace('.', '_')}_smem"
        alloc = SmemAllocator(None, arch=arch_str, global_sym_name=sym)
        lds_off = alloc._align(alloc.ptr, 16)
        alloc.ptr = lds_off + tile_bytes

        @flyc.kernel(known_block_size=[_BLOCK_THREADS, 1, 1])
        def quantize_2d_32x1_kernel(
            x: fx.Tensor,        # (M, K) bf16 or f32, row-major
            q: fx.Tensor,        # (M, K) fp8_e4m3fn, col-major — addressed as i32 packed
            scales: fx.Tensor,   # (K, M // 32) uint8 E8M0
        ):
            # block_idx.x = m_block (which 32-row chunk of M this WG handles)
            # block_idx.y = k_tile  (which 64-col chunk of K this WG handles)
            m_block = fx.block_idx.x
            k_tile = fx.block_idx.y
            tid = fx.thread_idx.x   # 0..63, owns K-col (k_tile * 64 + tid)

            x_rsrc = buffer_ops.create_buffer_resource(x)
            q_rsrc = buffer_ops.create_buffer_resource(q)
            s_rsrc = buffer_ops.create_buffer_resource(scales)

            # LDS tile: shape (M_BLOCK, K_TILE), input dtype, row-major.
            # bf16 input -> bf16 LDS, f32 input -> f32 LDS. Cast (if needed)
            # happens in PHASE 2 via arith.extf on a 4-vector.
            base = alloc.get_base()
            lds = SmemPtr(base, lds_off, in_dtype.ir_type, shape=(_M_BLOCK, _K_TILE))
            lds.get()

            tid_idx = ArithValue(tid).index_cast(T.index)

            # Global element offset of input[m_block * M_BLOCK + i, k_tile * K_TILE + tid].
            # In row-major (M, K), the offset = (m_block * M_BLOCK + i) * K + (k_tile * K_TILE + tid).
            # We pre-compute the constant part once outside the row loop.
            row_base = m_block * fx.Int32(_M_BLOCK)            # m_block * 32
            k_global = k_tile * fx.Int32(_K_TILE) + tid        # k_tile * 64 + tid

            # ---------- PHASE 1: cooperative LDS load (row-by-row, coalesced) ----------
            # Mirror LDS Step 4: store input dtype directly into LDS (no cast).
            # 32 wave-loads, one per row of the tile; each wave-load = 64 lanes
            # × 1 element = 1 cache line, perfectly coalesced.
            for i in range_constexpr(0, _M_BLOCK):
                row_idx = row_base + fx.Int32(i)
                g_off = row_idx * fx.Int32(K) + k_global
                v_in = buffer_ops.buffer_load(x_rsrc, g_off, vec_width=1, dtype=in_dtype)
                i_idx = ArithValue(fx.Int32(i)).index_cast(T.index)
                lds.store(v_in, [i_idx, tid_idx])

            gpu.barrier()  # may be elided (1 wave per workgroup)

            # ---------- PHASE 2: per-lane column read + compute ----------
            # Mirror LDS Step 4: read 4 scalars from LDS, build a vec<4>,
            # cast to f32 via arith.extf (exact for bf16 -> f32, identity for
            # f32 -> f32 via a noop ext we elide).
            chunks = []
            local_amax = fx.Float32(0.0)
            for c in range_constexpr(0, _CHUNKS_PER_BLOCK):
                elems = []
                for j in range_constexpr(0, _VEC):
                    row_lds = c * _VEC + j
                    row_idx = ArithValue(fx.Int32(row_lds)).index_cast(T.index)
                    elems.append(lds.load([row_idx, tid_idx]))
                if input_dtype_name == "torch.bfloat16":
                    vec_in = vector.from_elements(T.vec(4, T.bf16), elems)
                    vec_f32 = arith.extf(T.vec(4, T.f32), vec_in)
                else:
                    vec_f32 = vector.from_elements(T.vec(4, T.f32), elems)
                chunks.append(vec_f32)
                # Per-chunk amax via abs + intra-vector max (Step 2 path).
                chunk_amax = fx.math.absf(vec_f32).reduce(ReductionOp.MAX)
                local_amax = local_amax.maximumf(chunk_amax)

            # ---------- SCALE derivation (Step 3 path) ----------
            bits = ArithValue(local_amax).bitcast(T.i32)
            exp_biased = (bits.shrui(fx.Int32(23))) & fx.Int32(0xFF)
            E_amax = exp_biased - fx.Int32(127)
            scale_unb = E_amax - fx.Int32(8)
            scale_unb = arith.maxsi(arith.unwrap(scale_unb), arith.unwrap(fx.Int32(-127)))
            scale_unb = arith.minsi(scale_unb, arith.unwrap(fx.Int32(128)))
            scale_biased = scale_unb + fx.Int32(_E8M0_BIAS)
            scale_u8 = arith.trunci(T.i8, arith.unwrap(scale_biased))

            # inv_scale = 2 ^ (-scale_unb) — compiler typically lowers to v_ldexp_f32.
            neg_unb = fx.Int32(0) - scale_unb
            neg_unb_f = arith.sitofp(T.f32, arith.unwrap(neg_unb))
            inv_scale = fx.math.exp2(neg_unb_f)

            # ---------- PHASE 3: quantize, clamp, pack, store col-major ----------
            f32x4_ty = T.vec(4, T.f32)
            f8_max_v = vector.broadcast(f32x4_ty, arith.unwrap(fx.Float32(F8_MAX)))
            f8_min_v = vector.broadcast(f32x4_ty, arith.unwrap(fx.Float32(-F8_MAX)))

            # Output q is col-major (M, K) with stride (1, M); we receive a
            # 1D i32 view of the underlying storage (length M*K/4). For a
            # value at logical (m, k), the byte offset is k*M + m, and the
            # i32-unit offset is (k*M + m) // 4.
            #
            # Lane k_global writes 32 contiguous fp8 bytes for its column at
            # bytes [k_global*M + row_base, ..., k_global*M + row_base + 31].
            # That's 8 i32s starting at i32-offset (k_global*M + row_base) // 4.
            col_byte_base = k_global * fx.Int32(M) + row_base
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
                # Write 1 i32 = 4 fp8 bytes at column-relative i32-offset (col_i32_base + c).
                buffer_ops.buffer_store(out, q_rsrc, col_i32_base + fx.Int32(c))

            # ---------- PHASE 4: scale store ----------
            # Scales shape (K, M // 32). Element [k_global, m_block] at offset
            # k_global * (M//32) + m_block.
            scale_off = k_global * fx.Int32(M // _M_BLOCK) + m_block
            buffer_ops.buffer_store(scale_u8, s_rsrc, scale_off)

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
                block=(_BLOCK_THREADS, 1, 1),
                stream=stream,
            )

        return launch_quantize

else:
    def _compile_mxfp8_quantize_flydsl_2d_32x1(*_args, **_kwargs):
        missing = _missing_flydsl_runtime_packages()
        raise ImportError(
            "FlyDSL is not available. Missing package(s): "
            f"{', '.join(missing)}."
        )


def mxfp8_quantize_flydsl_2d_32x1(
    x: torch.Tensor,
    block_size: int = 32,
    scaling_mode: str = "floor",
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Quantize a 2D tensor to MXFP8 (32x1) using a FlyDSL kernel on AMD GPUs.

    AMD counterpart of ``mxfp8_quantize_cutedsl_2d_32x1``. Quantizes along
    the M dimension (each (K-column, M-block-of-32) gets one E8M0 scale).
    Output data is column-major to match the cutedsl version.

    Args:
        x: Input tensor, shape ``(M, K)``, dtype ``bfloat16`` or ``float32``,
            row-major contiguous, on a HIP/CUDA device.
        block_size: MXFP8 block size along M. Only ``32`` is supported.
        scaling_mode: ``"floor"`` only in this baseline (``"rceil"`` planned).

    Returns:
        Tuple ``(q_data, scales)``:
            * ``q_data``: column-major ``(M, K)`` ``torch.float8_e4m3fn``
              (stride ``(1, M)``).
            * ``scales``: ``(K, M // 32)`` viewed as ``torch.float8_e8m0fnu``.
    """
    assert x.dtype in (torch.bfloat16, torch.float32), (
        f"Input dtype must be bfloat16 or float32, got {x.dtype}"
    )
    assert x.is_cuda, "Input tensor must be on a CUDA/HIP device"
    assert block_size == _M_BLOCK, (
        f"Only block_size={_M_BLOCK} is supported (got {block_size})"
    )
    assert x.is_contiguous(), "Input must be contiguous (row-major)"
    assert x.ndim == 2, f"Expected 2D input, got shape {tuple(x.shape)}"

    M, K = x.shape
    assert M % _M_BLOCK == 0, (
        f"M ({M}) must be divisible by block_size ({_M_BLOCK})"
    )
    assert K % _K_TILE == 0, (
        f"K ({K}) must be divisible by {_K_TILE} in this baseline kernel."
    )

    M_BLOCKS = M // _M_BLOCK

    # Allocate the kernel's output buffer as a contiguous 1D int32 tensor
    # (the kernel issues 32-bit packed-fp8 stores; it expects an int32 view).
    # We then alias its storage as a col-major (M, K) fp8 tensor for the
    # caller — same bytes, different metadata. We can't call .view(int32)
    # directly on a col-major fp8 tensor because col-major stride[-1] = M != 1.
    q_i32_flat = torch.empty(M * K // 4, dtype=torch.int32, device=x.device)
    # Scales: (K, M // 32) row-major.
    scales_u8 = torch.empty((K, M_BLOCKS), device=x.device, dtype=torch.uint8)

    launch = _compile_mxfp8_quantize_flydsl_2d_32x1(
        str(x.dtype), scaling_mode, int(M), int(K),
    )

    import flydsl.compiler as flyc

    # Row-major (M, K): leading_dim is K = dim 1.
    x_fly = flyc.from_dlpack(x).mark_layout_dynamic(leading_dim=1, divisibility=2)

    grid_m = M // _M_BLOCK
    grid_k = K // _K_TILE
    stream = torch.cuda.current_stream()
    launch(x_fly, q_i32_flat, scales_u8, int(grid_m), int(grid_k), stream=stream)

    # Alias the i32 buffer's storage as col-major (M, K) fp8 — this is a
    # zero-copy reinterpret: the bytes the kernel wrote are now visible at
    # q_data[m, k] via the col-major (stride (1, M)) layout.
    q_data = torch.empty(0, dtype=torch.float8_e4m3fn, device=x.device)
    q_data.set_(q_i32_flat.untyped_storage(), 0, (M, K), (1, M))

    return q_data, scales_u8.view(torch.float8_e8m0fnu)
