# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD 3-Clause license found in the
# LICENSE file in the root directory of this source tree.

"""MXFP8 2D 1x32 quantization kernel implemented with FlyDSL (AMD GPUs).

AMD counterpart to ``cutedsl_quantize_2d_1x32.py``. For each row of an
``(M, K)`` tensor, derives one E8M0 scale per 32-element block of K and
emits row-major FP8 E4M3FN data.

Initial scope of this baseline (CDNA3+ / RDNA4+):
    * FLOOR scaling mode only. RCEIL is deferred — AMD has no hardware
      analogue to NVIDIA's ``cvt.rp.satfinite.ue8m0x2.f32``, so a software
      round-up implementation is needed; tracked as a follow-up.
    * Input dtype: bf16 or f32, row-major ``(M, K)``.
    * Output: row-major ``(M, K)`` ``torch.float8_e4m3fn`` data and
      ``(M, K // 32)`` E8M0 scales (uint8 container, ``view`` as
      ``torch.float8_e8m0fnu`` to match the cutedsl API).
    * Requires ``K % 2048 == 0`` (= 64 lanes × 32 elements per block).
      Common training shapes (4096, 8192, 16384) satisfy this. Tail
      handling will be added in a follow-up.
    * No ``offs`` / token-group support yet.

Parallelization:
    Grid:  (M,)            — one workgroup per row
    Block: 64 threads      — one AMD wave per workgroup
    Each lane owns one 32-element MXFP8 quantization block. The wave
    iterates over blocks-per-row in chunks of 64. No cross-lane reduction.

Per lane, per block (= one MXFP8 block of 32 elements = 8 chunks of 4):
    PASS 1 (scan):  for each of 8 chunks:
                       buffer_load 4 bf16 -> cast to f32 -> abs -> reduce-max
                       accumulate into running amax
    SCALE:           derive E8M0 byte (Step 3 algebra) and inv_scale
    PASS 2 (apply): for each of 8 chunks (kept in registers from pass 1):
                       multiply by inv_scale -> clamp ±448 -> v_cvt_pk_fp8_f32 ×2
                       buffer_store 32 bits of packed FP8
    SCALE STORE:     buffer_store_byte for the uint8 scale

The compiler typically:
    * fuses 8× ``buffer_store_dword`` into 2× ``buffer_store_dwordx4``
    * realizes ``inv_scale = exp2(-scale_unb)`` as ``v_ldexp_f32``
    * folds the cross-chunk amax accumulation into the per-chunk ``v_max3_f32``
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


# Tile constants. Wave size 64 × 32 elements per block = 2048 K elements/chunk.
_BLOCK_THREADS = 64
_BLOCK_SIZE = 32                                # MXFP8 quant block size
_VEC = 4                                         # 4 fp8 = 1 i32 (cvt_pk packing)
_CHUNKS_PER_BLOCK = _BLOCK_SIZE // _VEC          # 8
_K_PER_CHUNK = _BLOCK_THREADS * _BLOCK_SIZE      # 2048
_E8M0_BIAS = 127


# Module-level FlyDSL imports + kernel definition, guarded so this file is
# importable in environments without FlyDSL (mirroring cute_utils.py). The
# @flyc.kernel must live at module scope (not inside a closure) for the AST
# rewriter's free-var counting to work cleanly.
if _flydsl_runtime_available():
    import flydsl.compiler as flyc
    import flydsl.expr as fx
    from flydsl.expr import arith, buffer_ops, range_constexpr, rocdl, vector
    from flydsl.expr.arith import ArithValue
    from flydsl.expr.typing import T
    from flydsl.expr.vector import ReductionOp

    @functools.cache
    def _compile_mxfp8_quantize_flydsl_2d_1x32(
        input_dtype_name: str,
        scaling_mode: str,
        K: int,
    ):
        """JIT-compile the FlyDSL kernel for a given (dtype, mode, K).

        K is part of the cache key because ``range_constexpr`` loop bounds
        must be Python ints captured in the kernel closure. Common K values
        (1024/2048/4096/8192) yield only a handful of cached artifacts.
        """
        if scaling_mode != "floor":
            raise NotImplementedError(
                "FlyDSL MXFP8 quantize_2d_1x32 supports scaling_mode='floor' only "
                f"(got {scaling_mode!r}); RCEIL is a planned follow-up."
            )
        if input_dtype_name == "torch.bfloat16":
            in_dtype = fx.BFloat16
        elif input_dtype_name == "torch.float32":
            in_dtype = fx.Float32
        else:
            raise ValueError(f"Unsupported input dtype: {input_dtype_name}")

        # Closure-captured Python ints — these become loop bounds inside the
        # kernel. The AST rewriter inspects the function's free vars so we
        # keep the set small and concrete.
        K_BLOCKS_CONST = K // _BLOCK_SIZE              # number of MXFP8 blocks per row
        CHUNKS_PER_ROW = K_BLOCKS_CONST // _BLOCK_THREADS  # K-loop iterations per wave

        @flyc.kernel(known_block_size=[_BLOCK_THREADS, 1, 1])
        def quantize_2d_1x32_kernel(
            x: fx.Tensor,        # (M, K) bf16 or f32, row-major
            q: fx.Tensor,        # (M, K) fp8_e4m3fn — addressed as i32 packed
            scales: fx.Tensor,   # (M, K // 32) uint8 E8M0
        ):
            row = fx.block_idx.x      # workgroup id == row index
            tid = fx.thread_idx.x     # 0..63, lane in wave

            x_rsrc = buffer_ops.create_buffer_resource(x)
            q_rsrc = buffer_ops.create_buffer_resource(q)
            s_rsrc = buffer_ops.create_buffer_resource(scales)

            # Iterate over K in chunks of 2048 elements (= 64 lanes × 32 elem/block).
            # Each lane in the wave owns one MXFP8 block per chunk-iteration.
            for chunk_idx in range_constexpr(0, CHUNKS_PER_ROW):
                # Index of this lane's block within its row.
                block_in_row = chunk_idx * _BLOCK_THREADS + tid

                # First element index (within the row) that this lane owns.
                elem_base_in_row = block_in_row * _BLOCK_SIZE

                # Global element offset into the input/output flat arrays.
                elem_base = row * fx.Int32(K) + elem_base_in_row

                # ---------- PASS 1: scan ----------
                chunks = []                       # keep f32 chunks in regs
                local_amax = fx.Float32(0.0)
                for c in range_constexpr(0, _CHUNKS_PER_BLOCK):
                    off = elem_base + fx.Int32(c * _VEC)
                    vec_in = buffer_ops.buffer_load(
                        x_rsrc, off, vec_width=_VEC, dtype=in_dtype,
                    )
                    vec_f32 = vec_in.to(fx.Float32)
                    chunks.append(vec_f32)
                    chunk_amax = fx.math.absf(vec_f32).reduce(ReductionOp.MAX)
                    local_amax = local_amax.maximumf(chunk_amax)

                # ---------- SCALE derivation ----------
                # Inlined from flydsl_utils.compute_scale_floor_f32 for the same
                # closure-shape reason as above. The compiler folds the algebra
                # to ~3 ALU ops total (see Step 3 in scratch/flydsl_mxfp8_steps).
                bits = ArithValue(local_amax).bitcast(T.i32)
                exp_biased = (bits.shrui(fx.Int32(23))) & fx.Int32(0xFF)
                E_amax = exp_biased - fx.Int32(127)
                scale_unb = E_amax - fx.Int32(8)
                scale_unb = arith.maxsi(arith.unwrap(scale_unb), arith.unwrap(fx.Int32(-127)))
                scale_unb = arith.minsi(scale_unb, arith.unwrap(fx.Int32(128)))
                scale_biased = scale_unb + fx.Int32(_E8M0_BIAS)
                scale_u8 = arith.trunci(T.i8, arith.unwrap(scale_biased))

                # inv_scale = 2 ^ (-scale_unb)
                # Compiler typically lowers this whole expression to a single
                # v_ldexp_f32 used implicitly in the multiplies below.
                neg_unb = fx.Int32(0) - scale_unb
                neg_unb_f = arith.sitofp(T.f32, arith.unwrap(neg_unb))
                inv_scale = fx.math.exp2(neg_unb_f)

                # ---------- PASS 2: apply ----------
                f32x4_ty = T.vec(4, T.f32)
                f8_max_v = vector.broadcast(f32x4_ty, arith.unwrap(fx.Float32(F8_MAX)))
                f8_min_v = vector.broadcast(f32x4_ty, arith.unwrap(fx.Float32(-F8_MAX)))
                for c in range_constexpr(0, _CHUNKS_PER_BLOCK):
                    qv = chunks[c] * inv_scale
                    qv = arith.maximumf(qv, f8_min_v)   # FLOOR-mode clamp
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
                    # i32-units offset for the 32-bit packed-fp8 store.
                    i32_off = (elem_base + fx.Int32(c * _VEC)) // fx.Int32(4)
                    buffer_ops.buffer_store(out, q_rsrc, i32_off)

                # One uint8 scale per quant block at offset (row * K_BLOCKS + block_in_row).
                scale_off = row * fx.Int32(K_BLOCKS_CONST) + block_in_row
                buffer_ops.buffer_store(scale_u8, s_rsrc, scale_off)

        @flyc.jit
        def launch_quantize(
            x: fx.Tensor,
            q: fx.Tensor,
            scales: fx.Tensor,
            M: fx.Int32,
            stream: fx.Stream = fx.Stream(None),
        ):
            quantize_2d_1x32_kernel(x, q, scales).launch(
                grid=(M, 1, 1),
                block=(_BLOCK_THREADS, 1, 1),
                stream=stream,
            )

        return launch_quantize

else:
    def _compile_mxfp8_quantize_flydsl_2d_1x32(*_args, **_kwargs):
        missing = _missing_flydsl_runtime_packages()
        raise ImportError(
            "FlyDSL is not available. Missing package(s): "
            f"{', '.join(missing)}."
        )


def mxfp8_quantize_flydsl_2d_1x32(
    x: torch.Tensor,
    block_size: int = 32,
    scaling_mode: str = "floor",
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Quantize a 2D tensor to MXFP8 (1x32) using a FlyDSL kernel on AMD GPUs.

    AMD counterpart of ``mxfp8_quantize_cutedsl_2d_1x32``. Mirrors that
    function's API for direct interchange.

    Args:
        x: Input tensor, shape ``(M, K)``, dtype ``bfloat16`` or ``float32``,
            row-major contiguous, on a HIP/CUDA device.
        block_size: MXFP8 block size along K. Only ``32`` is supported.
        scaling_mode: ``"floor"`` only in this baseline (``"rceil"`` planned).

    Returns:
        Tuple ``(q_data, scales)``:
            * ``q_data``: row-major ``(M, K)`` ``torch.float8_e4m3fn``.
            * ``scales``: ``(M, K // 32)`` viewed as ``torch.float8_e8m0fnu``
              (uint8 container with E8M0 semantics).
    """
    assert x.dtype in (torch.bfloat16, torch.float32), (
        f"Input dtype must be bfloat16 or float32, got {x.dtype}"
    )
    assert x.is_cuda, "Input tensor must be on a CUDA/HIP device"
    assert block_size == _BLOCK_SIZE, (
        f"Only block_size={_BLOCK_SIZE} is supported (got {block_size})"
    )
    assert x.is_contiguous(), "Input must be contiguous (row-major)"
    assert x.ndim == 2, f"Expected 2D input, got shape {tuple(x.shape)}"

    M, K = x.shape
    assert K % _BLOCK_SIZE == 0, (
        f"K ({K}) must be divisible by block_size ({_BLOCK_SIZE})"
    )
    assert K % _K_PER_CHUNK == 0, (
        f"K ({K}) must be a multiple of {_K_PER_CHUNK} in this baseline kernel; "
        "tail handling is a planned follow-up."
    )

    K_BLOCKS = K // _BLOCK_SIZE

    q_data = torch.empty((M, K), device=x.device, dtype=torch.float8_e4m3fn)
    scales_u8 = torch.empty((M, K_BLOCKS), device=x.device, dtype=torch.uint8)

    launch = _compile_mxfp8_quantize_flydsl_2d_1x32(str(x.dtype), scaling_mode, int(K))

    import flydsl.compiler as flyc

    # Row-major (M, K): the stride-1 (leading) dim is K = dim 1.
    # divisibility=8 enables 128-bit vectorized loads (8 bf16 = 16 bytes).
    x_fly = flyc.from_dlpack(x).mark_layout_dynamic(leading_dim=1, divisibility=8)
    # Address q as an int32 view so the kernel can issue 32-bit packed-fp8 stores.
    q_i32_view = q_data.view(torch.int32)

    stream = torch.cuda.current_stream()
    launch(x_fly, q_i32_view, scales_u8, int(M), stream=stream)

    return q_data, scales_u8.view(torch.float8_e8m0fnu)
