# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD 3-Clause license found in the
# LICENSE file in the root directory of this source tree.

"""MXFP8 1x32 quantization kernel for AMD GPUs (FlyDSL counterpart of
``cutedsl_quantize_2d_1x32.py``).

For each row of an ``(M, K)`` tensor, derives one E8M0 scale per 32-element
block of K and emits row-major FP8 E4M3FN data. Scale derivation is FLOOR
mode only (RCEIL has no AMD hardware analog and is deferred).

Parallelization: 1 wave (= 64 lanes) per row; each lane owns one 32-element
quant block. The wave loops over blocks-per-row in chunks of 64 (one per lane).
No cross-lane reduction needed because each block fits in one lane's registers.
"""

from __future__ import annotations

import functools
from typing import Optional, Tuple

import torch

from .flydsl_utils import (
    AMD_WAVE_SIZE,
    BLOCK_SIZE,
    CHUNKS_PER_BLOCK,
    E8M0_EXPONENT_BIAS,
    VEC,
    _flydsl_runtime_available,
    _missing_flydsl_runtime_packages,
)

if _flydsl_runtime_available():
    from .flydsl_utils import current_stream_fast


# Each chunk-iteration of the K-loop processes one quant block per lane:
# AMD_WAVE_SIZE lanes × BLOCK_SIZE elements/block.
_K_PER_CHUNK = AMD_WAVE_SIZE * BLOCK_SIZE  # 2048


if _flydsl_runtime_available():
    import flydsl.compiler as flyc
    import flydsl.expr as fx
    from flydsl.expr import buffer_ops, range_constexpr
    from flydsl.expr.vector import ReductionOp

    # Module-level imports of the in-kernel helpers — see flydsl_utils.py for
    # why this matters (cutedsl uses the same pattern with cute_utils).
    from .flydsl_utils import (
        floor_scale_and_inv_scale,
        make_fp8_clamp_vectors,
        quantize_pack_chunk_to_i32,
    )

    @functools.cache
    def _compile_quantize_2d_1x32(
        input_dtype_name: str,
        scaling_mode: str,
        K: int,
    ):
        """JIT-compile the kernel for a given (dtype, mode, K).

        K is part of the cache key so the inner ``range_constexpr`` loop bound
        (chunks per row) is a Python int captured in the kernel closure.
        """
        if scaling_mode != "floor":
            raise NotImplementedError(
                "FlyDSL MXFP8 1x32 supports scaling_mode='floor' only "
                f"(got {scaling_mode!r}); RCEIL is a planned follow-up."
            )
        if input_dtype_name == "torch.bfloat16":
            in_dtype = fx.BFloat16
        elif input_dtype_name == "torch.float32":
            in_dtype = fx.Float32
        else:
            raise ValueError(f"Unsupported input dtype: {input_dtype_name}")

        K_BLOCKS = K // BLOCK_SIZE
        CHUNKS_PER_ROW = K_BLOCKS // AMD_WAVE_SIZE

        @flyc.kernel(known_block_size=[AMD_WAVE_SIZE, 1, 1])
        def quantize_2d_1x32_kernel(
            x: fx.Tensor,        # (M, K) bf16/f32 row-major
            q: fx.Tensor,        # (M, K) fp8_e4m3fn — addressed as i32 packed
            scales: fx.Tensor,   # (M, K // 32) uint8 E8M0
        ):
            row = fx.block_idx.x
            tid = fx.thread_idx.x

            x_rsrc = buffer_ops.create_buffer_resource(x)
            q_rsrc = buffer_ops.create_buffer_resource(q)
            s_rsrc = buffer_ops.create_buffer_resource(scales)

            f8_min_v, f8_max_v = make_fp8_clamp_vectors()

            for chunk_idx in range_constexpr(0, CHUNKS_PER_ROW):
                block_in_row = chunk_idx * AMD_WAVE_SIZE + tid
                elem_base = row * fx.Int32(K) + block_in_row * BLOCK_SIZE

                # Pass 1: load 8 vec4 chunks of input, accumulate per-block amax.
                chunks = []
                local_amax = fx.Float32(0.0)
                for c in range_constexpr(0, CHUNKS_PER_BLOCK):
                    off = elem_base + fx.Int32(c * VEC)
                    vec_in = buffer_ops.buffer_load(x_rsrc, off, vec_width=VEC, dtype=in_dtype)
                    vec_f32 = vec_in.to(fx.Float32)
                    chunks.append(vec_f32)
                    local_amax = local_amax.maximumf(
                        fx.math.absf(vec_f32).reduce(ReductionOp.MAX)
                    )

                scale_u8, inv_scale = floor_scale_and_inv_scale(local_amax)

                # Pass 2: quantize each retained chunk and write 32-bit packed FP8.
                # Compiler typically fuses 8x i32 stores into 2x dwordx4.
                for c in range_constexpr(0, CHUNKS_PER_BLOCK):
                    out = quantize_pack_chunk_to_i32(
                        chunks[c], inv_scale, f8_min_v, f8_max_v,
                    )
                    i32_off = (elem_base + fx.Int32(c * VEC)) // fx.Int32(VEC)
                    buffer_ops.buffer_store(out, q_rsrc, i32_off)

                buffer_ops.buffer_store(scale_u8, s_rsrc,
                                        row * fx.Int32(K_BLOCKS) + block_in_row)

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
                block=(AMD_WAVE_SIZE, 1, 1),
                stream=stream,
            )

        return launch_quantize

else:
    def _compile_quantize_2d_1x32(*_args, **_kwargs):
        missing = _missing_flydsl_runtime_packages()
        raise ImportError(
            f"FlyDSL is not available. Missing package(s): {', '.join(missing)}."
        )


def mxfp8_quantize_flydsl_2d_1x32(
    x: torch.Tensor,
    block_size: int = 32,
    scaling_mode: str = "floor",
    stage_count: int = 2,
    blocked_scale_output: bool = False,
    offs: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Quantize a 2D (M, K) tensor to MXFP8 along K using a FlyDSL kernel.

    AMD counterpart of :func:`mxfp8_quantize_cutedsl_2d_1x32`. Output data
    is row-major ``(M, K)``; scales are ``(M, K // 32)``.

    ``stage_count`` is accepted for API parity with the cutedsl wrapper
    (TMA pipeline depth) and ignored on AMD. ``blocked_scale_output`` and
    ``offs`` are not yet implemented and raise :class:`NotImplementedError`
    when set; the matching dispatcher in ``quant.py`` enforces the same
    contract one level up.
    """
    if scaling_mode != "floor":
        raise NotImplementedError(
            "mxfp8_quantize_flydsl_2d_1x32: "
            f"scaling_mode={scaling_mode!r} is not supported by the FlyDSL "
            "baseline; only 'floor' is implemented (RCEIL is a planned follow-up)."
        )
    if blocked_scale_output:
        raise NotImplementedError(
            "mxfp8_quantize_flydsl_2d_1x32: blocked_scale_output=True is "
            "tcgen05-specific to SM 10.x and not supported by the FlyDSL baseline."
        )
    if offs is not None:
        raise NotImplementedError(
            "mxfp8_quantize_flydsl_2d_1x32: token-group offs are not yet "
            "supported by the FlyDSL baseline."
        )
    del stage_count  # AMD has no TMA pipeline; accepted for API parity.
    assert x.dtype in (torch.bfloat16, torch.float32), (
        "Input tensor must be float32 or bfloat16"
    )
    assert x.is_cuda, "Input tensor must be CUDA"
    assert x.is_contiguous(), "Input tensor must be contiguous (row-major)"
    assert block_size == BLOCK_SIZE, f"Only block_size={BLOCK_SIZE} is supported"
    M, K = x.shape
    assert K % _K_PER_CHUNK == 0, f"K must be a multiple of {_K_PER_CHUNK}"

    q_data = torch.empty((M, K), device=x.device, dtype=torch.float8_e4m3fn)
    scales_u8 = torch.empty((M, K // BLOCK_SIZE), device=x.device, dtype=torch.uint8)

    launch = _compile_quantize_2d_1x32(str(x.dtype), scaling_mode, K)
    # Pass `x` raw (not via from_dlpack) so FlyDSL's bare-pointer fast path
    # avoids the per-call DLPack adapter overhead.
    launch(x, q_data.view(torch.int32), scales_u8, M,
           stream=current_stream_fast(x.device))
    return q_data, scales_u8.view(torch.float8_e8m0fnu)
