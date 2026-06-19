# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""Register-resident FWHT(32) + sign transform for the MXFP4 + RHT cast.

Implements, per 32-element block, the normalized Fast Walsh-Hadamard Transform
followed by an elementwise sign multiply:

    out = (FWHT(vals) / sqrt(32)) * sign

This equals ``vals @ hadamard_matrix(32) * sign`` where ``hadamard_matrix(32)``
is torchao's normalized (1/sqrt(32)) symmetric Sylvester/Walsh-Hadamard matrix
(see ``torchao/prototype/spinquant/hadamard_utils.py``).

The transform is a pure device helper (``fwht32_sign``) so the fused Task 4
kernel can call it per block from registers. ``fwht32_sign_host`` is a tiny
one-block-per-row kernel used only to validate the device helper against a dense
torch reference.
"""

import math

import torch

from .cute_utils import _cutedsl_runtime_available

# 1 / sqrt(32); applied once after the 5 butterfly stages.
_INV_SQRT_32 = 1.0 / math.sqrt(32.0)


if _cutedsl_runtime_available():
    import cutlass
    import cutlass.cute as cute

    # Normalization scalar as a device Float32 constant.
    INV_SQRT_32 = cutlass.Float32(_INV_SQRT_32)

    @cute.jit
    def fwht32_sign(vals: cute.Tensor, sign: cute.Tensor) -> cute.Tensor:
        """In-register normalized FWHT(32) followed by an elementwise sign mul.

        Computes, for the length-32 register fragment ``vals``::

            out[j] = (FWHT(vals) / sqrt(32))[j] * sign[j]

        which equals ``(vals @ hadamard_matrix(32))[j] * sign[j]``.

        The transform is the standard radix-2 decimation-in-time Walsh-Hadamard
        butterfly over strides ``s in {1, 2, 4, 8, 16}``. For each stride, every
        pair ``(i, i + s)`` with ``(i & s) == 0`` is touched exactly once::

            a, b = vals[i] + vals[i + s], vals[i] - vals[i + s]
            vals[i], vals[i + s] = a, b

        All pair indices are compile-time constants (the 5 stages and their
        pairings are fixed for a length-32 block), so the loops are unrolled via
        ``cutlass.range_constexpr``. This pairing/orientation is validated
        bit-for-bit-close against ``hadamard_matrix(32)`` by
        ``test_mxfp4_rht_cutedsl``.

        Args:
            vals: length-32 ``Float32`` register fragment (modified in place and
                also returned).
            sign: length-32 register fragment of ``{-1, +1}`` values (any
                arithmetic dtype; cast to ``Float32`` for the multiply).

        Returns:
            The transformed length-32 ``Float32`` fragment (the same ``vals``).
        """
        # 5-stage in-register butterfly. Strides are powers of two up to 16.
        for stage in cutlass.range_constexpr(5):
            s = 1 << stage
            for i in cutlass.range_constexpr(32):
                if cutlass.const_expr((i & s) == 0):
                    a = cutlass.Float32(vals[i])
                    b = cutlass.Float32(vals[i + s])
                    vals[i] = a + b
                    vals[i + s] = a - b

        # Normalize then apply the sign vector.
        for j in cutlass.range_constexpr(32):
            vals[j] = cutlass.Float32(vals[j]) * INV_SQRT_32 * cutlass.Float32(sign[j])

        return vals

    @cute.kernel
    def _fwht32_sign_kernel(
        gx: cute.Tensor,
        gsign: cute.Tensor,
        gout: cute.Tensor,
        num_rows: cutlass.Int32,
    ):
        """One thread per row: load 32 f32 + sign, apply ``fwht32_sign``, store.

        ``gx`` / ``gout`` are 2D ``(num_rows, 32)`` float32 views; ``gsign`` is a
        1D ``(32,)`` int32 view broadcast across all rows.
        """
        tidx, _, _ = cute.arch.thread_idx()
        bidx, _, _ = cute.arch.block_idx()
        bdim, _, _ = cute.arch.block_dim()
        row = bidx * bdim + tidx
        if row < num_rows:
            vals = cute.make_rmem_tensor((32,), cutlass.Float32)
            sign = cute.make_rmem_tensor((32,), cutlass.Float32)
            for j in cutlass.range_constexpr(32):
                vals[j] = cutlass.Float32(gx[row, j])
                sign[j] = cutlass.Float32(gsign[j])
            fwht32_sign(vals, sign)
            for j in cutlass.range_constexpr(32):
                gout[row, j] = cutlass.Float32(vals[j])

    @cute.jit
    def _fwht32_sign_launch(
        gx: cute.Tensor,
        gsign: cute.Tensor,
        gout: cute.Tensor,
        num_rows: cutlass.Int32,
        stream,
    ):
        threads = 128
        grid = (num_rows + threads - 1) // threads
        _fwht32_sign_kernel(gx, gsign, gout, num_rows).launch(
            grid=(grid, 1, 1),
            block=(threads, 1, 1),
            cluster=(1, 1, 1),
            stream=stream,
        )

    def fwht32_sign_host(x: torch.Tensor, sign: torch.Tensor) -> torch.Tensor:
        """Apply the normalized FWHT(32) + sign transform to each row of ``x``.

        Test/validation entry only -- the production kernel inlines the same
        ``fwht32_sign`` device helper per block. Matches (to fp32 rounding)
        ``(x @ hadamard_matrix(32)) * sign`` row-by-row.

        Args:
            x: 2D float32 CUDA tensor ``[N, 32]``.
            sign: length-32 integer CUDA tensor of ``{-1, +1}`` values.

        Returns:
            A ``(N, 32)`` float32 CUDA tensor.
        """
        import cuda.bindings.driver as cuda
        from cutlass.cute.runtime import from_dlpack

        assert x.is_cuda, "input must be a CUDA tensor"
        assert x.dim() == 2 and x.shape[1] == 32, "input must be 2D [N, 32]"
        assert x.dtype == torch.float32, "input must be float32"
        assert sign.is_cuda, "sign must be a CUDA tensor"
        assert sign.numel() == 32, "sign must have exactly 32 elements"

        num_rows = x.shape[0]
        x = x.contiguous()
        sign_i32 = sign.to(torch.int32).contiguous()
        out = torch.empty((num_rows, 32), device=x.device, dtype=torch.float32)

        gx = from_dlpack(x)
        gsign = from_dlpack(sign_i32)
        gout = from_dlpack(out)
        stream = cuda.CUstream(int(torch.cuda.current_stream().cuda_stream))
        _fwht32_sign_launch(gx, gsign, gout, cutlass.Int32(num_rows), stream)
        return out
