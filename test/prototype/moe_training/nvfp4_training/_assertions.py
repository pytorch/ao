# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD 3-Clause license found in the
# LICENSE file in the root directory of this source tree.

"""Shared assertions for the NVFP4 kernel tests.

Deliberately pytest-free so an out-of-tree comparison script can reuse them. The
vocabulary here is what each output can honestly be held to against the plain-PyTorch
reference in ``test.prototype.moe_training.nvfp4_training.nvfp4_reference``. Scales,
amaxes, and RTNE FP4 codes are all bitwise under TransformerEngine's default recipe.
"""

from typing import Optional

import torch


def assert_scales_bitwise(got: torch.Tensor, ref: torch.Tensor, label: str) -> None:
    """Compare E4M3 scales as raw bytes.

    Both sides are flattened: the kernels return 4-D swizzled ``(M//128, N//64, 32, 16)``
    while ``to_blocked`` returns a flat buffer, and the byte sequence is what matters.
    """
    got_b = got.flatten().contiguous().view(torch.uint8)
    ref_b = ref.flatten().contiguous().view(torch.uint8)
    assert got_b.shape == ref_b.shape, (
        f"{label}: shape mismatch {tuple(got_b.shape)} vs {tuple(ref_b.shape)}"
    )
    assert torch.equal(got_b, ref_b), (
        f"{label}: {(got_b != ref_b).sum().item()}/{got_b.numel()} fp8 scale bytes differ"
    )


def unpack_fp4_nibbles(codes: torch.Tensor) -> torch.Tensor:
    """(R, C//2) packed uint8 -> (R, C) nibbles, undoing ``pack_uint4``'s even-in-low order.

    Returns ``long``, not ``uint8``: callers subtract two code tensors, and in uint8 a
    difference of -1 wraps to 255.
    """
    lo = (codes & 0xF).long()
    hi = (codes >> 4).long()
    return torch.stack((lo, hi), dim=-1).reshape(codes.shape[0], -1)


def unpack_fp4_magnitudes(codes: torch.Tensor) -> torch.Tensor:
    """FP4 magnitude codes (0-7), sign bit stripped."""
    return unpack_fp4_nibbles(codes) & 0x7


def assert_codes_bitwise(got: torch.Tensor, ref: torch.Tensor, label: str) -> None:
    """Compare packed FP4 codes byte-for-byte against the TE-default reference."""
    assert got.shape == ref.shape, (
        f"{label}: shape mismatch {tuple(got.shape)} vs {tuple(ref.shape)}"
    )
    assert torch.equal(got, ref), (
        f"{label}: {(got != ref).sum().item()}/{got.numel()} code bytes differ"
    )


def dequantize(
    codes: torch.Tensor,
    scales: torch.Tensor,
    global_amax: torch.Tensor,
    *,
    is_swizzled: bool = True,
) -> torch.Tensor:
    """Dequantize NVFP4 codes + E4M3 scales back to float32."""
    from torchao.prototype.mx_formats.nvfp4_tensor import (
        NVFP4Tensor,
        per_tensor_amax_to_scale,
    )

    return (
        NVFP4Tensor(
            codes.contiguous(),
            scales.contiguous(),
            16,
            torch.bfloat16,
            per_tensor_scale=per_tensor_amax_to_scale(global_amax),
            is_swizzled_scales=is_swizzled,
        )
        .dequantize()
        .float()
    )


def assert_scales_finite(scales: torch.Tensor, label: str = "scales") -> None:
    """No lower-bound check: TE emits a zero per-vector scale for a zero or underflowing
    block, so pinning small scales to a nonzero floor would contradict the ground truth."""
    assert torch.isfinite(scales.to(torch.float32)).all(), f"{label} must be finite"


def assert_scales_adjacent(
    got: torch.Tensor, ref: torch.Tensor, label: str, *, max_ulps: int = 1
) -> None:
    """fp8 scale bytes equal or within ``max_ulps`` representable steps.

    For comparisons against ``mx_formats.nvfp4_quantize``, which multiplies by a
    reciprocal and applies an E4M3_EPS floor where the kernels follow TE's div_rn with no
    floor. Positive e4m3 bytes are magnitude-monotonic, so a byte delta is a ULP delta.
    """
    got_b = got.flatten().contiguous().view(torch.uint8).to(torch.int16)
    ref_b = ref.flatten().contiguous().view(torch.uint8).to(torch.int16)
    assert got_b.shape == ref_b.shape, (
        f"{label}: shape mismatch {tuple(got_b.shape)} vs {tuple(ref_b.shape)}"
    )
    diff = (got_b - ref_b).abs()
    assert (diff <= max_ulps).all(), (
        f"{label}: {(diff > max_ulps).sum().item()}/{diff.numel()} fp8 scale bytes "
        f"differ by >{max_ulps} ULP (max {diff.max().item()})"
    )


def assert_zero_quantized(
    codes: torch.Tensor,
    scales: torch.Tensor,
    dequantized: Optional[torch.Tensor] = None,
) -> None:
    """An all-zero input packs to zero codes, stores a zero block scale, and dequantizes
    to exactly zero."""
    assert torch.count_nonzero(codes) == 0, "zero input must pack to zero codes"
    assert torch.count_nonzero(scales.to(torch.float32)) == 0, (
        "zero input must store a zero block scale"
    )
    if dequantized is not None:
        assert torch.count_nonzero(dequantized) == 0, (
            "zero input must dequantize to zero"
        )
