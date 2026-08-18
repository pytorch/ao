# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD 3-Clause license found in the
# LICENSE file in the root directory of this source tree.

"""Host-side precondition validation for the cuDNN-frontend MXFP8 grouped-MLP ops.

The cuDNN FE grouped kernels hard-code a 256-row group granularity
(``FIX_PAD_SIZE = 256``): per-expert row counts that are only multiples of 128
SILENTLY and NONDETERMINISTICALLY corrupt results (the corruption locus
migrates between identical-input reruns, consistent with reads of stale memory
at sub-256 group boundaries). No smoke test can prove a misaligned
configuration safe, so the alignment contract is enforced in two tiers:

* ALWAYS-ON metadata-only checks (no host/device sync, FakeTensor-safe):
  dims, dtypes, devices, strides, the allocated row count ``R % 256 == 0``.
* OPT-IN offset-VALUE checks behind ``TORCHAO_MXFP8_VALIDATE_OFFSETS=1``
  (one D2H sync per call; validation runs only, skipped for fake tensors):
  offsets nondecreasing, every per-expert row count ``% 256 == 0``, and
  ``offsets[-1] <= R``. In a default build the offset values are a documented
  caller invariant provided by a pad_multiple=256 token dispatcher; there is
  NO device-side enforcement.

Checks raise ValueError rather than asserting, so ``python -O`` cannot strip
them. Metadata gates run before any ``data_ptr()`` gate so FakeTensor tracing
exercises the same checks.
"""

import os
from typing import Optional

import torch

__all__ = [
    "DIM_ALIGNMENT",
    "ROW_GROUP_ALIGNMENT",
    "SCALE_BLOCK_SIZE",
    "blocked_scale_numel",
    "host_offsets_validation_enabled",
    "validate_allocated_rows",
    "validate_blocked_scales",
    "validate_feature_dims",
    "validate_group_offsets",
    "validate_operand",
    "validate_ragged_colwise_scales",
    "_is_fake",
]

# MXFP8 scaling block: 32 values share one E8M0 scale.
SCALE_BLOCK_SIZE = 32
# tcgen05 blocked scale tile: 128 rows x 4 columns, 512 bytes.
SCALE_TILE_ROWS = 128
SCALE_TILE_COLS = 4
# Feature-dimension granularity (D and F).
DIM_ALIGNMENT = 128
# Row-count granularity: per-expert groups AND the allocated row count. This is
# the cuDNN FE kernels' FIX_PAD_SIZE, stricter than the archived family's 128.
ROW_GROUP_ALIGNMENT = 256
# Byte alignment for TMA/vectorized accesses.
_PTR_ALIGNMENT = 16

_SCALE_DTYPES = (torch.uint8, torch.float8_e8m0fnu)


def _round_up(x: int, to: int) -> int:
    return ((x + to - 1) // to) * to


def blocked_scale_numel(rows: int, cols: int) -> int:
    """Element count of the blocked E8M0 buffer for a logical [rows, cols] scale matrix.

    ``cols`` counts scale values, i.e. the reduced dimension divided by 32.
    """
    return _round_up(rows, SCALE_TILE_ROWS) * _round_up(cols, SCALE_TILE_COLS)


def host_offsets_validation_enabled() -> bool:
    """Opt-in host-side offset validation. Off by default: it forces a D2H sync."""
    return os.environ.get("TORCHAO_MXFP8_VALIDATE_OFFSETS", "0") == "1"


def _is_fake(tensor: torch.Tensor) -> bool:
    """True for meta/fake tensors, which have no usable data pointer or values."""
    if tensor.device.type == "meta":
        return True
    try:
        from torch._subclasses.fake_tensor import FakeTensor
    except ImportError:
        return False
    return isinstance(tensor, FakeTensor)


def validate_group_offsets(
    offsets: torch.Tensor,
    *,
    num_groups: int,
    allocated_rows: int,
    device: Optional[torch.device] = None,
    name: str = "offsets",
) -> None:
    """Validate the exclusive-end group offsets tensor.

    Metadata is always checked. The offset VALUES (nondecreasing, per-expert
    row counts % 256, offsets[-1] <= R) are checked only when
    ``host_offsets_validation_enabled()`` and the tensor is not fake, because
    reading them forces a D2H sync.
    """
    if not isinstance(offsets, torch.Tensor):
        raise ValueError(f"{name} must be a torch.Tensor, got {type(offsets)}")
    if num_groups < 1:
        raise ValueError(
            f"{name} must describe at least one expert group, got G={num_groups}"
        )
    if offsets.dtype != torch.int32:
        raise ValueError(f"{name} must be int32, got {offsets.dtype}")
    if not offsets.is_cuda:
        raise ValueError(f"{name} must be a CUDA tensor, got device {offsets.device}")
    if device is not None and offsets.device != device:
        raise ValueError(
            f"{name} must be on {device}, got {offsets.device}; all operands and "
            "destinations must share one CUDA device"
        )
    if offsets.ndim != 1:
        raise ValueError(f"{name} must be 1D, got shape {tuple(offsets.shape)}")
    if offsets.numel() != num_groups:
        raise ValueError(
            f"{name} must have one entry per local expert: expected {num_groups}, "
            f"got {offsets.numel()}"
        )
    if not offsets.is_contiguous():
        raise ValueError(f"{name} must be contiguous, got stride {offsets.stride()}")

    if not host_offsets_validation_enabled() or _is_fake(offsets):
        return

    values = offsets.tolist()  # d2h sync; opt-in debugging path only
    previous = 0
    for group, end in enumerate(values):
        if end < previous:
            raise ValueError(
                f"{name} must be nondecreasing, but entry {group} is {end} "
                f"after {previous}"
            )
        size = end - previous
        if size % ROW_GROUP_ALIGNMENT != 0:
            raise ValueError(
                f"per-expert row counts must be multiples of {ROW_GROUP_ALIGNMENT} "
                f"(cuDNN FE FIX_PAD_SIZE; sub-256 groups corrupt results "
                f"nondeterministically): expert {group} has {size} rows "
                f"(offsets {previous} -> {end})"
            )
        previous = end
    if previous > allocated_rows:
        raise ValueError(
            f"{name}[-1] ({previous}) exceeds the allocated row count "
            f"({allocated_rows})"
        )


def validate_operand(
    tensor: torch.Tensor,
    *,
    name: str,
    shape: tuple,
    dtype: torch.dtype,
    device: torch.device,
    stride: Optional[tuple] = None,
    check_pointer_alignment: bool = True,
) -> None:
    """Validate one operand's dtype, shape, device, optional exact stride, alignment.

    ``stride=None`` accepts any strides (the cuDNN FE wrappers consume both
    row-major and transposed-memory colwise operands; every combination the
    composite produces is probe-proven). Metadata gates run before the
    ``data_ptr()`` gate so FakeTensor tracing exercises the same checks.
    """
    if tensor.dtype != dtype:
        raise ValueError(f"{name} must be {dtype}, got {tensor.dtype}")
    if tuple(tensor.shape) != tuple(shape):
        raise ValueError(
            f"{name} must have shape {tuple(shape)}, got {tuple(tensor.shape)}"
        )
    if stride is not None and tuple(tensor.stride()) != tuple(stride):
        raise ValueError(
            f"{name} must have stride {tuple(stride)}, got {tuple(tensor.stride())}. "
            "This layout is part of the ABI; a values-equal tensor with a "
            "different stride is not interchangeable."
        )
    if tensor.device != device:
        raise ValueError(
            f"{name} must be on {device}, got {tensor.device}; all operands and "
            "destinations must share one CUDA device"
        )
    if check_pointer_alignment and not _is_fake(tensor):
        if tensor.data_ptr() % _PTR_ALIGNMENT != 0:
            raise ValueError(
                f"{name} must be {_PTR_ALIGNMENT}-byte aligned, but its data "
                f"pointer is {tensor.data_ptr() % _PTR_ALIGNMENT} bytes past an "
                "aligned address. A contiguous view with a nonzero storage "
                "offset can violate this."
            )


def validate_blocked_scales(
    scales: torch.Tensor,
    *,
    name: str,
    logical_rows: int,
    logical_cols: int,
    device: torch.device,
    groups: int = 1,
) -> None:
    """Validate a flat blocked E8M0 scale buffer with a statically known size.

    The buffer is carried flat (uint8 or float8_e8m0fnu); its logical shape is
    metadata. ``groups > 1`` describes per-expert weight buffers whose per-group
    blocks are concatenated.
    """
    if scales.dtype not in _SCALE_DTYPES:
        raise ValueError(
            f"{name} must be uint8 or float8_e8m0fnu (raw E8M0 bytes), "
            f"got {scales.dtype}"
        )
    expected = groups * blocked_scale_numel(logical_rows, logical_cols)
    if scales.numel() != expected:
        raise ValueError(
            f"{name} must hold {expected} blocked scale bytes for a logical "
            f"[{logical_rows}, {logical_cols}] scale matrix"
            + (f" across {groups} experts" if groups > 1 else "")
            + f", got {scales.numel()}"
        )
    if not scales.is_contiguous():
        raise ValueError(f"{name} must be contiguous, got stride {scales.stride()}")
    if scales.device != device:
        raise ValueError(f"{name} must be on {device}, got {scales.device}")


def validate_ragged_colwise_scales(
    scales: torch.Tensor,
    *,
    name: str,
    features: int,
    allocated_rows: int,
    device: torch.device,
) -> None:
    """Validate a per-group columnwise scale buffer whose size depends on offsets.

    Columnwise activation scale buffers are sized by the ROUTED row total
    ``offsets[-1]`` — a device value — not by the allocated ``R``: at
    ``offsets[-1] < R`` they legitimately cover only ``offsets[-1]/32`` scale
    columns (probe-verified). So only dtype/device/contiguity and divisibility
    are checked: the numel must be a nonnegative multiple of
    ``features * SCALE_TILE_COLS`` rows-block granularity and must not exceed
    the buffer implied by the allocated rows.
    """
    if scales.dtype not in _SCALE_DTYPES:
        raise ValueError(
            f"{name} must be uint8 or float8_e8m0fnu (raw E8M0 bytes), "
            f"got {scales.dtype}"
        )
    if not scales.is_contiguous():
        raise ValueError(f"{name} must be contiguous, got stride {scales.stride()}")
    if scales.device != device:
        raise ValueError(f"{name} must be on {device}, got {scales.device}")
    rows_pad = _round_up(features, SCALE_TILE_ROWS)
    # Each 256-row group contributes features_pad * (group_rows/32) bytes and
    # group_rows/32 is a multiple of 8, so the buffer is a multiple of
    # rows_pad * 8 bytes.
    granule = rows_pad * (ROW_GROUP_ALIGNMENT // SCALE_BLOCK_SIZE)
    if scales.numel() % granule != 0:
        raise ValueError(
            f"{name} numel {scales.numel()} is not a multiple of {granule} "
            f"(= round_up({features},128) x {ROW_GROUP_ALIGNMENT // SCALE_BLOCK_SIZE} "
            "scale columns per 256-row group)"
        )
    max_numel = rows_pad * (allocated_rows // SCALE_BLOCK_SIZE)
    if scales.numel() > max_numel:
        raise ValueError(
            f"{name} numel {scales.numel()} exceeds the maximum {max_numel} implied "
            f"by the allocated row count {allocated_rows}"
        )


def validate_feature_dims(*, model_dim: int, hidden_dim: int) -> None:
    """D and F must both be positive multiples of 128."""
    if model_dim <= 0 or model_dim % DIM_ALIGNMENT != 0:
        raise ValueError(
            f"model dimension D must be a positive multiple of {DIM_ALIGNMENT}, "
            f"got {model_dim}"
        )
    if hidden_dim <= 0 or hidden_dim % DIM_ALIGNMENT != 0:
        raise ValueError(
            f"routed-expert hidden dimension F must be a positive multiple of "
            f"{DIM_ALIGNMENT}, got {hidden_dim}"
        )


def validate_allocated_rows(rows: int, *, name: str = "R") -> None:
    """The allocated row count must be a multiple of 256 (may be zero).

    Per-expert groups are multiples of 256 (cuDNN FE FIX_PAD_SIZE) and the
    allocation must be reachable by a legal offsets vector plus an inactive
    tail; a non-256 allocation additionally breaks the whole-matrix ==
    per-group-concat identity of the rowwise blocked scales.
    """
    if rows % ROW_GROUP_ALIGNMENT != 0:
        raise ValueError(
            f"{name} must be a multiple of {ROW_GROUP_ALIGNMENT}, got {rows}"
        )
