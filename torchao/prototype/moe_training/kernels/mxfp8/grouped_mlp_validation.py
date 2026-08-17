# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD 3-Clause license found in the
# LICENSE file in the root directory of this source tree.

"""Host-side precondition validation for the MXFP8 routed-expert grouped-MLP kernels.

The grouped MXFP8 kernels require every per-expert row count to be a multiple of
128. That is not a convenience: the tcgen05 blocked scale layout permutes in
128-row tiles, so a group boundary off a 128 multiple splits a tile and the
blocked buffer for the group no longer matches what the GEMM reads.

Everything here is metadata-only so it costs no host/device synchronization and
stays traceable under torch.compile. The per-expert offset VALUES live in device
memory and are a documented caller invariant, not something these checks can
enforce: reading them requires a D2H sync, so they are validated only by the
opt-in TORCHAO_MXFP8_VALIDATE_OFFSETS=1 debugging path below, and there is NO
device-side enforcement in a default build. Malformed offsets are not guaranteed
to fault, either -- the ragged-K weight-gradient kernel in particular can return
a wrong result from a clean-looking launch. Callers (and any future selection
predicate) must guarantee the offsets contract, e.g. by padding every expert
group to a multiple of 128 rows at dispatch time.

Checks raise ValueError rather than asserting, so `python -O` cannot strip them.
"""

import os
from typing import Optional

import torch

__all__ = [
    "SCALE_BLOCK_SIZE",
    "SCALE_TILE_ROWS",
    "SCALE_TILE_COLS",
    "blocked_scale_numel",
    "host_offsets_validation_enabled",
    "validate_group_offsets",
    "validate_grouped_operand",
    "validate_blocked_scales",
    "validate_destination",
]

# MXFP8 scaling block: 32 values share one E8M0 scale.
SCALE_BLOCK_SIZE = 32
# tcgen05 blocked scale tile: 128 rows x 4 columns, 512 bytes.
SCALE_TILE_ROWS = 128
SCALE_TILE_COLS = 4
# Row-count granularity every expert group must respect.
GROUP_ALIGNMENT = 128
# Byte alignment the launchers promise for TMA/vectorized accesses.
_PTR_ALIGNMENT = 32


def _round_up(x: int, to: int) -> int:
    return ((x + to - 1) // to) * to


def blocked_scale_numel(rows: int, cols: int) -> int:
    """Element count of the blocked E8M0 buffer for a logical [rows, cols] scale matrix.

    `cols` is a count of scale values, i.e. the reduced dimension divided by 32.
    """
    return _round_up(rows, SCALE_TILE_ROWS) * _round_up(cols, SCALE_TILE_COLS)


def host_offsets_validation_enabled() -> bool:
    """Opt-in host-side offset validation. Off by default: it forces a D2H sync."""
    return os.environ.get("TORCHAO_MXFP8_VALIDATE_OFFSETS", "0") == "1"


def validate_group_offsets(
    offsets: torch.Tensor,
    *,
    num_groups: int,
    allocated_rows: int,
    device: Optional[torch.device] = None,
    name: str = "offsets",
) -> None:
    """Validate the exclusive-end group offsets tensor's metadata.

    Metadata is always checked, including that at least one expert group
    exists. The offset *values* are checked only when
    host_offsets_validation_enabled(), because reading them forces a D2H sync;
    otherwise they are a documented caller invariant with no default-build
    enforcement anywhere (see the module docstring).
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

    if not host_offsets_validation_enabled():
        return

    values = offsets.tolist()  # d2h sync; opt-in debugging path only
    previous = 0
    for group, end in enumerate(values):
        if end < previous:
            raise ValueError(
                f"{name} must be nondecreasing, but entry {group} is {end} after {previous}"
            )
        size = end - previous
        if size % GROUP_ALIGNMENT != 0:
            raise ValueError(
                f"per-expert row counts must be multiples of {GROUP_ALIGNMENT}: "
                f"expert {group} has {size} rows (offsets {previous} -> {end})"
            )
        previous = end
    if previous > allocated_rows:
        raise ValueError(
            f"{name}[-1] ({previous}) exceeds the allocated row count ({allocated_rows})"
        )


def validate_grouped_operand(
    tensor: torch.Tensor,
    *,
    name: str,
    shape: tuple,
    stride: tuple,
    dtype: torch.dtype,
    device: torch.device,
    check_pointer_alignment: bool = True,
) -> None:
    """Validate one quantized operand's dtype, shape, exact stride, device, alignment.

    Order matters: every metadata gate runs before the data_ptr() gate so that
    FakeTensor tracing exercises the same checks (a fake tensor has no pointer).
    """
    if tensor.dtype != dtype:
        raise ValueError(f"{name} must be {dtype}, got {tensor.dtype}")
    if tuple(tensor.shape) != tuple(shape):
        raise ValueError(
            f"{name} must have shape {tuple(shape)}, got {tuple(tensor.shape)}"
        )
    if tuple(tensor.stride()) != tuple(stride):
        raise ValueError(
            f"{name} must have stride {tuple(stride)}, got {tuple(tensor.stride())}. "
            "This layout is part of the ABI; a values-equal tensor with a different "
            "stride is not interchangeable."
        )
    if tensor.device != device:
        raise ValueError(
            f"{name} must be on {device}, got {tensor.device}; all operands and "
            "destinations must share one CUDA device"
        )
    if check_pointer_alignment and not _is_fake(tensor):
        if tensor.data_ptr() % _PTR_ALIGNMENT != 0:
            raise ValueError(
                f"{name} must be {_PTR_ALIGNMENT}-byte aligned, but its data pointer is "
                f"{tensor.data_ptr() % _PTR_ALIGNMENT} bytes past an aligned address. A "
                "contiguous view with a nonzero storage offset can violate this."
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
    """Validate a blocked E8M0 scale buffer's dtype, element count, and device.

    The buffer is carried flat: its logical shape is metadata, not its physical
    shape, so only the element count is constrained. `groups` > 1 describes the
    per-expert weight buffers, which are [G, per_group_numel].
    """
    if scales.dtype not in (torch.uint8, torch.float8_e8m0fnu):
        raise ValueError(
            f"{name} must be uint8 or float8_e8m0fnu (raw E8M0 bytes), got {scales.dtype}"
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


def validate_destination(
    tensor: torch.Tensor,
    *,
    name: str,
    shape: tuple,
    stride: tuple,
    dtype: torch.dtype,
    device: torch.device,
) -> None:
    """Validate a caller-allocated destination in the destination-passing entry points.

    Destinations are validated exactly like inputs. Skipping this is how a private
    entry point turns a caller's shape mistake into an out-of-bounds write.
    """
    validate_grouped_operand(
        tensor,
        name=name,
        shape=shape,
        stride=stride,
        dtype=dtype,
        device=device,
    )


def validate_feature_dims(*, model_dim: int, hidden_dim: int) -> None:
    """D and F must both be multiples of 128 for the initial supported predicate."""
    if model_dim % GROUP_ALIGNMENT != 0:
        raise ValueError(
            f"model dimension D must be a multiple of {GROUP_ALIGNMENT}, got {model_dim}"
        )
    if hidden_dim % GROUP_ALIGNMENT != 0:
        raise ValueError(
            f"routed-expert hidden dimension F must be a multiple of {GROUP_ALIGNMENT}, "
            f"got {hidden_dim}"
        )


def validate_allocated_rows(rows: int, *, name: str = "R") -> None:
    """The allocated row count must itself be 128-aligned.

    Group sizes are multiples of 128 and the active row count is their sum, so a
    non-128 allocation can only describe an inactive tail that no legal offsets
    vector can reach; rejecting it early keeps the tail contract simple.
    """
    if rows % GROUP_ALIGNMENT != 0:
        raise ValueError(f"{name} must be a multiple of {GROUP_ALIGNMENT}, got {rows}")


def _is_fake(tensor: torch.Tensor) -> bool:
    """True for meta/fake tensors, which have no usable data pointer."""
    if tensor.device.type == "meta":
        return True
    try:
        from torch._subclasses.fake_tensor import FakeTensor
    except ImportError:
        return False
    return isinstance(tensor, FakeTensor)
