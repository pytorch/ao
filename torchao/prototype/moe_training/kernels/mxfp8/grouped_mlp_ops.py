# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD 3-Clause license found in the
# LICENSE file in the root directory of this source tree.

"""Public custom-op surface for the MXFP8 routed-expert grouped-MLP kernels.

Three ops, one per fused kernel:

* ``torchao::mxfp8_grouped_gemm_swiglu_fwd``  -- FC1 grouped GEMM + SwiGLU +
  rowwise/columnwise MXFP8 quantization
* ``torchao::mxfp8_grouped_gemm_dswiglu_bwd`` -- FC2 dgrad grouped GEMM +
  dSwiGLU + rowwise/columnwise MXFP8 quantization
* ``torchao::mxfp8_grouped_gemm_wgrad``       -- grouped MXFP8 weight-gradient
  GEMM, invoked once for FC1 and once for FC2

Each op allocates its destinations through the ``_allocate_*_outputs`` helpers
below, which are pure torch and are shared by the real implementation and by
``register_fake``. Meta shapes and strides therefore cannot drift from eager --
a class of bug that matters here because several outputs are column-major and a
row-major fake would silently change what ``torch.compile`` traces.

Layout, alignment and numerical semantics are defined by the kernel contract;
the load-bearing preconditions (every per-expert row count a multiple of 128,
exact strides, blocked E8M0 scales in the tcgen05 128x4 layout) are enforced by
``grouped_mlp_validation`` before any launch, because violating them otherwise
corrupts the CUDA context rather than raising.
"""

from typing import Tuple

import torch

from torchao.prototype.moe_training.kernels.mxfp8.grouped_mlp_validation import (
    blocked_scale_numel,
    validate_allocated_rows,
    validate_blocked_scales,
    validate_feature_dims,
    validate_group_offsets,
    validate_grouped_operand,
)

__all__ = [
    "mxfp8_grouped_gemm_swiglu_fwd",
    "mxfp8_grouped_gemm_dswiglu_bwd",
    "mxfp8_grouped_gemm_wgrad",
]

_E4M3 = torch.float8_e4m3fn
_E8M0 = torch.float8_e8m0fnu
_SCALE_BLOCK = 32


def _empty_blocked_scales(
    logical_rows: int, logical_cols: int, *, device, groups: int = 1
) -> torch.Tensor:
    """Allocate a flat blocked E8M0 scale buffer.

    The buffer is flat by ABI: its logical shape is metadata. Kernels write every
    byte, including the inactive-tail rows, so an uninitialized allocation is
    safe here -- but only because that write obligation is part of the contract.
    """
    numel = groups * blocked_scale_numel(logical_rows, logical_cols)
    shape = (groups, numel // groups) if groups > 1 else (numel,)
    return torch.empty(shape, dtype=_E8M0, device=device)


# --------------------------------------------------------------------------
# Kernel A: FC1 grouped GEMM + SwiGLU + dual quantization
# --------------------------------------------------------------------------


def _allocate_swiglu_fwd_outputs(
    rows: int, hidden: int, device
) -> Tuple[torch.Tensor, ...]:
    z = torch.empty_strided(
        (rows, hidden, 2), (2 * hidden, 2, 1), dtype=torch.bfloat16, device=device
    )
    h_row_q = torch.empty_strided(
        (rows, hidden), (hidden, 1), dtype=_E4M3, device=device
    )
    h_row_sf = _empty_blocked_scales(rows, hidden // _SCALE_BLOCK, device=device)
    # Column-major: the 32x1 quantized operand is consumed as its own transpose.
    h_col_q = torch.empty_strided((rows, hidden), (1, rows), dtype=_E4M3, device=device)
    h_col_sf = _empty_blocked_scales(hidden, rows // _SCALE_BLOCK, device=device)
    return z, h_row_q, h_row_sf, h_col_q, h_col_sf


def _validate_swiglu_fwd_inputs(
    x_q, x_sf, w13_t_q, w13_t_sf, offsets
) -> Tuple[int, int, int, int]:
    if x_q.ndim != 2:
        raise ValueError(f"x_q must be 2D [R, D], got shape {tuple(x_q.shape)}")
    if w13_t_q.ndim != 3:
        raise ValueError(
            f"w13_t_q must be 3D [G, D, 2F], got shape {tuple(w13_t_q.shape)}"
        )
    rows, model_dim = x_q.shape
    groups, w_k, two_hidden = w13_t_q.shape
    if w_k != model_dim:
        raise ValueError(
            f"w13_t_q contraction dim {w_k} must match x_q's D {model_dim}"
        )
    if two_hidden % 2 != 0:
        raise ValueError(
            f"w13_t_q's N dim must be 2F with interleaved gate/up channels, got {two_hidden}"
        )
    hidden = two_hidden // 2
    device = x_q.device

    validate_feature_dims(model_dim=model_dim, hidden_dim=hidden)
    validate_allocated_rows(rows)
    validate_group_offsets(offsets, num_groups=groups, allocated_rows=rows)
    validate_grouped_operand(
        x_q,
        name="x_q",
        shape=(rows, model_dim),
        stride=(model_dim, 1),
        dtype=_E4M3,
        device=device,
    )
    validate_grouped_operand(
        w13_t_q,
        name="w13_t_q",
        shape=(groups, model_dim, two_hidden),
        stride=(model_dim * two_hidden, 1, model_dim),
        dtype=_E4M3,
        device=device,
    )
    validate_blocked_scales(
        x_sf,
        name="x_sf",
        logical_rows=rows,
        logical_cols=model_dim // _SCALE_BLOCK,
        device=device,
    )
    validate_blocked_scales(
        w13_t_sf,
        name="w13_t_sf",
        logical_rows=two_hidden,
        logical_cols=model_dim // _SCALE_BLOCK,
        device=device,
        groups=groups,
    )
    return rows, model_dim, hidden, groups


@torch.library.custom_op("torchao::mxfp8_grouped_gemm_swiglu_fwd", mutates_args=())
def _mxfp8_grouped_gemm_swiglu_fwd(
    x_q: torch.Tensor,
    x_sf: torch.Tensor,
    w13_t_q: torch.Tensor,
    w13_t_sf: torch.Tensor,
    offsets: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    rows, _model_dim, hidden, _groups = _validate_swiglu_fwd_inputs(
        x_q, x_sf, w13_t_q, w13_t_sf, offsets
    )
    outputs = _allocate_swiglu_fwd_outputs(rows, hidden, x_q.device)

    if rows == 0:
        # R == 0 is a required correctness case. Every destination is empty in
        # its row dimension and both scale buffers are zero-length, so there is
        # nothing to write; launching would build a degenerate (0, D, 1) layout
        # and fail inside the SF layout builder with an opaque MLIR error.
        return outputs

    from torchao.prototype.moe_training.kernels.mxfp8.cutedsl_grouped_mlp import (
        launch_grouped_gemm_swiglu_fwd,
    )

    launch_grouped_gemm_swiglu_fwd(x_q, x_sf, w13_t_q, w13_t_sf, offsets, *outputs)
    return outputs


@_mxfp8_grouped_gemm_swiglu_fwd.register_fake
def _(x_q, x_sf, w13_t_q, w13_t_sf, offsets):
    rows, _model_dim, hidden, _groups = _validate_swiglu_fwd_inputs(
        x_q, x_sf, w13_t_q, w13_t_sf, offsets
    )
    return _allocate_swiglu_fwd_outputs(rows, hidden, x_q.device)


# --------------------------------------------------------------------------
# Kernel B: FC2 dgrad grouped GEMM + dSwiGLU + dual quantization
# --------------------------------------------------------------------------


def _allocate_dswiglu_bwd_outputs(
    rows: int, hidden: int, device
) -> Tuple[torch.Tensor, ...]:
    two_hidden = 2 * hidden
    dz_row_q = torch.empty_strided(
        (rows, two_hidden), (two_hidden, 1), dtype=_E4M3, device=device
    )
    dz_row_sf = _empty_blocked_scales(rows, two_hidden // _SCALE_BLOCK, device=device)
    dz_col_q = torch.empty_strided(
        (rows, two_hidden), (1, rows), dtype=_E4M3, device=device
    )
    dz_col_sf = _empty_blocked_scales(two_hidden, rows // _SCALE_BLOCK, device=device)
    return dz_row_q, dz_row_sf, dz_col_q, dz_col_sf


def _validate_dswiglu_bwd_inputs(do_q, do_sf, w2_q, w2_sf, z_bf16, offsets):
    if do_q.ndim != 2:
        raise ValueError(f"do_q must be 2D [R, D], got shape {tuple(do_q.shape)}")
    if w2_q.ndim != 3:
        raise ValueError(
            f"w2_dgrad_q must be 3D [G, D, F], got shape {tuple(w2_q.shape)}"
        )
    if z_bf16.ndim != 3 or z_bf16.shape[-1] != 2:
        raise ValueError(
            f"z_bf16 must be [R, F, 2] with gate at index 0 and up at index 1, "
            f"got shape {tuple(z_bf16.shape)}"
        )
    rows, model_dim = do_q.shape
    groups, w_k, hidden = w2_q.shape
    if w_k != model_dim:
        raise ValueError(
            f"w2_dgrad_q contraction dim {w_k} must match do_q's D {model_dim}"
        )
    if tuple(z_bf16.shape) != (rows, hidden, 2):
        raise ValueError(
            f"z_bf16 must be [{rows}, {hidden}, 2] to match do_q and w2_dgrad_q, "
            f"got {tuple(z_bf16.shape)}"
        )
    device = do_q.device

    validate_feature_dims(model_dim=model_dim, hidden_dim=hidden)
    validate_allocated_rows(rows)
    validate_group_offsets(offsets, num_groups=groups, allocated_rows=rows)
    validate_grouped_operand(
        do_q,
        name="do_q",
        shape=(rows, model_dim),
        stride=(model_dim, 1),
        dtype=_E4M3,
        device=device,
    )
    validate_grouped_operand(
        w2_q,
        name="w2_dgrad_q",
        shape=(groups, model_dim, hidden),
        stride=(model_dim * hidden, 1, model_dim),
        dtype=_E4M3,
        device=device,
    )
    # z_bf16 is the exact destination Kernel A wrote, so its stride is pinned too.
    validate_grouped_operand(
        z_bf16,
        name="z_bf16",
        shape=(rows, hidden, 2),
        stride=(2 * hidden, 2, 1),
        dtype=torch.bfloat16,
        device=device,
    )
    validate_blocked_scales(
        do_sf,
        name="do_sf",
        logical_rows=rows,
        logical_cols=model_dim // _SCALE_BLOCK,
        device=device,
    )
    validate_blocked_scales(
        w2_sf,
        name="w2_dgrad_sf",
        logical_rows=hidden,
        logical_cols=model_dim // _SCALE_BLOCK,
        device=device,
        groups=groups,
    )
    return rows, model_dim, hidden, groups


@torch.library.custom_op("torchao::mxfp8_grouped_gemm_dswiglu_bwd", mutates_args=())
def _mxfp8_grouped_gemm_dswiglu_bwd(
    do_q: torch.Tensor,
    do_sf: torch.Tensor,
    w2_dgrad_q: torch.Tensor,
    w2_dgrad_sf: torch.Tensor,
    z_bf16: torch.Tensor,
    offsets: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    rows, _model_dim, hidden, _groups = _validate_dswiglu_bwd_inputs(
        do_q, do_sf, w2_dgrad_q, w2_dgrad_sf, z_bf16, offsets
    )
    outputs = _allocate_dswiglu_bwd_outputs(rows, hidden, do_q.device)

    if rows == 0:
        # See the R == 0 note in the forward op: nothing to write, and launching
        # would build a degenerate SF layout.
        return outputs

    from torchao.prototype.moe_training.kernels.mxfp8.cutedsl_grouped_mlp import (
        launch_grouped_gemm_dswiglu_bwd,
    )

    launch_grouped_gemm_dswiglu_bwd(
        do_q, do_sf, w2_dgrad_q, w2_dgrad_sf, z_bf16, offsets, *outputs
    )
    return outputs


@_mxfp8_grouped_gemm_dswiglu_bwd.register_fake
def _(do_q, do_sf, w2_dgrad_q, w2_dgrad_sf, z_bf16, offsets):
    rows, _model_dim, hidden, _groups = _validate_dswiglu_bwd_inputs(
        do_q, do_sf, w2_dgrad_q, w2_dgrad_sf, z_bf16, offsets
    )
    return _allocate_dswiglu_bwd_outputs(rows, hidden, do_q.device)


# --------------------------------------------------------------------------
# Kernel C: grouped MXFP8 wgrad
# --------------------------------------------------------------------------


def _validate_wgrad_inputs(dy_col_q, dy_col_sf, x_col_q, x_col_sf, offsets):
    if dy_col_q.ndim != 2 or x_col_q.ndim != 2:
        raise ValueError(
            "dy_col_q and x_col_q must both be 2D logical [R, N] / [R, K], got "
            f"{tuple(dy_col_q.shape)} and {tuple(x_col_q.shape)}"
        )
    rows, out_features = dy_col_q.shape
    x_rows, in_features = x_col_q.shape
    if x_rows != rows:
        raise ValueError(
            f"dy_col_q and x_col_q must share the row dim: {rows} vs {x_rows}"
        )
    groups = offsets.numel()
    device = dy_col_q.device

    validate_allocated_rows(rows)
    validate_group_offsets(offsets, num_groups=groups, allocated_rows=rows)
    # Both operands are column-major so their transposes are free.
    validate_grouped_operand(
        dy_col_q,
        name="dy_col_q",
        shape=(rows, out_features),
        stride=(1, rows),
        dtype=_E4M3,
        device=device,
    )
    validate_grouped_operand(
        x_col_q,
        name="x_col_q",
        shape=(rows, in_features),
        stride=(1, rows),
        dtype=_E4M3,
        device=device,
    )
    validate_blocked_scales(
        dy_col_sf,
        name="dy_col_sf",
        logical_rows=out_features,
        logical_cols=rows // _SCALE_BLOCK,
        device=device,
    )
    validate_blocked_scales(
        x_col_sf,
        name="x_col_sf",
        logical_rows=in_features,
        logical_cols=rows // _SCALE_BLOCK,
        device=device,
    )
    return rows, out_features, in_features, groups


@torch.library.custom_op("torchao::mxfp8_grouped_gemm_wgrad", mutates_args=())
def _mxfp8_grouped_gemm_wgrad(
    dy_col_q: torch.Tensor,
    dy_col_sf: torch.Tensor,
    x_col_q: torch.Tensor,
    x_col_sf: torch.Tensor,
    offsets: torch.Tensor,
) -> torch.Tensor:
    rows, out_features, in_features, groups = _validate_wgrad_inputs(
        dy_col_q, dy_col_sf, x_col_q, x_col_sf, offsets
    )
    dw = torch.empty(
        (groups, out_features, in_features),
        dtype=torch.bfloat16,
        device=dy_col_q.device,
    )

    if rows == 0:
        # Unlike A and B, this destination is NOT empty at R == 0: every expert
        # has zero rows, and an expert with zero rows is defined to produce an
        # all-zero slice. Zero it here rather than launching over an empty
        # contraction.
        return dw.zero_()

    from torchao.prototype.moe_training.kernels.mxfp8.cutedsl_grouped_mlp import (
        launch_grouped_gemm_wgrad,
    )

    launch_grouped_gemm_wgrad(dy_col_q, dy_col_sf, x_col_q, x_col_sf, offsets, dw)
    return dw


@_mxfp8_grouped_gemm_wgrad.register_fake
def _(dy_col_q, dy_col_sf, x_col_q, x_col_sf, offsets):
    _rows, out_features, in_features, groups = _validate_wgrad_inputs(
        dy_col_q, dy_col_sf, x_col_q, x_col_sf, offsets
    )
    return torch.empty(
        (groups, out_features, in_features),
        dtype=torch.bfloat16,
        device=dy_col_q.device,
    )


# --------------------------------------------------------------------------
# Public wrappers
# --------------------------------------------------------------------------


def mxfp8_grouped_gemm_swiglu_fwd(x_q, x_sf, w13_t_q, w13_t_sf, offsets):
    """FC1 grouped GEMM + SwiGLU + rowwise/columnwise MXFP8 quantization.

    Returns ``(z_bf16, h_row_q, h_row_sf, h_col_q, h_col_sf)``. ``z_bf16`` is the
    BF16 pre-activation saved for backward; pass it unchanged to
    :func:`mxfp8_grouped_gemm_dswiglu_bwd`.
    """
    return torch.ops.torchao.mxfp8_grouped_gemm_swiglu_fwd(
        x_q, x_sf, w13_t_q, w13_t_sf, offsets
    )


def mxfp8_grouped_gemm_dswiglu_bwd(
    do_q, do_sf, w2_dgrad_q, w2_dgrad_sf, z_bf16, offsets
):
    """FC2 dgrad grouped GEMM + dSwiGLU + rowwise/columnwise MXFP8 quantization.

    Returns ``(dz_row_q, dz_row_sf, dz_col_q, dz_col_sf)`` with gate/up gradients
    element-interleaved along the 2F axis.
    """
    return torch.ops.torchao.mxfp8_grouped_gemm_dswiglu_bwd(
        do_q, do_sf, w2_dgrad_q, w2_dgrad_sf, z_bf16, offsets
    )


def mxfp8_grouped_gemm_wgrad(dy_col_q, dy_col_sf, x_col_q, x_col_sf, offsets):
    """Grouped MXFP8 weight-gradient GEMM, returning BF16 ``[G, N, K]``.

    Used once for FC1 (``N=2F, K=D``) and once for FC2 (``N=D, K=F``). An expert
    with zero rows yields an all-zero slice.
    """
    return torch.ops.torchao.mxfp8_grouped_gemm_wgrad(
        dy_col_q, dy_col_sf, x_col_q, x_col_sf, offsets
    )
