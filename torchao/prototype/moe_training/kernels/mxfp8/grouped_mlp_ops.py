# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD 3-Clause license found in the
# LICENSE file in the root directory of this source tree.

"""Custom-op surface for the cuDNN-frontend MXFP8 routed-expert grouped MLP.

Four ops, each wrapping one ``cudnn.grouped_gemm_*_wrapper_sm100`` CuTe DSL
kernel from the standalone cudnn-frontend python package (>= 1.27; no
TransformerEngine involvement):

* ``torchao::mxfp8_grouped_gemm_swiglu_fwd``   -- FC1 ragged grouped GEMM +
  SwiGLU + rowwise AND columnwise MXFP8 RCEIL quantization + BF16 pre-GLU save
  (``grouped_gemm_glu_wrapper_sm100``).
* ``torchao::mxfp8_grouped_gemm``        -- ragged grouped GEMM on
  prequantized operands to BF16 (``grouped_gemm_quant_wrapper_sm100``); used
  for both FC2 forward and FC1 dgrad.
* ``torchao::mxfp8_grouped_gemm_dswiglu_bwd``   -- FC2 dgrad + dSwiGLU + dual
  MXFP8 quantization of dz (``grouped_gemm_dglu_wrapper_sm100``).
* ``torchao::mxfp8_grouped_gemm_wgrad`` -- ragged-reduction grouped
  weight gradient (``grouped_gemm_wgrad_wrapper_sm100``, dense output mode);
  called once for FC1 and once for FC2.

All scale arguments are FLAT blocked E8M0 buffers (uint8 or float8_e8m0fnu);
the ops build the kernel-native 6-D / 2-D views internally with probe-proven
recipes. ``offsets`` is int32 CUDA ``[G]`` exclusive-end rows; per-expert row
counts must be multiples of 256 (cuDNN FE ``FIX_PAD_SIZE``; see the validation
module for the two-tier enforcement and the misalignment hazard). Rows in
``[offsets[-1], R)``: caller-allocated outputs (the grouped-mm result and the
weight gradients) keep their tails untouched, while kernel-allocated outputs
(z, h, dz and their scales) carry garbage tails that are read-forbidden --
both behaviors probe-verified with NaN-poisoned tails.

Shared ``_validate_*`` helpers back ``register_fake`` so torch.compile rejects
unsupported calls at capture time, and shared output-spec helpers keep fake
and eager metadata identical (eager normalizes the wrapper's returned tensors
and checks them against the same spec the fake allocates from).

The user-facing wrappers live in
``torchao.prototype.moe_training.mxfp8_grouped_mlp``; importing that module
(or this one) registers the ops. ``import cudnn`` happens lazily inside op
bodies at first real launch.
"""

from typing import Tuple

import torch

from torchao.prototype.moe_training.kernels.mxfp8.grouped_mlp_validation import (
    ROW_GROUP_ALIGNMENT,
    SCALE_BLOCK_SIZE,
    validate_allocated_rows,
    validate_blocked_scales,
    validate_feature_dims,
    validate_group_offsets,
    validate_operand,
    validate_ragged_colwise_scales,
)

__all__ = ["ROW_GROUP_ALIGNMENT", "SCALE_BLOCK_SIZE"]

_E4M3 = torch.float8_e4m3fn
_E8M0 = torch.float8_e8m0fnu
_BLOCK = SCALE_BLOCK_SIZE

# Small per-(groups, dtype, device) caches for the kernels' alpha/beta and
# norm-const tensors. Never cached: the CUDA stream (looked up per call).
_ones_cache: dict = {}


def _cached_ones(numel: int, dtype: torch.dtype, device: torch.device) -> torch.Tensor:
    key = (numel, dtype, device)
    out = _ones_cache.get(key)
    if out is None:
        out = torch.ones(numel, dtype=dtype, device=device)
        _ones_cache[key] = out
    return out


def _require_cuda_device(device: torch.device, name: str) -> None:
    if device.type != "cuda":
        raise ValueError(
            f"{name} must be a CUDA tensor, got device {device}; these kernels "
            "run only on CUDA SM100 devices"
        )


def _as_e8m0(scales: torch.Tensor) -> torch.Tensor:
    return scales if scales.dtype == _E8M0 else scales.view(_E8M0)


def _act_scale_view(sf_flat: torch.Tensor, rows: int, cols: int) -> torch.Tensor:
    """Flat blocked scales of a logical [rows, cols/32] matrix -> the wrapper's
    6-D activation view (32, 4, rows/128, 4, cols/128, 1)."""
    return (
        _as_e8m0(sf_flat)
        .view(1, rows // 128, cols // 128, 32, 4, 4)
        .permute(3, 4, 1, 5, 2, 0)
    )


def _weight_scale_view(
    sf_flat: torch.Tensor, groups: int, n: int, k: int
) -> torch.Tensor:
    """Per-group-concat flat blocked scales of logical [n, k/32] per expert ->
    the wrapper's 6-D weight view (32, 4, n/128, 4, k/128, G)."""
    return (
        _as_e8m0(sf_flat)
        .view(groups, n // 128, k // 128, 32, 4, 4)
        .permute(3, 4, 1, 5, 2, 0)
    )


def _flat_scales(sf_6d: torch.Tensor) -> torch.Tensor:
    """Kernel-returned 6-D scale view -> the flat blocked buffer (a free view:
    the inverse permute restores the allocation's contiguous order)."""
    return sf_6d.permute(5, 2, 4, 0, 1, 3).reshape(-1)


def _check_normalized(
    tensor: torch.Tensor, *, name: str, shape: tuple, dtype: torch.dtype
) -> torch.Tensor:
    """Guard against wrapper-output metadata drifting from the fake spec."""
    if tuple(tensor.shape) != tuple(shape) or tensor.dtype != dtype:
        raise RuntimeError(
            f"cudnn wrapper output {name} has shape {tuple(tensor.shape)} dtype "
            f"{tensor.dtype}; expected {tuple(shape)} {dtype}. The installed "
            "cudnn-frontend's output contract changed; the registered fake no "
            "longer matches eager."
        )
    if not tensor.is_contiguous():
        return tensor.contiguous()
    return tensor


def _stream() -> int:
    return torch.cuda.current_stream().cuda_stream


# --------------------------------------------------------------------------
# Op 1: FC1 grouped GEMM + SwiGLU + dual quantization (glu wrapper)
# --------------------------------------------------------------------------


def _fwd_output_specs(rows: int, hidden: int):
    two_hidden = 2 * hidden
    return (
        ("z_bf16", (rows, two_hidden), torch.bfloat16),
        ("h_row_q", (rows, hidden), _E4M3),
        ("h_row_sf", (rows * hidden // _BLOCK,), _E8M0),
        ("h_col_q", (rows, hidden), _E4M3),
        ("h_col_sf", (hidden * rows // _BLOCK,), _E8M0),
    )


def _allocate_from_specs(specs, device) -> Tuple[torch.Tensor, ...]:
    return tuple(
        torch.empty(shape, dtype=dtype, device=device) for _, shape, dtype in specs
    )


def _validate_fwd_inputs(x_q, x_sf, w13_q, w13_sf, offsets):
    if x_q.ndim != 2:
        raise ValueError(f"x_q must be 2D [R, D], got shape {tuple(x_q.shape)}")
    if w13_q.ndim != 3:
        raise ValueError(f"w13_q must be 3D [G, 2F, D], got shape {tuple(w13_q.shape)}")
    rows, model_dim = x_q.shape
    groups, two_hidden, w_k = w13_q.shape
    if w_k != model_dim:
        raise ValueError(f"w13_q contraction dim {w_k} must match x_q's D {model_dim}")
    if two_hidden % 2 != 0:
        raise ValueError(
            f"w13_q's row dim must be 2F (32-block interleaved gate/up), "
            f"got {two_hidden}"
        )
    hidden = two_hidden // 2
    device = x_q.device

    _require_cuda_device(device, "x_q")
    validate_feature_dims(model_dim=model_dim, hidden_dim=hidden)
    validate_allocated_rows(rows)
    if rows * two_hidden >= 2**31:
        raise ValueError(
            f"R * 2F = {rows * two_hidden} does not fit an int32 element index"
        )
    validate_group_offsets(
        offsets, num_groups=groups, allocated_rows=rows, device=device
    )
    validate_operand(
        x_q,
        name="x_q",
        shape=(rows, model_dim),
        stride=(model_dim, 1),
        dtype=_E4M3,
        device=device,
    )
    # The rowwise weight cast delivers a contiguous [G, 2F, D] stack; the
    # kernel-facing (2F, D, G) view is built from exactly that layout.
    validate_operand(
        w13_q,
        name="w13_q",
        shape=(groups, two_hidden, model_dim),
        stride=(two_hidden * model_dim, model_dim, 1),
        dtype=_E4M3,
        device=device,
    )
    validate_blocked_scales(
        x_sf,
        name="x_sf",
        logical_rows=rows,
        logical_cols=model_dim // _BLOCK,
        device=device,
    )
    validate_blocked_scales(
        w13_sf,
        name="w13_sf",
        logical_rows=two_hidden,
        logical_cols=model_dim // _BLOCK,
        device=device,
        groups=groups,
    )
    return rows, model_dim, hidden, groups


@torch.library.custom_op("torchao::mxfp8_grouped_gemm_swiglu_fwd", mutates_args=())
def _mxfp8_grouped_gemm_swiglu_fwd(
    x_q: torch.Tensor,
    x_sf: torch.Tensor,
    w13_q: torch.Tensor,
    w13_sf: torch.Tensor,
    offsets: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """FC1 grouped GEMM + SwiGLU + dual MXFP8 RCEIL quantization (one cuDNN launch).

    Inputs (all CUDA, one device; prequantized outside):
      x_q     E4M3 ``[R, D]`` stride ``(D, 1)``, rowwise 1x32 quantized.
      x_sf    flat blocked E8M0 scales for logical ``[R, D/32]`` (whole-matrix
              blocked == per-group concat because R and every group are %256).
      w13_q   E4M3 ``[G, 2F, D]`` contiguous, rowwise quantized, rows in the
              cuDNN 32-BLOCK GLU order ``[gate0(32) | up0(32) | gate1 | ...]``.
      w13_sf  per-group flat blocked E8M0, logical ``[2F, D/32]`` per expert.
      offsets int32 CUDA ``[G]`` exclusive end rows; per-expert counts %256
              (caller invariant; see the validation module).

    Returns ``(z_bf16, h_row_q, h_row_sf, h_col_q, h_col_sf)``:
      z_bf16   BF16 ``[R, 2F]`` contiguous pre-activation in the same 32-block
               order; consumed unchanged by the bwd op.
      h_row_q  E4M3 ``[R, F]`` contiguous; h_row_sf flat blocked for ``[R, F/32]``.
      h_col_q  E4M3 ``[R, F]`` contiguous columnwise-quantized bytes
               (un-transposed kernel layout); h_col_sf PER-GROUP flat blocked
               for ``[F, rows_g/32]`` per expert.

    Rows past ``offsets[-1]`` of every output are GARBAGE (kernel-computed from
    the quantized input tail) and read-forbidden. ``R == 0`` returns empty
    outputs without touching cudnn; ``G == 0`` raises ValueError.
    """
    rows, model_dim, hidden, groups = _validate_fwd_inputs(
        x_q, x_sf, w13_q, w13_sf, offsets
    )
    specs = _fwd_output_specs(rows, hidden)
    if rows == 0:
        return _allocate_from_specs(specs, x_q.device)

    import cudnn

    out = cudnn.grouped_gemm_glu_wrapper_sm100(
        a_tensor=x_q.unsqueeze(0).permute(1, 2, 0),
        sfa_tensor=_act_scale_view(x_sf, rows, model_dim),
        padded_offsets=offsets,
        alpha_tensor=_cached_ones(groups, torch.bfloat16, x_q.device),
        b_tensor=w13_q.permute(1, 2, 0),
        sfb_tensor=_weight_scale_view(w13_sf, groups, 2 * hidden, model_dim),
        norm_const_tensor=_cached_ones(1, torch.float32, x_q.device),
        prob_tensor=None,
        acc_dtype=torch.float32,
        c_dtype=torch.bfloat16,
        d_dtype=_E4M3,
        cd_major="n",
        sf_vec_size=_BLOCK,
        act_func="swiglu",
        discrete_col_sfd=True,
        use_dynamic_sched=True,
        current_stream=_stream(),
    )
    results = (
        out["c_tensor"].view(rows, 2 * hidden),
        out["d_tensor"].view(rows, hidden),
        _flat_scales(out["sfd_row_tensor"]),
        out["d_col_tensor"].view(rows, hidden),
        _flat_scales(out["sfd_col_tensor"]),
    )
    return tuple(
        _check_normalized(t, name=spec[0], shape=spec[1], dtype=spec[2])
        for t, spec in zip(results, specs)
    )


@_mxfp8_grouped_gemm_swiglu_fwd.register_fake
def _(x_q, x_sf, w13_q, w13_sf, offsets):
    rows, _model_dim, hidden, _groups = _validate_fwd_inputs(
        x_q, x_sf, w13_q, w13_sf, offsets
    )
    return _allocate_from_specs(_fwd_output_specs(rows, hidden), x_q.device)


# --------------------------------------------------------------------------
# Op 2: grouped GEMM on prequantized operands -> BF16 (quant wrapper)
# --------------------------------------------------------------------------


def _validate_mm_inputs(a_q, a_sf, b_q, b_sf, offsets):
    if a_q.ndim != 2:
        raise ValueError(f"a_q must be 2D [R, K], got shape {tuple(a_q.shape)}")
    if b_q.ndim != 3:
        raise ValueError(f"b_q must be 3D [G, N, K], got shape {tuple(b_q.shape)}")
    rows, contraction = a_q.shape
    groups, out_features, b_k = b_q.shape
    if b_k != contraction:
        raise ValueError(f"b_q contraction dim {b_k} must match a_q's K {contraction}")
    device = a_q.device

    _require_cuda_device(device, "a_q")
    # N and K are both feature dims here (D/F/2F at the two call sites).
    validate_feature_dims(model_dim=out_features, hidden_dim=contraction)
    validate_allocated_rows(rows)
    if rows * max(out_features, contraction) >= 2**31:
        raise ValueError(
            f"R * max(N, K) = {rows * max(out_features, contraction)} does not "
            "fit an int32 element index"
        )
    validate_group_offsets(
        offsets, num_groups=groups, allocated_rows=rows, device=device
    )
    validate_operand(
        a_q,
        name="a_q",
        shape=(rows, contraction),
        stride=(contraction, 1),
        dtype=_E4M3,
        device=device,
    )
    # b_q strides are free: rowwise weight casts arrive [G, N, K] contiguous
    # and dim1-colwise casts arrive transposed to [G, N, K] (also row-major in
    # this orientation); the wrapper reads the strides (both probe-proven).
    validate_operand(
        b_q,
        name="b_q",
        shape=(groups, out_features, contraction),
        dtype=_E4M3,
        device=device,
    )
    validate_blocked_scales(
        a_sf,
        name="a_sf",
        logical_rows=rows,
        logical_cols=contraction // _BLOCK,
        device=device,
    )
    validate_blocked_scales(
        b_sf,
        name="b_sf",
        logical_rows=out_features,
        logical_cols=contraction // _BLOCK,
        device=device,
        groups=groups,
    )
    return rows, out_features, contraction, groups


@torch.library.custom_op("torchao::mxfp8_grouped_gemm", mutates_args=())
def _mxfp8_grouped_gemm(
    a_q: torch.Tensor,
    a_sf: torch.Tensor,
    b_q: torch.Tensor,
    b_sf: torch.Tensor,
    offsets: torch.Tensor,
) -> torch.Tensor:
    """Ragged grouped GEMM ``out[r] = dequant(a[r]) @ dequant(b[g(r)]).T`` -> BF16.

    Inputs:
      a_q  E4M3 ``[R, K]`` stride ``(K, 1)``, rowwise 1x32 quantized; a_sf flat
           blocked for logical ``[R, K/32]``.
      b_q  E4M3 ``[G, N, K]``-logical, quantized ALONG K, any strides (rowwise
           weight casts pass as-is; dim1-colwise casts pass transposed into
           this orientation).
      b_sf per-group flat blocked for the ``[N, K/32]``-oriented scale matrix
           (uniform for both quantization axes).
      offsets int32 CUDA ``[G]`` exclusive end rows.

    Covers FC2 forward (b = w2 rowwise: N=D, K=F) and FC1 dgrad (b = w13
    colwise: N=D, K=2F). Returns contiguous BF16 ``[R, N]``; rows past
    ``offsets[-1]`` are left uninitialized (probe-verified untouched).
    ``R == 0`` returns an empty output without touching cudnn.
    """
    rows, out_features, contraction, groups = _validate_mm_inputs(
        a_q, a_sf, b_q, b_sf, offsets
    )
    out = torch.empty(rows, out_features, dtype=torch.bfloat16, device=a_q.device)
    if rows == 0:
        return out

    import cudnn

    cudnn.grouped_gemm_quant_wrapper_sm100(
        a_tensor=a_q.unsqueeze(0).permute(1, 2, 0),
        sfa_tensor=_act_scale_view(a_sf, rows, contraction),
        padded_offsets=offsets,
        alpha_tensor=_cached_ones(groups, torch.bfloat16, a_q.device),
        b_tensor=b_q.permute(1, 2, 0),
        sfb_tensor=_weight_scale_view(b_sf, groups, out_features, contraction),
        norm_const_tensor=None,
        prob_tensor=None,
        acc_dtype=torch.float32,
        d_dtype=torch.bfloat16,
        d_tensor=out.as_strided(
            (rows, out_features, 1), (out_features, 1, rows * out_features)
        ),
        cd_major="n",
        sf_vec_size=_BLOCK,
        use_dynamic_sched=True,
        current_stream=_stream(),
    )
    return out


@_mxfp8_grouped_gemm.register_fake
def _(a_q, a_sf, b_q, b_sf, offsets):
    rows, out_features, _contraction, _groups = _validate_mm_inputs(
        a_q, a_sf, b_q, b_sf, offsets
    )
    return torch.empty(rows, out_features, dtype=torch.bfloat16, device=a_q.device)


# --------------------------------------------------------------------------
# Op 3: FC2 dgrad + dSwiGLU + dual quantization (dglu wrapper)
# --------------------------------------------------------------------------


def _bwd_output_specs(rows: int, hidden: int):
    two_hidden = 2 * hidden
    return (
        ("dz_row_q", (rows, two_hidden), _E4M3),
        ("dz_row_sf", (rows * two_hidden // _BLOCK,), _E8M0),
        ("dz_col_q", (rows, two_hidden), _E4M3),
        ("dz_col_sf", (two_hidden * rows // _BLOCK,), _E8M0),
    )


def _validate_bwd_inputs(dy_q, dy_sf, w2_col_q, w2_col_sf, z_bf16, offsets):
    if dy_q.ndim != 2:
        raise ValueError(f"dy_q must be 2D [R, D], got shape {tuple(dy_q.shape)}")
    if w2_col_q.ndim != 3:
        raise ValueError(
            f"w2_col_q must be 3D [G, D, F], got shape {tuple(w2_col_q.shape)}"
        )
    rows, model_dim = dy_q.shape
    groups, w_d, hidden = w2_col_q.shape
    if w_d != model_dim:
        raise ValueError(f"w2_col_q's D dim {w_d} must match dy_q's D {model_dim}")
    if z_bf16.ndim != 2 or tuple(z_bf16.shape) != (rows, 2 * hidden):
        raise ValueError(
            f"z_bf16 must be [{rows}, {2 * hidden}] (32-block interleaved, the "
            f"exact fwd-op output), got shape {tuple(z_bf16.shape)}"
        )
    device = dy_q.device

    _require_cuda_device(device, "dy_q")
    validate_feature_dims(model_dim=model_dim, hidden_dim=hidden)
    validate_allocated_rows(rows)
    if rows * 2 * hidden >= 2**31:
        raise ValueError(
            f"R * 2F = {rows * 2 * hidden} does not fit an int32 element index"
        )
    validate_group_offsets(
        offsets, num_groups=groups, allocated_rows=rows, device=device
    )
    validate_operand(
        dy_q,
        name="dy_q",
        shape=(rows, model_dim),
        stride=(model_dim, 1),
        dtype=_E4M3,
        device=device,
    )
    # Colwise-quantized w2; strides free (dim1-native layout probe-proven).
    validate_operand(
        w2_col_q,
        name="w2_col_q",
        shape=(groups, model_dim, hidden),
        dtype=_E4M3,
        device=device,
    )
    validate_operand(
        z_bf16,
        name="z_bf16",
        shape=(rows, 2 * hidden),
        stride=(2 * hidden, 1),
        dtype=torch.bfloat16,
        device=device,
    )
    validate_blocked_scales(
        dy_sf,
        name="dy_sf",
        logical_rows=rows,
        logical_cols=model_dim // _BLOCK,
        device=device,
    )
    # Colwise weight scales: logical [F, D/32] per expert.
    validate_blocked_scales(
        w2_col_sf,
        name="w2_col_sf",
        logical_rows=hidden,
        logical_cols=model_dim // _BLOCK,
        device=device,
        groups=groups,
    )
    return rows, model_dim, hidden, groups


@torch.library.custom_op("torchao::mxfp8_grouped_gemm_dswiglu_bwd", mutates_args=())
def _mxfp8_grouped_gemm_dswiglu_bwd(
    dy_q: torch.Tensor,
    dy_sf: torch.Tensor,
    w2_col_q: torch.Tensor,
    w2_col_sf: torch.Tensor,
    z_bf16: torch.Tensor,
    offsets: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """FC2 dgrad grouped GEMM + dSwiGLU + dual MXFP8 quantization (one launch).

    Inputs:
      dy_q / dy_sf      rowwise-quantized FC2 output gradient ``[R, D]``.
      w2_col_q          E4M3 ``[G, D, F]``-logical, quantized along D, any
                        strides (dim1-native accepted).
      w2_col_sf         per-group flat blocked for logical ``[F, D/32]``.
      z_bf16            the EXACT ``[R, 2F]`` output of the fwd op (32-block
                        interleaved). Rows past ``offsets[-1]`` never read.
      offsets           int32 CUDA ``[G]``.

    Returns ``(dz_row_q, dz_row_sf, dz_col_q, dz_col_sf)``: the FC1 gradient
    ``[R, 2F]`` in the same 32-block order, rowwise + columnwise quantized
    (columnwise: un-transposed kernel bytes, PER-GROUP flat blocked scales).
    Tails garbage/read-forbidden as in the fwd op.
    """
    rows, model_dim, hidden, groups = _validate_bwd_inputs(
        dy_q, dy_sf, w2_col_q, w2_col_sf, z_bf16, offsets
    )
    specs = _bwd_output_specs(rows, hidden)
    if rows == 0:
        return _allocate_from_specs(specs, dy_q.device)

    import cudnn

    out = cudnn.grouped_gemm_dglu_wrapper_sm100(
        a_tensor=dy_q.unsqueeze(0).permute(1, 2, 0),
        c_tensor=z_bf16.unsqueeze(0).permute(1, 2, 0),
        sfa_tensor=_act_scale_view(dy_sf, rows, model_dim),
        padded_offsets=offsets,
        alpha_tensor=_cached_ones(groups, torch.bfloat16, dy_q.device),
        beta_tensor=_cached_ones(groups, torch.bfloat16, dy_q.device),
        prob_tensor=None,
        dprob_tensor=None,
        b_tensor=w2_col_q.permute(2, 1, 0),
        sfb_tensor=_weight_scale_view(w2_col_sf, groups, hidden, model_dim),
        norm_const_tensor=_cached_ones(1, torch.float32, dy_q.device),
        acc_dtype=torch.float32,
        d_dtype=_E4M3,
        cd_major="n",
        sf_vec_size=_BLOCK,
        act_func="dswiglu",
        discrete_col_sfd=True,
        use_dynamic_sched=True,
        current_stream=_stream(),
    )
    results = (
        out["d_row_tensor"].view(rows, 2 * hidden),
        _flat_scales(out["sfd_row_tensor"]),
        out["d_col_tensor"].view(rows, 2 * hidden),
        _flat_scales(out["sfd_col_tensor"]),
    )
    return tuple(
        _check_normalized(t, name=spec[0], shape=spec[1], dtype=spec[2])
        for t, spec in zip(results, specs)
    )


@_mxfp8_grouped_gemm_dswiglu_bwd.register_fake
def _(dy_q, dy_sf, w2_col_q, w2_col_sf, z_bf16, offsets):
    rows, _model_dim, hidden, _groups = _validate_bwd_inputs(
        dy_q, dy_sf, w2_col_q, w2_col_sf, z_bf16, offsets
    )
    return _allocate_from_specs(_bwd_output_specs(rows, hidden), dy_q.device)


# --------------------------------------------------------------------------
# Op 4: grouped weight gradient (wgrad wrapper, dense output mode)
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
    groups = offsets.numel() if isinstance(offsets, torch.Tensor) else 0
    device = dy_col_q.device

    _require_cuda_device(device, "dy_col_q")
    validate_allocated_rows(rows)
    validate_feature_dims(model_dim=out_features, hidden_dim=in_features)
    if rows * max(out_features, in_features) >= 2**31:
        raise ValueError(
            f"R * max(N, K) = {rows * max(out_features, in_features)} does not "
            "fit an int32 element index"
        )
    validate_group_offsets(
        offsets, num_groups=groups, allocated_rows=rows, device=device
    )
    # Both operands accept ANY major: dim1-native transposed memory, the fwd/
    # bwd ops' un-transposed kernel bytes, and mixes -- all four combinations
    # probe-proven.
    validate_operand(
        dy_col_q,
        name="dy_col_q",
        shape=(rows, out_features),
        dtype=_E4M3,
        device=device,
    )
    validate_operand(
        x_col_q,
        name="x_col_q",
        shape=(rows, in_features),
        dtype=_E4M3,
        device=device,
    )
    # Columnwise scale buffers are sized by offsets[-1] (a device value), not
    # by R: only dtype/device/divisibility are host-checkable.
    validate_ragged_colwise_scales(
        dy_col_sf,
        name="dy_col_sf",
        features=out_features,
        allocated_rows=rows,
        device=device,
    )
    validate_ragged_colwise_scales(
        x_col_sf,
        name="x_col_sf",
        features=in_features,
        allocated_rows=rows,
        device=device,
    )
    # No cross-buffer size check: a kernel-produced operand's scales are sized
    # by the ALLOCATED rows while a composite-produced operand's are sized by
    # the ROUTED total offsets[-1] -- mixing the two is legitimate and
    # probe-proven (tail case); the kernel reads only within offsets.
    return rows, out_features, in_features, groups


@torch.library.custom_op("torchao::mxfp8_grouped_gemm_wgrad", mutates_args=())
def _mxfp8_grouped_gemm_wgrad(
    dy_col_q: torch.Tensor,
    dy_col_sf: torch.Tensor,
    x_col_q: torch.Tensor,
    x_col_sf: torch.Tensor,
    offsets: torch.Tensor,
) -> torch.Tensor:
    """Grouped MXFP8 weight gradient ``dw[g] = dequant(dy_g).T @ dequant(x_g)``.

    Inputs:
      dy_col_q  E4M3 logical ``[R, N]``, columnwise (32x1) quantized, ANY major.
      dy_col_sf PER-GROUP flat blocked scales (each expert's ``[N, rows_g/32]``
                block concatenated; the K-groups layout). Sized by the routed
                total ``offsets[-1]``, which may be < R.
      x_col_q / x_col_sf  likewise for logical ``[R, K]``.
      offsets   int32 CUDA ``[G]`` exclusive ends over the shared row axis.

    Do NOT feed whole-matrix ``to_blocked`` scales here: the per-group and
    whole-matrix orders coincide in byte count but not content whenever G > 1,
    and the mismatch is silent (probe: 2-5 dB instead of 100+).

    Returns contiguous BF16 ``dw [G, N, K]`` (FP32 accumulation). Zero-token
    experts ARE written (all-zero slices, probe-verified); ``R == 0`` returns
    zeros without launching. FC1 wgrad: N=2F, K=D. FC2 wgrad: N=D, K=F.
    """
    rows, out_features, in_features, groups = _validate_wgrad_inputs(
        dy_col_q, dy_col_sf, x_col_q, x_col_sf, offsets
    )
    dw = torch.empty(
        (groups, out_features, in_features),
        dtype=torch.bfloat16,
        device=dy_col_q.device,
    )
    if rows == 0:
        return dw.zero_()

    import cudnn

    cudnn.grouped_gemm_wgrad_wrapper_sm100(
        a_tensor=dy_col_q.t(),
        b_tensor=x_col_q,
        sfa_tensor=_as_e8m0(dy_col_sf).view(out_features, -1),
        sfb_tensor=_as_e8m0(x_col_sf).view(in_features, -1),
        offsets_tensor=offsets,
        acc_dtype=torch.float32,
        sf_vec_size=_BLOCK,
        accumulate_on_output=False,
        output_mode="dense",
        wgrad_tensor=dw,
        wgrad_dtype=torch.bfloat16,
        current_stream=_stream(),
    )
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
