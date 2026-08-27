# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD 3-Clause license found in the
# LICENSE file in the root directory of this source tree.

"""MXFP8 routed-expert grouped-MLP ops over the cuDNN-frontend CuTe DSL kernels.

Four custom ops, each one launch of a ``cudnn.grouped_gemm_*_wrapper_sm100``
kernel from the standalone cudnn-frontend python package (>= 1.27, Blackwell
SM 10.0 exactly -- the wrappers are sm100-specific; no TransformerEngine
dependency); the matching public wrappers live at the bottom of this module:

* :func:`mxfp8_grouped_gemm_swiglu_fwd_cudnn`   -- FC1 ragged grouped GEMM + SwiGLU
  + rowwise 1x32 AND columnwise 32x1 MXFP8 RCEIL quantization + BF16 pre-GLU.
* :func:`mxfp8_grouped_gemm_cudnn`              -- ragged grouped GEMM on
  prequantized operands to BF16 (FC2 forward and FC1 dgrad).
* :func:`mxfp8_grouped_gemm_dswiglu_bwd_cudnn`  -- FC2 dgrad + dSwiGLU + dual MXFP8
  quantization of the FC1 gradient.
* :func:`mxfp8_grouped_gemm_wgrad_cudnn`        -- ragged-reduction grouped weight
  gradient (dense output mode; called once for FC1 and once for FC2).

CONTRACT: every per-expert row count and the allocated row count must be
multiples of **256** -- the cuDNN FE kernels hard-code ``FIX_PAD_SIZE = 256``,
and groups that are only 128-row aligned corrupt results SILENTLY and
NONDETERMINISTICALLY (the corruption locus migrates between identical-input
reruns; no smoke test can prove a misaligned config safe). Use a token
dispatcher with ``pad_multiple=256``.

All scale arguments are FLAT blocked E8M0 buffers (uint8 or float8_e8m0fnu);
the ops build the kernel-native 6-D / 2-D views internally with probe-proven
recipes. The FC1 weight is E4M3 ``[G, 2F, D]`` with rows in the cuDNN
32-block GLU order ``[gate0(32) | up0(32) | gate1(32) | ...]`` (gate = the
SiLU'd operand). ``offsets`` is int32 CUDA ``[G]`` exclusive-end rows. Rows
in ``[offsets[-1], R)``: caller-allocated outputs (the grouped-mm result and
the weight gradients) keep their tails untouched, while kernel-allocated
outputs (z, h, dz and their scales) carry garbage tails that are
read-forbidden -- both behaviors probe-verified with NaN-poisoned tails.

Importing this module registers the four ``torchao::`` custom ops; the
``cudnn`` package itself is imported lazily inside the op bodies at first
real launch. :func:`is_supported` is the static shape predicate to call
before selecting this family.
"""

import importlib.util
from typing import Tuple

import torch

__all__ = [
    "DIM_ALIGNMENT",
    "ROW_GROUP_ALIGNMENT",
    "SCALE_BLOCK_SIZE",
    "is_supported",
    "mxfp8_grouped_gemm_cudnn",
    "mxfp8_grouped_gemm_dswiglu_bwd_cudnn",
    "mxfp8_grouped_gemm_swiglu_fwd_cudnn",
    "mxfp8_grouped_gemm_wgrad_cudnn",
]

# MXFP8 scaling block: 32 values share one E8M0 scale.
SCALE_BLOCK_SIZE = 32
# Feature-dimension granularity (D and F).
DIM_ALIGNMENT = 128
# Row-count granularity: per-expert groups AND the allocated row count (the
# cuDNN FE kernels' FIX_PAD_SIZE).
ROW_GROUP_ALIGNMENT = 256

_E4M3 = torch.float8_e4m3fn
_E8M0 = torch.float8_e8m0fnu
_BLOCK = SCALE_BLOCK_SIZE


# --------------------------------------------------------------------------
# Availability probe and the static shape predicate.
# --------------------------------------------------------------------------

_REQUIRED_WRAPPERS = (
    "grouped_gemm_glu_wrapper_sm100",
    "grouped_gemm_quant_wrapper_sm100",
    "grouped_gemm_dglu_wrapper_sm100",
    "grouped_gemm_wgrad_wrapper_sm100",
)
# 1.27 is required: earlier frontends reject prob_tensor=None.
_MIN_FE_VERSION = (1, 27)


def _fe_version_tuple(version: str) -> tuple:
    """Numeric prefix as a tuple ('1.27.0' -> (1, 27, 0)); never compare
    version STRINGS ('1.100' < '1.27' lexicographically)."""
    parts = []
    for piece in version.split("."):
        digits = ""
        for ch in piece:
            if not ch.isdigit():
                break
            digits += ch
        if not digits:
            break
        parts.append(int(digits))
    return tuple(parts)


def _is_sm100() -> bool:
    # Exactly capability (10, 0): the cudnn wrappers are *_sm100-specific and
    # unproven on other SM 10.x parts.
    return torch.cuda.is_available() and torch.cuda.get_device_capability() == (10, 0)


def _probe_cudnn_frontend() -> str:
    """Empty string when usable; else the reason it is not."""
    if importlib.util.find_spec("cudnn") is None:
        return "the cudnn-frontend python package ('cudnn') is not installed"
    try:
        import cudnn
    except Exception as exc:  # pragma: no cover - environment-specific
        return f"'import cudnn' failed: {exc!r}"
    version = getattr(cudnn, "__version__", "0")
    if _fe_version_tuple(version) < _MIN_FE_VERSION:
        return (
            f"cudnn-frontend {version} is too old; >= "
            f"{'.'.join(map(str, _MIN_FE_VERSION))} is required "
            "(prob_tensor=None support)"
        )
    missing = [name for name in _REQUIRED_WRAPPERS if not hasattr(cudnn, name)]
    if missing:
        return "cudnn-frontend lacks required wrappers: " + ", ".join(missing)
    return ""


_mxfp8_grouped_mlp_unavailable_reason = (
    _probe_cudnn_frontend()
    if _is_sm100()
    else (
        "requires an SM 10.0 (Blackwell) GPU; the cudnn wrappers are sm100-specific"
        if torch.cuda.is_available()
        else "CUDA is not available"
    )
)
_mxfp8_grouped_mlp_kernels_available = _mxfp8_grouped_mlp_unavailable_reason == ""


def _require_available() -> None:
    if not _mxfp8_grouped_mlp_kernels_available:
        raise NotImplementedError(
            "cuDNN-frontend MXFP8 grouped-MLP kernels are unavailable: "
            + _mxfp8_grouped_mlp_unavailable_reason
        )


def is_supported(model_dim: int, hidden_dim: int) -> bool:
    """True when D and F are positive multiples of 128. Integration code must
    ALSO guarantee the runtime row contract (per-expert groups and the row
    allocation padded to multiples of 256): row counts live in device memory
    and are not checkable here. Environment availability is a separate
    concern (``_mxfp8_grouped_mlp_kernels_available``)."""
    return (
        model_dim > 0
        and hidden_dim > 0
        and model_dim % DIM_ALIGNMENT == 0
        and hidden_dim % DIM_ALIGNMENT == 0
    )


# Small per-(groups, dtype, device) caches for the kernels' alpha/beta and
# norm-const tensors, each stored with the event recorded after its fill:
# the fill runs on the FIRST caller's stream, so a cache hit on any other
# stream must order after it (the buffer is immutable once filled, so one
# event covers every later consumer). Never cached: the CUDA stream itself
# (looked up per call).
_ones_cache: dict = {}


def _cached_ones(numel: int, dtype: torch.dtype, device: torch.device) -> torch.Tensor:
    key = (numel, dtype, device)
    hit = _ones_cache.get(key)
    if hit is None:
        out = torch.ones(numel, dtype=dtype, device=device)
        event = torch.cuda.Event()
        event.record(torch.cuda.current_stream(device))
        _ones_cache[key] = (out, event)
        return out
    out, event = hit
    event.wait(torch.cuda.current_stream(device))
    return out


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


def _fwd_dims(x_q, w13_q):
    rows, model_dim = x_q.shape
    groups, two_hidden, _ = w13_q.shape
    return rows, model_dim, two_hidden // 2, groups


@torch.library.custom_op(
    "torchao::mxfp8_grouped_gemm_swiglu_fwd_cudnn", mutates_args=()
)
def _mxfp8_grouped_gemm_swiglu_fwd_cudnn(
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
              (caller invariant).

    Returns ``(z_bf16, h_row_q, h_row_sf, h_col_q, h_col_sf)``:
      z_bf16   BF16 ``[R, 2F]`` contiguous pre-activation in the same 32-block
               order; consumed unchanged by the bwd op.
      h_row_q  E4M3 ``[R, F]`` contiguous; h_row_sf flat blocked for ``[R, F/32]``.
      h_col_q  E4M3 ``[R, F]`` contiguous columnwise-quantized bytes
               (un-transposed kernel layout); h_col_sf PER-GROUP flat blocked
               for ``[F, rows_g/32]`` per expert.

    Rows past ``offsets[-1]`` of every output are GARBAGE (kernel-computed from
    the quantized input tail) and read-forbidden. ``R == 0`` returns empty
    outputs without touching cudnn.
    """
    rows, model_dim, hidden, groups = _fwd_dims(x_q, w13_q)
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


@_mxfp8_grouped_gemm_swiglu_fwd_cudnn.register_fake
def _(x_q, x_sf, w13_q, w13_sf, offsets):
    rows, _model_dim, hidden, _groups = _fwd_dims(x_q, w13_q)
    return _allocate_from_specs(_fwd_output_specs(rows, hidden), x_q.device)


# --------------------------------------------------------------------------
# Op 2: grouped GEMM on prequantized operands -> BF16 (quant wrapper)
# --------------------------------------------------------------------------


def _mm_dims(a_q, b_q):
    rows, contraction = a_q.shape
    groups, out_features, _ = b_q.shape
    return rows, out_features, contraction, groups


@torch.library.custom_op("torchao::mxfp8_grouped_gemm_cudnn", mutates_args=())
def _mxfp8_grouped_gemm_cudnn(
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
    rows, out_features, contraction, groups = _mm_dims(a_q, b_q)
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


@_mxfp8_grouped_gemm_cudnn.register_fake
def _(a_q, a_sf, b_q, b_sf, offsets):
    rows, out_features, _contraction, _groups = _mm_dims(a_q, b_q)
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


def _bwd_dims(dy_q, w2_col_q):
    rows, model_dim = dy_q.shape
    groups, _, hidden = w2_col_q.shape
    return rows, model_dim, hidden, groups


@torch.library.custom_op(
    "torchao::mxfp8_grouped_gemm_dswiglu_bwd_cudnn", mutates_args=()
)
def _mxfp8_grouped_gemm_dswiglu_bwd_cudnn(
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
    rows, model_dim, hidden, groups = _bwd_dims(dy_q, w2_col_q)
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


@_mxfp8_grouped_gemm_dswiglu_bwd_cudnn.register_fake
def _(dy_q, dy_sf, w2_col_q, w2_col_sf, z_bf16, offsets):
    rows, _model_dim, hidden, _groups = _bwd_dims(dy_q, w2_col_q)
    return _allocate_from_specs(_bwd_output_specs(rows, hidden), dy_q.device)


# --------------------------------------------------------------------------
# Op 4: grouped weight gradient (wgrad wrapper, dense output mode)
# --------------------------------------------------------------------------


def _wgrad_dims(dy_col_q, x_col_q, offsets):
    rows, out_features = dy_col_q.shape
    in_features = x_col_q.shape[1]
    return rows, out_features, in_features, offsets.numel()


@torch.library.custom_op("torchao::mxfp8_grouped_gemm_wgrad_cudnn", mutates_args=())
def _mxfp8_grouped_gemm_wgrad_cudnn(
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
                block concatenated; the K-groups layout), padded with dead
                bytes to ``round_up(N,128) * R/32`` when ``offsets[-1] < R``.
      x_col_q / x_col_sf  likewise for logical ``[R, K]``.
      offsets   int32 CUDA ``[G]`` exclusive ends over the shared row axis.

    Do NOT feed whole-matrix ``to_blocked`` scales here: the per-group and
    whole-matrix orders coincide in byte count but not content whenever G > 1,
    and the mismatch is silent (probe: 2-5 dB instead of 100+).

    Returns contiguous BF16 ``dw [G, N, K]`` (FP32 accumulation). Zero-token
    experts ARE written (all-zero slices, probe-verified); ``R == 0`` returns
    zeros without launching. FC1 wgrad: N=2F, K=D. FC2 wgrad: N=D, K=F.
    """
    rows, out_features, in_features, groups = _wgrad_dims(dy_col_q, x_col_q, offsets)
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


@_mxfp8_grouped_gemm_wgrad_cudnn.register_fake
def _(dy_col_q, dy_col_sf, x_col_q, x_col_sf, offsets):
    _rows, out_features, in_features, groups = _wgrad_dims(dy_col_q, x_col_q, offsets)
    return torch.empty(
        (groups, out_features, in_features),
        dtype=torch.bfloat16,
        device=dy_col_q.device,
    )


# --------------------------------------------------------------------------
# Public wrappers: availability-gated entry points over the four custom ops.
# --------------------------------------------------------------------------


def mxfp8_grouped_gemm_swiglu_fwd_cudnn(x_q, x_sf, w13_q, w13_sf, offsets):
    """FC1 grouped GEMM + SwiGLU + rowwise/columnwise MXFP8 quantization.

    See ``torchao::mxfp8_grouped_gemm_swiglu_fwd_cudnn`` for the full ABI. ``w13_q``
    is E4M3 ``[G, 2F, D]`` contiguous with rows in 32-block GLU order; returns
    ``(z_bf16 [R, 2F], h_row_q [R, F], h_row_sf, h_col_q [R, F], h_col_sf)``
    where the columnwise scales are PER-GROUP blocked. Rows past
    ``offsets[-1]`` of every output are garbage and read-forbidden.
    """
    _require_available()
    return torch.ops.torchao.mxfp8_grouped_gemm_swiglu_fwd_cudnn(
        x_q, x_sf, w13_q, w13_sf, offsets
    )


def mxfp8_grouped_gemm_cudnn(a_q, a_sf, b_q, b_sf, offsets):
    """Ragged grouped GEMM on prequantized MXFP8 operands, BF16 output.

    ``b_q`` is ``[G, N, K]``-logical quantized along K with free strides
    (rowwise casts as-is; dim1-colwise casts transposed into this
    orientation); ``b_sf`` is always the per-group blocked ``[N, K/32]``
    orientation. Returns BF16 ``[R, N]`` with rows past ``offsets[-1]``
    uninitialized.
    """
    _require_available()
    return torch.ops.torchao.mxfp8_grouped_gemm_cudnn(a_q, a_sf, b_q, b_sf, offsets)


def mxfp8_grouped_gemm_dswiglu_bwd_cudnn(
    dy_q, dy_sf, w2_col_q, w2_col_sf, z_bf16, offsets
):
    """FC2 dgrad + dSwiGLU + dual MXFP8 quantization of the FC1 gradient.

    ``z_bf16`` must be the exact fwd-op output. Returns
    ``(dz_row_q [R, 2F], dz_row_sf, dz_col_q [R, 2F], dz_col_sf)`` in the same
    32-block order.
    """
    _require_available()
    return torch.ops.torchao.mxfp8_grouped_gemm_dswiglu_bwd_cudnn(
        dy_q, dy_sf, w2_col_q, w2_col_sf, z_bf16, offsets
    )


def mxfp8_grouped_gemm_wgrad_cudnn(dy_col_q, dy_col_sf, x_col_q, x_col_sf, offsets):
    """Grouped MXFP8 weight gradient ``dw[g] = dequant(dy_g).T @ dequant(x_g)``.

    Both operands columnwise (32x1) quantized with PER-GROUP blocked scales
    (never whole-matrix ``to_blocked`` -- same byte count, silently wrong
    block order). Returns contiguous BF16 ``[G, N, K]``.
    """
    _require_available()
    return torch.ops.torchao.mxfp8_grouped_gemm_wgrad_cudnn(
        dy_col_q, dy_col_sf, x_col_q, x_col_sf, offsets
    )
