# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD 3-Clause license found in the
# LICENSE file in the root directory of this source tree.

"""cuDNN-frontend MXFP8 grouped-MLP operations for routed-expert training.

Four custom ops, each one launch of a ``cudnn.grouped_gemm_*_wrapper_sm100``
CuTe DSL kernel from the standalone cudnn-frontend python package (>= 1.27,
Blackwell SM 10.x; no TransformerEngine dependency):

* :func:`mxfp8_grouped_gemm_swiglu_fwd`   -- FC1 ragged grouped GEMM + SwiGLU +
  rowwise 1x32 AND columnwise 32x1 MXFP8 RCEIL quantization + BF16 pre-GLU.
* :func:`mxfp8_grouped_gemm`        -- ragged grouped GEMM on prequantized
  MXFP8 operands to BF16 (FC2 forward and FC1 dgrad).
* :func:`mxfp8_grouped_gemm_dswiglu_bwd`   -- FC2 dgrad + dSwiGLU + dual MXFP8
  quantization of the FC1 gradient.
* :func:`mxfp8_grouped_gemm_wgrad` -- ragged-reduction grouped weight
  gradient (FC1 and FC2).

CONTRACT (stricter than the archived custom-kernel family): every per-expert
row count and the allocated row count must be multiples of **256** — the cuDNN
FE kernels hard-code ``FIX_PAD_SIZE = 256``, and groups that are only 128-row
aligned corrupt results SILENTLY and NONDETERMINISTICALLY (the corruption
locus migrates between identical-input reruns; no smoke test can prove a
misaligned config safe). Use a token dispatcher with ``pad_multiple=256`` and
see ``grouped_mlp_validation`` for the two-tier enforcement
(``TORCHAO_MXFP8_VALIDATE_OFFSETS=1`` for the opt-in synchronized check).

The FC1 weight must be provided in the cuDNN 32-block GLU row order
``[gate0(32) | up0(32) | gate1(32) | ...]`` along the 2F axis (gate = the
SiLU'd operand). Trainer integration (converter selection, the autograd
composite, weight-layout remaps, expert padding) lives in the consumer;
:func:`is_supported` is the static shape predicate to call before selecting
this family.

Importing this module registers the four ``torchao::`` custom ops. The
``cudnn`` package itself is imported lazily inside the op bodies.
"""

import importlib.util

import torch

# Importing the ops module registers the custom ops as a side effect. It is
# importable with no cudnn-frontend installed; `import cudnn` is deferred into
# the op bodies.
from torchao.prototype.moe_training.kernels.mxfp8 import (
    grouped_mlp_ops as _grouped_mlp_ops,  # noqa: F401
)
from torchao.prototype.moe_training.kernels.mxfp8.grouped_mlp_validation import (
    DIM_ALIGNMENT,
    ROW_GROUP_ALIGNMENT,
)

__all__ = [
    "DIM_ALIGNMENT",
    "ROW_GROUP_ALIGNMENT",
    "is_supported",
    "mxfp8_grouped_gemm_dswiglu_bwd",
    "mxfp8_grouped_gemm_swiglu_fwd",
    "mxfp8_grouped_gemm_wgrad",
    "mxfp8_grouped_gemm",
]

_REQUIRED_WRAPPERS = (
    "grouped_gemm_glu_wrapper_sm100",
    "grouped_gemm_quant_wrapper_sm100",
    "grouped_gemm_dglu_wrapper_sm100",
    "grouped_gemm_wgrad_wrapper_sm100",
)
# 1.27 is required: earlier frontends reject prob_tensor=None.
_MIN_FE_VERSION = (1, 27)


def _fe_version_tuple(version: str) -> tuple:
    """Numeric prefix of a version string as a tuple ('1.27.0' -> (1, 27, 0)).

    Never compare version STRINGS: '1.100' < '1.27' lexicographically.
    """
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


def _is_sm_10x() -> bool:
    return torch.cuda.is_available() and torch.cuda.get_device_capability()[0] == 10


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
    if _is_sm_10x()
    else (
        "requires an SM 10.x (Blackwell) GPU"
        if torch.cuda.is_available()
        else "CUDA is not available"
    )
)
_mxfp8_grouped_mlp_kernels_available = _mxfp8_grouped_mlp_unavailable_reason == ""


def _require_available() -> None:
    """Raise a clean NotImplementedError when the kernels cannot run here."""
    if not _mxfp8_grouped_mlp_kernels_available:
        raise NotImplementedError(
            "cuDNN-frontend MXFP8 grouped-MLP kernels are unavailable: "
            + _mxfp8_grouped_mlp_unavailable_reason
        )


def is_supported(model_dim: int, hidden_dim: int) -> bool:
    """Static shape predicate for selecting this operator family.

    True when D and F are positive multiples of 128. Integration code must
    ALSO guarantee the runtime row contract (per-expert groups and the row
    allocation padded to multiples of 256, e.g. dispatcher pad_multiple=256):
    row counts live in device memory and are not checkable here.

    Environment availability (cudnn-frontend >= 1.27, SM 10.x) is a separate
    concern: combine with ``_mxfp8_grouped_mlp_kernels_available``.
    """
    return (
        model_dim > 0
        and hidden_dim > 0
        and model_dim % DIM_ALIGNMENT == 0
        and hidden_dim % DIM_ALIGNMENT == 0
    )


def mxfp8_grouped_gemm_swiglu_fwd(x_q, x_sf, w13_q, w13_sf, offsets):
    """FC1 grouped GEMM + SwiGLU + rowwise/columnwise MXFP8 quantization.

    See ``torchao::mxfp8_grouped_gemm_swiglu_fwd`` for the full ABI. ``w13_q``
    is E4M3 ``[G, 2F, D]`` contiguous with rows in 32-block GLU order; returns
    ``(z_bf16 [R, 2F], h_row_q [R, F], h_row_sf, h_col_q [R, F], h_col_sf)``
    where the columnwise scales are PER-GROUP blocked. Rows past
    ``offsets[-1]`` of every output are garbage and read-forbidden.
    """
    _require_available()
    return torch.ops.torchao.mxfp8_grouped_gemm_swiglu_fwd(
        x_q, x_sf, w13_q, w13_sf, offsets
    )


def mxfp8_grouped_gemm(a_q, a_sf, b_q, b_sf, offsets):
    """Ragged grouped GEMM on prequantized MXFP8 operands, BF16 output.

    ``b_q`` is ``[G, N, K]``-logical quantized along K with free strides
    (rowwise casts as-is; dim1-colwise casts transposed into this
    orientation); ``b_sf`` is always the per-group blocked ``[N, K/32]``
    orientation. Returns BF16 ``[R, N]`` with rows past ``offsets[-1]``
    uninitialized.
    """
    _require_available()
    return torch.ops.torchao.mxfp8_grouped_gemm(a_q, a_sf, b_q, b_sf, offsets)


def mxfp8_grouped_gemm_dswiglu_bwd(dy_q, dy_sf, w2_col_q, w2_col_sf, z_bf16, offsets):
    """FC2 dgrad + dSwiGLU + dual MXFP8 quantization of the FC1 gradient.

    ``z_bf16`` must be the exact fwd-op output. Returns
    ``(dz_row_q [R, 2F], dz_row_sf, dz_col_q [R, 2F], dz_col_sf)`` in the same
    32-block order.
    """
    _require_available()
    return torch.ops.torchao.mxfp8_grouped_gemm_dswiglu_bwd(
        dy_q, dy_sf, w2_col_q, w2_col_sf, z_bf16, offsets
    )


def mxfp8_grouped_gemm_wgrad(dy_col_q, dy_col_sf, x_col_q, x_col_sf, offsets):
    """Grouped MXFP8 weight gradient ``dw[g] = dequant(dy_g).T @ dequant(x_g)``.

    Both operands columnwise (32x1) quantized with PER-GROUP blocked scales
    (never whole-matrix ``to_blocked`` — same byte count, silently wrong
    block order). Returns contiguous BF16 ``[G, N, K]``.
    """
    _require_available()
    return torch.ops.torchao.mxfp8_grouped_gemm_wgrad(
        dy_col_q, dy_col_sf, x_col_q, x_col_sf, offsets
    )
