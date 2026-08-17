# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD 3-Clause license found in the
# LICENSE file in the root directory of this source tree.

"""Fused MXFP8 grouped-MLP operations for the routed-expert training path.

Three physically fused CuTe DSL kernels for Blackwell (SM 10.x), each exactly
one GPU kernel launch for a nonempty supported input:

* :func:`mxfp8_grouped_gemm_swiglu_fwd`  -- FC1 ragged grouped GEMM (MXFP8
  operands, FP32 accumulation) + BF16 pre-activation save + SwiGLU + rowwise
  1x32 AND columnwise 32x1 MXFP8 RCEIL quantization of the activation.
* :func:`mxfp8_grouped_gemm_dswiglu_bwd` -- FC2 dgrad ragged grouped GEMM +
  dSwiGLU (from the saved pre-activation) + dual MXFP8 quantization of the
  FC1 gradient.
* :func:`mxfp8_grouped_gemm_wgrad`       -- generic ragged-reduction grouped
  weight gradient, called once for FC1 and once for FC2.

This module owns the kernels and their operator wrappers only. Trainer
integration (converter selection, the autograd composite, saved-activation
ownership, expert padding configuration) is follow-up work in the consumer;
:func:`is_supported` is the shape predicate that integration should call
before selecting this operator family.

Importing this module registers the three ``torchao::`` custom ops.
"""

import importlib.util

import torch

# Importing the ops module registers the custom ops as a side effect. It is
# importable with no CuTe DSL installed; the DSL is imported lazily inside the
# op bodies at first real launch.
from torchao.prototype.moe_training.kernels.mxfp8 import (
    grouped_mlp_ops as _grouped_mlp_ops,  # noqa: F401
)
from torchao.utils import is_cuda_version_at_least

__all__ = [
    "is_supported",
    "mxfp8_grouped_gemm_dswiglu_bwd",
    "mxfp8_grouped_gemm_swiglu_fwd",
    "mxfp8_grouped_gemm_wgrad",
]

# Every per-expert row count, the row allocation, and both feature dims must be
# multiples of this: the tcgen05 blocked scale layout permutes in 128-row tiles
# and the kernels tile all three GEMM axes at 128 with no tail path.
GROUP_ALIGNMENT = 128

# Runtime package detection, deliberately independent of the quantizer
# modules' availability flags: probing specs never imports the DSL.
_CUTEDSL_RUNTIME_PACKAGES = {
    "cuda.bindings.driver": "cuda-python",
    "cutlass": "nvidia-cutlass-dsl",
    "cutlass.cute": "nvidia-cutlass-dsl",
    "tvm_ffi": "apache-tvm-ffi",
}


def _missing_cutedsl_runtime_packages() -> list:
    """Names of the pip packages required by the CuTe DSL runtime but absent."""
    missing = []
    for module_name, package_name in _CUTEDSL_RUNTIME_PACKAGES.items():
        try:
            spec = importlib.util.find_spec(module_name)
        except (ModuleNotFoundError, ValueError):
            spec = None
        if spec is None and package_name not in missing:
            missing.append(package_name)
    return missing


def _is_sm_10x() -> bool:
    return torch.cuda.is_available() and torch.cuda.get_device_capability()[0] == 10


_mxfp8_grouped_mlp_kernels_available = (
    _is_sm_10x()
    and is_cuda_version_at_least(12, 8)
    and not _missing_cutedsl_runtime_packages()
)


def _require_available() -> None:
    """Raise a clean NotImplementedError when the kernels cannot run here."""
    if _mxfp8_grouped_mlp_kernels_available:
        return
    reasons = []
    if not torch.cuda.is_available():
        reasons.append("CUDA is not available")
    elif not _is_sm_10x():
        reasons.append(
            "requires an SM 10.x (Blackwell) GPU, found compute capability "
            f"{torch.cuda.get_device_capability()}"
        )
    if torch.cuda.is_available() and not is_cuda_version_at_least(12, 8):
        reasons.append(f"requires CUDA >= 12.8, found {torch.version.cuda}")
    missing = _missing_cutedsl_runtime_packages()
    if missing:
        reasons.append("missing required packages: " + ", ".join(missing))
    if not reasons:
        reasons.append("kernels are disabled on this system")
    raise NotImplementedError(
        "MXFP8 grouped-MLP kernels are unavailable: " + "; ".join(reasons)
    )


def is_supported(
    model_dim: int, hidden_dim: int, allocated_rows: int, num_groups: int
) -> bool:
    """Pure shape predicate for selecting this operator family.

    True when the static shapes satisfy the kernel contract: at least one
    expert group and D, F, R all positive multiples of 128. Integration code
    (e.g. a quantization converter choosing between this fused path and the
    unfused grouped-mm path) should call this BEFORE selecting the ops and
    fall back when it is False -- and must also guarantee the runtime offsets
    invariant, i.e. configure expert padding to 128 rows, because per-expert
    row counts live in device memory and are not host-checkable here.

    Environment availability (CUDA, SM 10.x, the CuTe DSL runtime) is a
    separate concern: combine with ``_mxfp8_grouped_mlp_kernels_available``.
    """
    return (
        num_groups >= 1
        and model_dim > 0
        and hidden_dim > 0
        and allocated_rows > 0
        and model_dim % GROUP_ALIGNMENT == 0
        and hidden_dim % GROUP_ALIGNMENT == 0
        and allocated_rows % GROUP_ALIGNMENT == 0
    )


def mxfp8_grouped_gemm_swiglu_fwd(x_q, x_sf, w13_t_q, w13_t_sf, offsets):
    """FC1 grouped GEMM + SwiGLU + rowwise/columnwise MXFP8 quantization.

    One kernel launch for any nonempty supported input. Arguments (all CUDA
    tensors on one device, prequantized by the caller):

    * ``x_q``: E4M3 ``[R, D]``, stride ``(D, 1)`` -- rowwise 1x32-quantized
      activations, expert-major packed rows.
    * ``x_sf``: flat blocked E8M0 scales for logical ``[R, D/32]``.
    * ``w13_t_q``: E4M3 ``[G, D, 2F]``, stride ``(2*D*F, 1, D)`` -- quantized
      ``w13.reshape(G, 2F, D).transpose(-2, -1)``; the ``2F`` axis is
      element-interleaved gate/up (gate at even indices).
    * ``w13_t_sf``: blocked E8M0 ``[G, round_up(2F,128)*round_up(D/32,4)]``.
    * ``offsets``: int32 ``[G]`` exclusive per-expert end rows; every expert's
      row count must be a nonnegative multiple of 128 and
      ``offsets[-1] <= R`` (documented caller invariant; see
      ``TORCHAO_MXFP8_VALIDATE_OFFSETS=1`` for the opt-in synchronized check).

    Returns ``(z_bf16, h_row_q, h_row_sf, h_col_q, h_col_sf)``; ``z_bf16``
    ``[R, F, 2]`` (gate index 0, up index 1) is the BF16 pre-activation saved
    for backward -- pass it unchanged to
    :func:`mxfp8_grouped_gemm_dswiglu_bwd`. ``h_col_q`` is column-major with
    whole-matrix blocked scales, ready for
    :func:`mxfp8_grouped_gemm_wgrad`. Inactive tail rows of every output are
    written as zeros. ``R == 0`` returns empty outputs; ``G == 0`` raises
    ``ValueError``.
    """
    _require_available()
    return torch.ops.torchao.mxfp8_grouped_gemm_swiglu_fwd(
        x_q, x_sf, w13_t_q, w13_t_sf, offsets
    )


def mxfp8_grouped_gemm_dswiglu_bwd(
    do_q, do_sf, w2_dgrad_q, w2_dgrad_sf, z_bf16, offsets
):
    """FC2 dgrad grouped GEMM + dSwiGLU + rowwise/columnwise MXFP8 quantization.

    One kernel launch for any nonempty supported input. Arguments:

    * ``do_q`` / ``do_sf``: rowwise 1x32-quantized FC2 output-gradient, E4M3
      ``[R, D]`` stride ``(D, 1)`` with blocked scales for ``[R, D/32]``.
    * ``w2_dgrad_q`` / ``w2_dgrad_sf``: E4M3 ``[G, D, F]`` stride
      ``(D*F, 1, D)`` (dgrad orientation of w2) with per-expert blocked scales
      for logical ``[F, D/32]``.
    * ``z_bf16``: the exact ``[R, F, 2]`` pre-activation saved by
      :func:`mxfp8_grouped_gemm_swiglu_fwd`; rows past ``offsets[-1]`` are
      never read.
    * ``offsets``: as in the forward op.

    Returns ``(dz_row_q, dz_row_sf, dz_col_q, dz_col_sf)`` -- the FC1 gradient
    ``[R, 2F]`` with gate/up gradients element-interleaved to match ``z_bf16``,
    quantized both rowwise (row-major qdata) and columnwise (column-major
    qdata, whole-matrix blocked scales, ready for the FC1 wgrad call).
    """
    _require_available()
    return torch.ops.torchao.mxfp8_grouped_gemm_dswiglu_bwd(
        do_q, do_sf, w2_dgrad_q, w2_dgrad_sf, z_bf16, offsets
    )


def mxfp8_grouped_gemm_wgrad(dy_col_q, dy_col_sf, x_col_q, x_col_sf, offsets):
    """Grouped MXFP8 weight gradient: ``dw[g] = dequant(dy_g).T @ dequant(x_g)``.

    One kernel launch per nonempty invocation, with the per-expert row ranges
    of the shared ``R`` axis forming the ragged reduction. Both operands are
    columnwise (32x1) quantized: E4M3 logical ``[R, N]`` / ``[R, K]`` with
    column-major stride ``(1, R)`` and WHOLE-MATRIX blocked E8M0 scales for
    logical ``[N, R/32]`` / ``[K, R/32]`` -- exactly what the forward and
    backward ops emit. Do not feed torchao's per-group K-groups scale
    rearrangement here: it has the same byte count but a different block
    order, and produces a silently wrong ``dw``.

    Generic over both call sites with no mode flag: FC1 wgrad is ``N=2F,
    K=D``; FC2 wgrad is ``N=D, K=F``. Returns contiguous BF16 ``[G, N, K]``
    (FP32 accumulation); zero-token experts yield all-zero slices, and
    ``R == 0`` returns an all-zero result without launching.
    """
    _require_available()
    return torch.ops.torchao.mxfp8_grouped_gemm_wgrad(
        dy_col_q, dy_col_sf, x_col_q, x_col_sf, offsets
    )
