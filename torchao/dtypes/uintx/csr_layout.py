# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD 3-Clause license found in the
# LICENSE file in the root directory of this source tree.
"""
Compressed-Sparse-Row (CSR) layout support for *affine quantized tensors* in
TorchAO.

Key pieces:
* ``CSRLayout`` – a stateless ``Layout`` subclass signalling that a tensor is
  stored in CSR form.
* ``CSR_AQTTensorImpl`` – a ``TensorImpl`` that actually holds the compressed
  data and quantization parameters (scale / zero-point).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch
from torch.utils._python_dispatch import return_and_correct_aliasing

from torchao.dtypes.affine_quantized_tensor import (
    AffineQuantizedTensor,
    register_layout,
)
from torchao.dtypes.uintx.plain_layout import (
    PlainAQTTensorImpl,
    _aqt_is_int8_reduced_range,
)
from torchao.dtypes.utils import Layout, PlainLayout

aten = torch.ops.aten

# -----------------------------------------------------------------------------
# Op-level helpers – dispatched during aten.linear if both inputs satisfy the
# predicate.  In the first version we delegate to ``torch.sparse.mm``; follow-up
# patches may swap in vendor SpMM kernels (FBGEMM / oneDNN).
# -----------------------------------------------------------------------------


def _linear_int8_act_int8_weight_csr_sparse_check(
    input_tensor: torch.Tensor,
    weight_tensor: torch.Tensor,
    bias: Optional[torch.Tensor],
) -> bool:
    """Check if we can execute csr-optimised path."""
    return (
        isinstance(input_tensor, AffineQuantizedTensor)
        and _aqt_is_int8_reduced_range(input_tensor)
        and isinstance(weight_tensor, AffineQuantizedTensor)
        and input_tensor.dtype == weight_tensor.dtype
        and isinstance(input_tensor._layout, PlainLayout)
        and isinstance(weight_tensor._layout, CSRLayout)
    )


def _linear_int8_act_int8_weight_csr_sparse_impl(
    input_tensor: AffineQuantizedTensor,
    weight_tensor: AffineQuantizedTensor,
    bias: Optional[torch.Tensor],
):
    """Reference implementation using ``torch.sparse.mm``.

    * ``input_tensor`` is **plain** INT8 AQT with per-tensor scale.
    * ``weight_tensor`` is **CSR** INT8 AQT – compressed row representation.
    * Produces *fp32* output multiplied by both quant scales, then casts back
      to original activation dtype.
    """
    x_vals_int8 = input_tensor.tensor_impl.int_data
    x_scale = input_tensor.tensor_impl.scale  # shape: (1,)

    # The weight CSR tensor is stored as ``int_data`` in CSR layout.
    w_csr = weight_tensor.tensor_impl.int_data  # 2-D sparse CSR INT8
    w_scale = weight_tensor.tensor_impl.scale  # shape: (1,) or (row,)

    # Reshape activations to 2-D (batch*seqlen, in_features)
    x2d = x_vals_int8.reshape(-1, x_vals_int8.shape[-1]).to(torch.int32)

    # As for now we don't have any kernel for spmm form int8 computation for ARM so we are upscaling it to fp32 for computation
    # Once we intigrate the kernel in torch then we can use that kernel instead of torch.mm()
    y_int32 = torch.mm(
        w_csr.to(torch.float32), x2d.t().to(torch.float32).contiguous()
    ).t()

    # Dequantise: y = (x_scale * w_scale) * y_int32
    scale_mat = x_scale.view(-1, 1) * w_scale.view(1, -1)

    y_fp32 = (y_int32.to(torch.float32) * scale_mat).reshape(
        *x_vals_int8.shape[:-1],  # (B,)
        y_int32.shape[-1],  # O
    )

    if bias is not None:
        y_fp32 += bias

    # Cast back to activation dtype (usually fp32 or bf16)
    return y_fp32.to(input_tensor.dtype).contiguous()


@dataclass(frozen=True)
class CSRLayout(Layout):
    """Layout marker for *Compressed Sparse Row* INT8 weights.

    Parameters
    ----------
    target_sparsity : float ∈ (0, 1) or *None*
        If given, this overrides the global env-var and hard-coded default.
    """

    target_sparsity: Optional[float] = None

    def __post_init__(self):
        if self.target_sparsity is not None:
            if not (0.0 < self.target_sparsity < 1.0):
                raise ValueError(
                    "CSRLayout(target_sparsity=…) expects a value in (0, 1) "
                    f"but got {self.target_sparsity}"
                )

    def pre_process(self, input: torch.Tensor) -> torch.Tensor:
        """Magnitude‑based pruning prior to CSR packing.

        Parameters
        ----------
        input : torch.Tensor
            A **dense** weight matrix (any dtype) that we intend to quantise
            and pack as CSR.  The function returns a *copy* with elements below
            a global magnitude threshold set to zero so that subsequent
            `torch._to_csr` compression produces the desired sparsity.

        Heuristic
        ---------
        *Prune* the smallest‑magnitude values until we reach the *target
        sparsity* fraction.  The target defaults to **90 %** but can be
        overridden by setting the argument variable target_sparsity
        ``TORCHAO_CSR_TARGET_SPARSITY`` (float in 0 – 1).  If the target is
        outside that range the function becomes a no‑op and simply returns the
        input unchanged.
        """
        import os

        # 1. explicit per-instance override wins
        if self.target_sparsity is not None:
            target = self.target_sparsity
        else:
            # 2. fall back to env var or 0.9 default
            target = float(os.getenv("TORCHAO_CSR_TARGET_SPARSITY", "0.9"))

        # Validate target range; fall back to no pruning if mis‑configured
        if not (0.0 < target < 1.0):
            return input

        # Clone to avoid mutating the caller's tensor in‑place
        temp = input.detach().clone()

        # Compute global threshold that keeps the largest (1‑target) fraction
        flat = temp.abs().view(-1)
        k = int(flat.numel() * (1 - target))
        if k <= 0:  # requested sparsity so high that everything would be zero
            return temp.zero_()

        # `kthvalue` is 1‑indexed: kthvalue(1) gives minimum, kthvalue(n) max
        thresh = flat.kthvalue(k).values
        mask = temp.abs() >= thresh
        temp.mul_(mask)
        return temp


@register_layout(CSRLayout)
class CSR_AQTTensorImpl(PlainAQTTensorImpl):
    """TensorImpl for CSR-compressed INT8 weights inside an AffineQuantizedTensor.

    The internal representation follows PyTorch's native CSR layout: we store
    *one* 2-D ``torch.sparse_csr_tensor`` holding the INT8 values, plus the
    usual quantisation scale & zero-point.
    """

    # NEW — provide a Python attribute so AffineQuantizedTensor
    # can fetch the layout without touching the C++ side.
    @property
    def layout(self):
        """
        Return a *torch.layout* enum so that
        `AffineQuantizedTensor.__new__` can forward it to
        `torch.Tensor._make_wrapper_subclass` without type errors.

        Note: the **custom** sparsity marker we care about
        (`CSRLayout()` instance) is still stored in `self._layout`.
        """
        return torch.sparse_csr

    @classmethod
    def __torch_dispatch__(cls, func, types, args, kwargs):
        kwargs = kwargs or {}

        # ➊ Intercept nn.Linear lowering (aten.linear)
        if func is aten.linear.default and len(args) == 3:
            inp, W, bias = args  # (input, weight, bias)
            # If W is an AffineQuantizedTensor whose impl is CSR_AQTTensorImpl → run our kernel
            if hasattr(W, "tensor_impl") and isinstance(W.tensor_impl, cls):
                return _linear_int8_act_int8_weight_csr_sparse_impl(inp, W, bias)

        # (optional but recommended) also cover addmm/mm fallbacks
        if func is aten.addmm.default and len(args) >= 3:
            bias, inp, Wt = args[:3]  # PyTorch often does bias + inp @ W.t()
            if hasattr(Wt, "tensor_impl") and isinstance(Wt.tensor_impl, cls):
                return _linear_int8_act_int8_weight_csr_sparse_impl(inp, Wt, bias)

        if func is aten.mm.default and len(args) == 2:
            lhs, rhs = args
            if hasattr(lhs, "tensor_impl") and isinstance(lhs.tensor_impl, cls):
                return _linear_int8_act_int8_weight_csr_sparse_impl(rhs, lhs, None)
            if hasattr(rhs, "tensor_impl") and isinstance(rhs.tensor_impl, cls):
                return _linear_int8_act_int8_weight_csr_sparse_impl(lhs, rhs, None)

        # keep your existing detach fast-path
        if func is aten.detach.default:
            return return_and_correct_aliasing(
                func, args, kwargs, args[0]._apply_fn_to_data(torch.detach)
            )

        raise NotImplementedError(
            f"CSR_AQTTensorImpl dispatch: attempting to run {func}, not supported"
        )

    @classmethod
    def _quantized_linear_op(input_tensor, weight_tensor, bias):
        # # 1. dynamic-quantise the activation
        qt = weight_tensor.input_quant_func(input_tensor, **weight_tensor.quant_kwargs)
        aqt_weight = weight_tensor.original_weight_tensor
        return _linear_int8_act_int8_weight_csr_sparse_impl(
            qt,  # plain-layout INT-8 activation
            aqt_weight,  # CSR-packed weight
            bias,
        )

    # ------------------------------ helpers ---------------------------

    def get_plain(self):
        """Return dense INT8 matrix alongside scale/zero_point.

        This is a slow path used by debugging utilities (e.g. ``to_dense()``).
        We materialise the CSR tensor to dense form.
        """
        int_data_expanded = self.int_data.to_dense()
        return int_data_expanded, self.scale, self.zero_point

    # --------------------------- construction -------------------------

    @classmethod
    def from_plain(
        cls,
        int_data: torch.Tensor,  # expected *dense* INT8 matrix (out, in)
        scale: torch.Tensor,
        zero_point: Optional[torch.Tensor],
        _layout: Layout,
    ):
        """Pack a *dense* INT8 matrix ``int_data`` into CSR layout."""
        assert isinstance(_layout, CSRLayout), "layout must be CSRLayout"

        # Use PyTorch util to convert to CSR; keep on same device / dtype.
        csr_tensor = int_data.to_sparse_csr()
        return cls(csr_tensor, scale, zero_point, _layout)
