# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD 3-Clause license found in the
# LICENSE file in the root directory of this source tree.
"""
Compressed-Sparse-Row (CSR) layout support for *affine quantized tensors* in
TorchAO.

This file mirrors the structure of ``SemiSparseLayout`` but targets a *general
unstructured* sparsity pattern encoded with the CSR format.  It enables INT8
weights packed as CSR to participate in the dynamic-activation / INT8-weight
workflow on CPU back-ends (FBGEMM / oneDNN) and provides a fall-back path that
relies on PyTorch's native ``torch.sparse.mm`` when no vendor kernel is
available.

Key pieces:

* ``CSRLayout`` – a stateless ``Layout`` subclass signalling that a tensor is
  stored in CSR form.
* ``CSR_AQTTensorImpl`` – a ``TensorImpl`` that actually holds the compressed
  data and quantization parameters (scale / zero-point).
* Helper ``_linear_int8_act_int8_weight_csr_sparse_check``
  and ``_linear_int8_act_int8_weight_csr_sparse_impl`` which will be invoked by
  TorchDispatch guards when ``aten.linear`` sees csr-packed weights.  These are
  minimal and will be expanded by later PRs to call vendor kernels.

NOTE: This file keeps CUDA-only pieces out – the CPU path is the first target.
Vendor kernels are *optional*; the reference implementation lowers to
``torch.sparse.mm`` so functional correctness is guaranteed even on exotic
architectures.
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
    input_tensor: torch.Tensor, weight_tensor: torch.Tensor, bias: Optional[torch.Tensor]
) -> bool:
    """FX-based guard – true if we can execute csr-optimised path."""
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
    w_scale = weight_tensor.tensor_impl.scale   # shape: (1,) or (row,)

    # Reshape activations to 2-D (batch*seqlen, in_features)
    x2d = x_vals_int8.reshape(-1, x_vals_int8.shape[-1]).to(torch.int32)

    # Sparse matmul (int8*int8 -> int32).  PyTorch up-casts INT8 CSR to INT32
    # during mm; if vendor kernel is registered this will be replaced at runtime.
    y_int32 = torch.sparse.mm(w_csr.to(torch.int32), x2d.t()).t()

    # Dequantise: y = (x_scale * w_scale) * y_int32
    y_fp32 = (y_int32.to(torch.float32) * x_scale * w_scale).reshape(
        *x_vals_int8.shape[:-1], y_int32.shape[-1]
    )

    if bias is not None:
        y_fp32 += bias

    # Cast back to activation dtype (usually fp32 or bf16)
    return y_fp32.to(input_tensor.dtype).contiguous()




@dataclass(frozen=True)
class CSRLayout(Layout):
    """Layout marker for *Compressed Sparse Row* INT8 matrices.

    The layout itself is **stateless**; all structural metadata (crow_indices,
    col_indices) lives inside the associated ``TensorImpl``.
    """

    def pre_process(self, input: torch.Tensor) -> torch.Tensor:
        """Optionally prune *very small* weights to zero.

        Developers may plug in magnitude pruning or learnable sparsity here.
        For now we simply return the input unchanged so users can control the
        sparsity schedule externally.
        """
        return input


@register_layout(CSRLayout)
class CSR_AQTTensorImpl(PlainAQTTensorImpl):
    """TensorImpl for CSR-compressed INT8 weights inside an AffineQuantizedTensor.

    The internal representation follows PyTorch's native CSR layout: we store
    *one* 2-D ``torch.sparse_csr_tensor`` holding the INT8 values, plus the
    usual quantisation scale & zero-point.
    """

    # ------------------------------------------------------------------
    # Override torch dispatch for basic ops we need (detach, to_plain).
    # ------------------------------------------------------------------

    @classmethod
    def __torch_dispatch__(cls, func, types, args, kwargs):
        kwargs = kwargs or {}

        if func is aten.detach.default:
            return return_and_correct_aliasing(
                func, args, kwargs, args[0]._apply_fn_to_data(torch.detach)
            )

        raise NotImplementedError(
            f"CSR_AQTTensorImpl dispatch: attempted to run {func}, not supported."
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
        # Suggestion: ensure row-major order (out_features × in_features).
        csr_tensor = torch._to_csr(int_data) if hasattr(torch, "_to_csr") else (
            torch.sparse_csr_tensor(*torch._convert_indices_from_coo_to_csr(int_data))
        )

        return cls(csr_tensor, scale, zero_point, _layout)
