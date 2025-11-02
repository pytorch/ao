# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD 3-Clause license found in the
# LICENSE file in the root directory of this source tree.

from typing import Optional

import torch

from torchao.quantization.quant_primitives import (
    _choose_qparams_affine_floatx,
)
from torchao.utils import TorchAOBaseTensor

__all__ = ["Float8SemiSparseTensor"]

aten = torch.ops.aten


class Float8SemiSparseTensor(TorchAOBaseTensor):
    """
    float8 quantized tensor with 2:4 semi-structured sparsity layout

    Tensor Attributes:
        qdata: float8 data in dense format
        qdata_compressed: SparseSemiStructuredTensor (compressed format for matmul)
        scale: scale factors for dequantization

    Non-Tensor Attributes:
        block_size: block size for quantization granularity
        original_shape: original uncompressed shape
        float8_dtype: float8 dtype variant
    """

    tensor_data_names = ["qdata", "qdata_compressed", "scale"]
    tensor_attribute_names = ["block_size", "original_shape"]
    optional_tensor_attribute_names = ["dtype"]

    def __new__(
        cls: type,
        qdata: torch.Tensor,
        qdata_compressed: torch.Tensor,
        scale: torch.Tensor,
        block_size: list[int],
        original_shape: tuple[int, ...],
        dtype=None,
        float8_dtype: torch.dtype = torch.float8_e4m3fn,
    ):
        kwargs = {
            "device": qdata.device,
            "dtype": dtype or scale.dtype,
            "requires_grad": False,
        }
        return torch.Tensor._make_wrapper_subclass(cls, original_shape, **kwargs)

    def __init__(
        self,
        qdata: torch.Tensor,
        qdata_compressed: torch.Tensor,
        scale: torch.Tensor,
        block_size: list[int],
        original_shape: tuple[int, ...],
        dtype=None,
        float8_dtype: torch.dtype = torch.float8_e4m3fn,
    ):
        super().__init__()
        self.qdata = qdata  # dense fp8 for dequantization
        self.qdata_compressed = qdata_compressed  # compressed for matmul
        self.scale = scale
        self.block_size = block_size
        self.original_shape = original_shape
        self.float8_dtype = float8_dtype

    def __repr__(self):
        return (
            f"{self.__class__.__name__}("
            f"qdata_shape={self.qdata.shape}, {self.scale=}, "
            f"{self.block_size=}, {self.shape=}, {self.device=}, {self.dtype=})"
        )

    @property
    def qdata_fp8(self):
        """For test compatibility"""
        return self.qdata

    @classmethod
    def from_hp(
        cls,
        w: torch.Tensor,
        block_size: list[int],
        float8_dtype: torch.dtype = torch.float8_e4m3fn,
    ):
        if w.dim() != 2 or len(block_size) != 2:
            raise ValueError("Expected 2D tensor and block_size length 2")

        if not w.is_cuda:
            raise ValueError("Semi-sparse layout requires CUDA tensors")

        # Verify dimensions are compatible with 2:4 compression
        rows, cols = w.shape
        if rows % 32 != 0 or cols % 32 != 0:
            raise ValueError(
                "Tensor dimensions must be multiples of 32 for CUDA sparse compression"
            )

        # Validate block_size
        if not all(bs > 0 for bs in block_size):
            raise ValueError(f"block_size must be positive, got {block_size}")

        if rows % block_size[0] != 0 or cols % block_size[1] != 0:
            raise ValueError(
                f"Dimensions {w.shape} must be divisible by block_size {block_size}"
            )

        # Apply 2:4 sparsity pruning
        with torch.no_grad():
            w_sparse = w.clone()

        pruning_inds = w_sparse.abs().view(-1, 4).argsort(dim=1)[:, :2]
        w_sparse.view(-1, 4).scatter_(1, pruning_inds, value=0)

        # Quantize to float8
        if float8_dtype == torch.float8_e4m3fn:
            ebits, mbits = 4, 3
            max_val = 448.0
        elif float8_dtype == torch.float8_e5m2:
            ebits, mbits = 5, 2
            max_val = 57344.0
        else:
            raise ValueError(f"Unsupported float8 dtype: {float8_dtype}")

        scale = _choose_qparams_affine_floatx(w_sparse, ebits=ebits, mbits=mbits)

        if scale.isnan().any():
            raise ValueError("Scale contains NaN")
        if not (scale > 0).all():
            raise ValueError(f"Scale contains non-positive values: min={scale.min()}")

        scale_expanded = scale.unsqueeze(1)
        scaled_data = w_sparse / scale_expanded
        scaled_data = scaled_data.clamp(-max_val, max_val)
        fp8_data = scaled_data.to(float8_dtype).contiguous()

        if fp8_data.isnan().any():
            raise ValueError("fp8_data contains NaN after quantization")

        # Store fp8 data in both dense and compressed formats
        fp8_data_fp16 = fp8_data.to(torch.float16)
        from torch.sparse import to_sparse_semi_structured

        fp8_compressed = to_sparse_semi_structured(fp8_data_fp16)

        return cls(
            fp8_data,  # dense for dequantization
            fp8_compressed,  # compressed for matmul
            scale,
            block_size,
            original_shape=w.shape,
            dtype=w.dtype,
            float8_dtype=float8_dtype,
        )

    def dequantize(self, output_dtype: Optional[torch.dtype] = None) -> torch.Tensor:
        if output_dtype is None:
            output_dtype = self.dtype

        # Use dense fp8 data
        qdata_fp = self.qdata.to(output_dtype)
        scale_expanded = self.scale.view(-1, 1).to(output_dtype)
        return qdata_fp * scale_expanded


implements = Float8SemiSparseTensor.implements
implements_torch_function = Float8SemiSparseTensor.implements_torch_function


@implements(aten.linear.default)
@implements_torch_function(torch.nn.functional.linear)
def _(func, types, args, kwargs):
    activation_tensor, weight_tensor, bias = (
        args[0],
        args[1],
        args[2] if len(args) > 2 else None,
    )

    assert isinstance(weight_tensor, Float8SemiSparseTensor)
    if not isinstance(activation_tensor, Float8SemiSparseTensor):
        raise TypeError(
            "Float8SemiSparseTensor requires pre-quantized activations (static quantization only). "
            "Activation must be Float8SemiSparseTensor."
        )
    assert activation_tensor.shape[-1] == weight_tensor.original_shape[1], (
        f"Shape mismatch: {activation_tensor.shape} @ {weight_tensor.original_shape}"
    )

    # Use compressed data for matmul
    x_vals_dense = activation_tensor.qdata_compressed.to_dense()
    x_vals_fp8 = x_vals_dense.view(torch.float8_e4m3fn)
    x_scales = activation_tensor.scale

    w_vals_dense = weight_tensor.qdata_compressed.to_dense()
    w_vals_fp8 = w_vals_dense.view(torch.float8_e4m3fn)
    w_scales = weight_tensor.scale

    # Prepare activation for sparse matmul
    tmp = x_vals_fp8
    if tmp.dim() > 2:
        tmp = tmp.view(-1, tmp.shape[-1])
    row = tmp.shape[0]

    from torch.sparse import SparseSemiStructuredTensorCUSPARSELT

    tmp_padded = SparseSemiStructuredTensorCUSPARSELT._pad_dense_input(tmp)

    # Convert weight fp8 to fp16 with scale for matmul
    w_scaled = w_vals_fp8.to(torch.float16) * w_scales.unsqueeze(1)
    w_sparse_scaled = torch.sparse.to_sparse_semi_structured(w_scaled)

    # Matmul with sparse weight
    y = torch.matmul(
        tmp_padded.to(torch.bfloat16), w_sparse_scaled.t().to(torch.bfloat16)
    )
    y = y[:row, :]

    # Apply activation scale
    y = y * x_scales.unsqueeze(1)

    # Reshape to original activation shape
    if x_vals_fp8.dim() > 2:
        y = y.view(*x_vals_fp8.shape[:-1], y.shape[-1])

    output_dtype = activation_tensor.dtype
    y = y.to(output_dtype).contiguous()

    if bias is not None:
        y += bias
    return y


@implements(aten.slice.Tensor)
def _(func, types, args, kwargs):
    """Slice operation - not supported for compressed sparse format"""
    # TODO: Build this tensor utility operation
    raise NotImplementedError(
        "Slicing not supported for Float8SemiSparseTensor. "
        "Decompress first using dequantize() if needed."
    )


@implements(aten.select.int)
def _(func, types, args, kwargs):
    """Select operation - not supported for compressed sparse format"""
    # TODO: Build this tensor utility operation
    raise NotImplementedError(
        "Select not supported for Float8SemiSparseTensor. "
        "Decompress first using dequantize() if needed."
    )


Float8SemiSparseTensor.__module__ = "torchao.quantization"
torch.serialization.add_safe_globals([Float8SemiSparseTensor])
