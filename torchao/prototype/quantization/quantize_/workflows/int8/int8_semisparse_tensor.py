# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD 3-Clause license found in the
# LICENSE file in the root directory of this source tree.

from typing import Optional

import torch

from torchao.quantization.quant_primitives import (
    MappingType,
    _maybe_expand_scale_to_tensor_shape,
    choose_qparams_affine,
    quantize_affine,
)
from torchao.utils import TorchAOBaseTensor

__all__ = ["Int8SemiSparseTensor"]

aten = torch.ops.aten


class Int8SemiSparseTensor(TorchAOBaseTensor):
    """
    int8 quantized tensor with 2:4 semi-structured sparsity layout

    Tensor Attributes:
        qdata: SparseSemiStructuredTensor (compressed format, fp16)
        scale: scale factors for dequantization

    Non-Tensor Attributes:
        block_size: block size for quantization granularity
        original_shape: original uncompressed shape
    """

    tensor_data_names = ["qdata", "scale"]
    tensor_attribute_names = ["block_size", "original_shape"]
    optional_tensor_attribute_names = ["dtype"]

    def __new__(
        cls: type,
        qdata: torch.Tensor,
        scale: torch.Tensor,
        block_size: list[int],
        original_shape: tuple[int, ...],
        dtype=None,
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
        scale: torch.Tensor,
        block_size: list[int],
        original_shape: tuple[int, ...],
        dtype=None,
    ):
        super().__init__()
        self.qdata = qdata
        self.scale = scale
        self.block_size = block_size
        self.original_shape = original_shape

    def __repr__(self):
        return (
            f"{self.__class__.__name__}("
            f"qdata_shape={self.qdata.shape}, {self.scale=}, "
            f"{self.block_size=}, {self.shape=}, {self.device=}, {self.dtype=})"
        )

    @property
    def qdata_int8(self):
        return self.qdata.to_dense().to(torch.int8)

    @classmethod
    def from_hp(
        cls,
        w: torch.Tensor,
        block_size: list[int],
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

        # Apply 2:4 sparsity pruning (row-wise for weight matrix)
        with torch.no_grad():
            w_sparse = w.clone()

        pruning_inds = w_sparse.abs().view(-1, 4).argsort(dim=1)[:, :2]
        w_sparse.view(-1, 4).scatter_(1, pruning_inds, value=0)

        # Quantize the sparse weight
        scale, zero_point = choose_qparams_affine(
            input=w_sparse,
            mapping_type=MappingType.SYMMETRIC,
            block_size=block_size,
            target_dtype=torch.int8,
            quant_min=-128,
            quant_max=127,
            scale_dtype=w.dtype,
            zero_point_dtype=torch.int8,
        )

        int_data = quantize_affine(
            w_sparse,
            block_size=block_size,
            scale=scale,
            zero_point=zero_point,
            output_dtype=torch.int8,
        ).contiguous()

        int_data_fp16 = int_data.to(torch.float16)
        from torch.sparse import to_sparse_semi_structured

        int_data_compressed = to_sparse_semi_structured(int_data_fp16)

        return cls(
            int_data_compressed,
            scale,
            block_size,
            original_shape=w.shape,
            dtype=w.dtype,
        )

    def dequantize(self, output_dtype: Optional[torch.dtype] = None) -> torch.Tensor:
        if output_dtype is None:
            output_dtype = self.dtype

        # Decompress and convert to int8
        qdata_dense = self.qdata.to_dense().to(torch.int8)
        qdata_fp = qdata_dense.to(output_dtype)

        scale_expanded = _maybe_expand_scale_to_tensor_shape(
            self.scale, self.original_shape
        )
        return qdata_fp * scale_expanded.to(output_dtype)


implements = Int8SemiSparseTensor.implements
implements_torch_function = Int8SemiSparseTensor.implements_torch_function


@implements(aten.linear.default)
@implements_torch_function(torch.nn.functional.linear)
def _(func, types, args, kwargs):
    activation_tensor, weight_tensor, bias = (
        args[0],
        args[1],
        args[2] if len(args) > 2 else None,
    )

    assert isinstance(weight_tensor, Int8SemiSparseTensor)
    assert isinstance(activation_tensor, Int8SemiSparseTensor), (
        "Int8SemiSparseTensor requires pre-quantized activations (static quantization only)"
    )
    assert activation_tensor.shape[-1] == weight_tensor.original_shape[1], (
        f"Shape mismatch: {activation_tensor.shape} @ {weight_tensor.original_shape}"
    )

    # Extract quantized data
    x_vals_dense = activation_tensor.qdata.to_dense().to(torch.int8)
    x_scales = activation_tensor.scale
    w_vals_dense = weight_tensor.qdata.to_dense().to(torch.int8)
    w_scales = weight_tensor.scale

    # Prepare activation for sparse matmul
    tmp = x_vals_dense.view(-1, x_vals_dense.shape[-1])
    row, col = tmp.shape

    from torch.sparse import SparseSemiStructuredTensorCUSPARSELT

    tmp_padded = SparseSemiStructuredTensorCUSPARSELT._pad_dense_input(tmp)

    # Perform sparse matmul with int8, output as bfloat16 with weight scale
    w_scaled = w_vals_dense.to(torch.float16) * w_scales.view(-1, 1)
    w_sparse_scaled = torch.sparse.to_sparse_semi_structured(w_scaled)
    y_bf16 = torch.matmul(
        tmp_padded.to(torch.bfloat16), w_sparse_scaled.t().to(torch.bfloat16)
    )
    y_bf16 = y_bf16[:row, :]

    # Apply activation scale
    y = (y_bf16 * x_scales.view(-1, 1)).view(*x_vals_dense.shape[:-1], y_bf16.shape[-1])

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
        "Slicing not supported for Int8SemiSparseTensor. "
        "Decompress first using dequantize() if needed."
    )


@implements(aten.select.int)
def _(func, types, args, kwargs):
    """Select operation - not supported for compressed sparse format"""
    # TODO: Build this tensor utility operation
    raise NotImplementedError(
        "Select not supported for Int8SemiSparseTensor. "
        "Decompress first using dequantize() if needed."
    )


Int8SemiSparseTensor.__module__ = "torchao.quantization"
torch.serialization.add_safe_globals([Int8SemiSparseTensor])
