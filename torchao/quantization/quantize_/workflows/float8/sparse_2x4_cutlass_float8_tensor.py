# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD 3-Clause license found in the
# LICENSE file in the root directory of this source tree.


from typing import List, Optional

import torch

from torchao.float8.inference import (
    FP8Granularity,
)
from torchao.ops import (
    rowwise_scaled_linear_sparse_cutlass_f8f8,
    to_sparse_semi_structured_cutlass_sm9x_f8,
)
from torchao.quantization.granularity import PerRow
from torchao.quantization.quant_primitives import (
    _choose_scale_float8,
    _quantize_affine_float8,
)
from torchao.quantization.quantize_.common import (
    _choose_quant_func_and_quantize_tensor,
)
from torchao.quantization.utils import get_block_size
from torchao.utils import (
    TorchAOBaseTensor,
    is_sm_at_least_90,
)

__all__ = [
    "Sparse2x4CUTLASSFloat8Tensor",
]

aten = torch.ops.aten


from .float8_tensor import QuantizeTensorToFloat8Kwargs


class Sparse2x4CUTLASSFloat8Tensor(TorchAOBaseTensor):
    """
    Float8 Quantized + 2:4 sparse (weight) Tensor using CUTLASS kernels, with float8 dynamic quantization for activation.

    Tensor Attributes:
        qdata: float8 raw data
        sparse_metadata: metadata for 2:4 sparse tensor
        scale: the scale for float8 Tensor

    Non-Tensor Attributes:
        block_size (List[int]): the block size for float8 quantization, meaning the shape of the elements
        sharing the same set of quantization parameters (scale), have the same rank as qdata or
        is an empty list (representing per tensor quantization)
        act_quant_kwargs (QuantizeTensorToFloat8Kwargs): the kwargs for Sparse2x4Float8Tensor.from_hp
        packing_format (Float8PackingFormat): the preference for quantize, mm etc. kernel to use,
        by default, this will be chosen for user based on hardware, library availabilities etc.
        dtype: Original Tensor dtype
    """

    tensor_data_names = ["qdata", "sparse_metadata", "scale"]
    tensor_attribute_names = []
    optional_tensor_attribute_names = [
        "block_size",
        "act_quant_kwargs",
        "dtype",
    ]

    def __new__(
        cls,
        qdata: torch.Tensor,
        sparse_metadata: torch.Tensor,
        scale: torch.Tensor,
        block_size: Optional[List[int]] = None,
        act_quant_kwargs: Optional[QuantizeTensorToFloat8Kwargs] = None,
        dtype: Optional[torch.dtype] = None,
    ):
        shape = qdata.shape[0], 2 * qdata.shape[1]

        kwargs = {}
        kwargs["device"] = qdata.device
        kwargs["dtype"] = dtype
        kwargs["requires_grad"] = False
        return torch.Tensor._make_wrapper_subclass(cls, shape, **kwargs)  # type: ignore[attr-defined]

    def __init__(
        self,
        qdata: torch.Tensor,
        sparse_metadata: torch.Tensor,
        scale: torch.Tensor,
        block_size: Optional[List[int]] = None,
        act_quant_kwargs: Optional[QuantizeTensorToFloat8Kwargs] = None,
        dtype: Optional[torch.dtype] = None,
    ):
        super().__init__()
        self.qdata = qdata
        self.sparse_metadata = sparse_metadata
        self.scale = scale
        self.block_size = block_size
        self.act_quant_kwargs = act_quant_kwargs

    def __repr__(self):
        return (
            f"{self.__class__.__name__}({self.act_quant_kwargs=}, {self.qdata=}, {self.sparse_metadata=}, {self.scale=}, "
            f"{self.block_size=}, "
            f"{self.shape=}, {self.device=}, {self.dtype=})"
        )

    def _quantization_type(self):
        return f"{self.act_quant_kwargs=}, {self.block_size=}, {self.scale.shape=}"

    def dequantize(self, output_dtype: Optional[torch.dtype] = None) -> torch.Tensor:
        # No support in CUTLASS to convert back to dense from sparse
        # semi-structured format, so multiplying with identity matrix,
        # and using identity scale factors, for the conversion.
        cols = self.shape[1]
        input = torch.eye(cols, dtype=self.qdata.dtype, device=self.qdata.device)
        input_scale = torch.ones(
            (cols,), dtype=self.scale.dtype, device=self.qdata.device
        )

        out_dtype = torch.bfloat16
        dense = (
            rowwise_scaled_linear_sparse_cutlass_f8f8(
                input,
                input_scale,
                self.qdata,
                self.sparse_metadata,
                self.scale,
                out_dtype=out_dtype,
            )
            .to(output_dtype)
            .t()
            .contiguous()
        )
        return dense

    @classmethod
    def from_hp(
        cls,
        hp_tensor: torch.Tensor,
        float8_dtype: torch.dtype = torch.float8_e4m3fn,
        granularity: FP8Granularity = PerRow(),
        hp_value_lb: Optional[float] = None,
        hp_value_ub: Optional[float] = None,
        act_quant_kwargs: Optional[QuantizeTensorToFloat8Kwargs] = None,
    ):
        block_size = get_block_size(hp_tensor.shape, granularity)
        block_size = list(block_size)
        scale = _choose_scale_float8(
            hp_tensor,
            float8_dtype=float8_dtype,
            block_size=block_size,
            hp_value_lb=hp_value_lb,
            hp_value_ub=hp_value_ub,
        )
        data = _quantize_affine_float8(hp_tensor, scale, float8_dtype)
        hp_dtype = hp_tensor.dtype

        assert is_sm_at_least_90(), (
            "CUTLASS sparse kernel requires hardware >= SM 9.0 (>= H100)"
        )
        assert isinstance(granularity, PerRow), (
            "CUTLASS sparse kernel only supports per-row quantization"
        )
        # CUTLASS path only supports quantizing along the last dim
        assert granularity.dim in (-1, len(hp_tensor.shape) - 1), (
            "CUTLASS sparse kernel only supports quantizing along the last dimension"
        )
        assert float8_dtype == torch.float8_e4m3fn, (
            "CUTLASS sparse kernel only supports float8_e4m3fn dtype"
        )
        assert hp_value_lb is None, "CUTLASS sparse kernel does not support hp_value_lb"

        # Use CUTLASS rowwise fp8 + 2:4 sparse mm kernel
        qdata, sparse_metadata = to_sparse_semi_structured_cutlass_sm9x_f8(data)

        return Sparse2x4CUTLASSFloat8Tensor(
            qdata,
            sparse_metadata,
            scale,
            block_size=block_size,
            act_quant_kwargs=act_quant_kwargs,
            dtype=hp_dtype,
        )


implements = Sparse2x4CUTLASSFloat8Tensor.implements
implements_torch_function = Sparse2x4CUTLASSFloat8Tensor.implements_torch_function


@implements(aten.linear.default)
@implements_torch_function(torch.nn.functional.linear)
def _(func, types, args, kwargs):
    input_tensor = kwargs.get("input", args[0] if len(args) > 0 else None)
    weight_tensor = kwargs.get("weight", args[1] if len(args) > 1 else None)
    bias = kwargs.get("bias", args[2] if len(args) > 2 else None)

    assert input_tensor is not None, "input tensor must not be None"
    assert weight_tensor is not None, "weight tensor must not be None"

    act_quant_kwargs = weight_tensor.act_quant_kwargs
    # quantize activation, if `act_quant_kwargs` is specified
    if act_quant_kwargs is not None:
        assert not isinstance(input_tensor, TorchAOBaseTensor), (
            "input tensor was already quantized"
        )
        input_tensor = _choose_quant_func_and_quantize_tensor(
            input_tensor, act_quant_kwargs
        )
    input = input_tensor.qdata
    input_scale = input_tensor.scale.squeeze(1)
    weight = weight_tensor.qdata
    weight_meta = weight_tensor.sparse_metadata
    weight_scale = weight_tensor.scale.squeeze(1)
    out_dtype = input_tensor.dtype

    out = rowwise_scaled_linear_sparse_cutlass_f8f8(
        input, input_scale, weight, weight_meta, weight_scale, bias, out_dtype
    )
    return out


@implements(aten.to.dtype_layout)
def _(func, types, args, kwargs):
    return (
        args[0]
        .dequantize()
        .to(
            *args[1:],
            dtype=kwargs.get("dtype", args[0].dtype),
            device=kwargs.get("device", args[0].device),
        )
    )


# implement to.dtype for cases where dtype specified in args[1]
@implements(aten.to.dtype)
def _(func, types, args, kwargs):
    dtype = kwargs.get("dtype", args[1] if len(args) > 1 else None)
    assert dtype is not None, "dtype must not be None"

    return (
        args[0]
        .dequantize()
        .to(
            dtype=dtype,
        )
    )


@implements(aten.abs.default)
def _(func, types, args, kwargs):

    return torch.abs(args[0].dequantize())


# Allow a model with Float8Tensor weights to be loaded with `weights_only=True`
torch.serialization.add_safe_globals([Sparse2x4CUTLASSFloat8Tensor])
