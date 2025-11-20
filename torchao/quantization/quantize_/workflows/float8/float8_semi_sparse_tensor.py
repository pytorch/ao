# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD 3-Clause license found in the
# LICENSE file in the root directory of this source tree.


from typing import List, Optional

import torch
from torch.utils._python_dispatch import return_and_correct_aliasing

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
    KernelPreference,
    _choose_quant_func_and_quantize_tensor,
)
from torchao.quantization.utils import get_block_size
from torchao.utils import (
    TorchAOBaseTensor,
    is_sm_at_least_90,
)

__all__ = [
    "Float8SemiSparseTensor",
]

aten = torch.ops.aten


from .float8_tensor import QuantizeTensorToFloat8Kwargs


class Float8SemiSparseTensor(TorchAOBaseTensor):
    """
    Float8 Quantized + 2:4 sparse (weight) Tensor, with float8 dynamic quantization for activation.

    Tensor Attributes:
        sparse_quantized_data: float8 raw data
        sparse_metadata: metadata for 2:4 sparse tensor
        scale: the scale for float8 Tensor

    Non-Tensor Attributes:
        block_size (List[int]): the block size for float8 quantization, meaning the shape of the elements
        sharing the same set of quantization parameters (scale), have the same rank as sparse_quantized_data or
        is an empty list (representing per tensor quantization)
        act_quant_kwargs (QuantizeTensorToFloat8Kwargs): the kwargs for Float8SemiSparseTensor.from_hp
        kernel_preference (KernelPreference): the preference for quantize, mm etc. kernel to use,
        by default, this will be chosen for user based on hardware, library availabilities etc.
        dtype: Original Tensor dtype
    """

    tensor_data_names = ["sparse_quantized_data", "sparse_metadata", "scale"]
    tensor_attribute_names = []
    optional_tensor_attribute_names = [
        "block_size",
        "act_quant_kwargs",
        "kernel_preference",
        "dtype",
    ]

    def __new__(
        cls,
        sparse_quantized_data: torch.Tensor,
        sparse_metadata: torch.Tensor,
        scale: torch.Tensor,
        block_size: Optional[List[int]] = None,
        act_quant_kwargs: Optional[QuantizeTensorToFloat8Kwargs] = None,
        kernel_preference: KernelPreference = KernelPreference.AUTO,
        dtype: Optional[torch.dtype] = None,
    ):
        shape = sparse_quantized_data.shape[0], 2 * sparse_quantized_data.shape[1]
        kwargs = {}
        kwargs["device"] = sparse_quantized_data.device
        kwargs["dtype"] = dtype
        kwargs["requires_grad"] = False
        return torch.Tensor._make_wrapper_subclass(cls, shape, **kwargs)  # type: ignore[attr-defined]

    def __init__(
        self,
        sparse_quantized_data: torch.Tensor,
        sparse_metadata: torch.Tensor,
        scale: torch.Tensor,
        block_size: Optional[List[int]] = None,
        act_quant_kwargs: Optional[QuantizeTensorToFloat8Kwargs] = None,
        kernel_preference: KernelPreference = KernelPreference.AUTO,
        dtype: Optional[torch.dtype] = None,
    ):
        super().__init__()
        self.sparse_quantized_data = sparse_quantized_data
        self.sparse_metadata = sparse_metadata
        self.scale = scale
        self.block_size = block_size
        self.act_quant_kwargs = act_quant_kwargs
        self.kernel_preference = kernel_preference

    def __repr__(self):
        return (
            f"{self.__class__.__name__}({self.act_quant_kwargs=}, {self.sparse_quantized_data=}, {self.sparse_metadata=}, {self.scale=}, "
            f"{self.block_size=}, {self.kernel_preference=} "
            f"{self.shape=}, {self.device=}, {self.dtype=})"
        )

    def _quantization_type(self):
        return f"{self.act_quant_kwargs=}, {self.block_size=}, {self.scale.shape=}, {self.kernel_preference=}"

    def dequantize(self, output_dtype: Optional[torch.dtype] = None) -> torch.Tensor:
        # No support in CUTLASS to convert back to dense from sparse
        # semi-structured format, so multiplying with identity matrix,
        # and using identity scale factors, for the conversion.
        cols = self.shape[1]
        plain_input = torch.eye(cols, device=self.sparse_quantized_data.device)
        input = plain_input.to(dtype=self.sparse_quantized_data.dtype)
        plain_input_scale = torch.ones(
            (cols,), device=self.sparse_quantized_data.device
        )
        input_scale = plain_input_scale.to(dtype=self.scale.dtype)

        out_dtype = torch.bfloat16
        dense = (
            rowwise_scaled_linear_sparse_cutlass_f8f8(
                input,
                input_scale,
                self.sparse_quantized_data,
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
        kernel_preference: KernelPreference = KernelPreference.AUTO,
        act_quant_kwargs: Optional[QuantizeTensorToFloat8Kwargs] = None,
    ):
        block_size = get_block_size(hp_tensor.shape, granularity)
        block_size = list(block_size)

        kernel_choice = None
        if (
            kernel_preference == KernelPreference.AUTO
            and is_sm_at_least_90()
            and isinstance(granularity, PerRow)
            # fbgemm path only supports quantizing along the last dim
            and granularity.dim in (-1, len(hp_tensor.shape) - 1)
            and float8_dtype == torch.float8_e4m3fn
            and hp_value_lb is None
        ):
            # if kernel_preference is AUTO and per row quantization and we are on sm90
            # we'll use CUTLASS rowwise fp8 + 2:4 sparse mm kernel
            kernel_choice = "sparse_cutlass"
        elif kernel_preference == KernelPreference.SPARSE_CUTLASS:
            # if user explicitly chose FBGEMM kernel preference, we'll also use fbgemm kernel
            assert is_sm_at_least_90(), (
                "Specified sparse_cutlass kernel and hardware is not >= SM 9.0 (>= H100)"
            )
            kernel_choice = "sparse_cutlass"

        scale = _choose_scale_float8(
            hp_tensor,
            float8_dtype=float8_dtype,
            block_size=block_size,
            hp_value_lb=hp_value_lb,
            hp_value_ub=hp_value_ub,
        )
        data = _quantize_affine_float8(hp_tensor, scale, float8_dtype)
        if kernel_choice == "sparse_cutlass":
            sparse_quantized_data, sparse_metadata = (
                to_sparse_semi_structured_cutlass_sm9x_f8(data)
            )
        else:
            raise ValueError("Only sparse_cutlass kernel is supported currently!")
        hp_dtype = hp_tensor.dtype

        return Float8SemiSparseTensor(
            sparse_quantized_data,
            sparse_metadata,
            scale,
            block_size=block_size,
            act_quant_kwargs=act_quant_kwargs,
            kernel_preference=kernel_preference,
            dtype=hp_dtype,
        )


implements = Float8SemiSparseTensor.implements
implements_torch_function = Float8SemiSparseTensor.implements_torch_function


@implements(aten.linear.default)
@implements_torch_function(torch.nn.functional.linear)
def _(func, types, args, kwargs):
    input_tensor, weight_tensor, bias = (
        args[0],
        args[1],
        args[2] if len(args) > 2 else None,
    )
    from torchao.ops import rowwise_scaled_linear_sparse_cutlass_f8f8

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
    weight = weight_tensor.sparse_quantized_data
    weight_meta = weight_tensor.sparse_metadata
    weight_scale = weight_tensor.scale.squeeze(1)
    out_dtype = input_tensor.dtype

    out = rowwise_scaled_linear_sparse_cutlass_f8f8(
        input, input_scale, weight, weight_meta, weight_scale, bias, out_dtype
    )
    return out


@implements(aten.clone.default)
def _(func, types, args, kwargs):
    return return_and_correct_aliasing(
        func, args, kwargs, args[0]._apply_fn_to_data(torch.clone)
    )


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


# Allow a model with Float8Tensor weights to be loaded with `weights_only=True`
torch.serialization.add_safe_globals(
    [Float8SemiSparseTensor, QuantizeTensorToFloat8Kwargs]
)
