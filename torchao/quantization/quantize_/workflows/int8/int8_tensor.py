# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD 3-Clause license found in the
# LICENSE file in the root directory of this source tree.

from dataclasses import dataclass
from typing import Optional

import torch
from torch.utils._python_dispatch import return_and_correct_aliasing

from torchao.float8.inference import (
    _slice_scale_for_dimension,
)
from torchao.kernel import int_scaled_matmul
from torchao.quantization.granularity import PerRow
from torchao.quantization.quant_primitives import (
    MappingType,
    _maybe_expand_scale_to_tensor_shape,
    choose_qparams_affine,
    quantize_affine,
)
from torchao.quantization.quantize_.common import (
    QuantizeTensorKwargs,
    _choose_quant_func_and_quantize_tensor,
)
from torchao.quantization.utils import get_block_size
from torchao.utils import TorchAOBaseTensor, fill_defaults

__all__ = ["Int8Tensor", "QuantizeTensorToInt8Kwargs"]

aten = torch.ops.aten


@dataclass
class QuantizeTensorToInt8Kwargs(QuantizeTensorKwargs):
    """Tensor kwargs for creating int8 tensor (either activation or weight)

    Args:
        block_size (list[int]): block size for quantization granularity
        granularity: the granularity for the Tensor, currently either PerRow() or PerTensor()
        # TODO: Static quantization support using `static_scale`, `static_zero_point`
    """

    block_size: list[int]
    granularity = PerRow()


class Int8Tensor(TorchAOBaseTensor):
    """
    int8 quantized tensor with plain layout

    Tensor Attributes:
        qdata: (N, K) or (B, N, K) int8 quantized weight data (2D or 3D)
        scale: scale factors for dequantization

    Non-Tensor Attributes:
        block_size: block size for quantization granularity
        act_quant_kwargs: flags for dynamic activation quantization
    """

    tensor_data_names = ["qdata", "scale"]
    tensor_attribute_names = ["block_size"]
    optional_tensor_attribute_names = [
        "act_quant_kwargs",
        "dtype",
    ]

    def __new__(
        cls: type,
        qdata: torch.Tensor,
        scale: torch.Tensor,
        block_size: list[int],
        act_quant_kwargs=None,
        dtype=None,
    ):
        kwargs = {
            "device": qdata.device,
            "dtype": dtype or scale.dtype,
            "requires_grad": False,
        }
        return torch.Tensor._make_wrapper_subclass(cls, qdata.shape, **kwargs)

    def __init__(
        self,
        qdata: torch.Tensor,
        scale: torch.Tensor,
        block_size: list[int],
        act_quant_kwargs=None,
        dtype=None,
    ):
        super().__init__()
        self.qdata = qdata
        self.scale = scale
        self.block_size = block_size
        self.act_quant_kwargs = act_quant_kwargs

    def __repr__(self):
        return (
            f"{self.__class__.__name__}({self.act_quant_kwargs=}, {self.qdata=}, {self.scale=}, "
            f"{self.block_size=}, {self.shape=}, {self.device=}, {self.dtype=})"
        )

    @classmethod
    def from_hp(
        cls,
        w_hp: torch.Tensor,
        granularity=PerRow(),
        act_quant_kwargs: Optional[QuantizeTensorToInt8Kwargs] = None,
    ):
        block_size = list(get_block_size(w_hp.shape, granularity))

        if w_hp.dim() not in [2, 3] or len(block_size) != w_hp.dim():
            raise ValueError("Expected 2D or 3D tensor with same block_size length")

        scale, zero_point = choose_qparams_affine(
            input=w_hp,
            mapping_type=MappingType.SYMMETRIC,
            block_size=block_size,
            target_dtype=torch.int8,
            quant_min=-128,
            quant_max=127,
            scale_dtype=w_hp.dtype,
            zero_point_dtype=torch.int8,
        )

        int_data = quantize_affine(
            w_hp,
            block_size=block_size,
            scale=scale,
            zero_point=zero_point,
            output_dtype=torch.int8,
        )

        return cls(
            int_data,
            scale,
            block_size,
            act_quant_kwargs=act_quant_kwargs,
            dtype=w_hp.dtype,
        )

    def dequantize(self, output_dtype: Optional[torch.dtype] = None) -> torch.Tensor:
        """Dequantize int8 tensor to floating point"""

        if output_dtype is None:
            output_dtype = self.dtype

        qdata_fp = self.qdata.to(output_dtype)
        scale = self.scale
        while scale.ndim < qdata_fp.ndim:
            scale = scale.unsqueeze(-1)

        scale_expanded = _maybe_expand_scale_to_tensor_shape(scale, qdata_fp.shape)
        return qdata_fp * scale_expanded.to(output_dtype)


implements = Int8Tensor.implements
implements_torch_function = Int8Tensor.implements_torch_function


@implements(aten.linear.default)
@implements_torch_function(torch.nn.functional.linear)
def _(func, types, args, kwargs):
    """INT8 quantization: dynamic activation or weight-only"""
    activation_tensor, weight_tensor, bias = (
        args[0],
        args[1],
        args[2] if len(args) > 2 else None,
    )

    assert isinstance(weight_tensor, Int8Tensor), (
        f"Expected weight to be Int8Tensor, got {type(weight_tensor)}"
    )

    if weight_tensor.act_quant_kwargs is not None:
        if not isinstance(activation_tensor, Int8Tensor):
            # Dynamic activation quantization
            act_kwargs = weight_tensor.act_quant_kwargs
            input_ndim = activation_tensor.ndim

            # Ensure block_size matches input tensor dimensions
            if len(act_kwargs.block_size) != input_ndim:
                if input_ndim == 3 and len(act_kwargs.block_size) == 2:
                    block_size_updated = [1] + list(act_kwargs.block_size)
                else:
                    block_size_updated = list(act_kwargs.block_size)[-input_ndim:]
                act_kwargs = QuantizeTensorToInt8Kwargs(block_size=block_size_updated)

            activation_tensor = _choose_quant_func_and_quantize_tensor(
                activation_tensor, act_kwargs
            )

        # 1. do the matrix form of dot(X_i, W_j)
        #
        # 2. rescale the output
        #
        # in cases with large matrices, y_dot_int32 can grow sufficiently
        # large that y_dot_int32 * a FP16 scale is greater than the maximum
        # value of a FP16, (which results in a value of inf even if multiplying
        # by the other scale would bring it within the expected range)

        x_vals_int8 = activation_tensor.qdata
        x_scales = activation_tensor.scale
        w_vals_int8_t = weight_tensor.qdata.contiguous().t()
        w_scales = weight_tensor.scale
        tmp = x_vals_int8.reshape(-1, x_vals_int8.shape[-1])
        x_scales_dtype = x_scales.dtype
        # Cast FP16 scale to float to avoid overflow in int_scaled_matmul
        intermediate_dtype = (
            torch.float if x_scales_dtype == torch.half else x_scales_dtype
        )
        y_dot_scaled = int_scaled_matmul(
            tmp, w_vals_int8_t, x_scales.reshape(-1, 1).to(intermediate_dtype)
        )
        y_dot_scaled = y_dot_scaled.to(x_scales_dtype)

        y = (y_dot_scaled * w_scales).reshape(
            *x_vals_int8.shape[:-1], y_dot_scaled.shape[-1]
        )

        # can downcast only at the very end
        output_dtype = activation_tensor.dtype
        y = y.to(output_dtype)
        if bias is not None:
            y += bias
        return y
    else:
        # FP × INT8 (weight-only)
        w_vals_int8_t = weight_tensor.qdata.t()
        m = torch.mm(
            activation_tensor.reshape(-1, activation_tensor.shape[-1]),
            w_vals_int8_t.to(activation_tensor.dtype),
        )
        y = m * weight_tensor.scale.to(m.dtype)
        y = y.reshape(*activation_tensor.shape[:-1], weight_tensor.qdata.shape[0])
        if bias is not None:
            y += bias
        return y


@implements(aten.slice.Tensor)
def _(func, types, args, kwargs):
    """Slice operation for Int8Tensor"""
    self, dim, start, end, step = fill_defaults(args, 5, [0, None, None, 1])

    if step != 1:
        raise NotImplementedError("Slicing with step > 1 is not supported")

    assert dim in [0, 1, 2], f"Only dim=0,1,2 are supported, got: dim={dim}"
    assert self.qdata.ndim in [2, 3], (
        f"Expected qdata to have dim=2,3 got: dim={self.qdata.ndim}"
    )

    if end >= self.shape[dim]:
        end = self.shape[dim]

    sliced_qdata = aten.slice.Tensor(self.qdata, dim, start, end, step)
    sliced_scale = _slice_scale_for_dimension(
        self.scale, self.qdata.shape, dim, start, end, step
    )

    block_size = list(self.block_size)
    for i in range(len(block_size)):
        block_size[i] = min(block_size[i], sliced_qdata.shape[i])

    return return_and_correct_aliasing(
        func,
        args,
        kwargs,
        Int8Tensor(
            sliced_qdata,
            sliced_scale,
            block_size,
            self.act_quant_kwargs,
            dtype=self.dtype,
        ),
    )


@implements(aten.select.int)
def _(func, types, args, kwargs):
    self, dim, index = args
    assert dim == 0, f"Only dim=0 supported, got {dim}"

    selected_qdata = self.qdata[index]
    selected_scale = _slice_scale_for_dimension(
        self.scale, self.qdata.shape, dim, index, index + 1, step=1
    ).squeeze(0)

    return return_and_correct_aliasing(
        func,
        args,
        kwargs,
        Int8Tensor(
            selected_qdata,
            selected_scale,
            self.block_size[1:],
            self.act_quant_kwargs,
            self.dtype,
        ),
    )


Int8Tensor.__module__ = "torchao.quantization"
torch.serialization.add_safe_globals([Int8Tensor, QuantizeTensorToInt8Kwargs])
