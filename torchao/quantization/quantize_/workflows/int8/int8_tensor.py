# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD 3-Clause license found in the
# LICENSE file in the root directory of this source tree.

from dataclasses import dataclass
from typing import Optional

import torch
from torch.utils._python_dispatch import return_and_correct_aliasing

from torchao.float8.inference import _slice_scale_for_dimension
from torchao.kernel import int_scaled_matmul
from torchao.quantization.granularity import Granularity, PerRow
from torchao.quantization.quant_primitives import (
    MappingType,
    choose_qparams_affine,
    dequantize_affine,
    quantize_affine,
)
from torchao.quantization.quantize_.common import QuantizeTensorKwargs
from torchao.quantization.utils import get_block_size
from torchao.utils import TorchAOBaseTensor, fill_defaults

__all__ = ["Int8Tensor", "QuantizeTensorToInt8Kwargs"]

aten = torch.ops.aten


@dataclass
class QuantizeTensorToInt8Kwargs(QuantizeTensorKwargs):
    """Tensor kwargs for creating int8 tensor (either activation or weight)

    Args:
        granularity: the granularity for the Tensor, currently either PerRow() or PerTensor()
    """

    granularity: Granularity = PerRow()


class Int8Tensor(TorchAOBaseTensor):
    """
    int8 quantized tensor with plain layout

    Tensor Attributes:
        qdata: (N, K) or (B, N, K) int8 quantized weight data (2D or 3D)
        scale: scale factors for dequantization
        # TODO: Static quantization support using `static_scale`

    Non-Tensor Attributes:
        granularity: the granularity for quantization (e.g., PerRow(), PerTensor())
        act_quant_kwargs: flags for dynamic activation quantization
    """

    # TODO: Static quantization support using `static_scale`
    tensor_data_names = ["qdata", "scale"]
    tensor_attribute_names = ["granularity"]
    optional_tensor_attribute_names = ["act_quant_kwargs", "block_size", "dtype"]

    def __new__(
        cls: type,
        qdata: torch.Tensor,
        scale: torch.Tensor,
        granularity: Optional[Granularity] = None,
        block_size: Optional[torch.Size] = None,
        act_quant_kwargs: Optional[QuantizeTensorToInt8Kwargs] = None,
        dtype: Optional[torch.dtype] = None,
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
        granularity: Granularity,
        block_size: Optional[torch.Size] = None,
        act_quant_kwargs: Optional[QuantizeTensorToInt8Kwargs] = None,
        dtype: Optional[torch.dtype] = None,
    ):
        super().__init__()
        self.qdata = qdata
        self.scale = scale
        self.granularity = granularity
        self.block_size = block_size or get_block_size(qdata.shape, granularity)
        self.act_quant_kwargs = act_quant_kwargs

    def __repr__(self):
        return (
            f"{self.__class__.__name__}("
            f"act_quant_kwargs={self.act_quant_kwargs}, "
            f"qdata={self.qdata}, "
            f"scale={self.scale}, "
            f"granularity={self.granularity}, "
            f"block_size={self.block_size}, "
            f"shape={self.shape}, "
            f"device={self.device}, "
            f"dtype={self.dtype})"
        )

    @classmethod
    def from_hp(
        cls,
        w_hp: torch.Tensor,
        granularity: Granularity = PerRow(),
        act_quant_kwargs: Optional[QuantizeTensorToInt8Kwargs] = None,
    ):
        """Create Int8Tensor from high-precision tensor"""
        block_size = get_block_size(w_hp.shape, granularity)

        if w_hp.dim() not in [2, 3] or len(block_size) != w_hp.dim():
            raise ValueError(
                f"Expected 2D or 3D tensor with matching block_size dimensions, "
                f"got tensor dim={w_hp.dim()}, block_size length={len(block_size)}"
            )

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
            granularity,
            block_size=block_size,
            act_quant_kwargs=act_quant_kwargs,
            dtype=w_hp.dtype,
        )

    def dequantize(self, output_dtype: Optional[torch.dtype] = None) -> torch.Tensor:
        """Dequantize int8 tensor to floating point"""
        if output_dtype is None:
            output_dtype = self.dtype

        block_size = get_block_size(self.qdata.shape, self.granularity)

        return dequantize_affine(
            input=self.qdata,
            block_size=block_size,
            scale=self.scale,
            zero_point=None,
            input_dtype=torch.int8,
            quant_min=-128,
            quant_max=127,
            output_dtype=output_dtype,
        )


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

    if not isinstance(weight_tensor, Int8Tensor):
        raise TypeError(f"Expected weight to be Int8Tensor, got {type(weight_tensor)}")

    output_dtype = activation_tensor.dtype

    if weight_tensor.act_quant_kwargs is not None:
        activation_tensor = Int8Tensor.from_hp(
            activation_tensor, weight_tensor.act_quant_kwargs.granularity
        )
        # Dynamic activation quantization path

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
        # Cast FP16 scale to float to avoid overflow in int_scaled_matmul
        intermediate_dtype = (
            torch.float if x_scales.dtype == torch.half else x_scales.dtype
        )
        y_dot_scaled = int_scaled_matmul(
            tmp, w_vals_int8_t, x_scales.reshape(-1, 1).to(intermediate_dtype)
        ).to(output_dtype)
        y = (y_dot_scaled * w_scales).reshape(
            *x_vals_int8.shape[:-1], y_dot_scaled.shape[-1]
        )

    else:
        # FP × INT8 (weight-only)
        w_vals_int8_t = weight_tensor.qdata.t()
        m = torch.mm(
            activation_tensor.reshape(-1, activation_tensor.shape[-1]),
            w_vals_int8_t.to(output_dtype),
        )
        y = m * weight_tensor.scale.to(m.dtype)
        y = y.reshape(*activation_tensor.shape[:-1], weight_tensor.qdata.shape[0])

    if bias is not None:
        y += bias

    return y.to(output_dtype)


@implements(aten.slice.Tensor)
def _(func, types, args, kwargs):
    """Slice operation for Int8Tensor"""
    self, dim, start, end, step = fill_defaults(args, 5, [0, None, None, 1])

    if step != 1:
        raise NotImplementedError(
            f"Slicing with step != 1 is not supported, got step={step}"
        )

    if dim not in [0, 1, 2]:
        raise ValueError(f"Only dim in [0, 1, 2] supported, got dim={dim}")

    if self.qdata.ndim not in [2, 3]:
        raise ValueError(f"Expected qdata to be 2D or 3D, got {self.qdata.ndim}D")

    if end is None or end > self.shape[dim]:
        end = self.shape[dim]

    sliced_qdata = aten.slice.Tensor(self.qdata, dim, start, end, step)
    sliced_scale = _slice_scale_for_dimension(
        self.scale, self.qdata.shape, dim, start, end, step
    )

    return return_and_correct_aliasing(
        func,
        args,
        kwargs,
        Int8Tensor(
            sliced_qdata,
            sliced_scale,
            self.granularity,
            block_size=get_block_size(sliced_qdata.shape, self.granularity),
            act_quant_kwargs=self.act_quant_kwargs,
            dtype=self.dtype,
        ),
    )


@implements(aten.select.int)
def _(func, types, args, kwargs):
    """Select operation for Int8Tensor"""
    self, dim, index = args
    if dim != 0:
        raise NotImplementedError(f"Only dim=0 supported, got dim={dim}")

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
            self.granularity,
            block_size=get_block_size(selected_qdata.shape, self.granularity),
            act_quant_kwargs=self.act_quant_kwargs,
            dtype=self.dtype,
        ),
    )


Int8Tensor.__module__ = "torchao.quantization"
torch.serialization.add_safe_globals([Int8Tensor, QuantizeTensorToInt8Kwargs])
