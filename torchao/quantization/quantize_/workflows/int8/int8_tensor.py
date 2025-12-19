# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD 3-Clause license found in the
# LICENSE file in the root directory of this source tree.

from dataclasses import dataclass
from typing import List, Optional

import torch
from torch.utils._python_dispatch import return_and_correct_aliasing

from torchao.float8.inference import _slice_scale_for_dimension
from torchao.kernel import int_scaled_matmul
from torchao.quantization.granularity import Granularity
from torchao.quantization.quant_primitives import (
    MappingType,
    choose_qparams_affine,
    dequantize_affine,
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
    """Tensor kwargs for creating int8 tensor for activation.

    Args:
        granularity: the granularity for the Tensor, currently either PerRow() or PerTensor()
        mapping_type: whether to use symmetric or asymmetric quant, only symmetric is supported currently
    """

    granularity: Granularity
    mapping_type: MappingType = MappingType.SYMMETRIC


class Int8Tensor(TorchAOBaseTensor):
    """
    int8 quantized tensor with plain layout.

    Currently only Symmetric quantization is supported.

    Tensor Attributes:
        qdata: (N, K) or (B, N, K) int8 quantized weight data (2D or 3D)
        scale: scale factors for dequantization

    Non-Tensor Attributes:
        granularity: the granularity for quantization (e.g., PerRow(), PerTensor())
        act_quant_kwargs: flags for dynamic activation quantization
    """

    tensor_data_names = ["qdata", "scale"]
    optional_tensor_data_names = ["act_scale"]
    tensor_attribute_names = ["block_size", "dtype"]
    optional_tensor_attribute_names = [
        "act_quant_kwargs",
    ]

    def __new__(
        cls: type,
        qdata: torch.Tensor,
        scale: torch.Tensor,
        block_size: List[int],
        dtype: torch.dtype,
        act_scale=None,
        act_quant_kwargs: Optional[QuantizeTensorToInt8Kwargs] = None,
    ):
        kwargs = {
            "device": qdata.device,
            "dtype": dtype,
            "requires_grad": False,
        }
        return torch.Tensor._make_wrapper_subclass(cls, qdata.shape, **kwargs)

    def __init__(
        self,
        qdata: torch.Tensor,
        scale: torch.Tensor,
        block_size: List[int],
        dtype: torch.dtype,
        act_scale=None,
        act_quant_kwargs: Optional[QuantizeTensorToInt8Kwargs] = None,
    ):
        super().__init__()
        self.qdata = qdata
        self.scale = scale
        self.block_size = block_size
        # don't set dtype because this gets done in __new__
        self.act_quant_kwargs = act_quant_kwargs
        self.act_scale = act_scale

    def __repr__(self):
        return (
            f"{self.__class__.__name__}("
            f"act_quant_kwargs={self.act_quant_kwargs}, "
            f"qdata={self.qdata}, "
            f"scale={self.scale}, "
            f"act_scale={self.act_scale}, "
            f"block_size={self.block_size}, "
            f"shape={self.shape}, "
            f"device={self.device}, "
            f"dtype={self.dtype})"
        )

    @classmethod
    def from_hp(
        cls,
        hp_tensor: torch.Tensor,
        granularity: Granularity,
        mapping_type=MappingType.SYMMETRIC,
        scale: Optional[torch.Tensor] = None,
        act_quant_kwargs: Optional[QuantizeTensorToInt8Kwargs] = None,
        act_scale: Optional[torch.Tensor] = None,
    ):
        """Create Int8Tensor from high-precision tensor"""
        block_size = get_block_size(hp_tensor.shape, granularity)
        block_size = list(block_size)

        if scale is None:
            scale, zero_point = choose_qparams_affine(
                input=hp_tensor,
                mapping_type=mapping_type,
                block_size=block_size,
                target_dtype=torch.int8,
                quant_min=-128,
                quant_max=127,
                scale_dtype=hp_tensor.dtype,
                zero_point_dtype=torch.int8,
                keepdim=True,
            )
        else:
            # Scale can be provided in the case of static quant
            assert scale.ndim == hp_tensor.ndim
            assert all(
                (hp_tensor.shape[i] // block_size[i]) == scale.shape[i]
                for i in range(hp_tensor.ndim)
            )
            zero_point = torch.zeros_like(scale, dtype=torch.int8)

        int_data = quantize_affine(
            hp_tensor,
            block_size=block_size,
            scale=scale,
            zero_point=zero_point,
            output_dtype=torch.int8,
        )

        return cls(
            int_data,
            scale,
            block_size,
            hp_tensor.dtype,
            act_scale=act_scale,
            act_quant_kwargs=act_quant_kwargs,
        )

    def dequantize(self, output_dtype: Optional[torch.dtype] = None) -> torch.Tensor:
        """Dequantize int8 tensor to floating point"""
        return dequantize_affine(
            input=self.qdata,
            block_size=self.block_size,
            scale=self.scale,
            zero_point=None,
            input_dtype=torch.int8,
            quant_min=-128,
            quant_max=127,
            output_dtype=output_dtype if output_dtype is not None else self.dtype,
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

    assert isinstance(weight_tensor, Int8Tensor), (
        f"Expected weight to be Int8Tensor, got {type(weight_tensor)}"
    )

    output_dtype = activation_tensor.dtype

    if weight_tensor.act_quant_kwargs is not None:
        activation_tensor = _choose_quant_func_and_quantize_tensor(
            activation_tensor,
            weight_tensor.act_quant_kwargs,
            scale=weight_tensor.act_scale,
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
        y = (y_dot_scaled * w_scales.flatten()).reshape(
            *x_vals_int8.shape[:-1], y_dot_scaled.shape[-1]
        )

    else:
        # FP × INT8 (weight-only)
        w_vals_int8_t = weight_tensor.qdata.t()
        m = torch.mm(
            activation_tensor.reshape(-1, activation_tensor.shape[-1]),
            w_vals_int8_t.to(output_dtype),
        )
        y = m * weight_tensor.scale.to(m.dtype).flatten()
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
    if self.scale.numel() == 1:
        # Per-tensor quantization - scale doesn't change
        sliced_scale = self.scale
    else:
        # Block-wise quantization - need to slice the scale appropriately
        sliced_scale = _slice_scale_for_dimension(
            self.scale, self.qdata.shape, dim, start, end, step
        )

    # adjust block_size since the shape has changed, block_size[i] should not be greater than shape[i]
    block_size = self.block_size.copy()
    for i in range(len(self.block_size)):
        block_size[i] = min(block_size[i], sliced_qdata.shape[i])

    return return_and_correct_aliasing(
        func,
        args,
        kwargs,
        Int8Tensor(
            sliced_qdata,
            sliced_scale,
            block_size,
            self.dtype,
            act_quant_kwargs=self.act_quant_kwargs,
        ),
    )


@implements(aten.index.Tensor)
def _(func, types, args, kwargs):
    return return_and_correct_aliasing(
        func,
        args,
        kwargs,
        args[0]._apply_fn_to_data(lambda x: func(x, *args[1:], **kwargs)),
    )


@implements(aten.is_pinned.default)
def _(func, types, args, kwargs):
    is_pinned = args[0].qdata.is_pinned() and args[0].scale.is_pinned()
    if args[0].act_scale is not None:
        is_pinned = is_pinned and args[0].act_scale.is_pinned()
    return is_pinned


@implements(aten._pin_memory.default)
def _(func, types, args, kwargs):
    pinned_qdata = args[0].qdata.pin_memory()
    pinned_scale = args[0].scale.pin_memory()

    pinned_act_scale = None
    if args[0].act_scale is not None:
        pinned_act_scale = args[0].act_scale.pin_memory()

    return Int8Tensor(
        pinned_qdata,
        pinned_scale,
        args[0].block_size,
        args[0].dtype,
        act_scale=pinned_act_scale,
        act_quant_kwargs=args[0].act_quant_kwargs,
    )


@implements(aten.select.int)
def _(func, types, args, kwargs):
    """Select operation for Int8Tensor"""
    old_int8_tensor, dim, index = args
    assert dim == 0, f"Int8Tensor aten.select.int with {dim=} is not yet supported"
    assert len(old_int8_tensor.qdata.shape) == len(old_int8_tensor.scale.shape), (
        "unsupported"
    )
    assert len(old_int8_tensor.qdata.shape) == len(old_int8_tensor.block_size), (
        "unsupported"
    )
    new_int8_tensor = Int8Tensor(
        old_int8_tensor.qdata[index],
        old_int8_tensor.scale[index],
        old_int8_tensor.block_size[1:],
        old_int8_tensor.dtype,
        old_int8_tensor.act_quant_kwargs,
    )
    return return_and_correct_aliasing(func, args, kwargs, new_int8_tensor)


Int8Tensor.__module__ = "torchao.quantization"
torch.serialization.add_safe_globals([Int8Tensor, QuantizeTensorToInt8Kwargs])
