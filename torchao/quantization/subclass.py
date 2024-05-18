# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import warnings

import torch
from torch.utils._python_dispatch import return_and_correct_aliasing

from .quant_primitives import (
    dequantize_per_channel,
    dynamically_quantize_per_channel,
    groupwise_affine_quantize_tensor,
    quant_int8_dynamic_per_token_linear,
    pack_tinygemm_scales_and_zeros,
    unpack_tinygemm_scales_and_zeros,
    groupwise_affine_quantize_tensor_from_qparams,
    choose_qparams_affine,
    quantize_affine,
    dequantize_affine,
    ZeroPointDomain,
    MappingType,
)
from torchao.kernel.intmm import int_scaled_matmul
from .utils import find_multiple
from typing import Tuple, Optional, Callable, Dict, Any
from collections import defaultdict
import functools


__all__ = [
    "Int8DynamicallyQuantizedLinearWeight",
    "Int8WeightOnlyQuantizedLinearWeight",
    "Int4WeightOnlyQuantizedLinearWeight",
    "AffineQuantizedTensor",
]


aten = torch.ops.aten

def _aqt_is_int8(aqt):
    """Check if an AffineQuantizedTensor is int8 quantized Tensor"""
    return (
        aqt.int_data.dtype == torch.int8 and
        aqt.quant_min is None or aqt.quant_min == -128 and
        aqt.quant_max is None or aqt.quant_max == 127
    )

def _aqt_is_int8_reduced_range(aqt):
    return (
        aqt.int_data.dtype == torch.int8 and
        aqt.quant_min == -127 and
        aqt.quant_max is None or aqt.quant_max == 127
    )

def _aqt_is_uint4(aqt):
    """Check if an AffineQuantizedTensor is uint4 quantized Tensor"""
    # TODO: use torch.uint4
    return (
        aqt.int_data.dtype == torch.int32 and
        aqt.quant_min is None or aqt.quant_min == 0 and
        aqt.quant_max is None or aqt.quant_max == 15
    )


class QuantizedLinearWeightBase(torch.Tensor):
    """
    Base quantized tensor subclass for quantized linear weights. When the from_float method is used,
    to create an instance of any QuantizedLinearWeightBase, we assume the input
    weight is oriented the way it is in a normal linear op, i.e. out-channels x in-channels.

    The shape and dtype of the tensor subclass represent how the tensor subclass looks externally,
    regardless of the internal representation's type or orientation.
    """

    @staticmethod
    def __new__(cls, int_data, transposed, shape, *args, **kwargs):
        kwargs["device"] = int_data.device
        kwargs["layout"] = (
            kwargs.get("layout") if kwargs.get("layout", False) else int_data.layout
        )
        assert "dtype" in kwargs
        assert not kwargs.get("requires_grad", False)
        kwargs["requires_grad"] = False
        return torch.Tensor._make_wrapper_subclass(cls, shape, **kwargs)  # type: ignore[attr-defined]

    def __init__(self, int_data, transposed, *args, **kwargs):

        self.int_data = int_data

        self.transposed = transposed

    @staticmethod
    def _quantized_op(act_mat, w_qtensor, bias):
        pass

    def __repr__(self):
        return (
            f"{self.__class__.__name__}(data={self.dequantize()}, shape={self.shape}, "
            f"device={self.device}, dtype={self.dtype}, requires_grad={self.requires_grad})"
        )

    def dequantize(self):
        pass

    def int_repr(self):
        pass

    def q_params(self):
        pass

    def half(self):
        return self.to(torch.float16)

    def _get_to_kwargs(self, *args, **kwargs):
        device, dtype, _, memory_format = torch._C._nn._parse_to(*args, **kwargs)
        device = self.device if device is None else device
        dtype = self.dtype if dtype is None else dtype
        memory_format = (
            memory_format if memory_format is not None else torch.preserve_format
        )
        kwargs = {
            "device": device,
            "dtype": dtype,
            "memory_format": memory_format,
        }
        return kwargs

    def _apply_fn_to_data(self, fn):
        pass

    def _change_shape(self):
        pass

    def __tensor_flatten__(self):
        pass

    @classmethod
    def __tensor_unflatten__(
        cls, tensor_data_dict, tensor_attributes, outer_size, outer_stride
    ):
        pass

    @classmethod
    def from_float(cls, input_float):
        pass

    # __torch_function__ = torch._C._disabled_torch_function_impl

    @classmethod
    def __torch_function__(cls, func, types, args=(), kwargs=None):
        kwargs = {} if kwargs is None else kwargs

        if func is torch.nn.functional.linear:
            mat1, w_qtensor, bias = (
                args[0],
                args[1],
                args[2] if len(args) > 2 else None,
            )
            assert w_qtensor.transposed == False
            return cls._quantized_op(mat1, w_qtensor, bias)

        try:
            with torch._C.DisableTorchFunctionSubclass():
                return func(*args, **kwargs)
        except:
            print(f"ERR: subclass doesn't implement {func}")

    @classmethod
    def __torch_dispatch__(cls, func, types, args, kwargs):
        # two scenarios where we currently fall back to vanilla mm:
        # 1 - when tensor is on CPU: we are missing qmm for CPU, but we should have a CPU implementation
        #     for consistency and to allow people to test
        # 2 - we're given non-floats - quantizing long to int8 is crazy
        if (
            func in [aten.mm.default, aten.addmm.default]
            and args[0].is_floating_point()
            and args[0].is_cuda
        ):
            if func == aten.addmm.default:
                assert args[1].shape[-1] == args[2].shape[0], (
                    f"need mat1 shape: {args[1].shape} final"
                    f"dim to match mat2 shape: {args[2].shape} first dim "
                )
                mat1, w_qtensor, bias = (
                    args[1],
                    args[2],
                    args[0],
                )
            else:
                assert args[0].shape[-1] == args[1].shape[0], (
                    f"need mat1 shape: {args[0].shape} final dim"
                    f"to match mat2 shape: {args[1].shape} first dim"
                )
                mat1, w_qtensor, bias = (
                    args[0],
                    args[1],
                    None if len(args) == 2 else args[2],
                )
            # call the quantized op for the specific type
            # of quantized tensor subclass
            return cls._quantized_op(mat1, w_qtensor, bias)

        if func is aten.detach.default:
            return return_and_correct_aliasing(
                func, args, kwargs, args[0]._apply_fn_to_data(torch.detach)
            )

        if func is aten.clone.default:
            return return_and_correct_aliasing(
                func, args, kwargs, args[0]._apply_fn_to_data(torch.clone)
            )

        if func is aten.t.default:
            args[0].transposed = not args[0].transposed
            new = args[0]._change_shape(args[0].shape[::-1])
            return return_and_correct_aliasing(func, args, kwargs, new)

        if func is aten._to_copy.default:
            return return_and_correct_aliasing(
                func,
                args,
                kwargs,
                args[0].to(*args[1:], **kwargs)._apply_fn_to_data(torch.clone),
            )

class ConstructTensorSubclass(torch.nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__()
        self.args = args
        self.kwargs = kwargs

    def forward(self, x):
        pass

    def right_inverse(self, tensor_subclass_instance):
        fields, _ = tensor_subclass_instance.__tensor_flatten__()
        return [getattr(tensor_subclass_instance, field) for field in fields]


@torch._dynamo.allow_in_graph
def from_qtensor_components_int8dyn(*args, **kwargs):
    return Int8DynamicallyQuantizedLinearWeight(*args, **kwargs)


class ConstructTensorSubclassInt8Dyn(ConstructTensorSubclass):
    def forward(self, int_data, q_scales):
        return from_qtensor_components_int8dyn(int_data, q_scales, *self.args, **self.kwargs)


class Int8DynamicallyQuantizedLinearWeight(QuantizedLinearWeightBase):
    """
    A Tensor subclass that when applied to a weight used in a linear op/module, changes the
    linear op to a dynamically quantized linear op with symmetric per-token and per-channel
    quantization on the activation and weight respectively.
    """
    subclass_constructor = ConstructTensorSubclassInt8Dyn

    @staticmethod
    def __new__(cls, int_data, q_scales, transposed, shape, dtype=None, **kwargs):
        if dtype is None:
            dtype = qscales.dtype
        kwargs["dtype"] = dtype
        return super().__new__(cls, int_data, transposed, shape, **kwargs)  # type: ignore[attr-defined]

    def __init__(self, int_data, q_scales, transposed, shape, dtype=None, **kwargs):

        self.q_scales = q_scales
        super().__init__(int_data, transposed)

    @staticmethod
    def _quantized_op(act_mat, w_qtensor, bias):
        return quant_int8_dynamic_per_token_linear(
            act_mat, w_qtensor.int_data, w_qtensor.q_scales, bias, act_mat.dtype
        )

    def dequantize(self, dtype=None):
        """
        Obtain the dequantized version of the quantized tensor subclass
        """
        zero_points = torch.zeros(self.q_scales.shape, device=self.q_scales.device, dtype=self.q_scales.dtype)
        # zero_points = 0
        # TODO: fix dtype here? `to(self.dtype)` is not overwritten by `dtype` arg?
        dq_t = dequantize_per_channel(
            self.int_data.t(), self.q_scales, zero_points, self.dtype if dtype is None else dtype
        ).to(self.dtype)
        # data was transposed to dequantize so make sure shape is correct
        return dq_t if not self.transposed else dq_t.t()

    def int_repr(self):
        """
        Get the internal integer representation of the quantized tensor
        """
        return self.int_data if self.transposed else self.int_data.t()

    def q_params(self):
        """
        Get the quantization scales for the quantized tensor
        """
        return {"q_scales": self.q_scales}

    def to(self, *args, **kwargs):
        kwargs = self._get_to_kwargs(*args, **kwargs)
        return self.__class__(
            self.int_data.to(kwargs["device"]),
            self.q_scales.to(kwargs["device"]),
            self.transposed,
            self.shape,
            **kwargs,
        )

    def _apply_fn_to_data(self, fn):
        return self.__class__(
            fn(self.int_data),
            fn(self.q_scales),
            self.transposed,
            self.shape,
            dtype=self.dtype,
        )

    #  `QuantizedLinearWeightBase` inconsistently.

    def _change_shape(self, shape):
        return self.__class__(
            self.int_data, self.q_scales, self.transposed, shape, dtype=self.dtype
        )

    def __tensor_flatten__(self):
        # note: the order of args must match the order of args in __init__
        return ["int_data", "q_scales"], [self.transposed, self.shape, self.dtype]

    @classmethod
    def __tensor_unflatten__(
        cls, tensor_data_dict, tensor_attributes, outer_size=None, outer_stride=None
    ):
        int_data, q_scales = tensor_data_dict["int_data"], tensor_data_dict["q_scales"]
        transposed, shape, dtype = tensor_attributes
        return cls(
            int_data,
            q_scales,
            transposed,
            shape if outer_size is None else outer_size,
            dtype=dtype,
            strides=outer_stride,
        )

    @classmethod
    def from_float(cls, input_float, qmin=-128, qmax=127, dtype=None):
        """
        Method used to convert a linear weight tensor to an instance of the
        Int8DynamicallyQuantizedLinearWeight subclass.

        Example usage::

            model.lin_mod.weight = (
                Int8DynamicallyQuantizedLinearWeight.from_float(model.lin_mod.weight)
            )
        """
        if dtype is None:
            dtype = input_float.dtype

        # because we call transpose in dequantization
        w_int_repr, w_scales, _ = dynamically_quantize_per_channel(
            input_float, qmin, qmax, torch.int8
        )
        # the desired representation shape for fast quantized matmul is
        # transposed compared to how it's stored as a linear weight,
        # i.e. we want in_channels as dim=0 and out_channels (and quantized axis) as dim=1
        # however the external representation of our tensor will maintain the correct
        # shape attribute which needs to be tracked directly.
        int_data = w_int_repr.contiguous().t()
        if not issubclass(cls, Int8DynamicallyQuantizedLinearWeight):
            int_data = int_data.contiguous()
        return cls(
            int_data, w_scales, False, input_float.shape, dtype=dtype,
        )


@torch._dynamo.allow_in_graph
def from_qtensor_components_int8wo(*args, **kwargs):
    return Int8WeightOnlyQuantizedLinearWeight(*args, **kwargs)


class ConstructTensorSubclassInt8wo(ConstructTensorSubclass):
    def forward(self, int_data, q_scales):
        return from_qtensor_components_int8wo(int_data, q_scales, *self.args, **self.kwargs)


class Int8WeightOnlyQuantizedLinearWeight(Int8DynamicallyQuantizedLinearWeight):
    """
    A Tensor subclass that when applied to a weight used in a linear op/module,
    changes the linear op to a weight-only quantized linear op with symmetric
    per-channel quantization on the weight.
    """
    subclass_constructor = ConstructTensorSubclassInt8wo

    @staticmethod
    def _quantized_op(act_mat, w_qtensor, bias):
        orig_dtype = act_mat.dtype
        y = (
            torch.mm(
                act_mat.reshape(-1, act_mat.shape[-1]),
                w_qtensor.int_data.to(act_mat.dtype),
            )
            * w_qtensor.q_scales
        )
        y = y.reshape(*act_mat.shape[:-1], y.shape[-1])
        if bias is not None:
            y += bias
        return y.to(orig_dtype)


@torch._dynamo.allow_in_graph
def from_qtensor_components_int4wo(*args, **kwargs):
    return Int4WeightOnlyQuantizedLinearWeight(*args, **kwargs)

class ConstructTensorSubclassInt4wo(ConstructTensorSubclass):
    def forward(self, int_data, scales_and_zeros):
        return from_qtensor_components_int4wo(int_data, scales_and_zeros, *self.args, **self.kwargs)

class Int4WeightOnlyQuantizedLinearWeight(QuantizedLinearWeightBase):
    """
    A Tensor subclass that when applied to a weight used in a linear op/module,
    changes that linear op to a weight-only int4 quantized linear op with groupwise
    affine quantization on the weight.
    """
    subclass_constructor = ConstructTensorSubclassInt4wo

    @staticmethod
    def __new__(
        cls,
        int_data,
        scales_and_zeros,
        transposed,
        shape,
        groupsize=128,
        inner_k_tiles=8,
        dtype=None,
        **kwargs,
    ):
        if dtype is None:
            dtype = scales_and_zeros.dtype
        kwargs["dtype"] = dtype
        return super().__new__(cls, int_data, transposed, shape, **kwargs)  # type: ignore[attr-defined]

    def __init__(
        self,
        int_data,
        scales_and_zeros,
        transposed,
        shape,
        groupsize,
        inner_k_tiles,
        dtype,
        **kwargs,
    ):
        # the transposed flag tracks whether the tensor subclass has been transposed relative
        # to how a weight is normally stored in a linear i.e. [out_features, in_features].
        # tracking both transposed and shape is slightly redundant but corner cases like
        # square matrices can cause issues otherwise

        self.scales_and_zeros = scales_and_zeros
        self.groupsize = groupsize
        self.inner_k_tiles = inner_k_tiles
        super().__init__(int_data, transposed)

    @staticmethod
    def _quantized_op(act_mat, w_qtensor, bias):
        orig_act_size = act_mat.size()
        orig_dtype = act_mat.dtype

        # reshape and pad activation
        act_mat = act_mat.reshape(-1, act_mat.shape[-1]).to(torch.bfloat16)
        pad_size = find_multiple(act_mat.shape[-1], 1024)
        act_mat = torch.nn.functional.pad(act_mat, (0, pad_size - act_mat.shape[-1]))

        # matmul
        y = aten._weight_int4pack_mm(
            act_mat.contiguous(),
            w_qtensor.int_data,
            w_qtensor.groupsize,
            w_qtensor.scales_and_zeros,
        )

        # remove out_feature padding
        orig_out_features = (
            w_qtensor.shape[-1] if w_qtensor.transposed else w_qtensor.shape[-2]
        )
        y = y[:, :orig_out_features]

        y = y.reshape(*orig_act_size[:-1], orig_out_features)
        if bias is not None:
            y += bias
        return y.to(orig_dtype)

    def dequantize(self):
        eye_shape = self.shape[1] if not self.transposed else self.shape[0]
        w_dq = self._quantized_op(
            torch.eye(eye_shape, device=self.device, dtype=self.dtype), self, None
        )
        # we dequantized using linear with the identity matrix, output has shape [in_channels, out_channels]
        # so we need to transpose back to get the original shape unless self.transposed is set.
        w_dq = w_dq if self.transposed else w_dq.t()
        return w_dq.to(self.dtype)

    def int_repr(self):
        return self.int_data

    def q_params(self):
        scales, zero_points = unpack_tinygemm_scales_and_zeros(
            self.scales_and_zeros,
        )
        return {"q_scales": scales, "q_zero_points": zero_points}

    def to(self, *args, **kwargs):
        kwargs = self._get_to_kwargs(*args, **kwargs)
        return self.__class__(
            self.int_data.to(kwargs["device"]),
            self.scales_and_zeros.to(kwargs["device"]),
            self.transposed,
            self.shape,
            self.groupsize,
            self.inner_k_tiles,
            **kwargs,
        )

    def _apply_fn_to_data(self, fn):
        return self.__class__(
            fn(self.int_data),
            fn(self.scales_and_zeros),
            self.transposed,
            self.shape,
            self.groupsize,
            self.inner_k_tiles,
            dtype=self.dtype,
        )

    #  `QuantizedLinearWeightBase` inconsistently.

    def _change_shape(self, shape):
        return self.__class__(
            self.int_data,
            self.scales_and_zeros,
            self.transposed,
            shape,
            self.groupsize,
            self.inner_k_tiles,
            dtype=self.dtype,
        )

    def __tensor_flatten__(self):
        return ["int_data", "scales_and_zeros"], (
            self.transposed,
            self.shape,
            self.groupsize,
            self.inner_k_tiles,
            self.dtype,
        )

    @classmethod

    #  `QuantizedLinearWeightBase` inconsistently.

    def __tensor_unflatten__(
        cls, tensor_data_dict, attributes, outer_size=None, outer_stride=None
    ):
        int_data, scales_and_zeros = (
            tensor_data_dict["int_data"],
            tensor_data_dict["scales_and_zeros"],
        )
        transposed, shape, groupsize, inner_k_tiles, dtype = attributes
        return cls(
            int_data,
            scales_and_zeros,
            transposed,
            shape if outer_size is None else outer_size,
            groupsize,
            inner_k_tiles,
            dtype=dtype,
            strides=outer_stride,
        )

    @classmethod
    def from_float(cls, input_float, groupsize=128, inner_k_tiles=8, dtype=None):
        """
        Method used to convert a linear weight tensor to an instance of the
        Int4WeightOnlyQuantizedLinearWeight subclass.

        Example usage::

            model.lin_mod.weight = (
                Int4WeightOnlyQuantizedLinearWeight.from_float(model.lin_mod.weight)
            )
        """
        if dtype is None:
            dtype = input_float.dtype

        int_data, scales_and_zeros, transposed, groupsize, inner_k_tils = cls.to_qtensor_components(input_float, groupsize, inner_k_tiles)
        return cls(
            int_data,
            scales_and_zeros,
            transposed,
            input_float.shape,
            groupsize,
            inner_k_tiles,
            dtype=dtype,
        )

    @classmethod
    def to_qtensor_components(cls, input_float, groupsize=128, inner_k_tiles=8):
        assert groupsize in [256, 128, 64, 32]
        assert inner_k_tiles in [8, 4, 2]
        orig_out_features, orig_in_features = input_float.shape

        # padding
        in_features = find_multiple(orig_in_features, 1024)
        out_features = find_multiple(orig_out_features, 8)
        input_float = torch.nn.functional.pad(
            input_float,
            (0, in_features - orig_in_features, 0, out_features - orig_out_features),
        )

        # quantization and packing
        input_int4x8, scales_and_zeros = groupwise_affine_quantize_tensor(
            input_float, 4, groupsize, dtype=input_float.dtype
        )
        int_data = aten._convert_weight_to_int4pack(input_int4x8, inner_k_tiles)
        return int_data, scales_and_zeros, False, groupsize, inner_k_tiles

def to_aqt(
    input_float,
    mapping_type,
    block_size,
    target_dtype,
    quant_min = None,
    quant_max = None,
    eps = None,
    scale_dtype = None,
    zero_point_dtype = None,
    preserve_zero = True,
    zero_point_domain = ZeroPointDomain.INT,
):
    return AffineQuantizedTensor.from_float(
        input_float,
        mapping_type,
        block_size,
        target_dtype,
        quant_min=quant_min,
        quant_max=quant_max,
        eps=eps,
        scale_dtype=scale_dtype,
        zero_point_dtype=zero_point_dtype,
        preserve_zero=preserve_zero,
        zero_point_domain=zero_point_domain
    )

# TODO: merge with nf4 implements decorator
# aten op to their __torch_dispatch__ implemnetations for the tensor subclass
_ATEN_OPS_TABLE: Dict[Callable, Dict[Any, Any]] = defaultdict(dict)

def implements_aten_ops(cls, aten_ops):
    """Use this decorator to implement a function for an aten op in __torch_dispatch__"""

    def decorator(func):
        for op in aten_ops:
            _ATEN_OPS_TABLE[cls][op] = func
        return func

    return decorator

_TORCH_FUNCTIONS_TABLE: Dict[Callable, Dict[Any, Any]] = defaultdict(dict)

def implements_torch_function(cls, torch_function):
    def decorator(func):
        functools.update_wrapper(func, torch_function)
        _TORCH_FUNCTIONS_TABLE[cls][torch_function] = func
        return func

    return decorator

def implements_aqt_aten_ops(aten_ops):
    return implements_aten_ops(AffineQuantizedTensor, aten_ops)

def implements_aqt_torch_function(torch_function):
    return implements_torch_function(AffineQuantizedTensor, torch_function)


class AffineQuantizedTensor(torch.Tensor):
    """
    Base affine quantized tensor subclass. When the from_float method is used,
    to create an instance of any AffineQuantizedTensor

    The shape and dtype of the tensor subclass represent how the tensor subclass looks externally,
    regardless of the internal representation's type or orientation.

    Affine quantization means we quantize the floating point tensor with an affine transformation:
       quantized_tensor = float_tensor / scale + zero_point

    fields:
      int_data (torch.Tensor): the quantized integer data Tensor
      scale (torch.Tensor): the scale Tensor used to map between floating point tensor to quantized tensor
      zero_point (torch.Tensor): the zero_point Tensor used to map between floating point tensor to quantized tensor
      block_size (Tuple[int, ...]): granularity of quantization, this means the size of the tensor elements that's sharing the same qparam
         e.g. when size is the same as the input tensor dimension, we are using per tensor quantization
      shape (torch.Size): the shape for the Tensor
      quant_min (Optional[int]): minimum quantized value for the Tensor, if not specified, it will be derived from dtype of `int_data`
      quant_max (Optional[int]): maximum quantized value for the Tensor, if not specified, it will be derived from dtype of `int_data`
      zero_point_domain (ZeroPointDomain): the domain that zero_point is in, should be eitehr integer or float
        if zero_point is in integer domain, zero point is added to the quantized integer value during
        quantization
        if zero_point is in floating point domain, zero point is subtracted from the floating point (unquantized)
        value during quantization
        default is ZeroPointDomain.INT
      input_quant_func (Optional[Callable]): function for quantizing the input float Tensor to a quantized tensor subclass object, that takes float Tensor as input and outputs an AffineQuantizedTensor object
      dtype: dtype for external representation of the tensor, e.g. torch.float32
    """

    @staticmethod
    def __new__(
        cls,
        int_data: torch.Tensor,
        scale: torch.Tensor,
        zero_point: torch.Tensor,
        block_size: Tuple[int, ...],
        shape: torch.Size,
        quant_min: Optional[int] = None,
        quant_max: Optional[int] = None,
        zero_point_domain: ZeroPointDomain = ZeroPointDomain.INT,
        dtype=None,
        strides=None,
    ):
        kwargs = {}
        kwargs["device"] = int_data.device
        kwargs["layout"] = (
            kwargs.get("layout") if kwargs.get("layout", False) else int_data.layout
        )
        if dtype is None:
            dtype = scale.dtype
        kwargs["dtype"] = dtype
        if strides is not None:
            kwargs["strides"] = strides
        kwargs["requires_grad"] = False
        return torch.Tensor._make_wrapper_subclass(cls, shape, **kwargs)  # type: ignore[attr-defined]

    def __init__(
        self,
        int_data: torch.Tensor,
        scale: torch.Tensor,
        zero_point: torch.Tensor,
        block_size: Tuple[int, ...],
        shape: torch.Size,
        quant_min: Optional[int] = None,
        quant_max: Optional[int] = None,
        zero_point_domain: ZeroPointDomain = ZeroPointDomain.INT,
        dtype=None,
        strides=None,
    ):
        self.int_data = int_data
        self.scale = scale
        self.zero_point = zero_point
        self.block_size = block_size
        self.quant_min = quant_min
        self.quant_max = quant_max
        self.zero_point_domain = zero_point_domain

    def __repr__(self):
        return (
            f"{self.__class__.__name__}(data={self.dequantize()}, shape={self.shape}, "
            f"device={self.device}, dtype={self.dtype}, requires_grad={self.requires_grad})"
        )

    def dequantize(self, output_dtype=None):
        if output_dtype is None:
            output_dtype = self.dtype
        return dequantize_affine(self.int_data, self.block_size, self.scale, self.zero_point, self.int_data.dtype, self.quant_min, self.quant_max, self.zero_point_domain, output_dtype=output_dtype)

    def __tensor_flatten__(self):
        return ["int_data", "scales", "zero_point"], [self.block_size, self.shape, self.quant_min, self.quant_max, self.zero_point_domain, self.dtype]

    @classmethod
    def __tensor_unflatten__(
        cls, tensor_data_dict, tensor_attributes, outer_size, outer_stride
    ):
        int_data, scale, zero_point = tensor_data_dict["int_data"], tensor_data_dict["scale"], tensor_data_dict["zero_point"]
        block_size, shape, quant_min, quant_max, zero_point_domain, dtype = tensor_attributes
        return cls(
            int_data,
            scale,
            zero_point,
            block_size,
            shape if outer_size is None else outer_size,
            quant_min,
            quant_max,
            zero_point_domain,
            dtype=dtype,
            strides=outer_stride,
        )

    @classmethod
    def from_float(
        cls,
        input_float,
        mapping_type,
        block_size,
        target_dtype,
        quant_min = None,
        quant_max = None,
        eps = None,
        scale_dtype = None,
        zero_point_dtype = None,
        preserve_zero = True,
        zero_point_domain = ZeroPointDomain.INT,
    ):
        scale, zero_point = choose_qparams_affine(input_float, mapping_type, block_size, target_dtype, quant_min, quant_max, eps, scale_dtype, zero_point_dtype, preserve_zero, zero_point_domain)
        int_data = quantize_affine(input_float, block_size, scale, zero_point, target_dtype, quant_min, quant_max, zero_point_domain)
        return cls(
            int_data,
            scale,
            zero_point,
            block_size,
            input_float.shape,
            quant_min,
            quant_max,
            zero_point_domain,
            dtype=input_float.dtype
        )

    @classmethod
    def __torch_function__(cls, func, types, args=(), kwargs=None):
        kwargs = {} if kwargs is None else kwargs

        if func in _TORCH_FUNCTIONS_TABLE[cls]:
            return _TORCH_FUNCTIONS_TABLE[cls][func](*args, **kwargs)

        with torch._C.DisableTorchFunctionSubclass():
            return func(*args, **kwargs)


    def _get_to_kwargs(self, *args, **kwargs):
        device, dtype, _, memory_format = torch._C._nn._parse_to(*args, **kwargs)
        device = self.device if device is None else device
        dtype = self.dtype if dtype is None else dtype
        memory_format = (
            memory_format if memory_format is not None else torch.preserve_format
        )
        kwargs = {
            "device": device,
            "dtype": dtype,
            "memory_format": memory_format,
        }
        return kwargs

    def to(self, *args, **kwargs):
        kwargs = self._get_to_kwargs(*args, **kwargs)
        return self.__class__(
            self.int_data.to(kwargs["device"]),
            self.scale.to(kwargs["device"]),
            self.zero_point.to(kwargs["device"]),
            self.block_size,
            self.shape,
            self.quant_min,
            self.quant_max,
            self.zero_point_domain,
            **kwargs,
        )

    def _apply_fn_to_data(self, fn):
        return self.__class__(
            fn(self.int_data),
            fn(self.scale),
            fn(self.zero_point),
            self.block_size,
            self.shape,
            self.quant_min,
            self.quant_max,
            self.zero_point_domain,
            dtype=self.dtype,
            strides=self.stride(),
        )

    @classmethod
    def __torch_dispatch__(cls, func, types, args, kwargs):
        # Note: we only added cpu path here for 8da4w, this is for executorch, in the future
        # 1. we'll add cpu/cuda version (int4mm etc.)
        # 2. we'll need to hide the 8da4w executorch version under things like layouts (we also have multiple impl for cpu kernel as Michael mentioned), so it will be something like
        #   cpu device + et laytout --> gives current 8da4w executorch representation
        #   cpu device + avx layout --> gives optimized kernel for 8da4w in avx cpu etc.
        #   cuda device + some layout --> gives cuda kernel

        # two scenarios where we currently fall back to vanilla mm:
        # 1 - when tensor is on CUDA: we'll add this later, we'll also enable dispatching to optimized
        #     kernels in CPU as well, see the note above
        # 2 - we're given non-floats - quantizing long to int8 is crazy

        if func in _ATEN_OPS_TABLE[cls]:
            return _ATEN_OPS_TABLE[cls][func](func, *args, **kwargs)

        raise NotImplementedError(
            f"AffineQuantizedTensor dispatch: attempting to run {func}, this is not supported"
        )

@implements_aqt_torch_function(torch.nn.functional.linear)
def functional_linear(*args, **kwargs):
    input_tensor, weight_qtensor, bias = (
        args[0],
        args[1],
        args[2] if len(args) > 2 else None,
    )
    is_cuda = weight_qtensor.is_cuda
    is_cpu = weight_qtensor.device == torch.device("cpu")
    if isinstance(weight_qtensor, AffineQuantizedTensor):
        weight_is_int8 = _aqt_is_int8(weight_qtensor)
        weight_is_uint4 = _aqt_is_uint4(weight_qtensor)

        if isinstance(input_tensor, AffineQuantizedTensor):
            # if input tensor is quantized, either dispatch to the int8 mm kernel
            # or just dequantize the input tensor
            input_is_int8 = _aqt_is_int8_reduced_range(input_tensor)
            input_tensor_dtype_is_expected = input_tensor.dtype in [
                torch.float,
                torch.bfloat16
            ]
            if (
                is_cuda and
                input_is_int8 and
                input_tensor_dtype_is_expected
            ):
                #
                # 1. do the matrix form of dot(X_i, W_j)
                #
                #
                # 2. rescale the output
                #
                # in cases with large matrices, y_dot_int32 can grow sufficiently
                # large that y_dot_int32 * a float16 scale is greater than the maximum
                # value of a float 16, (which results in a value of inf even if multiplying
                # by the other scale would bring it within the expected range)

                x_vals_int8 = input_tensor.int_data
                x_scales = input_tensor.scale
                w_vals_int8_t = weight_qtensor.int_data.contiguous().t()
                w_scales = weight_qtensor.scale
                tmp = x_vals_int8.reshape(-1, x_vals_int8.shape[-1])
                y_dot_scaled = int_scaled_matmul(tmp, w_vals_int8_t, x_scales.reshape(-1, 1))

                y = (y_dot_scaled * w_scales).reshape(
                    *x_vals_int8.shape[:-1], y_dot_scaled.shape[-1]
                )

                # can downcast only at the very end
                output_dtype = input_tensor.dtype
                y = y.to(output_dtype)
                if bias is not None:
                    y += bias
                return y
            else:
                input_tensor = input_tensor.dequantize()

        # weight only quantization
        # TODO: enable cpu and mps path as well
        # TODO: make sure weight dimension matches the expectation of the int4mm kernel
        # TODO: move this to TinygemmAffineQuantizedTensor
        if (
            is_cuda and
            weight_is_uint4 and
            weight_qtensor.dtype == torch.bfloat16 and
            len(weight_qtensor.shape) == 2 and
            weight_qtensor.block_size[0] == 1 and
            weight_qtensor.zero_point_domain == ZeroPointDomain.FLOAT
        ):
            # groupwise int4 quantization
            # TODO: currently doing packing on the fly, we'll need to figure out
            # the API to do packing before hand
            # TODO: expose the arg
            innerKTiles = 8
            packed_weight = torch.ops.aten._convert_weight_to_int4pack(weight_qtensor.int_data.to(torch.int32), innerKTiles)
            scales_and_zeros = pack_tinygemm_scales_and_zeros(weight_qtensor.scale, weight_qtensor.zero_point)
            groupsize = weight_qtensor.block_size[-1]
            return torch.ops.aten._weight_int4pack_mm(input_tensor.contiguous(), packed_weight, groupsize, scales_and_zeros)
        elif (
            is_cpu and
            weight_is_int8 and
            len(weight_qtensor.shape) == 2 and
            len(weight_qtensor.block_size) == 2 and
            weight_qtensor.block_size[0] == 1 and
            weight_qtensor.block_size[1] == weight_qtensor.shape[1]
        ):
            # TODO: enable mps path as well
            # per channel int8 weight only quantizated mm
            return torch.ops.aten._weight_int8pack_mm(input_tensor.contiguous(), weight_qtensor.int_data, weight_qtensor.scale)
        else:
            weight_tensor = weight_qtensor.dequantize()
            return torch.nn.functional.linear(input_tensor, weight_tensor, bias)
    else:
        if isinstance(input_tensor, AffineQuantizedTensor):
            input_tensor = input_tensor.dequantize()
        return torch.nn.functional.linear(input_tensor, weight_tensor, bias)


@implements_aqt_aten_ops([aten.mm.default, aten.addmm.default])
def aten_mm(func, *args, **kwargs):
    if not args[0].is_floating_point():
        raise NotImplementedError(f"{func} is not implemented for non floating point input")

    if func == aten.addmm.default:
        assert args[1].shape[-1] == args[2].shape[0], (
            f"need mat1 shape: {args[1].shape} final"
            f"dim to match mat2 shape: {args[2].shape} first dim "
        )
        input_tensor, weight_qtensor, bias = (
            args[1],
            args[2],
            args[0],
        )
    else:
        assert args[0].shape[-1] == args[1].shape[0], (
            f"need mat1 shape: {args[0].shape} final dim"
            f"to match mat2 shape: {args[1].shape} first dim"
        )
        input_tensor, weight_qtensor, bias = (
            args[0],
            args[1],
            None if len(args) == 2 else args[2],
        )
    weight_tensor = weight_qtensor.dequantize()
    return func(input_tensor, weight_tensor, bias)

@implements_aqt_aten_ops([aten.detach.default])
def detach(func, *args, **kwargs):
    return return_and_correct_aliasing(
        func, args, kwargs, args[0]._apply_fn_to_data(torch.detach)
    )


@implements_aqt_aten_ops([aten.clone.default])
def clone(func, *args, **kwargs):
    return return_and_correct_aliasing(
        func, args, kwargs, args[0]._apply_fn_to_data(torch.clone)
    )


@implements_aqt_aten_ops([aten._to_copy.default])
def _to_copy(func, *args, **kwargs):
    return return_and_correct_aliasing(
        func,
        args,
        kwargs,
        args[0].to(*args[1:], **kwargs)._apply_fn_to_data(torch.clone),
    )

@implements_aqt_aten_ops([aten.t.default])
def t(func, *args, **kwargs):
    # TODO: need to implement this
    # args[0].transposed = not args[0].transposed
    # new = args[0]._change_shape(args[0].shape[::-1])
    # return return_and_correct_aliasing(func, args, kwargs, new)
    raise Exception("transpose not implemented yet")


class LinearActQuantizedTensor(torch.Tensor):
    """
    Applies activation quantization for linear operator
    """
    def __new__(
        cls,
        original_weight_tensor: torch.Tensor,
        input_quant_func: Callable,
    ):
        kwargs = {}
        dtype = original_weight_tensor.dtype
        kwargs["dtype"] = dtype
        kwargs["requires_grad"] = False
        shape = original_weight_tensor.shape
        return torch.Tensor._make_wrapper_subclass(cls, shape, **kwargs)  # type: ignore[attr-defined]

    def __init__(
        self,
        original_weight_tensor: torch.Tensor,
        input_quant_func: Callable,
    ):
        self.original_weight_tensor = original_weight_tensor
        self.input_quant_func = input_quant_func

    def __tensor_flatten__(self):
        return ["original_weight_tensor"], [self.input_quant_func]

    @classmethod
    def __tensor_unflatten__(
        cls, tensor_data_dict, tensor_attributes, outer_size, outer_stride
    ):
        original_weight_tensor = tensor_data_dict["original_weight_tensor"]
        input_quant_func = tensor_attributes
        return cls(
            original_weight_tensor,
            input_quant_func,
        )

    @classmethod
    def from_float(
        cls,
        input_float,
        input_quant_func,
    ):
        return cls(
            input_float,
            input_quant_func,
        )

    @classmethod
    def __torch_function__(cls, func, types, args=(), kwargs=None):
        kwargs = {} if kwargs is None else kwargs

        if func is torch.nn.functional.linear:
            input_tensor, weight_tensor, bias = (
                args[0],
                args[1],
                args[2] if len(args) > 2 else None,
            )
            if isinstance(weight_tensor, LinearActQuantizedTensor):
                input_quant_func = weight_tensor.input_quant_func
                original_weight_tensor = weight_tensor.original_weight_tensor
                aqt = input_quant_func(input_tensor)
                return torch.nn.functional.linear(aqt, original_weight_tensor, bias)

        with torch._C.DisableTorchFunctionSubclass():
            return func(*args, **kwargs)

    def _apply_fn_to_data(self, fn):
        return self.__class__(
            fn(self.original_weight_tensor),
            self.input_quant_func,
        )

    def __torch_dispatch__(cls, func, types, args, kwargs):
        if (
            func in [aten.mm.default, aten.addmm.default]
            and args[0].is_floating_point()
        ):
            if func == aten.addmm.default:
                assert args[1].shape[-1] == args[2].shape[0], (
                    f"need mat1 shape: {args[1].shape} final"
                    f"dim to match mat2 shape: {args[2].shape} first dim "
                )
                input_tensor, weight_qtensor, bias = (
                    args[1],
                    args[2],
                    args[0],
                )
                aqt = self.input_quant_func(input_tensor)
                return func(bias, aqt, weight_tensor)
            else:
                assert args[0].shape[-1] == args[1].shape[0], (
                    f"need mat1 shape: {args[0].shape} final dim"
                    f"to match mat2 shape: {args[1].shape} first dim"
                )
                input_tensor, weight_qtensor, bias = (
                    args[0],
                    args[1],
                    None if len(args) == 2 else args[2],
                )
                aqt = self.input_quant_func(input_tensor)
                return func(aqt, weight_tensor, bias)

        if func is aten.detach.default:
            return return_and_correct_aliasing(
                func, args, kwargs, args[0]._apply_fn_to_data(torch.detach)
            )

        if func is aten.clone.default:
            return return_and_correct_aliasing(
                func, args, kwargs, args[0]._apply_fn_to_data(torch.clone)
            )

        raise NotImplementedError(
            f"LinearActQuantizedTensor dispatch: attempting to run {func}, this is not supported"
        )
