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
    unpack_tinygemm_scales_and_zeros,
)
from .utils import find_multiple


__all__ = [
    "Int8DynamicallyQuantizedLinearWeight",
    "Int8WeightOnlyQuantizedLinearWeight",
    "Int4WeightOnlyQuantizedLinearWeight",
]


aten = torch.ops.aten


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


class Int8DynamicallyQuantizedLinearWeight(QuantizedLinearWeightBase):
    """
    A Tensor subclass that when applied to a weight used in a linear op/module, changes the
    linear op to a dynamically quantized linear op with symmetric per-token and per-channel
    quantization on the activation and weight respectively.
    """

    @staticmethod
    def __new__(cls, int_data, q_scales, transposed, shape, **kwargs):
        kwargs["dtype"] = kwargs.get("dtype", q_scales.dtype)
        return super().__new__(cls, int_data, transposed, shape, **kwargs)  # type: ignore[attr-defined]

    def __init__(self, int_data, q_scales, transposed, shape, **kwargs):

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
        dq_t = dequantize_per_channel(
            self.int_data.t(), self.q_scales, 0, self.dtype if dtype is None else dtype
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
        return ["int_data", "q_scales"], [self.transposed, self.dtype, self.shape]

    @classmethod
    def __tensor_unflatten__(
        cls, tensor_data_dict, tensor_attributes, outer_size=None, outer_stride=None
    ):
        int_data, q_scales = tensor_data_dict["int_data"], tensor_data_dict["q_scales"]
        transposed, dtype, shape = tensor_attributes
        return cls(
            int_data,
            q_scales,
            transposed,
            shape if outer_size is None else outer_size,
            dtype=dtype,
            strides=outer_stride,
        )

    @classmethod
    def from_float(cls, input_float, qmin=-128, qmax=127):
        """
        Method used to convert a linear weight tensor to an instance of the
        Int8DynamicallyQuantizedLinearWeight subclass.

        Example usage::

            model.lin_mod.weight = (
                Int8DynamicallyQuantizedLinearWeight.from_float(model.lin_mod.weight)
            )
        """
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
            int_data, w_scales, False, input_float.shape, dtype=input_float.dtype
        )


class Int8WeightOnlyQuantizedLinearWeight(Int8DynamicallyQuantizedLinearWeight):
    """
    A Tensor subclass that when applied to a weight used in a linear op/module,
    changes the linear op to a weight-only quantized linear op with symmetric
    per-channel quantization on the weight.
    """

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


class Int4WeightOnlyQuantizedLinearWeight(QuantizedLinearWeightBase):
    """
    A Tensor subclass that when applied to a weight used in a linear op/module,
    changes that linear op to a weight-only int4 quantized linear op with groupwise
    affine quantization on the weight.
    """

    @staticmethod
    def __new__(
        cls,
        int_data,
        scales_and_zeros,
        transposed,
        shape,
        groupsize=128,
        inner_k_tiles=8,
        **kwargs,
    ):
        kwargs["dtype"] = kwargs.get("dtype", scales_and_zeros.dtype)
        return super().__new__(cls, int_data, transposed, shape, **kwargs)  # type: ignore[attr-defined]

    def __init__(
        self,
        int_data,
        scales_and_zeros,
        transposed,
        shape,
        groupsize,
        inner_k_tiles,
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
            self.groupsize,
            self.inner_k_tiles,
            self.dtype,
            self.shape,
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
        transposed, groupsize, inner_k_tiles, dtype, shape = attributes
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
    def from_float(cls, input_float, groupsize=128, inner_k_tiles=8):
        """
        Method used to convert a linear weight tensor to an instance of the
        Int4WeightOnlyQuantizedLinearWeight subclass.

        Example usage::

            model.lin_mod.weight = (
                Int4WeightOnlyQuantizedLinearWeight.from_float(model.lin_mod.weight)
            )
        """
        assert groupsize in [256, 128, 64, 32]
        assert inner_k_tiles in [8, 4, 2]
        orig_shape = input_float.shape
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
            input_float, 4, groupsize
        )
        int_data = aten._convert_weight_to_int4pack(input_int4x8, inner_k_tiles)

        return cls(
            int_data,
            scales_and_zeros,
            False,
            orig_shape,
            groupsize,
            inner_k_tiles,
            dtype=input_float.dtype,
        )
