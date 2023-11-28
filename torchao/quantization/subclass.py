# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import torch
from .quant_primitives import (
    dequantize_per_channel,
    dynamically_quantize_per_channel,
    quant_int8_dynamic_per_token_linear,
)
from torch.utils._python_dispatch import return_and_correct_aliasing

__all__ = [
    "DynamicallyQuantizedLinearWeight",
    "WeightOnlyQuantizedLinearWeight"
]


class Int8QuantizedLinearWeightBase(torch.Tensor):
    """
    Base Quantized Tensor subclass for int8 quantized Linear weights. The weight
    is quantized symmetrically per-channel. When the float_float method is used,
    to create an instance of any Int8QuantizedLinearWeightBase, we assume the input
    weight is oriented the way it is in a normal linear op, i.e. out-channels x in-channels.
    Subclasses which inherit from this class need to implement the _quantized_op method.
    """
    @staticmethod
    def __new__(cls, int_data, q_scales, transposed=False, **kwargs):
        # The `transposed` argument indicates that the int_data (attribute or argument)
        # is transposed compared to how we'd like the external representation
        # of the shape to be.
        # This is needed so we don't have to mutate the int_data when it gets
        # transposed/detached, instead we can just pass the int_data to the
        # new instance and alter the transposed flag where needed.
        kwargs["device"] = int_data.device
        kwargs["dtype"] = kwargs.get("dtype", q_scales.dtype)
        size = int_data.shape[::-1] if transposed else int_data.shape
        kwargs["layout"] = (
            kwargs.get("layout") if kwargs.get("layout", False) else int_data.layout
        )
        assert not kwargs.get("requires_grad", False)
        kwargs["requires_grad"] = False
        return torch.Tensor._make_wrapper_subclass(cls, size, **kwargs)  # type: ignore[attr-defined]

    def __init__(self, int_data, q_scales, transposed=False):
        self._transposed = transposed
        self.int_data = int_data
        self.q_scales = q_scales

    @staticmethod
    def _quantized_op(act_mat, int_w_mat, q_scales, bias):
        pass

    def __repr__(self):
        return (
            f"{self.__class__.__name__}(data={self.dequantize()}, shape={self.shape}, "
            f"device={self.device}, dtype={self.dtype}, requires_grad={self.requires_grad})"
        )

    def dequantize(self, dtype=None):
        """
        Obtain the dequantized version of the quantized tensor subclass
        """
        dq_t = dequantize_per_channel(
            self.int_data.t(), self.q_scales, 0, self.dtype if dtype is None else dtype
        )
        # note: data was already transposed to calculate out
        return dq_t if self._transposed else dq_t.t()

    def int_repr(self):
        """
        Get the internal integer representation of the quantized tensor
        """
        return self.int_data.t() if self._transposed else self.int_data

    def q_scales(self):
        """
        Get the quantization scales for the quantized tensor
        """
        return self.q_scales

    def _detach(self):
        return self.__class__(
            self.int_data, self.q_scales, transposed=self._transposed
        )

    def _transpose(self):
        return self.__class__(
            self.int_data, self.q_scales, transposed=(not self._transposed)
        )

    def __tensor_flatten__(self):
        return ["int_data", "q_scales"], self._transposed

    @classmethod
    def __tensor_unflatten__(cls, tensor_data, transposed):
        int_data, q_scales = tensor_data["int_data"], tensor_data["q_scales"]
        return cls(
            int_data, q_scales, transposed=transposed
        )

    __torch_function__ = torch._C._disabled_torch_function_impl

    @classmethod
    def __torch_dispatch__(cls, func, types, args, kwargs):
        # two scenarios where we currently fall back to vanilla mm:
        # 1 - when tensor is on CPU: we are missing qmm for CPU, but we should have a CPU implementation
        #     for consistency and to allow people to test
        # 2 - we're given non-floats - quantizing long to int8 is crazy
        if (
            func in [torch.ops.aten.mm.default, torch.ops.aten.addmm.default]
            and args[0].is_floating_point()
            and args[0].is_cuda
        ):
            if func == torch.ops.aten.addmm.default:
                assert (
                    args[1].shape[-1] == args[2].shape[0]
                ), (
                    f"need mat1 shape: {args[1].shape} final"
                    f"dim to match mat2 shape: {args[2].shape} first dim "
                )
                mat1, mat2, q_scales, bias = (
                    args[1],
                    args[2].int_data,
                    args[2].q_scales,
                    args[0],
                )
            else:
                assert (
                    args[0].shape[-1] == args[1].shape[0]
                ), (
                    f"need mat1 shape: {args[0].shape} final dim"
                    f"to match mat2 shape: {args[1].shape} first dim"
                )
                mat1, mat2, q_scales, bias = (
                    args[0],
                    args[1].int_data,
                    args[1].q_scales,
                    None,
                )
            # call the quantized op for the specific type
            # of quantized tensor subclass
            return cls._quantized_op(
                mat1, mat2, q_scales, bias
            )

        if func is torch.ops.aten.detach.default:
            return return_and_correct_aliasing(
                func, args, kwargs, args[0]._detach()
            )

        if func is torch.ops.aten.t.default:
            return return_and_correct_aliasing(
                func, args, kwargs, args[0]._transpose()
            )

    @classmethod
    def from_float(cls, input_float, qmin=-128, qmax=127, dtype=torch.int8):
        """
        Method used to convert a linear weight tensor to an instance of the
        desired Tensor subclass.

        Example usage::

            model.lin_mod.weight = DynamicallyQuantizedLinearWeight.from_float(model.lin_mod.weight)
        """
        w_int_repr, w_scales, _ = dynamically_quantize_per_channel(
            input_float, qmin, qmax, dtype
        )
        # the desired representation shape for fast quantized matmul is
        # transposed compared to how it's stored as a linear weight,
        # i.e. we want in_channels is dim=0 and out_channels (and quantized axis) is dim=1
        return cls(w_int_repr.contiguous().t(), w_scales, transposed=True)


class DynamicallyQuantizedLinearWeight(Int8QuantizedLinearWeightBase, torch.Tensor):
    """
    A Tensor subclass that when applied to a weight used in a linear op/module, changes the
    linear op to a dynamically quantized linear op with symmetric per-token and per-channel
    quantization on the activation and weight respectively.
    """
    @staticmethod
    def _quantized_op(act_mat, int_w_mat, q_scales, bias):
        return quant_int8_dynamic_per_token_linear(
            act_mat, int_w_mat, q_scales, bias, act_mat.dtype
        )


class WeightOnlyQuantizedLinearWeight(Int8QuantizedLinearWeightBase, torch.Tensor):
    """
    A Tensor subclass that when applied to a weight used in a linear op/module,
    changes the linear op to a weight-only quantized linear op with symmetric
    per-channel quantization on the weight.
    """
    @staticmethod
    def _quantized_op(act_mat, int_w_mat, q_scales, bias):
        act_mat = act_mat.view(-1, act_mat.shape[-1])
        y = torch.mm(act_mat, int_w_mat.to(act_mat.dtype)) * q_scales
        y = y.reshape(*act_mat.shape[:-1], -1)
        if bias is not None:
            y += bias
        return y
