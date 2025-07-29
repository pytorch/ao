# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD 3-Clause license found in the
# LICENSE file in the root directory of this source tree.
from typing import Tuple

import torch
from torch.utils._python_dispatch import return_and_correct_aliasing

from torchao.quantization.quant_primitives import _DTYPE_TO_QVALUE_BOUNDS
from torchao.utils import (
    TORCH_VERSION_AT_LEAST_2_5,
    TorchAOBaseTensor,
)

aten = torch.ops.aten


class Int8DynamicActivationLutTensor(TorchAOBaseTensor):
    """
    Tensor subclass that applies int8 dynamic activation quantization with lookup table quantization

    Args:
        original_weight_tensor (torch.Tensor): The weight tensor to be wrapped.
        scale (torch.Tensor): The scale tensor to be applied to activation.
    """

    packed_weight: torch.Tensor
    original_shape: Tuple[int, int]
    weight_scale_group_size: int
    bit_width: int

    def __new__(
        cls,
        packed_weight: torch.Tensor,
        original_shape: Tuple[int, int],
        weight_scale_group_size: int,
        bit_width: int,
    ):
        kwargs = {}
        kwargs["dtype"] = torch.float32
        kwargs["requires_grad"] = False
        kwargs["device"] = packed_weight.device
        return torch.Tensor._make_wrapper_subclass(cls, original_shape, **kwargs)  # type: ignore[attr-defined]

    def __init__(
        self,
        packed_weight: torch.Tensor,
        original_shape: Tuple[int, int],
        weight_scale_group_size,
        bit_width: int,
    ):
        self.packed_weight = packed_weight
        self.original_shape = original_shape
        self.weight_scale_group_size = weight_scale_group_size
        self.bit_width = bit_width

    @classmethod
    def from_plain(
        cls,
        weight_indices: torch.Tensor,
        weight_luts: torch.Tensor,
        weight_scale: torch.Tensor,
        weight_scale_group_size: int,
        bias,
    ):
        if len(weight_luts.shape) == 1:
            weight_luts = weight_luts.unsqueeze(0)
        assert len(weight_luts.shape) == 2, (
            "Expected weight_luts to be 2D tensor.  Each row in the tensor is an LUT"
        )
        bit_width = {2**b: b for b in range(1, 5)}[weight_luts.shape[1]]

        int8_min, int8_max = _DTYPE_TO_QVALUE_BOUNDS[torch.int8]
        assert torch.all(weight_luts >= int8_min)
        assert torch.all(weight_luts <= int8_max)
        weight_luts = weight_luts.to(torch.int8)

        n, k = weight_indices.shape
        # assert n % 8 == 0, f"Expected n to be divisible by 8, but got n={n}"
        assert k % 16 == 0, f"Expected k to be divisible by 16, but got k={k}"
        assert torch.all(weight_indices >= 0)
        assert torch.all(weight_indices < 2**bit_width)

        weight_scale = weight_scale.reshape(-1)
        assert k % weight_scale_group_size == 0, (
            f"Expected k to be divisible by weight_scale_group_size, but got k={k} and weight_scale_group_size={weight_scale_group_size}"
        )
        assert weight_scale.shape == (n * (k // weight_scale_group_size),)

        if bias is not None:
            assert bias.shape == (n,)

        packed_weight = getattr(
            torch.ops.torchao, f"_pack_8bit_act_{bit_width}bit_weight_with_lut"
        )(
            weight_indices,
            weight_luts,
            weight_scale,
            weight_scale_group_size,
            bias,
            None,
        )
        return cls(packed_weight, (n, k), weight_scale_group_size, bit_width)

    def __repr__(self):
        return "Int8DynamicActivationLutTensor"

    def __tensor_flatten__(self):
        return ["packed_weight"], [
            self.original_shape,
            self.weight_scale_group_size,
            self.bit_width,
        ]

    @classmethod
    def __tensor_unflatten__(
        cls, tensor_data_dict, tensor_attributes, outer_size, outer_stride
    ):
        packed_weight = tensor_data_dict["packed_weight"]
        original_shape, weight_scale_group_size, bitwidth = tensor_attributes
        return cls(packed_weight, original_shape, weight_scale_group_size, bitwidth)

    @staticmethod
    def _quantized_linear_op(
        input_tensor: torch.Tensor, weight_tensor: torch.Tensor, bias: torch.Tensor
    ):
        def _impl_2d(
            input_tensor: torch.Tensor, weight_tensor: torch.Tensor, bias: torch.Tensor
        ):
            original_dtype = torch.float32
            if input_tensor.dtype != torch.float32:
                original_dtype = input_tensor.dtype
                input_tensor = input_tensor.to(torch.float32)

            assert input_tensor.dim() == 2
            m, k = input_tensor.shape
            n, k_ = weight_tensor.original_shape
            assert k == k_, (
                f"Incompatible input shape. Expected second dimension to be equal to {k_}, but got {k}"
            )
            assert bias is None, (
                "Expected bias to be None because it should be packed with the weight tensor"
            )
            out = getattr(
                torch.ops.torchao,
                f"_linear_8bit_act_{weight_tensor.bit_width}bit_weight",
            )(
                input_tensor,
                weight_tensor.packed_weight,
                weight_tensor.weight_scale_group_size,
                n,
                k,
            )

            if original_dtype != torch.float32:
                out = out.to(original_dtype)
            return out

        assert input_tensor.dim() >= 2
        if input_tensor.dim() == 2:
            res = _impl_2d(input_tensor, weight_tensor, bias)
        else:
            assert input_tensor.dim() >= 3
            lead_shape = input_tensor.shape[0:-2]
            m, k = input_tensor.shape[-2], input_tensor.shape[-1]
            res = _impl_2d(input_tensor.reshape(-1, k), weight_tensor, bias)
            res = res.reshape(*lead_shape, m, -1)

        return res

    def _apply_fn_to_data(self, fn):
        return self.__class__(
            fn(self.packed_weight),
            self.original_shape,
            self.weight_scale_group_size,
            self.bit_width,
        )

    def to(self, *args, **kwargs):
        kwargs = self._get_to_kwargs(*args, **kwargs)
        device = kwargs.pop("device")
        return self.__class__(
            self.packed_weight.to(device),
            self.original_shape,
            self.weight_scale_group_size,
            self.bit_width,
        )


implements = Int8DynamicActivationLutTensor.implements


@implements(torch.nn.functional.linear)
def _(func, types, args, kwargs):
    input_tensor, weight_tensor, bias = (
        args[0],
        args[1],
        args[2] if len(args) > 2 else None,
    )
    if isinstance(weight_tensor, Int8DynamicActivationLutTensor):
        return weight_tensor._quantized_linear_op(input_tensor, weight_tensor, bias)

    raise NotImplementedError(
        "Int8DynamicActivationLutTensor: No specialized dispatch found for linear op"
    )


@implements(aten.detach.default)
def _(func, types, args, kwargs):
    return return_and_correct_aliasing(
        func, args, kwargs, args[0]._apply_fn_to_data(torch.detach)
    )


@implements(aten.clone.default)
def _(func, types, args, kwargs):
    return return_and_correct_aliasing(
        func, args, kwargs, args[0]._apply_fn_to_data(torch.clone)
    )


@implements(aten._to_copy.default)
def _(func, types, args, kwargs):
    return return_and_correct_aliasing(
        func,
        args,
        kwargs,
        args[0].to(*args[1:], **kwargs)._apply_fn_to_data(torch.clone),
    )


if TORCH_VERSION_AT_LEAST_2_5:
    # Allow a model with Int8DynamicActivationLutTensor weights to be loaded with `weights_only=True`
    torch.serialization.add_safe_globals([Int8DynamicActivationLutTensor])
