# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from typing import Optional

import torch
from torch.utils._python_dispatch import return_and_correct_aliasing

from torchao.quantization.quant_primitives import (
    choose_qparams_gguf,
    dequantize_gguf,
    quantize_gguf,
)
from torchao.utils import (
    TORCH_VERSION_AT_LEAST_2_5,
    TorchAOBaseTensor,
)

_QK_K = 256
aten = torch.ops.aten

__all__ = [
    "GGUFQuantizedTensor",
]


class GGUFQuantizedTensor(TorchAOBaseTensor):
    """
    A Tensor subclass that when applied to a weight used in a linear op/module,
    changes that linear op to a weight-only int4 quantized linear op with groupwise
    affine quantization on the weight.
    """

    @staticmethod
    def __new__(
        cls,
        n_blocks_per_superblock,
        super_block_scale_scale,
        super_block_min_scale,
        quantized_block_scale,
        quantized_block_min,
        int_data,
        shape,
        **kwargs,
    ):
        kwargs["device"] = kwargs.get("device", super_block_scale_scale.device)
        kwargs["dtype"] = kwargs.get("dtype", super_block_scale_scale.dtype)
        kwargs["requires_grad"] = False
        return torch.Tensor._make_wrapper_subclass(cls, shape, **kwargs)  # type: ignore[attr-defined]

    def __init__(
        self,
        n_blocks_per_superblock,
        super_block_scale_scale,
        super_block_min_scale,
        quantized_block_scale,
        quantized_block_min,
        int_data,
        shape,
        **kwargs,
    ):
        self.n_blocks_per_superblock = n_blocks_per_superblock
        self.super_block_scale_scale = super_block_scale_scale
        self.super_block_min_scale = super_block_min_scale
        self.quantized_block_scale = quantized_block_scale
        self.quantized_block_min = quantized_block_min
        self.int_data = int_data

    def _apply_fn_to_data(self, fn):
        return self.__class__(
            self.n_blocks_per_superblock,
            fn(self.super_block_scale_scale),
            fn(self.super_block_min_sclae),
            fn(self.quantized_block_scale),
            fn(self.quantized_block_min),
            fn(self.int_data),
            self.shape,
            dtype=self.dtype,
        )

    def __tensor_flatten__(self):
        return [
            "super_block_scale_scale",
            "super_block_min_scale",
            "quantized_block_scale",
            "quantized_block_min",
            "int_data",
        ], (
            self.n_blocks_per_superblock,
            self.dtype,
            self.shape,
        )

    @classmethod
    def __tensor_unflatten__(
        cls, tensor_data_dict, attributes, outer_size=None, outer_stride=None
    ):
        (
            super_block_scale_scale,
            super_block_min_scale,
            quantized_block_scale,
            quantized_block_min,
            int_data,
        ) = (
            tensor_data_dict["super_block_scale_scale"],
            tensor_data_dict["super_block_min_scale"],
            tensor_data_dict["quantized_block_scale"],
            tensor_data_dict["quantized_block_min"],
            tensor_data_dict["int_data"],
        )
        n_blocks_per_superblock, dtype, shape = attributes
        return cls(
            n_blocks_per_superblock,
            super_block_scale_scale,
            super_block_min_scale,
            quantized_block_scale,
            quantized_block_min,
            int_data,
            shape if outer_size is None else outer_size,
            dtype=dtype,
        )

    def dequantize(self, output_dtype: Optional[torch.dtype] = None) -> torch.Tensor:
        if output_dtype is None:
            output_dtype = self.dtype

        block_size = tuple(
            [1] * (self.int_data.ndim - 1) + [_QK_K // self.n_blocks_per_superblock]
        )
        return dequantize_gguf(
            self.int_data,
            block_size,
            self.dtype,
            self.super_block_scale_scale,
            self.super_block_min_scale,
            self.quantized_block_scale,
            self.quantized_block_min,
            output_dtype=output_dtype,
        )

    def to(self, *args, **kwargs):
        kwargs = self._get_to_kwargs(*args, **kwargs)
        device = kwargs.pop("device")
        return self.__class__(
            self.n_blocks_per_superblock,
            self.super_block_scale_scale.to(device),
            self.super_block_min_scale.to(device),
            self.quantized_block_scale.to(device),
            self.quantized_block_min.to(device),
            self.int_data.to(device),
            self.shape,
            **kwargs,
        )

    def _apply_fn_to_data(self, fn):
        """
        Returns a new `CodebookQuantizedTensor`.
        """
        return self.__class__(
            self.n_blocks_per_superblock,
            fn(self.super_block_scale_scale),
            fn(self.super_block_min_scale),
            fn(self.quantized_block_scale),
            fn(self.quantized_block_min),
            fn(self.int_data),
            self.shape,
            dtype=self.dtype,
        )

    def requires_grad_(self, requires_grad=False):
        """
        Modifies the tensor's `requires_grad` status in-place.
        """
        assert not requires_grad, "Only requires_grad == False is supported"
        return self

    @classmethod
    def from_float(cls, input_float, n_blocks_per_superblock, target_dtype):
        """
        Method used to convert a linear weight tensor to an instance of the
        GGMLInt4LinearWeight subclass.

        Example usage::

            model.lin_mod.weight = (
                GGMLInt4LinearWeight.from_float(model.lin_mod.weight)
            )
        """
        assert (
            target_dtype == torch.uint4
        ), "only uint4 quantization is supported right now"
        block_size = (1, _QK_K // n_blocks_per_superblock)
        (
            super_block_scale_scale,
            super_block_min_scale,
            quantized_block_scale,
            quantized_block_min,
        ) = choose_qparams_gguf(input_float, block_size, target_dtype)

        int_data = quantize_gguf(
            input_float,
            block_size,
            target_dtype,
            super_block_scale_scale,
            super_block_min_scale,
            quantized_block_scale,
            quantized_block_min,
        )
        return cls(
            n_blocks_per_superblock,
            super_block_scale_scale,
            super_block_min_scale,
            quantized_block_scale,
            quantized_block_min,
            int_data,
            input_float.shape,
        )


implements = GGUFQuantizedTensor.implements


@implements([aten.detach.default, aten.alias.default])
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


@implements([torch.nn.functional.linear, aten.linear.default])
def _(func, types, args, kwargs):
    input_tensor, weight_tensor, bias = (
        args[0],
        args[1],
        args[2] if len(args) > 2 else None,
    )
    if not input_tensor.is_floating_point():
        raise NotImplementedError(
            f"{func} is not implemented for non floating point input"
        )

    dtype = input_tensor.dtype

    if hasattr(weight_tensor, "dequantize"):
        weight_tensor = weight_tensor.dequantize(output_dtype=dtype)

    return torch.nn.functional.linear(input_tensor, weight_tensor, bias)


if TORCH_VERSION_AT_LEAST_2_5:
    # Allow a model with GGUFQuantizedTensor weights to be loaded with `weights_only=True`
    torch.serialization.add_safe_globals([GGUFQuantizedTensor])
