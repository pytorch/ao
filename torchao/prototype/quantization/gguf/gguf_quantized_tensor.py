# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from dataclasses import dataclass
from typing import Optional

import torch

from torchao.core.config import AOBaseConfig
from torchao.quantization.quant_primitives import (
    choose_qparams_gguf,
    dequantize_gguf,
    quantize_gguf,
)
from torchao.quantization.transform_module import register_quantize_module_handler
from torchao.utils import TorchAOBaseTensor

_QK_K = 256

__all__ = [
    "GGUFQuantizedTensor",
    "choose_qparams_gguf",
    "quantize_gguf",
    "dequantize_gguf",
    "GGUFWeightOnlyConfig",
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
        n_super_blocks,
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
        n_super_blocks,
        super_block_scale_scale,
        super_block_min_scale,
        quantized_block_scale,
        quantized_block_min,
        int_data,
        shape,
        **kwargs,
    ):
        self.n_super_blocks = n_super_blocks
        self.super_block_scale_scale = super_block_scale_scale
        self.super_block_min_scale = super_block_min_scale
        self.quantized_block_scale = quantized_block_scale
        self.quantized_block_min = quantized_block_min
        self.int_data = int_data

    def _apply_fn_to_data(self, fn):
        return self.__class__(
            self.n_super_blocks,
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
            self.n_super_blocks,
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
        n_super_blocks, dtype, shape = attributes
        return cls(
            n_super_blocks,
            super_block_scale_scale,
            super_block_min_scale,
            quantized_block_scale,
            quantized_block_min,
            int_data,
            shape if outer_size is None else outer_size,
            dtype=dtype,
        )

    def dequantize(self, output_dtype: Optional[torch.dtype] = None) -> torch.Tensor:
        block_size = tuple(
            [1] * (self.int_data.ndim - 1) + [_QK_K // self.n_super_blocks]
        )
        return dequantize_gguf(
            self.int_data,
            block_size,
            self.dtype,
            self.super_block_scale_scale,
            self.super_block_min_scale,
            self.quantized_block_scale,
            self.quantized_block_min,
        )

    def detach(self):
        """
        Returns a new `CodebookQuantizedTensor`.
        """
        return self.__class__(
            self.n_super_blocks,
            self.super_block_scale_scale.detach(),
            self.super_block_min_scale.detach(),
            self.quantized_block_scale.detach(),
            self.quantized_block_min.detach(),
            self.int_data.detach(),
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
    def from_float(cls, input_float, n_super_blocks, target_dtype):
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
        block_size = (1, _QK_K // n_super_blocks)
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
            n_super_blocks,
            super_block_scale_scale,
            super_block_min_scale,
            quantized_block_scale,
            quantized_block_min,
            int_data,
            input_float.shape,
            dtype=torch.float16,
        )


@dataclass
class GGUFWeightOnlyConfig(AOBaseConfig):
    dtype: torch.dtype = torch.uint4
    n_super_blocks: int = 8


@register_quantize_module_handler(GGUFWeightOnlyConfig)
def _gguf_weight_only_transform(
    module: torch.nn.Module,
    config: GGUFWeightOnlyConfig,
):
    """
    Applies gguf weight-only quantization to linear layers.

    Args:
        dtype: torch.uint1 to torch.uint8, torch.int32 supported.
        n_super_blocks: the number of super blocks in a 256 element block for gguf, e.g. when it is 8
            it means we have blocks of 32 and 8 blocks in a superblock of 256 elements.
    Returns:
        Callable for quantization transformation.
    """
    weight = module.weight
    if (weight.ndim != 2) or (weight.shape[-1] % 256 != 0):
        return module

    quantized_weight = GGUFQuantizedTensor.from_float(
        weight, n_super_blocks=config.n_super_blocks, target_dtype=config.dtype
    )
    module.weight = torch.nn.Parameter(quantized_weight, requires_grad=False)
    return module
