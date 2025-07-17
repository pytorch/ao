# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.


# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import logging
from typing import Any, Callable, Dict, Optional, Type

import torch
import torch.nn as nn
import torch.nn.functional as F

from torchao.dtypes.utils import is_device
from torchao.utils import (
    TORCH_VERSION_AT_LEAST_2_6,
    find_multiple,
)

from .quant_primitives import (
    MappingType,
    dequantize_affine,
)
from .unified import Quantizer
from .utils import (
    group_quantize_tensor_symmetric,
    groupwise_affine_quantize_tensor,
    per_token_dynamic_quant,
)

aten = torch.ops.aten

__all__ = [
    "WeightOnlyInt4Linear",
    "Int4WeightOnlyQuantizer",
    "Int8DynActInt4WeightQuantizer",
]


def _check_linear_int4_k(k, groupsize=1, inner_k_tiles=None):
    k_divisible_by_groupsize = k % groupsize == 0
    if inner_k_tiles is not None:
        k_divisible_by_16_times_inner_k_tiles = k % (inner_k_tiles * 16) == 0
        return k_divisible_by_groupsize and k_divisible_by_16_times_inner_k_tiles
    return k_divisible_by_groupsize


def linear_forward_int4(
    x: torch.Tensor,
    weight_int4pack: torch.Tensor,
    scales_and_zeros: torch.Tensor,
    out_features: int,
    groupsize: int,
    precision: torch.dtype = torch.bfloat16,
    scales_precision: torch.dtype = torch.bfloat16,
):
    origin_x_size = x.size()
    x = x.reshape(-1, origin_x_size[-1])
    if is_device(x.device.type, "cpu") and TORCH_VERSION_AT_LEAST_2_6:
        c = torch.ops.aten._weight_int4pack_mm_for_cpu(
            x.to(precision),
            weight_int4pack,
            groupsize,
            scales_and_zeros.to(scales_precision),
        ).to(dtype=x.dtype)
    else:
        c = torch.ops.aten._weight_int4pack_mm(
            x.to(precision),
            weight_int4pack,
            groupsize,
            scales_and_zeros.to(scales_precision),
        ).to(dtype=x.dtype)
    new_shape = origin_x_size[:-1] + (out_features,)
    c = c.reshape(new_shape)
    return c


class WeightOnlyInt4Linear(torch.nn.Module):
    __constants__ = ["in_features", "out_features"]
    in_features: int
    out_features: int
    weight: torch.Tensor

    def __init__(
        self,
        in_features: int,
        out_features: int,
        # TODO: remove dtype field, not used
        bias=False,
        device=None,
        dtype=None,
        groupsize: int = 128,
        inner_k_tiles: int = 8,
        precision: torch.dtype = torch.bfloat16,
        scales_precision: torch.dtype = torch.bfloat16,
    ) -> None:
        super().__init__()
        self.padding = not _check_linear_int4_k(in_features, groupsize, inner_k_tiles)
        if self.padding:
            self.origin_in_features = in_features
            in_features = find_multiple(in_features, 1024)

        self.in_features = in_features
        self.out_features = out_features
        assert not bias, "require bias=False"
        self.device = device
        self.groupsize = groupsize
        self.inner_k_tiles = inner_k_tiles
        self.precision = precision
        self.scales_precision = scales_precision

        if dtype is not None:
            raise ValueError("Please specify 'precision' instead of 'dtype'")

        assert out_features % 8 == 0, "require out_features % 8 == 0"
        assert in_features % (inner_k_tiles * 16) == 0, (
            "require in_features % (innerKTiles * 16) == 0"
        )
        if is_device(device.type, "cpu"):
            self.register_buffer(
                "weight",
                torch.zeros(
                    (
                        out_features,
                        in_features // 2,
                    ),
                    dtype=torch.uint8,
                    device=device,
                ),
            )
        else:
            self.register_buffer(
                "weight",
                torch.zeros(
                    (
                        out_features // 8,
                        in_features // (inner_k_tiles * 16),
                        32,
                        inner_k_tiles // 2,
                    ),
                    dtype=torch.int32,
                    device=device,
                ),
            )
        self.dtype = dtype
        self.register_buffer(
            "scales_and_zeros",
            torch.zeros(
                (in_features // groupsize, out_features, 2),
                dtype=self.scales_precision,
                device=device,
            ),
        )

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        if self.padding:
            input = F.pad(input, pad=(0, self.in_features - self.origin_in_features))
        return linear_forward_int4(
            input,
            self.weight,
            self.scales_and_zeros,
            self.out_features,
            self.groupsize,
            self.precision,
            self.scales_precision,
        )


def _replace_linear_int4(
    module: torch.nn.Module,
    groupsize: int,
    inner_k_tiles: Optional[int],
    padding_allowed: bool,
    skip_layer_func: Optional[Callable] = None,
    precision: torch.dtype = torch.bfloat16,
    scales_precision: torch.dtype = torch.bfloat16,
    linear_class: Type[torch.nn.Module] = WeightOnlyInt4Linear,
    copy_weights: bool = False,
):
    for name, child in module.named_children():
        # TODO: support linear bias
        if (
            isinstance(child, nn.Linear)
            and child.bias is None
            and (skip_layer_func is None or not skip_layer_func(child.weight))
        ):
            if (
                _check_linear_int4_k(child.in_features, groupsize, inner_k_tiles)
                or padding_allowed
            ):
                new_linear = linear_class(
                    child.in_features,
                    child.out_features,
                    bias=False,
                    device=child.weight.device,
                    groupsize=groupsize,
                    inner_k_tiles=inner_k_tiles,
                    precision=precision,
                    scales_precision=scales_precision,
                )
                # TODO: merge with 8da4w?
                # In distributed training, the model may be instantiated
                # on the meta device, in which case there is no need to
                # copy the weights, and doing so will result in an error
                if copy_weights and child.weight.device != torch.device("meta"):
                    new_linear.weight = child.weight
                setattr(module, name, new_linear)
        else:
            _replace_linear_int4(
                child,
                groupsize,
                inner_k_tiles,
                padding_allowed,
                skip_layer_func,
                precision,
                scales_precision,
                linear_class,
                copy_weights,
            )


def replace_linear_int4(
    module, groupsize, inner_k_tiles, padding_allowed, skip_layer_func=None
):
    _replace_linear_int4(
        module,
        groupsize,
        inner_k_tiles,
        padding_allowed,
        skip_layer_func,
        linear_class=WeightOnlyInt4Linear,
    )


class Int4WeightOnlyQuantizer(Quantizer):
    def __init__(
        self,
        groupsize: int = 256,
        padding_allowed: bool = True,
        inner_k_tiles: Optional[int] = 8,
        device: torch.device = torch.device("cuda"),
        precision: torch.dtype = torch.bfloat16,
    ) -> None:
        super().__init__()
        assert inner_k_tiles in [2, 4, 8]
        assert groupsize in [32, 64, 128, 256]

        self.inner_k_tiles = inner_k_tiles
        self.groupsize: int = groupsize
        self.padding_allowed: bool = padding_allowed
        self.device: torch.device = device
        # precision and dtype are being used interchangeably here
        self.precision: torch.dtype = precision

    @torch.no_grad()
    def _create_quantized_state_dict(
        self, model: torch.nn.Module
    ) -> Dict[str, torch.Tensor]:
        cur_state_dict = model.state_dict()
        for fqn, mod in model.named_modules():
            if isinstance(mod, torch.nn.Linear) and mod.bias is None:
                out_features = mod.out_features
                in_features = mod.in_features
                # assert out_features % 8 == 0, "require out_features % 8 == 0"
                logging.info(f"linear: {fqn}, in={in_features}, out={out_features}")

                assert in_features % self.groupsize == 0, (
                    f"require in_features:{in_features} % self.groupsize:{self.groupsize} == 0"
                )

                weight = mod.weight.data
                if not _check_linear_int4_k(
                    in_features, self.groupsize, self.inner_k_tiles
                ):
                    if self.padding_allowed:
                        import torch.nn.functional as F

                        logging.warning(
                            f"warning: {fqn} is padded to satisfy in_features % 1024 == 0"
                        )
                        padded_in_features = find_multiple(in_features, 1024)
                        weight = F.pad(
                            weight, pad=(0, padded_in_features - in_features)
                        )
                    else:
                        logging.warning(
                            f"warning: {fqn} is skipped, int4 requires that in_features is 32, 64, or is divisible by 1024, "
                            + "and that groupsize and inner_k_tiles*16 evenly divide into it"
                        )
                        continue
                (w_int4x8, scales_and_zeros) = groupwise_affine_quantize_tensor(
                    weight,
                    4,  # n_bit
                    self.groupsize,
                    self.precision,  # dtype for scales_and_zeros
                )
                # TODO: just get the device from mod.weight.device?
                if (
                    is_device(w_int4x8.device.type, "cpu")
                    and TORCH_VERSION_AT_LEAST_2_6
                ):
                    weight_int4pack = (
                        torch.ops.aten._convert_weight_to_int4pack_for_cpu(
                            w_int4x8.to(self.device), self.inner_k_tiles
                        )
                    )
                else:
                    weight_int4pack = torch.ops.aten._convert_weight_to_int4pack(
                        w_int4x8.to(self.device), self.inner_k_tiles
                    )
                cur_state_dict[f"{fqn}.weight"] = weight_int4pack.to(self.device)
                cur_state_dict[f"{fqn}.scales_and_zeros"] = scales_and_zeros.to(
                    self.device
                )
        return cur_state_dict

    def _convert_for_runtime(self, model: torch.nn.Module) -> torch.nn.Module:
        _replace_linear_int4(
            model,
            self.groupsize,
            self.inner_k_tiles,
            self.padding_allowed,
            skip_layer_func=None,
            precision=self.precision,
            scales_precision=self.precision,
        )
        return model

    def quantize(
        self, model: torch.nn.Module, *args: Any, **kwargs: Any
    ) -> torch.nn.Module:
        state_dict = self._create_quantized_state_dict(model)
        model = self._convert_for_runtime(model)
        # TODO: make it strict
        model.load_state_dict(state_dict, strict=False)
        return model


def linear_forward_8da4w(
    x,
    weight_int8,
    bias,
    scales,
    zeros,
    out_features,
    groupsize,
    output_precision,
):
    # uses fp32 to match torchao.quantization.quant_api._int8_asymm_per_token_quant
    # and activation_scale_dtype in QAT configs
    # TODO: in future add ability to specify activation_scale_dtype to PTQ configs
    # and enable similar change here
    x = per_token_dynamic_quant(
        x,
        scale_dtype=torch.float32,
        zero_point_dtype=torch.float32,
        eps=torch.finfo(torch.float32).eps,
    )

    # TODO: verify and remove following reshape code
    # origin_x_size = x.size()
    # x = x.reshape(-1, origin_x_size[-1])

    # TODO: better API
    # weight_int8 = torch.ops.quantized_decomposed.unpack_int4_to_int8(weight_int4packed)
    n_bit = 4
    quant_min = -(2 ** (n_bit - 1))
    quant_max = 2 ** (n_bit - 1) - 1
    block_size = (1, groupsize)

    w_dq = dequantize_affine(
        weight_int8,
        block_size,
        scales,
        zeros,
        torch.int8,
        quant_min,
        quant_max,
        output_dtype=output_precision,
    )

    # x = x.to(torch.float16)
    # w_dq = w_dq.to(torch.float16)
    c = torch.nn.functional.linear(x, w_dq, bias)

    # new_shape = origin_x_size[:-1] + (out_features,)
    # c = c.reshape(new_shape)

    return c


class Int8DynActInt4WeightLinear(torch.nn.Module):
    __constants__ = ["in_features", "out_features"]

    in_features: int
    out_features: int
    weight: torch.Tensor
    bias: torch.Tensor

    """
    This module implements a dynamic quantized linear layer with int4 weight.
    Weights are per channel groupwise quantized. Parameters of importance
    groupsize: the number of elements in each quantized group
    precision: precision of input and output. e.g. torch.float32 means input
    activation is float32 and output is float32.
    scales_precision: precision of per group scale.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias=True,
        device=None,
        # TODO: remove this field, not used
        dtype=None,
        groupsize: int = 256,
        precision: torch.dtype = torch.float32,
        scales_precision: torch.dtype = torch.float32,
    ) -> None:
        super().__init__()
        # always pad if needed since it becomes a noop at runtime if not needed
        # self.origin_in_features = in_features
        assert in_features % groupsize == 0, (
            f"require in_features:{in_features} % groupsize:{groupsize} == 0"
        )
        # in_features = _calc_padded_size_linear_int4(
        #    in_features, groupsize
        # )
        self.in_features = in_features
        self.out_features = out_features
        # TODO: align groupsize naming
        self.groupsize = groupsize
        # Precision of the activation which also indicates
        # output precision of the dynamically quantized linear layer
        # that his module represents.
        self.precision = precision

        if dtype is not None:
            raise ValueError("Please specify 'precision' instead of 'dtype'")

        # currently storing unpacked int8 weights
        self.register_buffer(
            "weight",
            torch.zeros((out_features, in_features), dtype=torch.int8),
        )
        self.register_buffer(
            "scales",
            torch.zeros(
                (out_features, in_features // groupsize),
                dtype=scales_precision,
            ),
        )
        self.register_buffer(
            "zeros",
            torch.zeros(
                (out_features, in_features // groupsize),
                dtype=scales_precision,
            ),
        )

        if bias:
            self.register_buffer("bias", torch.zeros(out_features, dtype=precision))
        else:
            self.bias = None

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        input = input.to(self.precision)
        # padding is removed for perf
        # input = F.pad(input, pad=(0, self.in_features - self.origin_in_features))
        return linear_forward_8da4w(
            input,
            self.weight,
            self.bias,
            self.scales,
            self.zeros,
            self.out_features,
            self.groupsize,
            self.precision,
        )


def _replace_linear_8da4w(
    module: torch.nn.Module,
    groupsize: int,
    padding_allowed: bool,
    precision: torch.dtype,
    scales_precision: torch.dtype,
    linear_class: Type[torch.nn.Module],
    copy_weights: bool = False,
):
    # import the util function here to avoid circular dependency
    from torchao.quantization.quant_api import _replace_with_custom_fn_if_matches_filter

    def filter_fn(child: torch.nn.Module, cur_fqn: str) -> bool:
        return isinstance(child, nn.Linear) and (
            _check_linear_int4_k(child.in_features, groupsize) or padding_allowed
        )

    def replacement_fn(child: torch.nn.Module) -> torch.nn.Module:
        new_linear = linear_class(
            child.in_features,
            child.out_features,
            bias=child.bias is not None,
            device=child.weight.device,
            groupsize=groupsize,
            precision=precision,
            scales_precision=scales_precision,
        )
        # In distributed training, the model may be instantiated
        # on the meta device, in which case there is no need to
        # copy the weights, and doing so will result in an error
        if copy_weights and child.weight.device != torch.device("meta"):
            new_linear.weight = child.weight
            new_linear.bias = child.bias
        return new_linear

    _replace_with_custom_fn_if_matches_filter(module, replacement_fn, filter_fn)


def replace_linear_8da4w(
    module: torch.nn.Module,
    groupsize: int,
    padding_allowed: bool,
    precision: torch.dtype,
    scales_precision: torch.dtype,
):
    _replace_linear_8da4w(
        module,
        groupsize,
        padding_allowed,
        precision,
        scales_precision,
        Int8DynActInt4WeightLinear,
    )


class Int8DynActInt4WeightQuantizer(Quantizer):
    def __init__(
        self,
        groupsize: int = 256,
        padding_allowed: bool = False,
        precision: torch.dtype = torch.float32,
        scales_precision: torch.dtype = torch.float32,
        device: torch.device = torch.device("cpu"),
        mapping_type: MappingType = MappingType.SYMMETRIC,
    ) -> None:
        super().__init__()
        self.groupsize: int = groupsize
        self.padding_allowed: bool = padding_allowed
        self.precision: torch.dtype = precision
        self.scales_precision: torch.dtype = scales_precision
        self.device: torch.device = device
        self.mapping_type: MappingType = mapping_type

    @torch.no_grad()
    def _create_quantized_state_dict(
        self, model: torch.nn.Module
    ) -> Dict[str, torch.Tensor]:
        cur_state_dict = model.state_dict()
        for fqn, mod in model.named_modules():
            if isinstance(mod, torch.nn.Linear):
                out_features = mod.out_features
                in_features = mod.in_features
                # assert out_features % 8 == 0, "require out_features % 8 == 0"
                logging.info(f"linear: {fqn}, in={in_features}, out={out_features}")

                assert in_features % self.groupsize == 0, (
                    f"require in_features:{in_features} % self.groupsize:{self.groupsize} == 0"
                )

                weight = mod.weight.data
                if not _check_linear_int4_k(in_features, self.groupsize):
                    if self.padding_allowed:
                        import torch.nn.functional as F

                        logging.warning(
                            f"warning: {fqn} is padded to satisfy in_features % 1024 == 0"
                        )
                        padded_in_features = find_multiple(in_features, 1024)
                        weight = F.pad(
                            weight, pad=(0, padded_in_features - in_features)
                        )
                    else:
                        logging.warning(
                            f"warning: {fqn} is skipped, int4 requires that in_features is 32, 64, or is divisible by 1024, "
                            + "and that groupsize and inner_k_tiles*16 evenly divide into it"
                        )
                        continue
                (
                    weight_int8,
                    scales,
                    zeros,
                ) = group_quantize_tensor_symmetric(
                    weight.to(self.precision),
                    4,  # n_bit
                    self.groupsize,
                    self.scales_precision,
                    mapping_type=self.mapping_type,
                )
                cur_state_dict[f"{fqn}.weight"] = weight_int8.to(self.device)
                cur_state_dict[f"{fqn}.scales"] = scales.to(self.device)
                cur_state_dict[f"{fqn}.zeros"] = zeros.to(self.device)

        return cur_state_dict

    def _convert_for_runtime(self, model: torch.nn.Module) -> torch.nn.Module:
        replace_linear_8da4w(
            model,
            self.groupsize,
            self.padding_allowed,
            self.precision,
            # TODO: this should be self.scales_precision?
            self.precision,
        )
        return model

    def quantize(
        self, model: torch.nn.Module, *args: Any, **kwargs: Any
    ) -> torch.nn.Module:
        state_dict = self._create_quantized_state_dict(model)
        model = self._convert_for_runtime(model)
        # TODO: make it strict
        model.load_state_dict(state_dict, strict=False)
        return model
