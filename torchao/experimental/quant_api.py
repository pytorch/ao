# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import logging
import sys
from dataclasses import dataclass
from enum import Enum
from typing import Optional, Union

import torch
import torch.nn as nn
from torch.ao.quantization.fx._decomposed import (
    quantize_per_channel_group,
)

from torchao.core.config import AOBaseConfig
from torchao.quantization.quant_primitives import (
    _choose_qparams_and_quantize_affine_hqq,
)
from torchao.quantization.transform_module import (
    register_quantize_module_handler,
)

logger = logging.getLogger(__name__)
logger.setLevel(logging.WARNING)


handler = logging.StreamHandler(sys.stdout)
formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
handler.setFormatter(formatter)
logger.addHandler(handler)


# can switch to StrEnum (https://docs.python.org/3/library/enum.html#enum.StrEnum)
# after python 3.10 is end of life (https://devguide.python.org/versions/)
class UIntxChooseQParamsAlgorithm(str, Enum):
    """Variant of quantization algorithm to calculate scale and zero_point for UIntx quantization."""

    """
    Default min-max quantization algorithm.
    Uses simple (max - min) / (qmax - qmin) scaling.
    """
    MIN_MAX = "min_max"

    """
    Half-Quadratic Quantization (HQQ) algorithm.
    Uses iterative optimization to find better scale/zero parameters.
    Typically produces better accuracy than min-max at the cost of slower quantization.
    See: https://mobiusml.github.io/hqq_blog/
    """
    HQQ = "hqq"


def _quantize(
    vals: torch.Tensor, group_size: int, nbit: int, has_weight_zeros: bool, signed=True
):
    assert nbit >= 1 and nbit <= 8
    if signed:
        qmin = -(1 << (nbit - 1))
        qmax = (1 << (nbit - 1)) - 1
    else:
        qmin = 0
        qmax = (1 << nbit) - 1

    n, k = vals.shape
    vals = vals.reshape(-1, group_size)
    vmins, _ = torch.min(vals, axis=1)
    vmaxs, _ = torch.max(vals, axis=1)
    group_scales = (vmaxs - vmins) / (qmax - qmin)

    if not has_weight_zeros:
        group_zeros = torch.zeros_like(group_scales)
    else:
        group_zeros = qmin - torch.round(vmins / group_scales)

    vals = vals.reshape(n, k)
    group_scales = group_scales.reshape(n, -1)
    group_zeros = group_zeros.reshape(n, -1)

    group_qvals = quantize_per_channel_group(
        input=vals,
        scales=group_scales,
        zero_points=group_zeros,
        quant_min=qmin,
        quant_max=qmax,
        dtype=torch.int8 if signed else torch.uint8,
        group_size=group_size,
    )

    if not has_weight_zeros:
        group_zeros = None

    return group_qvals, group_scales, group_zeros


class UIntxWeightOnlyQuantizedLinear(nn.Module):
    def __init__(
        self,
        pack_weight_op,
        linear_op,
        bias: Optional[torch.Tensor] = None,
    ):
        super().__init__()
        self._pack_weights_op = pack_weight_op
        self._linear_op = linear_op
        if bias is not None:
            self.bias = nn.Parameter(bias, requires_grad=False)
        else:
            self.register_parameter("bias", None)

    def quantize_and_pack_weights(
        self,
        weights,
        nbit,
        group_size,
        uintx_choose_qparams_algorithm: UIntxChooseQParamsAlgorithm = UIntxChooseQParamsAlgorithm.MIN_MAX,
    ):
        self.nbit = nbit
        self.group_size = group_size

        if uintx_choose_qparams_algorithm == UIntxChooseQParamsAlgorithm.HQQ:
            weight_qvals, weight_scales, weight_zeros, _ = (
                _choose_qparams_and_quantize_affine_hqq(
                    weights,
                    nbits=nbit,
                    group_size=group_size,
                    axis=1,
                    compute_dtype=weights.dtype,
                    device=weights.device,
                    verbose=False,
                    raw_output=True,  # Get raw HQQ format, not tinygemm format
                )
            )
            weight_qvals = weight_qvals.to(torch.uint8)
            # HQQ raw format: W_dequant = (W_q - zero) * scale
            # Kernel expects: W_dequant = W_q * scale + zeros
            # So: zeros = -zero * scale
            weight_zeros = -weight_zeros * weight_scales
        elif uintx_choose_qparams_algorithm == UIntxChooseQParamsAlgorithm.MIN_MAX:
            weight_qvals, weight_scales, weight_zeros = _quantize(
                weights, self.group_size, self.nbit, has_weight_zeros=True, signed=False
            )
            weight_zeros = -weight_zeros * weight_scales
        else:
            raise ValueError(
                f"Unsupported uintx_choose_qparams_algorithm: {uintx_choose_qparams_algorithm}"
            )
        self.weight_scales = nn.Parameter(weight_scales, requires_grad=False)
        self.weight_zeros = nn.Parameter(weight_zeros, requires_grad=False)
        packed_weights = self._pack_weights_op(weight_qvals.cpu()).to(
            device=weight_qvals.device
        )
        self.packed_weights = nn.Parameter(packed_weights, requires_grad=False)

    def forward(self, x):
        assert x.dim() >= 2
        if x.dim() == 2:
            output = self._linear_op(
                x,
                self.packed_weights,
                self.group_size,
                self.weight_scales,
                self.weight_zeros,
            )
            if self.bias is not None:
                output = output + self.bias
            return output

        lead_shape = x.shape[0:-1]
        k = x.shape[-1]
        n = self.weight_scales.shape[0]
        output = self._linear_op(
            x.reshape(-1, k),
            self.packed_weights,
            self.group_size,
            self.weight_scales,
            self.weight_zeros,
        ).reshape(*lead_shape, n)
        if self.bias is not None:
            output = output + self.bias
        return output


# TODO(mcandales): Consolidate with _replace_linear_with_quantized_linear
def _replace_linear_with_quantized_linear_mps(module: nn.Module, kwargs={}):
    group_size = kwargs["group_size"]
    nbit = kwargs["nbit"]

    assert not isinstance(module, nn.Linear)
    assert nbit >= 1 and nbit <= 7

    for name, child in module.named_children():
        if not isinstance(child, nn.Linear):
            _replace_linear_with_quantized_linear_mps(child, kwargs)
        else:
            if not child.weight.is_contiguous():
                raise ValueError(
                    f"UIntxWeightOnlyQuantizedLinear requires contiguous weights for layer '{name}'. "
                    "Please call .contiguous() on the weight tensor before quantization."
                )
            qlinear = UIntxWeightOnlyQuantizedLinear(
                pack_weight_op=getattr(torch.ops.torchao, f"_pack_weight_{nbit}bit"),
                linear_op=getattr(
                    torch.ops.torchao, f"_linear_fp_act_{nbit}bit_weight"
                ),
                bias=child.bias,
            )
            setattr(module, name, qlinear)
            qlinear.quantize_and_pack_weights(child.weight, nbit, group_size)


class UIntxWeightOnlyLinearQuantizer:
    def __init__(
        self,
        *,
        device: Optional[str] = None,
        precision: Optional[torch.dtype] = None,
        bitwidth: Optional[int] = None,
        groupsize: Optional[int] = None,
    ):
        if device and device != "mps":
            raise NotImplementedError(
                "Only device=mps is currently supported in UIntxWeightOnlyLinearQuantizer"
            )
        else:
            self.device = device

        if precision and precision not in [
            torch.float32,
            torch.float16,
            torch.bfloat16,
        ]:
            raise ValueError(
                "Only precisions float32, float16 & bfloat16 are supported in UIntxWeightOnlyLinearQuantizer"
            )
        else:
            self.precision = precision

        if bitwidth is None:
            bitwidth = 4
            logger.warning(f"bitwidth not specified, defaulting to {bitwidth}.")
        if bitwidth not in range(1, 8):
            raise ValueError(
                "Only bitwidts 1 to 7 are supported in UIntxWeightOnlyLinearQuantizer"
            )
        else:
            self.bitwidth = bitwidth

        if groupsize is None:
            groupsize = 128
            logger.warning(f"groupsize not specified, defaulting to {groupsize}.")
        if groupsize not in [32, 64, 128, 256]:
            raise ValueError(
                "Only groupsizes 32, 64, 128 & 256 are supported in UIntxWeightOnlyLinearQuantizer"
            )
        else:
            self.groupsize = groupsize

    def quantize(self, model: nn.Module) -> nn.Module:
        if self.device:
            model = model.to(self.device)
        if self.precision:
            model = model.to(self.precision)
        _replace_linear_with_quantized_linear_mps(
            model,
            kwargs={
                "group_size": self.groupsize,
                "nbit": self.bitwidth,
            },
        )
        return model


@dataclass
class UIntxWeightOnlyConfig(AOBaseConfig):
    """
    Configuration for applying uintx weight-only asymmetric per-group quantization
    to linear layers for MPS devices.

    Args:
        bitwidth (int): Number of bits for quantization, must be between 1 and 7 inclusive.
            Default is 4.
        group_size (int): Group size for quantization. Must be one of [32, 64, 128, 256].
            Default is 128.
        uintx_choose_qparams_algorithm (Union[UIntxChooseQParamsAlgorithm, str]): Algorithm for
            choosing quantization parameters. Options:
            - "min_max" (default): Simple min-max scaling
            - "hqq": Half-Quadratic Quantization for better accuracy
    """

    bitwidth: int = 4
    group_size: int = 128
    uintx_choose_qparams_algorithm: Union[UIntxChooseQParamsAlgorithm, str] = (
        UIntxChooseQParamsAlgorithm.MIN_MAX
    )

    def __post_init__(self):
        if self.bitwidth not in range(1, 8):
            raise ValueError(
                f"bitwidth must be between 1 and 7 inclusive, got {self.bitwidth}"
            )
        if self.group_size not in [32, 64, 128, 256]:
            raise ValueError(
                f"group_size must be one of [32, 64, 128, 256], got {self.group_size}"
            )
        # Convert string to enum if necessary
        if isinstance(self.uintx_choose_qparams_algorithm, str):
            self.uintx_choose_qparams_algorithm = UIntxChooseQParamsAlgorithm(
                self.uintx_choose_qparams_algorithm
            )


@register_quantize_module_handler(UIntxWeightOnlyConfig)
def _uintx_weight_only_mps_transform(
    module: torch.nn.Module, config: UIntxWeightOnlyConfig
) -> torch.nn.Module:
    nbit = config.bitwidth
    group_size = config.group_size
    uintx_choose_qparams_algorithm = config.uintx_choose_qparams_algorithm

    if not module.weight.is_contiguous():
        raise ValueError(
            "UIntxWeightOnlyQuantizedLinear requires contiguous weights. "
            "Please call .contiguous() on the weight tensor before quantization."
        )

    qlinear = UIntxWeightOnlyQuantizedLinear(
        pack_weight_op=getattr(torch.ops.torchao, f"_pack_weight_{nbit}bit"),
        linear_op=getattr(torch.ops.torchao, f"_linear_fp_act_{nbit}bit_weight"),
        bias=module.bias,
    )
    qlinear.quantize_and_pack_weights(
        module.weight, nbit, group_size, uintx_choose_qparams_algorithm
    )
    return qlinear
