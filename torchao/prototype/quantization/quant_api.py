# Copyright (c) Meta Platforms, Inc. and affiliates.
# Copyright 2024-2025 Arm Limited and affiliates.
# All rights reserved.
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""
Quantization workflow APIs moved from `torch/quantization/quant_api.py`
to prototype.
"""

import logging
import types
import warnings
from dataclasses import dataclass

import torch

import torchao
from torchao.core.config import AOBaseConfig
from torchao.dtypes import (
    PlainLayout,
    UintxLayout,
    to_affine_quantized_intx,
)
from torchao.quantization.quant_primitives import (
    _DTYPE_TO_QVALUE_BOUNDS,
    MappingType,
    ZeroPointDomain,
)
from torchao.quantization.transform_module import (
    register_quantize_module_handler,
)
from torchao.quantization.utils import (
    _linear_extra_repr,
)

logger = logging.getLogger(__name__)


@dataclass
class UIntXWeightOnlyConfig(AOBaseConfig):
    """
    Configuration for applying uintx weight-only asymmetric per-group quantization to linear layers, using uintx quantization where
    x is the number of bits specified by `dtype`

    Args:
        `dtype`: torch.uint1 to torch.uint7 sub byte dtypes
        `group_size`: parameter for quantization, controls the granularity of quantization, smaller
         size is more fine grained, defaults to 64
        `pack_dim`: the dimension we use for packing, defaults to -1
        `use_hqq`: whether to use hqq algorithm or the default algorithm to quantize the weight
        `set_inductor_config`: if True, adjusts `torchinductor` settings to recommended values.
    """

    dtype: torch.dtype
    group_size: int = 64
    pack_dim: int = -1
    use_hqq: bool = False
    set_inductor_config: bool = True

    def __post_init__(self):
        torch._C._log_api_usage_once("torchao.quantization.UIntXWeightOnlyConfig")
        warnings.warn(
            "`UIntXWeightOnlyConfig` will be deleted in a future release of torchao. Please see https://github.com/pytorch/ao/issues/2752 for more details."
        )


@register_quantize_module_handler(UIntXWeightOnlyConfig)
def _uintx_weight_only_transform(
    module: torch.nn.Module, config: UIntXWeightOnlyConfig
):
    dtype = config.dtype
    group_size = config.group_size
    pack_dim = config.pack_dim
    use_hqq = config.use_hqq
    if config.set_inductor_config:
        torchao.quantization.utils.recommended_inductor_config_setter()

    weight = module.weight

    SUPPORTED_DTYPES = {
        torch.uint1,
        torch.uint2,
        torch.uint3,
        torch.uint4,
        torch.uint5,
        torch.uint6,
        torch.uint7,
        torch.uint8,
    }
    assert dtype in SUPPORTED_DTYPES, f"Unsupported dtype for hqq: {dtype}"

    mapping_type = MappingType.ASYMMETRIC
    block_size = (1, group_size)

    if use_hqq:
        quant_min, quant_max = _DTYPE_TO_QVALUE_BOUNDS[dtype]
        dtype = torch.uint8
        eps = None
        zero_point_dtype = None
        zero_point_domain = ZeroPointDomain.FLOAT
        preserve_zero = False
        _layout = PlainLayout()
    else:
        quant_min, quant_max = None, None
        eps = torch.finfo(torch.float32).eps
        zero_point_dtype = torch.int32
        zero_point_domain = ZeroPointDomain.INT
        preserve_zero = True
        _layout = UintxLayout(dtype=dtype, pack_dim=pack_dim)

    new_weight = to_affine_quantized_intx(
        weight,
        mapping_type,
        block_size,
        dtype,
        quant_min=quant_min,
        quant_max=quant_max,
        eps=eps,
        zero_point_dtype=zero_point_dtype,
        zero_point_domain=zero_point_domain,
        preserve_zero=preserve_zero,
        _layout=_layout,
        use_hqq=use_hqq,
    )
    module.weight = torch.nn.Parameter(new_weight, requires_grad=False)
    module.extra_repr = types.MethodType(_linear_extra_repr, module)
    return module
