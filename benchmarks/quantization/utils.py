# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD 3-Clause license found in the
# LICENSE file in the root directory of this source tree.

import torch

from torchao.prototype.awq import AWQConfig
from torchao.prototype.mx_formats.inference_workflow import (
    MXDynamicActivationMXWeightConfig,
    NVFP4DynamicActivationNVFP4WeightConfig,
)
from torchao.prototype.smoothquant import SmoothQuantConfig
from torchao.quantization import (
    Float8DynamicActivationFloat8WeightConfig,
    Float8DynamicActivationInt4WeightConfig,
    Int4WeightOnlyConfig,
    Int8DynamicActivationInt8WeightConfig,
    Int8WeightOnlyConfig,
    PerRow,
)


def string_to_config(s):
    if s == "None":
        return None
    elif s == "float8_rowwise":
        return Float8DynamicActivationFloat8WeightConfig(granularity=PerRow())
    elif s == "int4_groupwise_weight_float8_rowwise_activation":
        return Float8DynamicActivationInt4WeightConfig()
    elif s == "int4_groupwise_hqq_weight_only":
        return Int4WeightOnlyConfig(
            group_size=32,
            int4_packing_format="tile_packed_to_4d",
            int4_choose_qparams_algorithm="hqq",
        )
    elif s == "int8_rowwise_weight_only":
        return Int8WeightOnlyConfig()
    elif s == "int8_rowwise":
        return Int8DynamicActivationInt8WeightConfig()
    elif s == "mxfp8":
        return MXDynamicActivationMXWeightConfig(
            activation_dtype=torch.float8_e4m3fn,
            weight_dtype=torch.float8_e4m3fn,
        )
    elif s == "nvfp4":
        return NVFP4DynamicActivationNVFP4WeightConfig(
            use_dynamic_per_tensor_scale=True,
            use_triton_kernel=True,
        )
    elif s == "awq_int4_weight_only":
        base_config = Int4WeightOnlyConfig(
            group_size=128,
            int4_packing_format="tile_packed_to_4d",
            int4_choose_qparams_algorithm="hqq",
        )
        return AWQConfig(base_config, step="prepare_for_loading")
    elif s == "smoothquant_int8":
        base_config = Int8DynamicActivationInt8WeightConfig(version=2)
        return SmoothQuantConfig(base_config, step="prepare_for_loading")
    else:
        raise AssertionError(f"unsupported {s}")
