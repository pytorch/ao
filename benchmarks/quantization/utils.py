# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD 3-Clause license found in the
# LICENSE file in the root directory of this source tree.

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
    else:
        raise AssertionError(f"unsupported {s}")
