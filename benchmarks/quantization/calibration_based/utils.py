# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD 3-Clause license found in the
# LICENSE file in the root directory of this source tree.
from torchao.quantization import (
    Int4WeightOnlyConfig,
    Int8DynamicActivationInt8WeightConfig,
)


def string_to_calibration_config(s):
    """Convert string to calibration-based quantization config."""
    if s == "awq_int4_weight_only":
        return Int4WeightOnlyConfig(
            group_size=128,
            int4_packing_format="tile_packed_to_4d",
            int4_choose_qparams_algorithm="hqq",
            version=2,
        )
    elif s == "smoothquant_int8":
        return Int8DynamicActivationInt8WeightConfig(version=2)
    else:
        raise AssertionError(f"unsupported config: {s}")
