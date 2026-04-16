# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD 3-Clause license found in the
# LICENSE file in the root directory of this source tree.

from torchao.prototype.mx_formats.config import ScaleCalculationMode

__all__ = [
    "ScaleCalculationMode",
    "MXDynamicActivationMXWeightConfig",
    "NVFP4DynamicActivationNVFP4WeightConfig",
    "NVFP4ObservedLinear",
    "NVFP4WeightOnlyConfig",
]


def __getattr__(name: str):
    if name in {
        "MXDynamicActivationMXWeightConfig",
        "NVFP4DynamicActivationNVFP4WeightConfig",
        "NVFP4ObservedLinear",
        "NVFP4WeightOnlyConfig",
    }:
        from torchao.prototype.mx_formats.inference_workflow import (
            MXDynamicActivationMXWeightConfig,
            NVFP4DynamicActivationNVFP4WeightConfig,
            NVFP4ObservedLinear,
            NVFP4WeightOnlyConfig,
        )

        return {
            "MXDynamicActivationMXWeightConfig": MXDynamicActivationMXWeightConfig,
            "NVFP4DynamicActivationNVFP4WeightConfig": NVFP4DynamicActivationNVFP4WeightConfig,
            "NVFP4ObservedLinear": NVFP4ObservedLinear,
            "NVFP4WeightOnlyConfig": NVFP4WeightOnlyConfig,
        }[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
