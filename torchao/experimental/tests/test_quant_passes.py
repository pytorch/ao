# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import unittest

import torch
from torch.testing import FileCheck

from torchao.dtypes import QDQLayout
from torchao.experimental.quant_passes import (
    replace_q_dq_patterns_with_quantized_linear_ops_pass,
)
from torchao.quantization.granularity import PerGroup, PerRow
from torchao.quantization.quant_api import (
    Int8DynamicActivationIntxWeightConfig,
    MappingType,
    ZeroPointDomain,
    quantize_,
)


class TestQuantPasses(unittest.TestCase):
    def test_replace_q_dq_patterns_with_quantized_linear_ops_pass(self):
        layers = []
        layer_to_weight_dtype = {}
        layer_to_weight_mapping_type = {}
        layer_to_weight_zero_point_domain = {}
        layer_to_weight_granularity = {}
        for weight_dtype in [getattr(torch, f"int{i}") for i in range(1, 9)]:
            for weight_mapping_type in [MappingType.ASYMMETRIC, MappingType.SYMMETRIC]:
                for weight_zero_point_domain in [
                    ZeroPointDomain.INT,
                    ZeroPointDomain.NONE,
                ]:
                    if (
                        weight_mapping_type == MappingType.SYMMETRIC
                        and weight_zero_point_domain == ZeroPointDomain.INT
                    ):
                        continue
                    for weight_granularity in [PerRow(), PerGroup(32)]:
                        for has_bias in [True, False]:
                            idx = len(layers)
                            layer_to_weight_dtype[idx] = weight_dtype
                            layer_to_weight_mapping_type[idx] = weight_mapping_type
                            layer_to_weight_zero_point_domain[idx] = (
                                weight_zero_point_domain
                            )
                            layer_to_weight_granularity[idx] = weight_granularity
                            layers.append(torch.nn.Linear(64, 64, bias=has_bias))

        activations = torch.randn(2, 1, 64, dtype=torch.float32)
        model = torch.nn.Sequential(*layers)
        for idx in range(len(layers)):
            quantize_(
                model,
                Int8DynamicActivationIntxWeightConfig(
                    weight_dtype=layer_to_weight_dtype[idx],
                    weight_mapping_type=layer_to_weight_mapping_type[idx],
                    weight_zero_point_domain=layer_to_weight_zero_point_domain[idx],
                    weight_granularity=layer_to_weight_granularity[idx],
                    layout=QDQLayout(),
                ),
                lambda m, fqn: fqn == str(idx),
            )

        eager_results = model(activations)
        exported = torch.export.export(model, (activations,), strict=True)
        exported = replace_q_dq_patterns_with_quantized_linear_ops_pass(exported, target="universal")

        # We should not find pack op because it gets constant folded
        FileCheck().check_not("torch.ops.torchao._pack_8bit_act").run(
            exported.graph_module.code
        )

        # We should find len(layers) torchao linear ops
        FileCheck().check_count(
            "torch.ops.torchao._linear_8bit_act_", count=len(layers), exactly=True
        ).run(exported.graph_module.code)

        # We should not find Q/DQ ops
        FileCheck().check_not("torch.ops.quant.quantize_affine.default").run(
            exported.graph_module.code
        )
        FileCheck().check_not("torch.ops.quant.dequantize_affine.default").run(
            exported.graph_module.code
        )
        FileCheck().check_not("torch.ops.quant.choose_qparams_affine.default").run(
            exported.graph_module.code
        )

        # Numerics should match
        exported_results = exported.module()(activations)
        self.assertTrue(torch.allclose(exported_results, eager_results))


if __name__ == "__main__":
    unittest.main()
