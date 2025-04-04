# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import unittest

import torch
from torch.testing import FileCheck

from torchao.experimental.q_dq_layout import QDQLayout
from torchao.experimental.quant_api import (
    Int8DynamicActivationIntxWeightConfig,
)
from torchao.experimental.quant_passes import (
    replace_q_dq_patterns_with_quantized_linear_ops_pass,
)
from torchao.quantization.granularity import PerGroup, PerRow
from torchao.quantization.quant_api import quantize_


class TestQuantPasses(unittest.TestCase):
    def replace_q_dq_patterns_with_quantized_linear_ops_pass(self):
        # setattr(torch.ops.pt2e_quant, "dequantize_affine", None)
        layers = [
            torch.nn.Linear(256, 128, bias=True),
            torch.nn.Linear(128, 64, bias=False),
            torch.nn.Linear(64, 32, bias=True),
        ]
        model = torch.nn.Sequential(*layers)
        activations = torch.randn(2, 1, 256, dtype=torch.float32)
        quantize_(
            model,
            Int8DynamicActivationIntxWeightConfig(
                weight_dtype=torch.int4,
                granularity=PerGroup(64),
                has_weight_zeros=True,
                layout=QDQLayout(),
            ),
            lambda m, fqn: fqn == "0",
        )
        quantize_(
            model,
            Int8DynamicActivationIntxWeightConfig(
                weight_dtype=torch.int3,
                granularity=PerRow(),
                has_weight_zeros=False,
                layout=QDQLayout(),
            ),
            lambda m, fqn: fqn == "1",
        )
        quantize_(
            model,
            Int8DynamicActivationIntxWeightConfig(
                weight_dtype=torch.int5,
                granularity=PerGroup(32),
                has_weight_zeros=False,
                layout=QDQLayout(),
            ),
            lambda m, fqn: fqn == "2",
        )

        eager_results = model(activations)
        exported = torch.export.export(model, (activations,), strict=True)
        exported = replace_q_dq_patterns_with_quantized_linear_ops_pass(exported)

        # We should not find pack op because it gets constant folded
        FileCheck().check_not("torch.ops.torchao._pack_8bit_act").run(
            exported.graph_module.code
        )

        # We should find 3 torchao linear ops
        FileCheck().check_count(
            "torch.ops.torchao._linear_8bit_act_", count=3, exactly=True
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
