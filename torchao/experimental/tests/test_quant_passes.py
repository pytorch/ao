# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import itertools
import unittest

import torch
from parameterized import param, parameterized
from torch.testing import FileCheck

from torchao.dtypes import QDQLayout
from torchao.experimental.quant_passes import (
    replace_q_dq_patterns_with_quantized_embedding_ops_pass,
    replace_q_dq_patterns_with_quantized_linear_ops_pass,
)
from torchao.quantization.granularity import PerAxis, PerGroup
from torchao.quantization.quant_api import (
    Int8DynamicActivationIntxWeightConfig,
    IntxWeightOnlyConfig,
    MappingType,
    quantize_,
)


class TestQuantPasses(unittest.TestCase):
    def test_replace_q_dq_patterns_with_quantized_linear_ops_pass(self):
        layers = []
        layer_to_weight_dtype = {}
        layer_to_weight_mapping_type = {}
        layer_to_weight_granularity = {}
        for (
            weight_dtype,
            weight_mapping_type,
            weight_granularity,
            has_bias,
        ) in itertools.product(
            [getattr(torch, f"int{i}") for i in range(1, 9)],
            [MappingType.ASYMMETRIC, MappingType.SYMMETRIC],
            [PerAxis(0), PerGroup(32)],
            [True, False],
        ):
            idx = len(layers)
            layer_to_weight_dtype[idx] = weight_dtype
            layer_to_weight_mapping_type[idx] = weight_mapping_type
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
                    weight_granularity=layer_to_weight_granularity[idx],
                    layout=QDQLayout(),
                ),
                lambda m, fqn: fqn == str(idx),
            )

        eager_results = model(activations)
        exported = torch.export.export(model, (activations,), strict=True)
        exported = replace_q_dq_patterns_with_quantized_linear_ops_pass(
            exported, target="universal"
        )

        # We should not find pack op because it gets constant folded
        FileCheck().check_not("torch.ops.torchao._pack_8bit_act").run(
            exported.graph_module.code
        )

        # We should find len(layers) torchao linear ops
        FileCheck().check_count(
            "torch.ops.torchao._linear_8bit_act_", count=len(layers), exactly=True
        ).run(exported.graph_module.code)

        # We should not find Q/DQ ops
        FileCheck().check_not("torch.ops.torchao.quantize_affine.default").run(
            exported.graph_module.code
        )
        FileCheck().check_not("torch.ops.torchao.dequantize_affine.default").run(
            exported.graph_module.code
        )
        # FileCheck().check_not("torch.ops.torchao.choose_qparams_affine.default").run(
        #     exported.graph_module.code
        # ) # TODO: Fix this

        # Numerics should match
        exported_results = exported.module()(activations)
        self.assertTrue(torch.allclose(exported_results, eager_results))

    @parameterized.expand(
        [
            param(weight_dtype=weight_dtype, granularity=granularity)
            for weight_dtype in [getattr(torch, f"int{i}") for i in range(1, 9)]
            for granularity in [PerAxis(0), PerGroup(32)]
        ],
        name_func=lambda f, _, params: f.__name__ + f"_{params.kwargs}",
    )
    def test_replace_q_dq_patterns_with_quantized_embedding_ops_pass(
        self, weight_dtype, granularity
    ):
        # Calling torch.export many times in a parametrized test causes
        # torch._dynamo.exc.FailOnRecompileLimitHit: recompile_limit reached error
        # Setting cache_size_limit to a large number to avoid this error
        torch._dynamo.config.cache_size_limit = 10000

        mapping_type = MappingType.ASYMMETRIC

        model = torch.nn.Sequential(
            *[torch.nn.Embedding(5000, 512), torch.nn.Linear(512, 512)]
        )
        indices = torch.randint(0, 5000, (4, 5, 17), dtype=torch.int32)

        quantize_(
            model,
            IntxWeightOnlyConfig(
                weight_dtype=weight_dtype,
                granularity=granularity,
                mapping_type=mapping_type,
                layout=QDQLayout(),
            ),
            lambda m, fqn: isinstance(m, torch.nn.Embedding),
        )
        eager_results = model(indices)

        exported = torch.export.export(model, (indices,), strict=True)
        exported = replace_q_dq_patterns_with_quantized_embedding_ops_pass(exported)

        # We should not find pack op because it gets constant folded
        FileCheck().check_not("torch.ops.torchao._pack_embedding").run(
            exported.graph_module.code
        )

        # We should find
        FileCheck().check_count(
            "torch.ops.torchao._embedding", count=1, exactly=True
        ).run(exported.graph_module.code)

        # We should not find Q/DQ ops
        FileCheck().check_not("torch.ops.torchao.dequantize_affine.default").run(
            exported.graph_module.code
        )

        # Numerics should match
        exported_results = exported.module()(indices)
        self.assertTrue(torch.allclose(exported_results, eager_results))


if __name__ == "__main__":
    unittest.main()
