# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import copy
import unittest

import torch
from torch.testing import FileCheck

from torchao.dtypes import PlainLayout
from torchao.experimental.q_dq_layout import QDQLayout
from torchao.experimental.quant_api import int8_dynamic_activation_intx_weight
from torchao.quantization.granularity import PerGroup
from torchao.quantization.quant_api import quantize_
from torchao.utils import unwrap_tensor_subclass


class TestQDQLayout(unittest.TestCase):
    def test_accuracy(self):
        """
        Checks the accuracy of TestQDQLayout() by comparing
        its results to the results of a reference model that uses PlainLayout()
        """
        granularity = PerGroup(128)
        m = 1
        n = 1071
        k = 4096
        activations = torch.randn(m, k)
        model = torch.nn.Sequential(*[torch.nn.Linear(k, n, bias=False)])

        for weight_dtype in [
            torch.int1,
            torch.int2,
            torch.int3,
            torch.int4,
            torch.int5,
            torch.int6,
            torch.int7,
            torch.int8,
        ]:
            for has_weight_zeros in [True, False]:
                print(
                    f"Testing weight_dtype={weight_dtype}, has_weight_zeros={has_weight_zeros}"
                )
                quantized_model = copy.deepcopy(model)
                quantize_(
                    quantized_model,
                    int8_dynamic_activation_intx_weight(
                        weight_dtype=weight_dtype,
                        granularity=granularity,
                        has_weight_zeros=has_weight_zeros,
                        layout=QDQLayout(),
                    ),
                )

                quantized_model_reference = copy.deepcopy(model)
                quantize_(
                    quantized_model_reference,
                    int8_dynamic_activation_intx_weight(
                        weight_dtype=weight_dtype,
                        granularity=granularity,
                        has_weight_zeros=has_weight_zeros,
                        layout=PlainLayout(),
                    ),
                )

                with torch.no_grad():
                    result = quantized_model(activations)
                    expected_result = quantized_model_reference(activations)

                num_mismatch_at_low_tol = 0
                num_total = result.reshape(-1).shape[0]
                for i in range(num_total):
                    actual_val = result.reshape(-1)[i]
                    expected_val = expected_result.reshape(-1)[i]
                    self.assertTrue(torch.allclose(actual_val, expected_val, atol=1e-6))
                    if not torch.allclose(actual_val, expected_val):
                        num_mismatch_at_low_tol += 1

                # Assert at most 5% of entries are not close at a low tolerance
                self.assertTrue(num_mismatch_at_low_tol / num_total <= 0.05)

    def test_export(self):
        """
        Checks that models quantized with TestQDQLayout() export as expected
        """
        granularity = PerGroup(64)
        weight_dtype = torch.int4
        has_weight_zeros = False
        layers = [
            torch.nn.Linear(512, 256, bias=False),
        ]
        model = torch.nn.Sequential(*layers)
        activations = torch.randn(1, 512, dtype=torch.float32)

        to_export_with_old_api = copy.deepcopy(model)

        print("Quantizing model")
        quantize_(
            model,
            int8_dynamic_activation_intx_weight(
                weight_dtype=weight_dtype,
                granularity=granularity,
                has_weight_zeros=has_weight_zeros,
                layout=QDQLayout(),
            ),
        )
        eager_results = model(activations)

        unwrap_tensor_subclass(model)

        print("Exporting quantized model")
        exported = torch.export.export(model, (activations,), strict=True)
        exported_results = exported.module()(activations)
        self.assertTrue(torch.allclose(eager_results, exported_results))

        expected_lines = [
            "torch.ops.quantized_decomposed.choose_qparams_per_token_asymmetric.default(input_1, torch.int8)",
            "torch.ops.quantized_decomposed.quantize_per_token.default(input_1, getitem, getitem_1, -128, 127, torch.int8)",
            "torch.ops.quantized_decomposed.dequantize_per_token.default(quantize_per_token, getitem, getitem_1, -128, 127, torch.int8, torch.float32)",
            "torch.ops.aten.to.dtype(dequantize_per_token, torch.float32)",
            "torch.ops.quantized_decomposed.dequantize_per_channel_group.default(p_fn_0_parametrizations_weight_original0, p_fn_0_parametrizations_weight_original1, None, -8, 7, torch.int8, 64, torch.float32)",
            "torch.ops.aten.linear.default(to, dequantize_per_channel_group)",
        ]
        for line in expected_lines:
            FileCheck().check_count(line, 1, exactly=True).run(
                exported.graph_module.code
            )

        # Compare exported graph with old API
        # TODO: delete after old API is deprecated
        from torchao.quantization.quant_api import Int8DynActInt4WeightQuantizer

        quantizer = Int8DynActInt4WeightQuantizer(
            groupsize=granularity.group_size,
            padding_allowed=False,
            precision=torch.float32,
            scales_precision=torch.float32,
            device=torch.device("cpu"),
            # mapping_type=MappingType.ASYMMETRIC,
        )
        quantizer.quantize(to_export_with_old_api)
        exported_from_old_api = torch.export.export(
            to_export_with_old_api,
            (activations,),
        )

        expected_lines_old_api = [
            "torch.ops.quantized_decomposed.choose_qparams_per_token_asymmetric.default(to, torch.int8)",
            "torch.ops.quantized_decomposed.quantize_per_token.default(to, getitem, getitem_1, -128, 127, torch.int8)",
            "torch.ops.quantized_decomposed.dequantize_per_token.default(quantize_per_token, getitem, getitem_1, -128, 127, torch.int8, torch.float32)",
            "torch.ops.aten.to.dtype(dequantize_per_token, torch.float32)",
            "torch.ops.quantized_decomposed.dequantize_per_channel_group.default(b_getattr_l__fn_____0___weight, b_getattr_l__fn_____0___scales, b_getattr_l__fn_____0___zeros, -8, 7, torch.int8, 64, torch.float32)",
            "torch.ops.aten.linear.default(to_1, dequantize_per_channel_group)",
        ]
        for line in expected_lines_old_api:
            FileCheck().check_count(line, 1, exactly=True).run(
                exported_from_old_api.graph_module.code
            )

        # TODO: there are slight differences in the results because exported_results uses
        # asymmetric with zero_point_domain NONE (has_weight_zeros=False)
        # and results_from_old_api uses symmetric (but with an asymmetric range)
        # I think the new API might make more sense, but need more thought
        # results_from_old_api = exported_from_old_api.module()(activations)
        # self.assertTrue(torch.allclose(exported_results, results_from_old_api))


if __name__ == "__main__":
    unittest.main()
