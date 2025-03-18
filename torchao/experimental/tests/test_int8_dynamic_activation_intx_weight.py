# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import copy
import tempfile
import unittest

import torch
from parameterized import param, parameterized
from torch.testing import FileCheck

from torchao.dtypes import PlainLayout
from torchao.experimental.packed_linear_int8_dynamic_activation_intx_weight_layout import (
    PackedLinearInt8DynamicActivationIntxWeightLayout,
)
from torchao.experimental.q_dq_layout import QDQLayout
from torchao.experimental.quant_api import int8_dynamic_activation_intx_weight
from torchao.quantization.granularity import PerGroup, PerRow
from torchao.quantization.quant_api import quantize_
from torchao.utils import unwrap_tensor_subclass


class TestInt8DynamicActivationIntxWeight(unittest.TestCase):
    TEST_ACCURACY_CASES = [
        param(
            layout=layout,
            weight_dtype=weight_dtype,
            has_weight_zeros=has_weight_zeros,
            granularity=granularity,
        )
        for layout in [
            PackedLinearInt8DynamicActivationIntxWeightLayout(),
            QDQLayout(),
        ]
        for weight_dtype in [
            torch.int1,
            torch.int2,
            torch.int3,
            torch.int4,
            torch.int5,
            torch.int6,
            torch.int7,
            torch.int8,
        ]
        for has_weight_zeros in [
            True,
            False,
        ]
        for granularity in [
            PerGroup(128),
            PerRow(),
        ]
    ]

    @parameterized.expand(
        TEST_ACCURACY_CASES,
        name_func=lambda f, _, params: f.__name__ + f"_{params.kwargs}",
    )
    def test_accuracy(self, layout, weight_dtype, has_weight_zeros, granularity):
        """
        Checks the accuracy of different layouts by comparing the results to PlainLayout()
        """
        m = 3
        n = 1071
        k = 2048
        activations = torch.randn(m, k)
        model = torch.nn.Sequential(
            *[torch.nn.Linear(k, k, bias=False), torch.nn.Linear(k, n, bias=True)]
        )

        quantized_model = copy.deepcopy(model)
        quantize_(
            quantized_model,
            int8_dynamic_activation_intx_weight(
                weight_dtype=weight_dtype,
                granularity=granularity,
                has_weight_zeros=has_weight_zeros,
                layout=layout,
            ),
        )

        quantized_model_reference = copy.deepcopy(model)
        quantize_(
            quantized_model_reference,
            int8_dynamic_activation_intx_weight(
                weight_dtype=weight_dtype,
                granularity=granularity,
                has_weight_zeros=has_weight_zeros,
                layout=self._reference_layout(),
            ),
        )

        with torch.no_grad():
            result = quantized_model(activations)
            expected_result = quantized_model_reference(activations)

        # When weight_dtype is int4, we need low tolerance when comparing
        # to the reference because KleidiAI kernels (based on bfloat16 scales)
        # may be used
        self._assert_close(result, expected_result, strict=(weight_dtype != torch.int4))

    def test_accuracy_aten(self):
        m = 3
        n = 1024
        k = 2048
        activations = torch.randn(m, k)
        model = torch.nn.Sequential(
            *[torch.nn.Linear(k, k, bias=False), torch.nn.Linear(k, n, bias=True)]
        )
        weight_dtype = torch.int4
        granularity = PerGroup(128)
        has_weight_zeros = False

        quantized_model = copy.deepcopy(model)
        quantize_(
            quantized_model,
            int8_dynamic_activation_intx_weight(
                weight_dtype=weight_dtype,
                granularity=granularity,
                has_weight_zeros=has_weight_zeros,
                layout=PackedLinearInt8DynamicActivationIntxWeightLayout(target="aten"),
            ),
        )

        quantized_model_reference = copy.deepcopy(model)
        quantize_(
            quantized_model_reference,
            int8_dynamic_activation_intx_weight(
                weight_dtype=weight_dtype,
                granularity=granularity,
                has_weight_zeros=has_weight_zeros,
                layout=self._reference_layout(),
            ),
        )

        with torch.no_grad():
            result = quantized_model(activations)
            expected_result = quantized_model_reference(activations)

        # KleidiAI aten kernels need low tolerance when comparing to reference
        # because they use bfloat16 scales
        self._assert_close(result, expected_result, strict=False)

    def _assert_close(self, result, expected_result, strict: bool = False):
        if strict:
            self.assertTrue(
                torch.nn.functional.mse_loss(result, expected_result) <= 1e-6
            )
            self.assertTrue(torch.allclose(result, expected_result, atol=1e-2))
        else:
            self.assertTrue(
                torch.nn.functional.mse_loss(result, expected_result) <= 1e-3
            )

    def _reference_layout(self):
        return PlainLayout()

    def test_export_compile_aoti_PackedLinearInt8DynamicActivationIntxWeightLayout(
        self,
    ):
        """
        Checks that models quantized with PackedLinearInt8DynamicActivationIntxWeightLayout() work with
        torch.export.export, torch.compile, and AOTI.
        """
        granularity = PerRow()
        m = 3
        k0 = 512
        k1 = 256
        k2 = 128
        k3 = 1024
        weight_dtype = torch.int4
        has_weight_zeros = True
        layers = [
            torch.nn.Linear(k0, k1, bias=False),
            torch.nn.Linear(k1, k2, bias=True),
            torch.nn.Linear(k2, k3, bias=False),
        ]
        model = torch.nn.Sequential(*layers)
        activations = torch.randn(2, 1, m, k0, dtype=torch.float32)

        quantize_(
            model,
            int8_dynamic_activation_intx_weight(
                weight_dtype=weight_dtype,
                granularity=granularity,
                has_weight_zeros=has_weight_zeros,
                layout=PackedLinearInt8DynamicActivationIntxWeightLayout(),
            ),
        )
        eager_results = model(activations)

        unwrapped_model = copy.deepcopy(model)
        unwrap_tensor_subclass(model)

        # Export
        exported = torch.export.export(model, (activations,), strict=True)
        exported_results = exported.module()(activations)
        self.assertTrue(torch.allclose(eager_results, exported_results))

        # Compile
        compiled = torch.compile(unwrapped_model)
        with torch.no_grad():
            compiled_results = compiled(activations)
        self.assertTrue(torch.allclose(eager_results, compiled_results))

        # AOTI
        with tempfile.TemporaryDirectory() as tmpdirname:
            package_path = f"{tmpdirname}/model.pt2"
            torch._inductor.aoti_compile_and_package(
                exported, package_path=package_path
            )
            fn = torch._inductor.aoti_load_package(package_path)
            aoti_results = fn(activations)
            self.assertTrue(torch.allclose(eager_results, aoti_results))

    def test_export_QDQLayout(self):
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
        exported = torch.export.export(model, (activations,), strict=True)
        exported_results = exported.module()(activations)
        self.assertTrue(torch.allclose(eager_results, exported_results))

        expected_lines = [
            "torch.ops.quant.choose_qparams_affine.default(input_1, 'ASYMMETRIC', [1, 512], torch.int32, -128, 127, None, torch.float32, torch.int32)",
            "torch.ops.quant.quantize_affine.default(input_1, [1, 512], getitem, getitem_1, torch.int32, -128, 127)",
            "torch.ops.quant.dequantize_affine.default(quantize_affine, [1, 512], getitem, getitem_1, torch.int32, -128, 127)",
            "torch.ops.quant.dequantize_affine.default(p_fn_0_parametrizations_weight_original0, [1, 64], p_fn_0_parametrizations_weight_original1, None, torch.int32, -8, 7, 'NONE')",
            "torch.ops.aten.linear.default(dequantize_affine, dequantize_affine_1)",
        ]
        for line in expected_lines:
            FileCheck().check_count(line, 1, exactly=True).run(
                exported.graph_module.code
            )


if __name__ == "__main__":
    unittest.main()
