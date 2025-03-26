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
from torchao.experimental.quant_api import (
    Int8DynamicActivationIntxWeightConfig,
    replace_q_dq_with_torchao_quantized_linear_ops,
)
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
            PackedLinearInt8DynamicActivationIntxWeightLayout(target="universal"),
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

        # We set round_weight_scale_to_bf16 to True for accuracy testing because
        # some kernels do this internally (e.g., KleidiAI kernels)
        round_weight_scale_to_bf16 = True

        quantized_model = copy.deepcopy(model)
        quantize_(
            quantized_model,
            Int8DynamicActivationIntxWeightConfig(
                weight_dtype=weight_dtype,
                granularity=granularity,
                has_weight_zeros=has_weight_zeros,
                layout=layout,
                round_weight_scale_to_bf16=round_weight_scale_to_bf16,
            ),
        )

        quantized_model_reference = copy.deepcopy(model)
        quantize_(
            quantized_model_reference,
            Int8DynamicActivationIntxWeightConfig(
                weight_dtype=weight_dtype,
                granularity=granularity,
                has_weight_zeros=has_weight_zeros,
                layout=self._reference_layout(),
                round_weight_scale_to_bf16=round_weight_scale_to_bf16,
            ),
        )

        with torch.no_grad():
            result = quantized_model(activations)
            expected_result = quantized_model_reference(activations)
        self._assert_close(result, expected_result)

    def test_accuracy_kleidiai(self):
        n = 1071
        k = 2048
        model = torch.nn.Sequential(
            *[torch.nn.Linear(k, k, bias=False), torch.nn.Linear(k, n, bias=True)]
        )
        weight_dtype = torch.int4
        granularity = PerGroup(128)
        has_weight_zeros = False

        # We set round_weight_scale_to_bf16 to True for accuracy testing because
        # some KleidiAI kernels do this internally
        round_weight_scale_to_bf16 = True

        quantized_model = copy.deepcopy(model)
        quantize_(
            quantized_model,
            int8_dynamic_activation_intx_weight(
                weight_dtype=weight_dtype,
                granularity=granularity,
                has_weight_zeros=has_weight_zeros,
                layout=PackedLinearInt8DynamicActivationIntxWeightLayout(
                    target="kleidiai"
                ),
                round_weight_scale_to_bf16=round_weight_scale_to_bf16,
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
                round_weight_scale_to_bf16=round_weight_scale_to_bf16,
            ),
        )

        with torch.no_grad():
            for m in [1, 3, 5, 9, 13]:
                activations = torch.randn(m, k)
                result = quantized_model(activations)
                expected_result = quantized_model_reference(activations)

                # KleidiAI kernels require much higher tolerance when comparing to reference,
                # especially for GEMM kernels
                self._assert_close(
                    result, expected_result, mse_tol=1e-2, atol=1e-2, rtol=1
                )

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

        # We set round_weight_scale_to_bf16 to True for accuracy testing because
        # some KleidiAI kernels do this internally
        round_weight_scale_to_bf16 = True

        quantized_model = copy.deepcopy(model)
        quantize_(
            quantized_model,
            Int8DynamicActivationIntxWeightConfig(
                weight_dtype=weight_dtype,
                granularity=granularity,
                has_weight_zeros=has_weight_zeros,
                layout=PackedLinearInt8DynamicActivationIntxWeightLayout(target="aten"),
                round_weight_scale_to_bf16=round_weight_scale_to_bf16,
            ),
        )

        quantized_model_reference = copy.deepcopy(model)
        quantize_(
            quantized_model_reference,
            Int8DynamicActivationIntxWeightConfig(
                weight_dtype=weight_dtype,
                granularity=granularity,
                has_weight_zeros=has_weight_zeros,
                layout=self._reference_layout(),
                round_weight_scale_to_bf16=round_weight_scale_to_bf16,
            ),
        )

        with torch.no_grad():
            result = quantized_model(activations)
            expected_result = quantized_model_reference(activations)

        self._assert_close(result, expected_result)

    def _assert_close(
        self, result, expected_result, mse_tol=1e-6, atol=1e-2, rtol=1e-5
    ):
        mse_loss = torch.nn.functional.mse_loss(result, expected_result)
        self.assertTrue(
            mse_loss <= mse_tol,
            f"Got mse_loss={mse_loss}, above mse tolerance {mse_tol}",
        )

        n_rand_idxs = 5
        rand_idxs = torch.randint(0, result.numel(), (n_rand_idxs,))
        self.assertTrue(
            torch.allclose(result, expected_result, atol=atol, rtol=rtol),
            f"Failed allclose at atol={atol}, rtol={rtol}.  On {n_rand_idxs} random indices, we have result={result.reshape(-1)[rand_idxs]} vs expected_result={expected_result.reshape(-1)[rand_idxs]}.",
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
            Int8DynamicActivationIntxWeightConfig(
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

    def test_export_dynamic_shape_PackedLinearInt8DynamicActivationIntxWeightLayout(
        self,
    ):
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
        dynamic_shapes = {
            "input": {
                0: torch.export.Dim("dim0"),
                1: None,
                2: torch.export.Dim("dim2"),
                3: None,
            }
        }

        quantize_(
            model,
            Int8DynamicActivationIntxWeightConfig(
                weight_dtype=weight_dtype,
                granularity=granularity,
                has_weight_zeros=has_weight_zeros,
                layout=PackedLinearInt8DynamicActivationIntxWeightLayout(),
            ),
        )
        eager_results = model(activations)

        unwrap_tensor_subclass(model)

        # Export
        exported = torch.export.export(
            model, (activations,), strict=True, dynamic_shapes=dynamic_shapes
        )
        exported_results = exported.module()(activations)
        self.assertTrue(torch.allclose(eager_results, exported_results))

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
            Int8DynamicActivationIntxWeightConfig(
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

    def test_replace_q_dq_with_torchao_quantized_linear_ops(self):
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

        unwrap_tensor_subclass(model)
        exported = torch.export.export(model, (activations,), strict=True)
        exported = replace_q_dq_with_torchao_quantized_linear_ops(exported)

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
