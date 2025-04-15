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

from torchao.dtypes import PackedLinearInt8DynamicActivationIntxWeightLayout, QDQLayout
from torchao.quantization.granularity import PerAxis, PerGroup
from torchao.quantization.quant_api import (
    Int8DynamicActivationIntxWeightConfig,
    MappingType,
    ZeroPointDomain,
    quantize_,
)


class TestInt8DynamicActivationIntxWeight(unittest.TestCase):
    TEST_ACCURACY_CASES = [
        param(
            layout=layout,
            weight_dtype=weight_dtype,
            weight_zero_point_domain=weight_zero_point_domain,
            weight_granularity=weight_granularity,
        )
        for layout in [
            PackedLinearInt8DynamicActivationIntxWeightLayout(),
            PackedLinearInt8DynamicActivationIntxWeightLayout(target="universal"),
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
        for weight_zero_point_domain in [
            ZeroPointDomain.NONE,
            ZeroPointDomain.INT,
        ]
        for weight_granularity in [
            PerGroup(128),
            PerAxis(0),
        ]
    ]

    @parameterized.expand(
        TEST_ACCURACY_CASES,
        name_func=lambda f, _, params: f.__name__ + f"_{params.kwargs}",
    )
    def test_accuracy(
        self, layout, weight_dtype, weight_zero_point_domain, weight_granularity
    ):
        """
        Checks the accuracy of packed layouts
        """
        m = 3
        n = 1071
        k = 2048
        activations = torch.randn(m, k)
        model = torch.nn.Sequential(
            *[torch.nn.Linear(k, k, bias=False), torch.nn.Linear(k, n, bias=True)]
        )
        weight_mapping_type = MappingType.ASYMMETRIC

        # We set round weights to bf16 and set scale dtype to bf16 because
        # some kernels do this internally (e.g., KleidiAI kernels)
        model = model.to(torch.bfloat16).to(torch.float32)
        weight_scale_dtype = torch.bfloat16

        quantized_model = copy.deepcopy(model)
        quantize_(
            quantized_model,
            Int8DynamicActivationIntxWeightConfig(
                weight_dtype=weight_dtype,
                weight_granularity=weight_granularity,
                weight_mapping_type=weight_mapping_type,
                weight_zero_point_domain=weight_zero_point_domain,
                weight_scale_dtype=weight_scale_dtype,
                layout=layout,
            ),
        )

        quantized_model_reference = copy.deepcopy(model)
        quantize_(
            quantized_model_reference,
            Int8DynamicActivationIntxWeightConfig(
                weight_dtype=weight_dtype,
                weight_granularity=weight_granularity,
                weight_mapping_type=weight_mapping_type,
                weight_zero_point_domain=weight_zero_point_domain,
                weight_scale_dtype=weight_scale_dtype,
                layout=self._reference_layout(),
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
        weight_granularity = PerGroup(128)
        weight_mapping_type = MappingType.ASYMMETRIC
        weight_zero_point_domain = ZeroPointDomain.NONE

        # KleidiAI kernels round scales to bf16 internally
        model = model.to(torch.bfloat16).to(torch.float32)
        weight_scale_dtype = torch.bfloat16

        quantized_model = copy.deepcopy(model)
        quantize_(
            quantized_model,
            Int8DynamicActivationIntxWeightConfig(
                weight_dtype=weight_dtype,
                weight_granularity=weight_granularity,
                weight_mapping_type=weight_mapping_type,
                weight_zero_point_domain=weight_zero_point_domain,
                weight_scale_dtype=weight_scale_dtype,
                layout=PackedLinearInt8DynamicActivationIntxWeightLayout(
                    target="kleidiai"
                ),
            ),
        )

        quantized_model_reference = copy.deepcopy(model)
        quantize_(
            quantized_model_reference,
            Int8DynamicActivationIntxWeightConfig(
                weight_dtype=weight_dtype,
                weight_granularity=weight_granularity,
                weight_mapping_type=weight_mapping_type,
                weight_zero_point_domain=weight_zero_point_domain,
                weight_scale_dtype=weight_scale_dtype,
                layout=self._reference_layout(),
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
        weight_granularity = PerGroup(128)
        weight_mapping_type = MappingType.ASYMMETRIC
        weight_zero_point_domain = ZeroPointDomain.NONE

        # We set round_weight_scale_to_bf16 to True for accuracy testing because
        # some KleidiAI kernels do this internally
        model = model.to(torch.bfloat16).to(torch.float32)
        weight_scale_dtype = torch.bfloat16

        quantized_model = copy.deepcopy(model)
        quantize_(
            quantized_model,
            Int8DynamicActivationIntxWeightConfig(
                weight_dtype=weight_dtype,
                weight_granularity=weight_granularity,
                weight_mapping_type=weight_mapping_type,
                weight_zero_point_domain=weight_zero_point_domain,
                weight_scale_dtype=weight_scale_dtype,
                layout=PackedLinearInt8DynamicActivationIntxWeightLayout(target="aten"),
            ),
        )

        quantized_model_reference = copy.deepcopy(model)
        quantize_(
            quantized_model_reference,
            Int8DynamicActivationIntxWeightConfig(
                weight_dtype=weight_dtype,
                weight_granularity=weight_granularity,
                weight_mapping_type=weight_mapping_type,
                weight_zero_point_domain=weight_zero_point_domain,
                weight_scale_dtype=weight_scale_dtype,
                layout=self._reference_layout(),
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
        return QDQLayout()

    def test_export_compile_aoti_PackedLinearInt8DynamicActivationIntxWeightLayout(
        self,
    ):
        """
        Checks that models quantized with PackedLinearInt8DynamicActivationIntxWeightLayout() work with
        torch.export.export, torch.compile, and AOTI.
        """
        m = 3
        k0 = 512
        k1 = 256
        k2 = 128
        k3 = 1024
        weight_dtype = torch.int4
        weight_granularity = PerAxis(0)
        weight_mapping_type = MappingType.ASYMMETRIC
        weight_zero_point_domain = ZeroPointDomain.INT
        layers = [
            torch.nn.Linear(k0, k1, bias=False),
            torch.nn.Linear(k1, k2, bias=True),
            torch.nn.Linear(k2, k3, bias=False),
        ]
        model = torch.nn.Sequential(*layers)
        activations = torch.randn(2, 1, m, k0)

        quantize_(
            model,
            Int8DynamicActivationIntxWeightConfig(
                weight_dtype=weight_dtype,
                weight_granularity=weight_granularity,
                weight_mapping_type=weight_mapping_type,
                weight_zero_point_domain=weight_zero_point_domain,
                weight_scale_dtype=torch.bfloat16,
                layout=PackedLinearInt8DynamicActivationIntxWeightLayout(),
            ),
        )
        eager_results = model(activations)

        # Export
        exported = torch.export.export(model, (activations,), strict=True)
        exported_results = exported.module()(activations)
        self.assertTrue(torch.allclose(eager_results, exported_results))

        # Compile
        compiled = torch.compile(model)
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
        m = 3
        k0 = 512
        k1 = 256
        k2 = 128
        k3 = 1024
        weight_dtype = torch.int4
        weight_granularity = PerAxis(0)
        weight_mapping_type = MappingType.ASYMMETRIC
        weight_zero_point_domain = ZeroPointDomain.INT

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
                weight_granularity=weight_granularity,
                weight_mapping_type=weight_mapping_type,
                weight_zero_point_domain=weight_zero_point_domain,
                weight_scale_dtype=torch.bfloat16,
                layout=PackedLinearInt8DynamicActivationIntxWeightLayout(),
            ),
        )
        eager_results = model(activations)

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
        layers = [
            torch.nn.Linear(512, 256, bias=False),
        ]
        model = torch.nn.Sequential(*layers)
        activations = torch.randn(1, 512, dtype=torch.float32)

        quantize_(
            model,
            Int8DynamicActivationIntxWeightConfig(
                weight_dtype=torch.int4,
                weight_granularity=PerGroup(64),
                weight_zero_point_domain=ZeroPointDomain.NONE,
                layout=QDQLayout(),
            ),
        )
        eager_results = model(activations)

        exported = torch.export.export(model, (activations,), strict=True)

        exported_results = exported.module()(activations)
        self.assertTrue(torch.allclose(eager_results, exported_results))

        expected_lines = [
            "torch.ops.torchao.choose_qparams_affine.default(input_1, 'ASYMMETRIC', [1, 512], torch.int8, None, None, None, torch.float64, torch.int64)",
            "torch.ops.torchao.quantize_affine.default(input_1, [1, 512], getitem, getitem_1, torch.int8)",
            "torch.ops.torchao.dequantize_affine.default(quantize_affine, [1, 512], getitem, getitem_1, torch.int8)",
            "torch.ops.torchao.dequantize_affine.default(access_subclass_inner_tensor_default_72, [1, 64], access_subclass_inner_tensor_default_73, None, torch.int8, -8, 7, 'NONE')",
            "torch.ops.aten.linear.default(dequantize_affine, dequantize_affine_1)",
        ]
        for line in expected_lines:
            FileCheck().check_count(line, 1, exactly=True).run(
                exported.graph_module.code
            )

    @parameterized.expand(
        [
            param(layout=layout)
            for layout in [
                PackedLinearInt8DynamicActivationIntxWeightLayout(),
                QDQLayout(),
            ]
        ],
        name_func=lambda f, _, params: f.__name__ + f"_{params.kwargs}",
    )
    def test_serialization(self, layout):
        layers = [
            torch.nn.Linear(512, 256),
        ]
        model = torch.nn.Sequential(*layers)
        model2 = torch.nn.Sequential(*layers)
        activations = torch.randn(1, 512, dtype=torch.float32)

        quantize_(
            model,
            Int8DynamicActivationIntxWeightConfig(
                weight_dtype=torch.int4,
                weight_granularity=PerGroup(64),
                layout=layout,
            ),
        )
        expected = model(activations)

        with tempfile.TemporaryDirectory() as tmpdirname:
            torch.save(model.state_dict(), f"{tmpdirname}/model.pt")
            state_dict = torch.load(
                f"{tmpdirname}/model.pt", map_location="cpu", weights_only=True
            )

            # Load deserialized weights into model2 and check result
            model2.load_state_dict(state_dict, assign=True)
            actual = model2(activations)
            self.assertTrue(torch.allclose(expected, actual))

    def test_moved_error(self):
        from torchao.experimental.quant_api import Int8DynamicActivationIntxWeightConfig

        with self.assertRaisesRegex(
            NotImplementedError,
            "Int8DynamicActivationIntxWeightConfig has moved from torchao.experimental.quant_api to torchao.quantization.quant_api",
        ):
            config = Int8DynamicActivationIntxWeightConfig(  # noqa: F841
                weight_dtype=torch.int4,
                granularity=PerGroup(64),
            )


if __name__ == "__main__":
    unittest.main()
