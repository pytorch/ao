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
from torchao.quantization.qat import (
    FakeQuantizeConfig,
    FromIntXQuantizationAwareTrainingConfig,
    Int8DynActInt4WeightQATQuantizer,
    IntXQuantizationAwareTrainingConfig,
)
from torchao.quantization.quant_api import (
    Int8DynamicActivationInt4WeightConfig,
    Int8DynamicActivationIntxWeightConfig,
    MappingType,
    quantize_,
)


class TestInt8DynamicActivationIntxWeight(unittest.TestCase):
    TEST_ACCURACY_CASES = [
        param(
            layout=layout,
            weight_dtype=weight_dtype,
            weight_mapping_type=weight_mapping_type,
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
        for weight_mapping_type in [
            MappingType.SYMMETRIC,
            MappingType.ASYMMETRIC,
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
        self, layout, weight_dtype, weight_mapping_type, weight_granularity
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
        weight_mapping_type = MappingType.SYMMETRIC

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
        weight_mapping_type = MappingType.SYMMETRIC

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
                weight_mapping_type=MappingType.SYMMETRIC,
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
            "torch.ops.torchao.dequantize_affine.default",
            "torch.ops.aten.linear.default(dequantize_affine, dequantize_affine_1)",
        ]
        for line in expected_lines:
            count = 1
            if line == "torch.ops.torchao.dequantize_affine.default":
                count = 2
            FileCheck().check_count(line, count, exactly=True).run(
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

    @parameterized.expand(
        [
            param(
                group_size=group_size,
                mapping_type=mapping_type,
                act_mapping_type=act_mapping_type,
            )
            for group_size, mapping_type, act_mapping_type in zip(
                [32, 64],
                [MappingType.ASYMMETRIC, MappingType.SYMMETRIC],
                [MappingType.ASYMMETRIC, MappingType.SYMMETRIC],
            )
        ],
        name_func=lambda f, _, params: f.__name__ + f"_{params.kwargs}",
    )
    def test_identical_to_Int8DynamicActivationInt4WeightConfig(
        self, group_size, mapping_type, act_mapping_type
    ):
        """
        Checks that Int8DynamicActivationIntxWeightConfig with weight_dtype=torch.int4 is identical to Int8DynamicActivationInt4WeightConfig
        """
        k0 = 512
        k1 = 256
        layers = [
            torch.nn.Linear(k0, k1),
        ]
        model = torch.nn.Sequential(*layers)
        activations = torch.randn(3, 1, k0)

        model_copy = copy.deepcopy(model)

        quantize_(
            model,
            Int8DynamicActivationIntxWeightConfig(
                weight_dtype=torch.int4,
                weight_granularity=PerGroup(group_size),
                weight_mapping_type=mapping_type,
                weight_scale_dtype=None,
                act_mapping_type=act_mapping_type,
            ),
        )
        quantize_(
            model_copy,
            Int8DynamicActivationInt4WeightConfig(
                group_size=group_size,
                mapping_type=mapping_type,
                act_mapping_type=act_mapping_type,
            ),
        )
        with torch.no_grad():
            torch.allclose(model(activations), model_copy(activations))

    @parameterized.expand(
        [
            param(
                weight_dtype=weight_dtype,
                group_size=group_size,
                mapping_type=mapping_type,
                act_mapping_type=act_mapping_type,
                scale_dtype=scale_dtype,
                model_dtype=model_dtype,
            )
            for weight_dtype in list(getattr(torch, f"int{x}") for x in range(1, 9))
            for group_size in [32, 64, 128]
            for mapping_type in [MappingType.SYMMETRIC, MappingType.ASYMMETRIC]
            for act_mapping_type in [MappingType.ASYMMETRIC, MappingType.SYMMETRIC]
            for scale_dtype in [torch.float32, torch.bfloat16, torch.float16]
            for model_dtype in [torch.float32, torch.bfloat16]
        ],
        name_func=lambda f, _, params: f.__name__ + f"_{params.kwargs}",
    )
    def test_identical_to_IntXQuantizationAwareTrainingConfig(
        self,
        weight_dtype,
        group_size,
        mapping_type,
        act_mapping_type,
        scale_dtype,
        model_dtype,
    ):
        # TODO: the QAT logic for asymmetric mapping is very different from PTQ, so we don't test that case here
        # Unify the two?
        if mapping_type == MappingType.ASYMMETRIC:
            return

        assert mapping_type in [MappingType.SYMMETRIC, MappingType.ASYMMETRIC]
        assert act_mapping_type in [MappingType.SYMMETRIC, MappingType.ASYMMETRIC]
        is_symmetric = mapping_type == MappingType.SYMMETRIC
        is_act_symmetric = act_mapping_type == MappingType.SYMMETRIC

        k0 = 512
        k1 = 256
        layers = [
            torch.nn.Linear(k0, k1),
        ]
        model = torch.nn.Sequential(*layers)
        activations = torch.randn(
            k0,
        )

        model = model.to(model_dtype)
        activations = activations.to(model_dtype)

        activation_config = FakeQuantizeConfig(
            torch.int8,
            "per_token",
            is_symmetric=is_act_symmetric,
        )
        weight_config = FakeQuantizeConfig(
            weight_dtype,
            group_size=group_size,
            is_symmetric=is_symmetric,
            scale_precision=scale_dtype,
        )

        quantize_(
            model,
            IntXQuantizationAwareTrainingConfig(activation_config, weight_config),
        )
        try:
            expected_out = model(activations)
        except NotImplementedError as e:
            # QAT does not support act_mapping_type == MappingType.SYMMETRIC yet
            if act_mapping_type == MappingType.SYMMETRIC:
                return
            raise e

        quantize_(model, FromIntXQuantizationAwareTrainingConfig())
        quantize_(
            model,
            Int8DynamicActivationIntxWeightConfig(
                weight_dtype=weight_dtype,
                weight_granularity=PerGroup(group_size),
                weight_mapping_type=mapping_type,
                weight_scale_dtype=scale_dtype,
                act_mapping_type=act_mapping_type,
            ),
        )
        actual_out = model(activations)
        self.assertTrue(torch.allclose(expected_out, actual_out))

    @parameterized.expand(
        [
            param(
                group_size=group_size,
                scale_dtype=scale_dtype,
                model_dtype=model_dtype,
            )
            for group_size in [32, 64, 128]
            for scale_dtype in [torch.float32, torch.bfloat16, torch.float16]
            for model_dtype in [torch.float32, torch.bfloat16]
        ],
        name_func=lambda f, _, params: f.__name__ + f"_{params.kwargs}",
    )
    def test_identical_to_Int8DynActInt4WeightQATQuantizer(
        self, group_size, scale_dtype, model_dtype
    ):
        k0 = 512
        k1 = 256
        layers = [
            torch.nn.Linear(k0, k1),
        ]
        model = torch.nn.Sequential(*layers)
        activations = torch.randn(
            k0,
        )

        model = model.to(model_dtype)
        activations = activations.to(model_dtype)

        qat_quantizer = Int8DynActInt4WeightQATQuantizer(
            groupsize=group_size, precision=model_dtype, scales_precision=scale_dtype
        )
        model = qat_quantizer.prepare(model)
        expected_out = model(activations)

        prepared_model_copy = copy.deepcopy(model)

        # Convert model method 1
        quantize_(model, FromIntXQuantizationAwareTrainingConfig())
        quantize_(
            model,
            Int8DynamicActivationIntxWeightConfig(
                weight_dtype=torch.int4,
                weight_granularity=PerGroup(group_size),
                weight_mapping_type=MappingType.SYMMETRIC,
                weight_scale_dtype=scale_dtype,
                act_mapping_type=MappingType.ASYMMETRIC,
            ),
        )
        actual_out1 = model(activations)
        self.assertTrue(torch.allclose(expected_out, actual_out1))

        # Convert model method 2
        qat_quantizer.convert(prepared_model_copy)
        actual_out2 = prepared_model_copy(activations)
        self.assertTrue(torch.allclose(expected_out, actual_out2))


if __name__ == "__main__":
    unittest.main()
