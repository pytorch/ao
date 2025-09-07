# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD 3-Clause license found in the
# LICENSE file in the root directory of this source tree.

import copy
import tempfile
import unittest

import torch
from parameterized import param, parameterized
from torch.testing import FileCheck
from torch.testing._internal.common_utils import (
    TestCase,
    run_tests,
)

from torchao.dtypes import QDQLayout
from torchao.quantization import (
    Int8DynamicActivationIntxWeightConfig,
    IntxWeightOnlyConfig,
    MappingType,
    quantize_,
)
from torchao.quantization.granularity import PerAxis, PerGroup
from torchao.quantization.qat import IntxFakeQuantizeConfig, QATConfig
from torchao.quantization.quantize_.workflows import IntxPackingFormat
from torchao.quantization.utils import compute_error
from torchao.utils import torch_version_at_least, unwrap_tensor_subclass


@unittest.skipIf(not torch_version_at_least("2.7.0"), "Need pytorch 2.7+")
class TestIntxUnpackedToInt8Tensor(TestCase):
    def setUp(self):
        self.config = IntxWeightOnlyConfig(
            weight_dtype=torch.int4,
            granularity=PerGroup(32),
        )

    def test_embedding(self):
        dtype = torch.bfloat16
        device = "cpu"
        input = torch.randint(low=0, high=128, size=(10,), device=device)
        embedding = torch.nn.Embedding(128, 256, dtype=dtype, device=device)
        original = embedding(input)
        quantize_(embedding, self.config)
        quantized = embedding(input)
        error = compute_error(original, quantized)
        self.assertTrue(error > 20)

    def test_linear(self):
        dtype = torch.bfloat16
        device = "cpu"
        input = torch.randn(1, 128, dtype=dtype, device=device)
        linear = torch.nn.Linear(128, 256, dtype=dtype, device=device)
        original = linear(input)
        quantize_(linear, self.config)
        quantized = linear(input)
        error = compute_error(original, quantized)
        self.assertTrue(error > 20)

    def test_slice(self):
        dtype = torch.bfloat16
        device = "cpu"
        dummy = torch.nn.Linear(256, 256, bias=False, dtype=dtype, device=device)

        dummy1 = torch.nn.Linear(256, 64, bias=False, dtype=dtype, device=device)
        dummy1.weight = torch.nn.Parameter(
            dummy.weight.narrow(0, 0, 64), requires_grad=False
        )

        dummy2 = torch.nn.Linear(128, 256, dtype=dtype, device=device)
        dummy2.weight = torch.nn.Parameter(
            dummy.weight.narrow(1, 0, 128), requires_grad=False
        )

        quantize_(dummy, self.config)
        weight1 = dummy.weight.narrow(0, 0, 64)
        weight2 = dummy.weight.narrow(1, 0, 128)

        self.assertEqual(weight1.qdata, dummy.weight.qdata.narrow(0, 0, 64))
        self.assertEqual(weight1.scale, dummy.weight.scale.narrow(0, 0, 64))

        self.assertEqual(weight2.qdata, dummy.weight.qdata.narrow(1, 0, 128))
        self.assertEqual(weight2.scale, dummy.weight.scale.narrow(1, 0, 4))

        # check for sliced weight, before and after float8 quantization
        # does not differ too much
        input = torch.randn(2, 256, dtype=dtype, device=device)
        res_ref = dummy1(input)
        dummy.weight = torch.nn.Parameter(weight1, requires_grad=False)
        res = dummy(input)
        assert compute_error(res, res_ref) > 20

        input = torch.randn(2, 128, dtype=dtype, device=device)
        res_ref = dummy2(input)
        dummy.weight = torch.nn.Parameter(weight2, requires_grad=False)
        res = dummy(input)
        assert compute_error(res, res_ref) > 15

    def test_slice_and_copy_(self):
        device = "cpu"
        l = torch.nn.Linear(1024, 1024).to(device).to(torch.bfloat16)
        quantize_(l, self.config)
        param = l.weight
        param_data = param.data
        param_data = param_data.narrow(0, 0, 512)
        assert param.data.qdata.data_ptr() == param_data.qdata.data_ptr()
        assert param.data.scale.data_ptr() == param_data.scale.data_ptr()
        assert param.data.zero_point.data_ptr() == param_data.zero_point.data_ptr()

        # dummy_l has random input (shouldn't be 0)
        dummy_l = torch.nn.Linear(1024, 1024).to(device).to(torch.bfloat16)
        quantize_(dummy_l, self.config)
        quantized = dummy_l.weight
        quantized = quantized.narrow(0, 0, 512)

        param_data.copy_(quantized)

        # making sure param.data is updated
        assert param.data.qdata[0][0] == quantized.qdata[0][0]

    def test_to_dtype(self):
        activations_bf16 = torch.randn(1, 128, dtype=torch.bfloat16)
        activations_fp32 = torch.randn(1, 128, dtype=torch.float32)
        activations_fp16 = torch.randn(1, 128, dtype=torch.float16)

        linear = torch.nn.Linear(128, 256)
        quantize_(linear, self.config)

        linear.to(dtype=torch.float16)
        linear(activations_fp16)

        linear.to(dtype=torch.float32)
        linear(activations_fp32)

        linear.to(dtype=torch.bfloat16)
        linear(activations_bf16)

    def test_export_intx_weight_only_config(self):
        linear = torch.nn.Linear(128, 256)
        quantize_(linear, self.config)
        ep = torch.export.export(linear, (torch.randn(1, 128),))
        assert "torch.ops.torchao.dequantize_affine.default" in ep.graph_module.code

    def test_export_int8_dyn_act_intx_weight_config(self):
        layers = [
            torch.nn.Linear(512, 256, bias=False),
        ]
        model = torch.nn.Sequential(*layers)
        activations = torch.randn(1, 512, dtype=torch.float32)

        quantize_(
            model,
            Int8DynamicActivationIntxWeightConfig(
                weight_dtype=torch.int4,
                weight_granularity=PerAxis(0),
                weight_mapping_type=MappingType.SYMMETRIC,
                packing_format=IntxPackingFormat.UNPACKED_TO_INT8,
            ),
        )
        eager_results = model(activations)

        exported = torch.export.export(model, (activations,))

        exported_results = exported.module()(activations)
        self.assertTrue(torch.allclose(eager_results, exported_results))

        expected_counts = {
            "torch.ops.torchao.choose_qparams_affine.default": 1,
            "torch.ops.torchao.quantize_affine.default": 1,
            "torch.ops.torchao.dequantize_affine.default": 2,
            "torch.ops.aten.linear.default": 1,
            "torch.ops.aten.reshape.default": 0,
        }
        for line, count in expected_counts.items():
            FileCheck().check_count(line, count, exactly=True).run(
                exported.graph_module.code
            )

    def test_export_int8_dyn_act_intx_weight_config_with_unwrap(self):
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
                packing_format=IntxPackingFormat.UNPACKED_TO_INT8,
            ),
        )
        eager_results = model(activations)

        unwrap_tensor_subclass(model)

        exported = torch.export.export(model, (activations,))

        exported_results = exported.module()(activations)
        self.assertTrue(torch.allclose(eager_results, exported_results))

        expected_counts = {
            "torch.ops.torchao.choose_qparams_affine.default": 1,
            "torch.ops.torchao.quantize_affine.default": 1,
            "torch.ops.torchao.dequantize_affine.default": 2,
            "torch.ops.aten.linear.default": 1,
            "torch.ops.aten.reshape.default": 0,
        }
        for line, count in expected_counts.items():
            FileCheck().check_count(line, count, exactly=True).run(
                exported.graph_module.code
            )

    def test_serialization_int8_dyn_act_intx_weight_config(self):
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
                packing_format=IntxPackingFormat.UNPACKED_TO_INT8,
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

    def test_serialization_intx_weight_only_config(self):
        layers = [
            torch.nn.Linear(512, 256),
        ]
        model = torch.nn.Sequential(*layers)
        model2 = torch.nn.Sequential(*layers)
        activations = torch.randn(1, 512, dtype=torch.float32)

        quantize_(
            model,
            IntxWeightOnlyConfig(
                weight_dtype=torch.int4,
                granularity=PerGroup(64),
                packing_format=IntxPackingFormat.UNPACKED_TO_INT8,
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

    @parameterized.expand(
        [
            param(
                weight_dtype=weight_dtype,
                group_size=group_size,
                mapping_type=mapping_type,
                scale_dtype=scale_dtype,
                model_dtype=model_dtype,
            )
            for weight_dtype in list(getattr(torch, f"int{x}") for x in range(1, 9))
            for group_size in [32, 64, 128]
            for mapping_type in [MappingType.SYMMETRIC]
            for scale_dtype in [torch.float32, torch.bfloat16, torch.float16]
            for model_dtype in [torch.float32, torch.bfloat16, torch.float16]
        ],
        name_func=lambda f, _, params: f.__name__ + f"_{params.kwargs}",
    )
    def test_qat_int8_dyn_act_intx_weight_config(
        self, weight_dtype, group_size, mapping_type, scale_dtype, model_dtype
    ):
        activation_config = IntxFakeQuantizeConfig(
            torch.int8, "per_token", is_symmetric=False, scale_precision=scale_dtype
        )
        weight_config = IntxFakeQuantizeConfig(
            weight_dtype,
            group_size=group_size,
            mapping_type=mapping_type,
            scale_precision=scale_dtype,
        )
        qat_config_prepare = QATConfig(
            activation_config=activation_config,
            weight_config=weight_config,
            step="prepare",
        )
        qat_config_convert = QATConfig(
            step="convert",
        )
        quant_config = Int8DynamicActivationIntxWeightConfig(
            weight_dtype=weight_config.dtype,
            weight_granularity=PerGroup(group_size),
            weight_mapping_type=mapping_type,
            weight_scale_dtype=scale_dtype,
            packing_format=IntxPackingFormat.UNPACKED_TO_INT8,
        )

        k0 = 512
        k1 = 256
        layers = [
            torch.nn.Linear(k0, k1),
            torch.nn.Linear(k1, k0),
        ]
        model = torch.nn.Sequential(*layers)
        activations = torch.randn(
            k0,
        )
        model = model.to(model_dtype)
        activations = activations.to(model_dtype)

        quantize_(model, qat_config_prepare)
        prepared_out = model(activations)

        quantize_(model, qat_config_convert)
        converted_out = model(activations)

        quantize_(
            model,
            quant_config,
        )
        quantizeed_out = model(activations)

        sqnr = compute_error(prepared_out, converted_out).item()
        sqnr = compute_error(prepared_out, quantizeed_out).item()

        if model_dtype == scale_dtype:
            self.assertTrue(
                sqnr == float("inf"),
                f"Got SQNR of {sqnr} between prepared and quantized",
            )
        else:
            # There is slight difference in how v2 does dynamic activation quantization
            # It uses the model_dtype, whereas v1 always uses float32
            self.assertTrue(
                sqnr > 35, f"Got SQNR of {sqnr} between prepared and quantized"
            )

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
            for mapping_type in [MappingType.SYMMETRIC]
            for act_mapping_type in [MappingType.ASYMMETRIC]
            for scale_dtype in [torch.float32, torch.bfloat16, torch.float16]
            for model_dtype in [torch.float32, torch.bfloat16, torch.float16]
        ],
        name_func=lambda f, _, params: f.__name__ + f"_{params.kwargs}",
    )
    def test_intx_unpacked_v2_is_close_to_qdq_v1(
        self,
        weight_dtype,
        group_size,
        mapping_type,
        act_mapping_type,
        scale_dtype,
        model_dtype,
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

        model_v1 = copy.deepcopy(model)
        quantize_(
            model_v1,
            Int8DynamicActivationIntxWeightConfig(
                weight_dtype=weight_dtype,
                weight_granularity=PerGroup(group_size),
                weight_mapping_type=mapping_type,
                weight_scale_dtype=scale_dtype,
                act_mapping_type=act_mapping_type,
                version=1,
                layout=QDQLayout(),
            ),
        )
        out_v1 = model_v1(activations)

        quantize_(
            model,
            Int8DynamicActivationIntxWeightConfig(
                weight_dtype=weight_dtype,
                weight_granularity=PerGroup(group_size),
                weight_mapping_type=mapping_type,
                weight_scale_dtype=scale_dtype,
                act_mapping_type=act_mapping_type,
                packing_format=IntxPackingFormat.UNPACKED_TO_INT8,
            ),
        )
        out_v2 = model(activations)
        sqnr = compute_error(out_v1, out_v2).item()

        if model_dtype == torch.float32 and model_dtype == torch.float32:
            self.assertTrue(sqnr == float("inf"), f"Got SQNR of {sqnr}")
        else:
            # There is slight difference in how v2 does dynamic activation quantization
            # It uses the model_dtype, whereas v1 always uses float32
            self.assertTrue(sqnr > 35, f"Got SQNR of {sqnr}")


if __name__ == "__main__":
    run_tests()
