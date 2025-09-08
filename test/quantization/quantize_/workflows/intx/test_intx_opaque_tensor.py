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
from torch.testing._internal.common_utils import (
    TestCase,
    run_tests,
)

from torchao.quantization.granularity import PerAxis, PerGroup
from torchao.quantization.quant_api import (
    Int8DynamicActivationIntxWeightConfig,
    MappingType,
    quantize_,
)
from torchao.quantization.quantize_.workflows import IntxPackingFormat
from torchao.quantization.quantize_.workflows.intx.intx_opaque_tensor import (
    _is_kernel_library_loaded,
)
from torchao.quantization.utils import compute_error


def _get_accuracy_test_cases():
    MODEL_DTYPES = [
        torch.float32,
        torch.bfloat16,
    ]

    PACKING_FORMATS = [
        IntxPackingFormat.UNPACKED_TO_INT8,
        IntxPackingFormat.OPAQUE_ATEN,
        IntxPackingFormat.OPAQUE_TORCHAO_AUTO,
        IntxPackingFormat.OPAQUE_TORCHAO_KLEIDIAI,
        IntxPackingFormat.OPAQUE_TORCHAO_LOWBIT,
    ]

    WEIGHT_DTYPES = [
        torch.int1,
        torch.int2,
        torch.int3,
        torch.int4,
        torch.int5,
        torch.int6,
        torch.int7,
        torch.int8,
    ]

    MAPPING_TYPES = [
        MappingType.SYMMETRIC,
        MappingType.ASYMMETRIC,
        MappingType.SYMMETRIC_NO_CLIPPING_ERR,
    ]

    GRANULARITIES = [PerGroup(128), PerAxis(0)]

    def _is_valid_test_combination(
        model_dtype,
        packing_format,
        weight_dtype,
        weight_mapping_type,
        weight_granularity,
    ):
        # ATEN restrictions
        if packing_format == IntxPackingFormat.OPAQUE_ATEN:
            if weight_dtype != torch.int4:
                return False
            if weight_mapping_type == MappingType.ASYMMETRIC:
                return False
            if model_dtype != torch.float32:
                return False

        # TORCHAO_KLEIDIAI restrictions
        if packing_format == IntxPackingFormat.OPAQUE_TORCHAO_KLEIDIAI:
            if weight_dtype != torch.int4:
                return False
            if weight_mapping_type == MappingType.ASYMMETRIC:
                return False

        # SYMMETRIC_NO_CLIPPING_ERR does not work well with int1
        if (
            weight_dtype == torch.int1
            and weight_mapping_type == MappingType.SYMMETRIC_NO_CLIPPING_ERR
        ):
            return False

        return True

    test_cases = [
        param(
            model_dtype=mdt,
            packing_format=pf,
            weight_dtype=dt,
            weight_mapping_type=mt,
            weight_granularity=gr,
        )
        for mdt in MODEL_DTYPES
        for pf in PACKING_FORMATS
        for dt in WEIGHT_DTYPES
        for mt in MAPPING_TYPES
        for gr in GRANULARITIES
        if _is_valid_test_combination(dt, pf, dt, mt, gr)
    ]

    return test_cases


@unittest.skipIf(not _is_kernel_library_loaded(), "Kernel library not loaded")
class TestIntxOpaqueTensor(TestCase):
    @parameterized.expand(
        _get_accuracy_test_cases(),
        name_func=lambda f, _, params: f.__name__ + f"_{params.kwargs}",
    )
    def test_accuracy(
        self,
        model_dtype,
        packing_format,
        weight_dtype,
        weight_mapping_type,
        weight_granularity,
    ):
        """
        Checks the accuracy of packed layouts
        """
        m = 3
        n = 1071
        k = 2048
        activations = torch.randn(m, k).to(model_dtype)
        model = torch.nn.Sequential(
            *[torch.nn.Linear(k, k, bias=False), torch.nn.Linear(k, n, bias=True)]
        ).to(model_dtype)

        quantized_model = copy.deepcopy(model)
        quantize_(
            quantized_model,
            Int8DynamicActivationIntxWeightConfig(
                weight_dtype=weight_dtype,
                weight_granularity=weight_granularity,
                weight_mapping_type=weight_mapping_type,
                packing_format=packing_format,
                version=2,
            ),
        )

        quantized_model_reference = copy.deepcopy(model)
        quantize_(
            quantized_model_reference,
            Int8DynamicActivationIntxWeightConfig(
                weight_dtype=weight_dtype,
                weight_granularity=weight_granularity,
                weight_mapping_type=weight_mapping_type,
                packing_format=IntxPackingFormat.UNPACKED_TO_INT8,
                version=2,
            ),
        )

        with torch.no_grad():
            result = quantized_model(activations)
            expected_result = quantized_model_reference(activations)

        sqnr = compute_error(result, expected_result)
        self.assertTrue(sqnr > 30, f"Got SQNR of {sqnr}")

    def test_export_compile_aoti(
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
                0: torch.export.Dim.AUTO,
                1: torch.export.Dim.STATIC,
                2: torch.export.Dim.AUTO,
                3: torch.export.Dim.STATIC,
            }
        }

        quantize_(
            model,
            Int8DynamicActivationIntxWeightConfig(
                weight_dtype=weight_dtype,
                weight_granularity=weight_granularity,
                weight_mapping_type=weight_mapping_type,
                packing_format=IntxPackingFormat.OPAQUE_TORCHAO_AUTO,
                version=2,
            ),
        )
        eager_results = model(activations)

        # Export
        exported = torch.export.export(
            model, (activations,), strict=True, dynamic_shapes=dynamic_shapes
        )
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

    @parameterized.expand(
        [
            param(packing_format=pf)
            for pf in [
                IntxPackingFormat.OPAQUE_TORCHAO_AUTO,
                IntxPackingFormat.OPAQUE_ATEN,
            ]
        ],
        name_func=lambda f, _, params: f.__name__ + f"_{params.kwargs}",
    )
    def test_serialization(self, packing_format):
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
                packing_format=packing_format,
                version=2,
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

    def test_moe_quant_intx(self):
        from torchao.prototype.moe_quant.quantizable_moe_modules import (
            MOEFeedForwardAOQuantizable,
        )
        from torchao.prototype.moe_quant.utils import (
            FakeExtraDimTensor,
            MoEQuantConfig,
            UseFakeExtraDimTensor,
            cond_ffn_filter,
        )
        from torchao.quantization.quant_api import (
            Int8DynamicActivationIntxWeightConfig,
            quantize_,
        )
        from torchao.quantization.utils import compute_error

        with torch.device("cpu"):
            model = MOEFeedForwardAOQuantizable(512, 256, 8, 2, empty_init=False).to(
                torch.float32
            )
            x = torch.randn(8, 512, dtype=torch.float32)

        out = model(x).clone()

        base_config = Int8DynamicActivationIntxWeightConfig(
            packing_format=IntxPackingFormat.OPAQUE_TORCHAO_AUTO,
            version=2,
        )
        moe_config = MoEQuantConfig(
            base_config, use_fake_extra_dim_tensor=UseFakeExtraDimTensor.TRUE
        )

        quantize_(model, moe_config, cond_ffn_filter)

        out_q = model(x).clone()
        assert isinstance(model.experts.w1, FakeExtraDimTensor)

        mod_c = torch.compile(model, mode="reduce-overhead")

        mod_c(x)
        mod_c(x)

        out_qc = mod_c(x).clone()

        self.assertTrue(compute_error(out_q, out) > 30)
        self.assertTrue(compute_error(out_qc, out) > 30)


if __name__ == "__main__":
    run_tests()
