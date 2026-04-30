# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD 3-Clause license found in the
# LICENSE file in the root directory of this source tree.
import copy
import logging
import unittest

import torch
from torch import nn
from torch.testing._internal import common_utils

try:
    from torch.testing._internal.common_cuda import PLATFORM_SUPPORTS_FP8_SPARSE
except ImportError:
    PLATFORM_SUPPORTS_FP8_SPARSE = False
from torchao.quantization import (
    Float8DynamicActivationFloat8WeightConfig,
)
from torchao.quantization.granularity import PerTensor
from torchao.quantization.quant_api import (
    quantize_,
)
from torchao.quantization.quantize_.workflows import (
    Float8PackingFormat,
)
from torchao.quantization.utils import compute_error
from torchao.sparsity import apply_fake_sparsity
from torchao.utils import (
    is_ROCM,
    torch_version_at_least,
)

logging.basicConfig(
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s", level=logging.INFO
)


@unittest.skipIf(
    not torch_version_at_least("2.10.0"),
    "Need torch >= 2.10.0",
)
class TestFloat8Sparse2x4_1DData1DMetadataTensor(common_utils.TestCase):
    def setUp(self):
        if not is_ROCM():
            self.skipTest("hipSPARSELt path requires ROCm")
        if not PLATFORM_SUPPORTS_FP8_SPARSE:
            self.skipTest("Need platform with FP8 sparse support (hipSPARSELt)")

    @common_utils.parametrize("compile", [True, False])
    def test_fp8_hipsparselt_sparse(self, compile):
        with torch.inference_mode():
            input = torch.rand((256, 256), dtype=torch.bfloat16, device="cuda")
            model = (
                nn.Sequential(
                    nn.Linear(256, 1024),
                    nn.Linear(1024, 256),
                )
                .bfloat16()
                .cuda()
                .eval()
            )

            apply_fake_sparsity(model)
            baseline_result = model(input)
            model_copy = copy.deepcopy(model)

            # Quantized (dense)
            quantize_(
                model_copy,
                Float8DynamicActivationFloat8WeightConfig(
                    granularity=PerTensor(),
                ),
            )
            dense_result = model_copy(input)
            dense_sqnr = compute_error(baseline_result, dense_result)

            # Sparse + quantized
            quantize_(
                model,
                Float8DynamicActivationFloat8WeightConfig(
                    version=2,
                    packing_format=Float8PackingFormat.SPARSE_1D_DATA_1D_METADATA,
                    granularity=PerTensor(),
                ),
            )
            if compile:
                model = torch.compile(model)
            sparse_result = model(input)
            sparse_sqnr = compute_error(baseline_result, sparse_result)

            self.assertEqual(dense_sqnr, sparse_sqnr)

    def test_fp8_hipsparselt_sparse_lowering_op_clone(self):
        """Validates clone dispatch correctly copies both sparse data and scale metadata."""
        with torch.inference_mode():
            model = nn.Linear(256, 1024).half().cuda().eval()
            apply_fake_sparsity(model)
            quantize_(
                model,
                Float8DynamicActivationFloat8WeightConfig(
                    version=2,
                    packing_format=Float8PackingFormat.SPARSE_1D_DATA_1D_METADATA,
                    granularity=PerTensor(),
                ),
            )

            original = model.weight.dequantize()
            cloned = model.weight.clone().dequantize()

            for o, c in zip(original, cloned):
                self.assertEqual(o, c)

    def test_fp8_hipsparselt_sparse_lowering_op_to(self):
        """Validates both to.dtype_layout and to.dtype dispatch paths correctly dequantize the sparse tensor."""
        with torch.inference_mode():
            model = nn.Linear(256, 1024).half().cuda().eval()
            apply_fake_sparsity(model)
            model_copy = copy.deepcopy(model)
            expected = model_copy.weight.to(dtype=torch.float)

            quantize_(
                model,
                Float8DynamicActivationFloat8WeightConfig(
                    version=2,
                    packing_format=Float8PackingFormat.SPARSE_1D_DATA_1D_METADATA,
                    granularity=PerTensor(),
                ),
            )

            original_by_to_dtype_layout = torch.ops.aten.to.dtype_layout(
                model.weight,
                dtype=torch.float,
                layout=torch.strided,
            )
            torch.testing.assert_close(
                expected, original_by_to_dtype_layout, atol=1e-1, rtol=1e-1
            )

            original_by_to_dtype = torch.ops.aten.to.dtype(
                model.weight,
                torch.float,
            )
            torch.testing.assert_close(
                expected, original_by_to_dtype, atol=1e-1, rtol=1e-1
            )


common_utils.instantiate_parametrized_tests(TestFloat8Sparse2x4_1DData1DMetadataTensor)

if __name__ == "__main__":
    unittest.main()
