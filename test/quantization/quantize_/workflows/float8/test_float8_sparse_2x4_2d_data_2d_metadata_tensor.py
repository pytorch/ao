# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD 3-Clause license found in the
# LICENSE file in the root directory of this source tree.
import copy
import importlib.util
import logging
import unittest

import torch
from torch import nn
from torch.ao.pruning import WeightNormSparsifier
from torch.testing._internal import common_utils

from torchao.ops import to_sparse_semi_structured_cutlass_sm9x_f8
from torchao.quantization import (
    Float8DynamicActivationFloat8WeightConfig,
)
from torchao.quantization.granularity import PerRow
from torchao.quantization.quant_api import (
    quantize_,
)
from torchao.quantization.quantize_.workflows import (
    Float8PackingFormat,
)
from torchao.quantization.quantize_.workflows.float8.kernels import (
    _to_sparse_semi_structured_cutedsl,
)
from torchao.quantization.utils import compute_error
from torchao.utils import is_sm_at_least_90

logging.basicConfig(
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s", level=logging.INFO
)


def apply_fake_sparsity(model):
    """Simulates 2:4 sparsity on all linear layers in a model (test setup helper)."""
    sparse_config = [
        {"tensor_fqn": f"{name}.weight"}
        for name, mod in model.named_modules()
        if isinstance(mod, nn.Linear)
    ]
    sparsifier = WeightNormSparsifier(
        sparsity_level=1.0, sparse_block_shape=(1, 4), zeros_per_block=2
    )
    sparsifier.prepare(model, sparse_config)
    sparsifier.step()
    sparsifier.squash_mask()


def create_semi_structured_tensor(r, c, dtype):
    """Returns a 1:2 sparse matrix of size (r, c), which is also 2:4 sparse."""
    choice_indices = torch.randint(0, 2, (r * c // 2,)).cuda()
    mask = (
        torch.nn.functional.one_hot(choice_indices, num_classes=2)
        .reshape(r, c)
        .contiguous()
        .to(torch.int32)
    )
    sparse_weight = mask + (torch.rand(r, c).cuda() * mask)
    return sparse_weight.to(dtype)


def _cutedsl_runtime_available():
    for module in ("cutlass", "cutlass.cute", "tvm_ffi"):
        try:
            if importlib.util.find_spec(module) is None:
                return False
        except ModuleNotFoundError:
            return False
    return True


class TestFloat8Sparse2x4_2DData2DMetadataTensor(common_utils.TestCase):
    @unittest.skipIf(not is_sm_at_least_90(), "Need H100 to run")
    @unittest.skipIf(not _cutedsl_runtime_available(), "CuTeDSL runtime unavailable")
    @common_utils.parametrize(
        "rows, cols",
        [
            (128, 64),
            (256, 128),
            (1024, 1024),
            (2048, 8192),
        ],
    )
    @common_utils.parametrize(
        "dtype",
        [torch.float8_e4m3fn, torch.float8_e5m2],
    )
    def test_cutedsl_sparse_conversion(self, rows, cols, dtype):
        weight = create_semi_structured_tensor(
            rows,
            cols,
            dtype=dtype,
        ).cuda()

        legacy_data, legacy_meta = to_sparse_semi_structured_cutlass_sm9x_f8(weight)
        cutedsl_data, cutedsl_meta = _to_sparse_semi_structured_cutedsl(weight)

        self.assertEqual(legacy_data, cutedsl_data)
        self.assertEqual(legacy_meta, cutedsl_meta)

    @unittest.skipIf(not is_sm_at_least_90(), "Need H100 to run")
    @unittest.skipIf(not _cutedsl_runtime_available(), "CuTeDSL runtime unavailable")
    @common_utils.parametrize("rows, cols", [(0, 64), (128, 0), (0, 0)])
    def test_cutedsl_sparse_conversion_empty(self, rows, cols):
        weight = torch.empty((rows, cols), dtype=torch.float8_e4m3fn, device="cuda")
        data, meta = _to_sparse_semi_structured_cutedsl(weight)
        self.assertEqual(data.shape, (rows, (cols + 31) // 32 * 16))
        self.assertEqual(meta.shape, ((rows + 63) // 64 * 64, (cols + 127) // 128 * 16))

    @unittest.skipIf(not is_sm_at_least_90(), "Need H100 to run")
    @unittest.skipIf(not _cutedsl_runtime_available(), "CuTeDSL runtime unavailable")
    def test_cutedsl_sparse_conversion_large(self):
        # rows * cols past 2^31 elements, where 32-bit element offsets would
        # overflow.
        rows, cols = 65600, 32768
        if torch.cuda.mem_get_info()[0] < 12 * 2**30:
            self.skipTest("Not enough free GPU memory")
        # Exactly two nonzero bytes in every group of four, in one of the six
        # 2:4 patterns. Nonzero values avoid 0x00 and 0x80 (+0 and -0), so the
        # compressed data is just the nonzero bytes in order.
        patterns = torch.tensor(
            [[1, 1, 0, 0], [1, 0, 1, 0], [1, 0, 0, 1], [0, 1, 1, 0], [0, 1, 0, 1]]
            + [[0, 0, 1, 1]],
            dtype=torch.uint8,
            device="cuda",
        )
        choice = torch.randint(0, 6, (rows, cols // 4), device="cuda")
        weight = patterns[choice].view(rows, cols)
        del choice
        weight *= torch.randint(1, 128, (rows, cols), dtype=torch.uint8, device="cuda")
        data, _ = _to_sparse_semi_structured_cutedsl(weight.view(torch.float8_e4m3fn))
        # Check the rows on both sides of the 2^31 boundary, and the last rows.
        for first in (2**31 // cols - 2, rows - 4):
            tail = weight[first : first + 4]
            expected = tail[tail != 0].view(4, cols // 2)
            self.assertEqual(data[first : first + 4].view(torch.uint8), expected)

    @unittest.skipIf(not is_sm_at_least_90(), "Need H100 to run")
    @unittest.skipIf(not torch.cuda.is_available(), "Need CUDA available")
    @common_utils.parametrize("compile", [True, False])
    def test_fp8_cutlass_sparse(self, compile):
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

            # Quantized
            quantize_(model_copy, Float8DynamicActivationFloat8WeightConfig())
            dense_result = model_copy(input)
            dense_sqnr = compute_error(baseline_result, dense_result)

            # Sparse + quantized
            quantize_(
                model,
                Float8DynamicActivationFloat8WeightConfig(
                    version=2,
                    packing_format=Float8PackingFormat.SPARSE_2D_DATA_2D_METADATA,
                    granularity=PerRow(),
                ),
            )
            if compile:
                model = torch.compile(model)
            sparse_result = model(input)
            sparse_sqnr = compute_error(baseline_result, sparse_result)

            self.assertEqual(dense_sqnr, sparse_sqnr)

    @unittest.skipIf(not is_sm_at_least_90(), "Need H100 to run")
    @unittest.skipIf(not torch.cuda.is_available(), "Need CUDA available")
    def test_fp8_cutlass_sparse_lowering_op_clone(self):
        with torch.inference_mode():
            model = nn.Linear(256, 1024).half().cuda().eval()
            apply_fake_sparsity(model)
            quantize_(
                model,
                Float8DynamicActivationFloat8WeightConfig(
                    version=2,
                    packing_format=Float8PackingFormat.SPARSE_2D_DATA_2D_METADATA,
                    granularity=PerRow(),
                ),
            )

            original = model.weight.dequantize()
            cloned = model.weight.clone().dequantize()

            for o, c in zip(original, cloned):
                self.assertEqual(o, c)

    @unittest.skipIf(not is_sm_at_least_90(), "Need H100 to run")
    @unittest.skipIf(not torch.cuda.is_available(), "Need CUDA available")
    def test_fp8_cutlass_sparse_lowering_op_to(self):
        # Need to run with inference mode to avoid dispatching to `aten.to_copy`
        with torch.inference_mode():
            model = nn.Linear(256, 1024).half().cuda().eval()
            apply_fake_sparsity(model)
            model_copy = copy.deepcopy(model)
            expected = model_copy.weight.to(dtype=torch.float)

            quantize_(
                model,
                Float8DynamicActivationFloat8WeightConfig(
                    version=2,
                    packing_format=Float8PackingFormat.SPARSE_2D_DATA_2D_METADATA,
                    granularity=PerRow(),
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


common_utils.instantiate_parametrized_tests(TestFloat8Sparse2x4_2DData2DMetadataTensor)

if __name__ == "__main__":
    unittest.main()
