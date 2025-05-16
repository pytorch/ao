# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD 3-Clause license found in the
# LICENSE file in the root directory of this source tree.
import unittest

import torch
from torch.testing._internal.common_quantization import QuantizationTestCase
from torch.testing._internal.common_utils import (
    instantiate_parametrized_tests,
    parametrize,
    run_tests,
)

from torchao.dtypes.uintx.uintx_layout import to_uintx
from torchao.quantization.quant_api import quantize_, uintx_weight_only
from torchao.quantization.quant_primitives import (
    MappingType,
    ZeroPointDomain,
    choose_qparams_affine,
    dequantize_affine,
    quantize_affine,
)
from torchao.utils import (
    TORCH_VERSION_AT_LEAST_2_3,
    TORCH_VERSION_AT_LEAST_2_5,
)

# torch.uintx dtypes are introduced in 2.3
if TORCH_VERSION_AT_LEAST_2_3:
    dtypes = (
        torch.uint1,
        torch.uint2,
        torch.uint3,
        torch.uint4,
        torch.uint5,
        torch.uint6,
        torch.uint7,
    )
else:
    dtypes = ()

group_sizes = [32, 64, 128]
devices = ["cpu", "cuda"]


class Linear16(torch.nn.Module):
    def __init__(self, scale, device):
        super().__init__()
        self.net = torch.nn.Sequential(
            torch.nn.Linear(
                scale * 2, scale, bias=False, dtype=torch.float16, device=device
            ),
            torch.nn.Linear(
                scale, scale, bias=False, dtype=torch.float16, device=device
            ),
            torch.nn.Linear(
                scale, scale // 2, bias=False, dtype=torch.float16, device=device
            ),
        )

    def forward(self, x):
        return self.net(x)


class TestUintx(QuantizationTestCase):
    def setUp(self):
        torch._dynamo.reset()

    def tearDown(self):
        torch._dynamo.reset()

    @parametrize("dtype", dtypes)
    @parametrize("group_size", group_sizes)
    @unittest.skipIf(not torch.cuda.is_available(), reason="CUDA not available")
    @unittest.skipIf(
        not TORCH_VERSION_AT_LEAST_2_5,
        reason="only works with fix in the nightly build",
    )
    def test_uintx_quant_on_cpu_then_move_to_cuda(self, dtype, group_size):
        scale = 512
        fp16_mod_on_cpu = Linear16(scale, "cpu")
        quantize_(fp16_mod_on_cpu, uintx_weight_only(dtype, group_size=group_size))
        test_input_on_cpu = torch.randn(scale * 2, dtype=torch.float16, device="cpu")
        output_on_cpu = fp16_mod_on_cpu(test_input_on_cpu)
        fp16_mod_on_cuda = fp16_mod_on_cpu.to("cuda")
        test_input_on_cuda = test_input_on_cpu.to("cuda")
        output_on_cuda = fp16_mod_on_cuda(test_input_on_cuda)
        self.assertTrue(
            torch.allclose(output_on_cpu, output_on_cuda.cpu(), atol=1.0e-3),
            "The output of the model on CPU and CUDA should be close",
        )

    @parametrize("dtype", dtypes)
    @parametrize("group_size", group_sizes)
    @parametrize("device", devices)
    @unittest.skipIf(not torch.cuda.is_available(), reason="CUDA not available")
    @unittest.skipIf(
        not TORCH_VERSION_AT_LEAST_2_5,
        reason="only works with fix in the nightly build",
    )
    def test_uintx_weight_only_model_quant(self, dtype, group_size, device):
        scale = 512
        fp16 = Linear16(scale, device)
        quantize_(fp16, uintx_weight_only(dtype, group_size=group_size))
        uintx = torch.compile(fp16, fullgraph=True)
        test_input = torch.randn(scale * 2, dtype=torch.float16, device=device)
        output = uintx.forward(test_input)
        self.assertTrue(output is not None, "model quantization failed")

    @parametrize("dtype", dtypes)
    @parametrize("group_size", group_sizes)
    @parametrize("device", devices)
    @unittest.skipIf(not torch.cuda.is_available(), reason="CUDA not available")
    @unittest.skipIf(
        not TORCH_VERSION_AT_LEAST_2_5,
        reason="only works with fix in the nightly build",
    )
    def test_uintx_weight_only_quant(self, dtype, group_size, device):
        input_float = torch.randn((1, 256), dtype=torch.float16, device=device)
        mapping_type = MappingType.SYMMETRIC
        eps = torch.finfo(torch.float32).eps
        zero_point_dtype = torch.int32
        zero_point_domain = ZeroPointDomain.INT
        block_size = (1, group_size)

        scale, zero_point = choose_qparams_affine(
            input_float,
            mapping_type,
            block_size,
            dtype,
            eps=eps,
            scale_dtype=torch.float32,
            zero_point_dtype=zero_point_dtype,
            preserve_zero=True,
            zero_point_domain=zero_point_domain,
        )

        aqt = quantize_affine(
            input_float,
            block_size,
            scale,
            zero_point,
            dtype,
            zero_point_domain=zero_point_domain,
        )
        # Note: output will be uint8 tensor for sub byte tensors for now

        q = to_uintx(aqt, dtype, -1)
        self.assertTrue(q is not None, "quantization failed")
        deqaunt = dequantize_affine(
            q, block_size, scale, zero_point, dtype, zero_point_domain=zero_point_domain
        )
        self.assertTrue(deqaunt is not None, "deqauntization failed")

    @parametrize("dtype", dtypes)
    @unittest.skipIf(not torch.cuda.is_available(), reason="Need CUDA available")
    @unittest.skipIf(
        not TORCH_VERSION_AT_LEAST_2_3, reason="sub byte dtype requires torch 2.3+"
    )
    def test_uintx_target_dtype(self, dtype):
        from torchao.quantization.quant_api import uintx_weight_only

        linear = torch.nn.Linear(128, 256, dtype=torch.bfloat16, device="cuda")
        # make sure it runs
        quantize_(linear, uintx_weight_only(dtype))
        linear(torch.randn(1, 128, dtype=torch.bfloat16, device="cuda"))

    @parametrize("dtype", dtypes)
    @unittest.skipIf(not torch.cuda.is_available(), reason="Need CUDA available")
    @unittest.skipIf(
        not TORCH_VERSION_AT_LEAST_2_5,
        reason="torch.compile without unwrap_tensor_subclass requires torch 2.5+",
    )
    def test_uintx_target_dtype_compile(self, dtype):
        from torchao.quantization.quant_api import uintx_weight_only

        linear = torch.nn.Linear(128, 256, dtype=torch.bfloat16, device="cuda")
        # make sure it runs
        quantize_(linear, uintx_weight_only(dtype))
        linear = torch.compile(linear)
        linear(torch.randn(1, 128, dtype=torch.bfloat16, device="cuda"))

    @parametrize("dtype", dtypes)
    @unittest.skipIf(not torch.cuda.is_available(), reason="Need CUDA available")
    @unittest.skipIf(
        not TORCH_VERSION_AT_LEAST_2_3, reason="sub byte dtype requires torch 2.3+"
    )
    def test_uintx_model_size(self, dtype):
        from torchao.quantization.quant_api import uintx_weight_only
        from torchao.utils import get_model_size_in_bytes

        # scale size = 1/64 * 2 bytes = 1/32 bytes
        # zero_point size = 1/64 * 4 bytes = 1/16 bytes
        # dtype data size = 1 * bit_width/8 = bit_width/8 bytes
        _dtype_to_ratio = {
            torch.uint1: (1 / 8 + 1 / 16 + 1 / 32) / 2,
            torch.uint2: (2 / 8 + 1 / 16 + 1 / 32) / 2,
            torch.uint3: (3 / 8 + 1 / 16 + 1 / 32) / 2,
            torch.uint4: (4 / 8 + 1 / 16 + 1 / 32) / 2,
            torch.uint5: (5 / 8 + 1 / 16 + 1 / 32) / 2,
            torch.uint6: (6 / 8 + 1 / 16 + 1 / 32) / 2,
            torch.uint7: (7 / 8 + 1 / 16 + 1 / 32) / 2,
        }
        linear = torch.nn.Sequential(
            torch.nn.Linear(128, 256, bias=False, dtype=torch.bfloat16, device="cuda")
        )
        bf16_size = get_model_size_in_bytes(linear)
        # make sure it runs
        quantize_(linear[0], uintx_weight_only(dtype))
        quantized_size = get_model_size_in_bytes(linear)
        self.assertTrue(bf16_size * _dtype_to_ratio[dtype] == quantized_size)


instantiate_parametrized_tests(TestUintx)

if __name__ == "__main__":
    run_tests()
