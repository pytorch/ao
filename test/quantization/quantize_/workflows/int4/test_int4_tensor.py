# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD 3-Clause license found in the
# LICENSE file in the root directory of this source tree.

import unittest

import torch
from torch.testing._internal.common_utils import (
    instantiate_parametrized_tests,
    parametrize,
    run_tests,
)

from torchao.quantization import (
    Float8DynamicActivationInt4WeightConfig,
    Int4WeightOnlyConfig,
    quantize_,
)
from torchao.quantization.quantize_.common import SupportsActivationPreScaling
from torchao.quantization.utils import compute_error
from torchao.testing.utils import TorchAOIntegrationTestCase
from torchao.utils import (
    _is_mslk_available,
    is_sm_at_least_90,
    is_sm_at_least_100,
    torch_version_at_least,
)

# Configs for plain int4 weight format
WEIGHT_ONLY_CONFIG = Int4WeightOnlyConfig(
    group_size=128,
    int4_packing_format="plain",
)

FP8_ACT_CONFIG = Float8DynamicActivationInt4WeightConfig(
    int4_packing_format="plain",
)


@unittest.skipIf(not torch_version_at_least("2.8.0"), "Need pytorch 2.8+")
@unittest.skipIf(not torch.cuda.is_available(), "Need CUDA available")
@unittest.skipIf(not is_sm_at_least_90(), "Need sm90+")
@unittest.skipIf(is_sm_at_least_100(), "MSLK kernel not compatible with sm100+")
@unittest.skipIf(not _is_mslk_available(), "Requires mslk >= 1.0.0")
class TestInt4Tensor(TorchAOIntegrationTestCase):
    GPU_DEVICES = ["cuda"] if torch.cuda.is_available() else []

    # ==================== Tests for both configs ====================

    @parametrize("config", [WEIGHT_ONLY_CONFIG, FP8_ACT_CONFIG])
    # sizes format: (M_shape, N, K) where input is (*M_shape, K) and linear is (K, N)
    @parametrize(
        "sizes",
        [
            ((1,), 256, 128),
            ((4, 32), 256, 128),
        ],
    )
    def test_linear(self, config, sizes):
        M, N, K = sizes
        dtype = torch.bfloat16
        device = "cuda"
        input = torch.randn(*M, K, dtype=dtype, device=device)
        linear = torch.nn.Linear(K, N, dtype=dtype, device=device)
        original = linear(input)
        quantize_(linear, config)
        quantized = linear(input)
        self.assertTrue(compute_error(original, quantized) > 18)

    @parametrize("config", [WEIGHT_ONLY_CONFIG, FP8_ACT_CONFIG])
    def test_slice(self, config):
        dtype = torch.bfloat16
        device = "cuda"
        dummy = torch.nn.Linear(256, 256, bias=False, dtype=dtype, device=device)
        dummy1 = torch.nn.Linear(256, 64, bias=False, dtype=dtype, device=device)
        dummy1.weight = torch.nn.Parameter(
            dummy.weight.narrow(0, 0, 64), requires_grad=False
        )
        dummy2 = torch.nn.Linear(128, 256, dtype=dtype, device=device)
        dummy2.weight = torch.nn.Parameter(
            dummy.weight.narrow(1, 0, 128), requires_grad=False
        )

        quantize_(dummy, config)
        weight1 = dummy.weight.narrow(0, 0, 64)
        weight2 = dummy.weight.narrow(1, 0, 128)
        self.assertEqual(weight1.qdata, dummy.weight.qdata.narrow(0, 0, 64))
        self.assertEqual(weight1.scale, dummy.weight.scale.narrow(1, 0, 64))
        self.assertEqual(weight1.zero_point, dummy.weight.zero_point.narrow(1, 0, 64))
        self.assertEqual(weight2.qdata, dummy.weight.qdata.narrow(1, 0, 64))
        self.assertEqual(weight2.scale, dummy.weight.scale.narrow(0, 0, 1))
        self.assertEqual(weight2.zero_point, dummy.weight.zero_point.narrow(0, 0, 1))

        # check for sliced weight, before and after quantization
        # does not differ too much
        input = torch.randn(2, 256, dtype=dtype, device=device)
        res_ref = dummy1(input)
        dummy.weight = torch.nn.Parameter(weight1.contiguous(), requires_grad=False)
        res = dummy(input)
        assert compute_error(res, res_ref) > 20

        input = torch.randn(2, 128, dtype=dtype, device=device)
        res_ref = dummy2(input)
        dummy.weight = torch.nn.Parameter(weight2.contiguous(), requires_grad=False)
        res = dummy(input)
        assert compute_error(res, res_ref) > 15

    @parametrize("config", [WEIGHT_ONLY_CONFIG, FP8_ACT_CONFIG])
    def test_slice_preserves_aliasing(self, config):
        l = torch.nn.Linear(1024, 1024).to("cuda").to(torch.bfloat16)
        l.weight = torch.nn.Parameter(
            torch.zeros(1024, 1024, dtype=torch.bfloat16, device="cuda")
        )
        quantize_(l, config)
        param = l.weight
        param_data = param.data
        param_data = param_data.narrow(0, 0, 512)
        # Making sure the aliasing is preserved in sliced quantized Tensor
        assert param.data.qdata.data_ptr() == param_data.qdata.data_ptr()
        assert param.data.scale.data_ptr() == param_data.scale.data_ptr()
        assert param.data.zero_point.data_ptr() == param_data.zero_point.data_ptr()

    @parametrize("config", [WEIGHT_ONLY_CONFIG, FP8_ACT_CONFIG])
    def test_slice_and_copy_similar_to_vllm(self, config):
        self._test_slice_and_copy_similar_to_vllm(config)

    @parametrize("config", [WEIGHT_ONLY_CONFIG, FP8_ACT_CONFIG])
    @parametrize(
        "sizes",
        [
            ((128,), 256, 128),
            ((32, 128), 64, 256),
            ((2, 32, 128), 64, 256),
        ],
    )
    def test_to_device(self, config, sizes):
        M, N, K = sizes
        dtype = torch.bfloat16
        for device in self.GPU_DEVICES:
            input_tensor = torch.randn(*M, K, dtype=dtype, device=device)
            linear = torch.nn.Linear(K, N, dtype=dtype)
            quantize_(linear, config)
            linear.to(device)
            linear(input_tensor)

            linear = torch.nn.Linear(K, N, dtype=dtype)
            quantize_(linear, config)
            linear.to(device=device)
            linear(input_tensor)

            linear = torch.nn.Linear(K, N, dtype=dtype)
            quantize_(linear, config)
            linear.to(device)
            linear(input_tensor)

    @parametrize("config", [WEIGHT_ONLY_CONFIG, FP8_ACT_CONFIG])
    def test_module_path(self, config):
        linear = torch.nn.Linear(128, 256, dtype=torch.bfloat16)
        quantize_(linear, config)
        self.assertEqual(
            str(type(linear.weight)),
            "<class 'torchao.quantization.Int4Tensor'>",
        )

    # ==================== Tests for bf16 activation only ====================

    def test_bmm(self):
        class M(torch.nn.Module):
            def __init__(self, weight):
                super().__init__()
                self.weight = torch.nn.Parameter(weight, requires_grad=False)

            def forward(self, x):
                return torch.bmm(x, self.weight)

        dtype = torch.bfloat16
        device = "cuda"
        input = torch.randn(10, 32, 128, dtype=dtype, device=device)
        weight = torch.randn(10, 128, 256, dtype=dtype, device=device)
        m = M(weight).eval()
        original = m(input)
        # we need to transpose the weight first for bmm
        m.weight = torch.nn.Parameter(m.weight.data.transpose(1, 2).contiguous())
        quantize_(m, WEIGHT_ONLY_CONFIG, filter_fn=lambda x, fqn: True)
        quantized = m(input)
        self.assertTrue(compute_error(original, quantized) > 18)

    @parametrize(
        "sizes",
        [
            ((128,), 256, 128),
            ((32, 128), 64, 256),
            ((2, 32, 128), 64, 256),
        ],
    )
    def test_cat(self, sizes):
        config = WEIGHT_ONLY_CONFIG
        dtype = torch.bfloat16
        device = "cuda"
        M, N, K = sizes
        linear1 = torch.nn.Linear(K, N, dtype=dtype, device=device)
        linear2 = torch.nn.Linear(K, N, dtype=dtype, device=device)
        input_cat1 = torch.randn(*M, K, dtype=dtype, device=device)

        cat_weight1 = torch.cat([linear1.weight, linear2.weight], dim=0)
        dummy_linear1 = torch.nn.Linear(K, N, bias=False, dtype=dtype, device=device)

        dummy_linear1.weight = torch.nn.Parameter(cat_weight1)
        quantize_(dummy_linear1, config)

        quantize_(linear1, config)
        quantize_(linear2, config)

        cat_qweight1 = torch.cat([linear1.weight, linear2.weight], dim=0)
        self.assertTrue(cat_qweight1.shape, (2 * N, K))
        self.assertEqual(
            dummy_linear1.weight.qdata,
            cat_qweight1.qdata,
        )
        self.assertEqual(
            dummy_linear1.weight.scale,
            cat_qweight1.scale,
        )
        self.assertEqual(
            dummy_linear1.weight.zero_point,
            cat_qweight1.zero_point,
        )

        # making sure cat_qweight1 can be used for inference
        dummy_linear1.weight = torch.nn.Parameter(cat_qweight1, requires_grad=False)
        dummy_linear1(input_cat1)

        # align the scale and zero_point before concatenation
        linear2.weight.scale = linear1.weight.scale
        linear2.weight.zero_point = linear1.weight.zero_point
        cat_qweight2 = torch.cat([linear1.weight, linear2.weight], dim=1)
        self.assertTrue(cat_qweight2.shape, (N, 2 * K))
        ref_data = torch.cat(
            [
                linear1.weight.qdata,
                linear2.weight.qdata,
            ],
            dim=1,
        )
        ref_scale = linear1.weight.scale
        ref_zero_point = linear1.weight.zero_point
        self.assertEqual(cat_qweight2.qdata, ref_data)
        self.assertEqual(cat_qweight2.scale, ref_scale)
        self.assertEqual(cat_qweight2.zero_point, ref_zero_point)

    def test_moe_weight_reshape_ops(self):
        self._test_moe_weight_reshape_ops(WEIGHT_ONLY_CONFIG)

    def test_activation_prescaling(self):
        dtype = torch.bfloat16
        device = "cuda"
        input = torch.randn(1, 128, dtype=dtype, device=device)
        linear = torch.nn.Linear(128, 256, bias=False, dtype=dtype, device=device)
        original = linear(input)
        quantize_(linear, WEIGHT_ONLY_CONFIG)
        qw = linear.weight
        assert isinstance(qw, SupportsActivationPreScaling), (
            "Expected int4 tensor supports activation prescaling"
        )
        assert qw.act_pre_scale is None, "Default `act_pre_scale` is None"
        _ACT_PRE_SCALE = 2
        qw.act_pre_scale = _ACT_PRE_SCALE
        quantized = linear(input)

        # making sure activation pre scaling is successfully applied to the activation
        self.assertTrue(compute_error(original * _ACT_PRE_SCALE, quantized) > 20)

    def test_weight_attributes_bf16(self):
        dtype = torch.bfloat16
        device = "cuda"
        linear = torch.nn.Linear(128, 256, dtype=dtype, device=device)
        quantize_(linear, WEIGHT_ONLY_CONFIG)

        weight = linear.weight
        self.assertEqual(weight.activation_dtype, torch.bfloat16)

    # ==================== Tests for fp8 activation only ====================

    def test_weight_attributes_fp8(self):
        dtype = torch.bfloat16
        device = "cuda"
        linear = torch.nn.Linear(128, 256, dtype=dtype, device=device)
        quantize_(linear, FP8_ACT_CONFIG)

        weight = linear.weight
        self.assertEqual(weight.activation_dtype, torch.float8_e4m3fn)


instantiate_parametrized_tests(TestInt4Tensor)

if __name__ == "__main__":
    run_tests()
