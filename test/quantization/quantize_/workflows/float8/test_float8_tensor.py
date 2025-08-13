# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD 3-Clause license found in the
# LICENSE file in the root directory of this source tree.

import copy
import unittest
from contextlib import nullcontext
from typing import Tuple

import torch
from torch.testing._internal import common_utils
from torch.testing._internal.common_utils import (
    run_tests,
)

from torchao.quantization import (
    Float8DynamicActivationFloat8WeightConfig,
    Float8WeightOnlyConfig,
    PerRow,
    PerTensor,
    quantize_,
)
from torchao.quantization.quantize_.common import KernelPreference
from torchao.quantization.utils import compute_error
from torchao.testing.utils import TorchAOIntegrationTestCase
from torchao.utils import (
    TORCH_VERSION_AT_LEAST_2_8,
    _is_fbgemm_genai_gpu_available,
    is_sm_at_least_89,
    is_sm_at_least_90,
)

# Needed since changing args to function causes recompiles
torch._dynamo.config.cache_size_limit = 128


class ToyLinearModel(torch.nn.Module):
    def __init__(self, in_features, out_features):
        super().__init__()
        self.linear1 = torch.nn.Linear(in_features, out_features, bias=False)
        self.linear2 = torch.nn.Linear(out_features, in_features, bias=False)

    def forward(self, x):
        x = self.linear1(x)
        x = self.linear2(x)
        return x


# TODO: move tests in test_affine_quantized_float.py here after we migrated all implementations
@unittest.skipIf(not TORCH_VERSION_AT_LEAST_2_8, "Need pytorch 2.8+")
@unittest.skipIf(not torch.cuda.is_available(), "Need CUDA available")
@unittest.skipIf(not is_sm_at_least_89(), "Need sm89+")
class TestFloat8Tensor(TorchAOIntegrationTestCase):
    def setUp(self):
        self.GPU_DEVICES = ["cuda"] if torch.cuda.is_available() else []

    @unittest.skipIf(not torch.cuda.is_available(), "Need CUDA available")
    @unittest.skipIf(
        not is_sm_at_least_89(), "Requires GPU with compute capability >= 8.9"
    )
    @common_utils.parametrize("dtype", [torch.bfloat16, torch.float32])
    @common_utils.parametrize("mode", ["dynamic", "weight-only"])
    @common_utils.parametrize("compile", [True, False])
    @common_utils.parametrize("granularity", [PerTensor(), PerRow()])
    @common_utils.parametrize(
        "kernel_preference",
        [KernelPreference.AUTO, KernelPreference.TORCH, KernelPreference.FBGEMM],
    )
    # Inputs are (M,..), K, N
    @common_utils.parametrize(
        "sizes",
        [
            ((128,), 256, 128),
            ((32, 128), 64, 256),
        ],
    )
    def test_fp8_linear_variants(
        self,
        dtype: torch.dtype,
        mode: str,
        compile: bool,
        granularity,
        kernel_preference: KernelPreference,
        sizes: Tuple,
    ):
        error_message = None
        if isinstance(granularity, PerRow):
            if mode == "dynamic" and dtype != torch.bfloat16:
                error_message = "PerRow quantization only works for bfloat16 precision"

        if mode == "weight-only" and kernel_preference != KernelPreference.AUTO:
            return unittest.skip(
                "weight only quant only uses AUTO kernel preference right now"
            )

        if kernel_preference == KernelPreference.FBGEMM and (
            (not _is_fbgemm_genai_gpu_available()) or (not is_sm_at_least_90())
        ):
            return unittest.skip(
                "Requires fbgemm_gpu_genai to run fbgemm kernel preference test"
            )

        error_context = (
            self.assertRaisesRegex(AssertionError, error_message)
            if error_message
            else nullcontext()
        )

        with error_context:
            M, N, K = sizes
            input_tensor = torch.randn(*M, K, dtype=dtype, device="cuda")

            # Create a linear layer with bfloat16 dtype
            model = ToyLinearModel(K, N).eval().to(dtype).to("cuda")

            quantized_model = copy.deepcopy(model)

            if mode == "dynamic":
                config = Float8DynamicActivationFloat8WeightConfig(
                    granularity=granularity,
                    kernel_preference=kernel_preference,
                )
            else:
                assert mode == "weight-only", f"Unsupported mode: {mode}"
                config = Float8WeightOnlyConfig()

            quantize_(quantized_model, config)

            if compile:
                quantized_model = torch.compile(quantized_model, fullgraph=True)

            output_original = model(input_tensor)
            output_quantized = quantized_model(input_tensor)

            error = compute_error(output_original, output_quantized)
            assert compute_error(output_original, output_quantized) > 20, (
                f"Quantization error is too high got a SQNR of {error}"
            )

    @common_utils.parametrize("granularity", [PerTensor(), PerRow()])
    @unittest.skipIf(
        not is_sm_at_least_90(),
        "Failing in SM89 right now: "
        "AssertionError: tensor(False, device='cuda:0') is not true : sqnr: -2.90625, will fix a bit later",
    )
    def test_slice(self, granularity):
        config = Float8DynamicActivationFloat8WeightConfig(granularity=granularity)
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
        weight1 = dummy.weight.clone().narrow(0, 0, 64)
        weight2 = dummy.weight.clone().narrow(1, 0, 128)
        self.assertEqual(
            weight1.qdata,
            dummy.weight.qdata.narrow(0, 0, 64),
        )
        self.assertEqual(
            weight2.qdata,
            dummy.weight.qdata.narrow(1, 0, 128),
        )
        if isinstance(granularity, PerRow):
            self.assertEqual(
                weight1.scale,
                dummy.weight.scale.narrow(0, 0, 64),
            )
            self.assertEqual(
                weight2.scale,
                dummy.weight.scale,
            )
        else:
            self.assertEqual(
                weight1.scale,
                dummy.weight.scale,
            )
            self.assertEqual(
                weight2.scale,
                dummy.weight.scale,
            )

        # check for sliced weight, before and after float8 quantization
        # does not differ too much
        input = torch.randn(2, 256, dtype=dtype, device=device)
        res_ref = dummy1(input)
        dummy.weight = torch.nn.Parameter(weight1.contiguous(), requires_grad=False)
        res = dummy(input)
        sqnr = compute_error(res, res_ref)
        self.assertTrue(sqnr > 25, f"sqnr: {sqnr}")

        input = torch.randn(2, 128, dtype=dtype, device=device)
        res_ref = dummy2(input)
        dummy.weight = torch.nn.Parameter(weight2.contiguous(), requires_grad=False)
        res = dummy(input)
        sqnr = compute_error(res, res_ref)
        self.assertTrue(sqnr > 15, f"sqnr: {sqnr}")

    @common_utils.parametrize("granularity", [PerTensor(), PerRow()])
    # Inputs are (M,..), K, N
    @common_utils.parametrize(
        "sizes",
        [
            ((128,), 256, 128),
            ((32, 128), 64, 256),
        ],
    )
    def test_kernel_preference_numerical_equivalence(self, granularity, sizes):
        """Test different kernel preferences have the same numerics for float8 dynamic activation
        and float8 weight config
        """
        M, N, K = sizes
        dtype = torch.bfloat16
        input_tensor = torch.randn(*M, K, dtype=dtype, device="cuda")
        # Create a linear layer with bfloat16 dtype
        model = ToyLinearModel(K, N).eval().to(dtype).to("cuda")

        # reference kernel preference and results
        # we are using KerenelPreference.TORCH as the reference
        kp_ref = KernelPreference.TORCH
        config = Float8DynamicActivationFloat8WeightConfig(
            granularity=granularity, kernel_preference=kp_ref
        )
        quantized_model = copy.deepcopy(model)
        quantize_(quantized_model, config)
        res_ref = quantized_model(input_tensor)

        other_kernel_preferences = [
            KernelPreference.AUTO,
        ]
        if _is_fbgemm_genai_gpu_available() and is_sm_at_least_90():
            other_kernel_preferences.append(KernelPreference.FBGEMM)

        quantized_outputs = {}
        for kp in other_kernel_preferences:
            config = Float8DynamicActivationFloat8WeightConfig(
                granularity=granularity, kernel_preference=kp
            )
            quantized_model = copy.deepcopy(model)
            quantize_(quantized_model, config)
            quantized_outputs[kp] = quantized_model(input_tensor)

        from torchao.quantization.utils import compute_error

        # comparing numerics between different kernel preferences, using TORCH as the standard
        kp_and_res = list(quantized_outputs.items())
        for i in range(len(kp_and_res)):
            kp, res = kp_and_res[i]
            self.assertTrue(
                compute_error(res, res_ref) > 28,
                f"mismatch between {kp=} and {kp_ref}, {sizes=}, {granularity=}",
            )

    @common_utils.parametrize("granularity", [PerTensor(), PerRow()])
    def test_slice_preserves_aliasing(self, granularity):
        config = Float8DynamicActivationFloat8WeightConfig(granularity=granularity)
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

    @common_utils.parametrize("granularity", [PerTensor(), PerRow()])
    def test_slice_and_copy_similar_to_vllm(self, granularity):
        config = Float8DynamicActivationFloat8WeightConfig(granularity=granularity)
        self._test_slice_and_copy_similar_to_vllm(config)

    @unittest.skipIf(not is_sm_at_least_90(), "Nedd sm90+")
    def test_bmm(self):
        # only support per row quantization
        config = Float8DynamicActivationFloat8WeightConfig(granularity=PerRow())

        class M(torch.nn.Module):
            def __init__(self, weight):
                super().__init__()
                self.weight = weight

            def forward(self, x):
                return torch.bmm(x, self.weight)

        dtype = torch.bfloat16
        device = "cuda"
        input = torch.randn(10, 32, 128, dtype=dtype, device=device)
        weight = torch.randn(10, 128, 256, dtype=dtype, device=device)
        m = M(weight).eval()
        original = m(input)
        # we need to transpose the weight first for bmm
        m.weight = torch.nn.Parameter(m.weight.transpose(1, 2).contiguous())
        quantize_(m, config, filter_fn=lambda x, fqn: True)
        quantized = m(input)
        self.assertTrue(compute_error(original, quantized) > 20)

    @common_utils.parametrize("granularity", [PerTensor(), PerRow()])
    @common_utils.parametrize(
        "sizes",
        [
            ((128,), 256, 128),
            ((32, 128), 64, 256),
            ((2, 32, 128), 64, 256),
        ],
    )
    def test_to_device(self, granularity, sizes):
        config = Float8DynamicActivationFloat8WeightConfig(granularity=granularity)
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

    @common_utils.parametrize("granularity", [PerTensor(), PerRow()])
    @common_utils.parametrize(
        "sizes",
        [
            ((128,), 256, 128),
            ((32, 128), 64, 256),
            ((2, 32, 128), 64, 256),
        ],
    )
    def test_cat(self, granularity, sizes):
        config = Float8DynamicActivationFloat8WeightConfig(granularity=granularity)
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

        # making sure cat_qweight1 can be used for inference
        dummy_linear1.weight = torch.nn.Parameter(cat_qweight1, requires_grad=False)
        dummy_linear1(input_cat1)

        # align the scale before concatenation
        linear2.weight.scale = linear1.weight.scale
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
        self.assertEqual(cat_qweight2.qdata, ref_data)
        self.assertEqual(cat_qweight2.scale, ref_scale)

    @unittest.skipIf(not is_sm_at_least_90(), "Nedd sm90+")
    def test_moe_weight_reshape_ops(self):
        # only per row quantization is supported for bmm
        granularity = PerRow()
        config = Float8DynamicActivationFloat8WeightConfig(granularity=granularity)
        self._test_moe_weight_reshape_ops(config)


common_utils.instantiate_parametrized_tests(TestFloat8Tensor)

if __name__ == "__main__":
    run_tests()
