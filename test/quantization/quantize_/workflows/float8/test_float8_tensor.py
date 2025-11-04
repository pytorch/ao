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
from torch._inductor.utils import run_and_get_code
from torch.testing import FileCheck
from torch.testing._internal import common_utils
from torch.testing._internal.common_utils import run_tests

from torchao.quantization import (
    Float8DynamicActivationFloat8WeightConfig,
    Float8WeightOnlyConfig,
    PerBlock,
    PerRow,
    PerTensor,
    quantize_,
)
from torchao.quantization.quantize_.common import KernelPreference
from torchao.quantization.quantize_.workflows.float8.float8_tensor import Float8Tensor
from torchao.quantization.utils import compute_error
from torchao.testing.utils import TorchAOIntegrationTestCase
from torchao.utils import (
    _is_fbgemm_gpu_genai_available,
    is_sm_at_least_89,
    is_sm_at_least_90,
    is_sm_at_least_100,
    torch_version_at_least,
)

# Needed since changing args to function causes recompiles
torch._dynamo.config.cache_size_limit = 128


class ToyLinearModel(torch.nn.Module):
    def __init__(self, in_features, out_features, bias):
        super().__init__()
        self.linear1 = torch.nn.Linear(in_features, out_features, bias=bias)
        self.linear2 = torch.nn.Linear(out_features, in_features, bias=bias)

    def forward(self, x):
        x = self.linear1(x)
        x = self.linear2(x)
        return x


class ToyConvModel(torch.nn.Module):
    def __init__(
        self, dim, in_channels, out_channels, kernel_size, bias, padding, dtype, device
    ):
        super().__init__()
        convs = {1: torch.nn.Conv1d, 2: torch.nn.Conv2d, 3: torch.nn.Conv3d}
        self.conv = convs[dim](
            in_channels,
            out_channels,
            kernel_size,
            bias=bias,
            padding=padding,
            dtype=dtype,
            device=device,
        )
        if dim == 3:
            self.conv = self.conv.to(memory_format=torch.channels_last_3d)

    def forward(self, x):
        return self.conv(x)


# TODO: move tests in test_affine_quantized_float.py here after we migrated all implementations
@unittest.skipIf(not torch_version_at_least("2.8.0"), "Need pytorch 2.8+")
@unittest.skipIf(not torch.cuda.is_available(), "Need CUDA available")
@unittest.skipIf(not is_sm_at_least_89(), "Need sm89+")
class TestFloat8Tensor(TorchAOIntegrationTestCase):
    def setUp(self):
        self.GPU_DEVICES = ["cuda"] if torch.cuda.is_available() else []
        torch.set_grad_enabled(False)

    @unittest.skipIf(not torch.cuda.is_available(), "Need CUDA available")
    @unittest.skipIf(
        not is_sm_at_least_89(), "Requires GPU with compute capability >= 8.9"
    )
    @common_utils.parametrize("dtype", [torch.bfloat16, torch.float32])
    @common_utils.parametrize("mode", ["dynamic", "weight-only"])
    @common_utils.parametrize("compile", [True, False])
    @common_utils.parametrize(
        "granularity",
        [PerTensor(), PerRow(), (PerBlock((1, 128)), PerBlock((128, 128)))],
    )
    @common_utils.parametrize(
        "kernel_preference",
        [KernelPreference.AUTO, KernelPreference.TORCH, KernelPreference.FBGEMM],
    )
    # Inputs are (M,..), K, N
    @common_utils.parametrize(
        "sizes",
        [
            ((128,), 256, 128),
            ((32, 128), 256, 512),
        ],
    )
    @common_utils.parametrize("bias", [False, True])
    @torch.no_grad()
    def test_fp8_linear_variants(
        self,
        dtype: torch.dtype,
        mode: str,
        compile: bool,
        granularity,
        kernel_preference: KernelPreference,
        sizes: Tuple,
        bias: bool,
    ):
        if isinstance(granularity, PerTensor):
            if kernel_preference is KernelPreference.FBGEMM:
                return unittest.skip(
                    "per tensor with fbgemm kernel preference does not work yet"
                )
            elif mode == "weight-only":
                return unittest.skip("unimplemented")

        elif granularity == (PerBlock((1, 128)), PerBlock((128, 128))):
            if dtype is not torch.bfloat16:
                return unittest.skip("unimplemented")
            elif mode != "dynamic":
                return unittest.skip("unimplemented")
            elif kernel_preference not in (
                KernelPreference.AUTO,
                KernelPreference.TORCH,
            ):
                return unittest.skip("unimplemented")

            if bias is True:
                sizes_to_keep = ((128,), 256, 128)
                if (
                    sizes != sizes_to_keep
                    or kernel_preference is not KernelPreference.TORCH
                ):
                    return unittest.skip(
                        "cut down on number of options to save test time"
                    )

        error_message = None
        if isinstance(granularity, PerRow):
            if mode == "dynamic" and dtype != torch.bfloat16:
                error_message = "PerRow quantization only works for bfloat16 precision"

        if mode == "weight-only" and kernel_preference != KernelPreference.AUTO:
            return unittest.skip(
                "weight only quant only uses AUTO kernel preference right now"
            )

        if kernel_preference == KernelPreference.FBGEMM and (
            (not _is_fbgemm_gpu_genai_available()) or (not is_sm_at_least_90())
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
            model = ToyLinearModel(K, N, bias).eval().to(dtype).to("cuda")

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

            # ensure weight scaling is what we expect
            qs1 = quantized_model.linear1.weight.scale
            qs2 = quantized_model.linear2.weight.scale
            if granularity == PerTensor():
                assert qs1.shape == (1, 1)
                assert qs2.shape == (1, 1)
            elif granularity == PerRow():
                assert qs1.shape == (N, 1)
                assert qs2.shape == (K, 1)
            else:
                assert granularity == (PerBlock((1, 128)), PerBlock((128, 128)))
                assert qs1.shape == (N // 128, K // 128)
                assert qs2.shape == (K // 128, N // 128)

            if compile:
                quantized_model = torch.compile(quantized_model, fullgraph=True)

            output_original = model(input_tensor)
            output_quantized = quantized_model(input_tensor)

            error = compute_error(output_original, output_quantized)
            assert compute_error(output_original, output_quantized) > 20, (
                f"Quantization error is too high got a SQNR of {error}"
            )

    @unittest.skipIf(not torch.cuda.is_available(), "Need CUDA available")
    @unittest.skipIf(
        not is_sm_at_least_100(), "Requires GPU with compute capability >= 10.0"
    )
    @common_utils.parametrize("dtype", [torch.bfloat16, torch.float32])
    @common_utils.parametrize("compile", [True, False])
    @common_utils.parametrize("granularity", [PerTensor()])
    @common_utils.parametrize("inference_mode", [True, False])
    @common_utils.parametrize(
        "kernel_preference",
        [KernelPreference.AUTO],
    )
    # only test for 3D conv for now
    # Inputs are (N, C_in, C_out, D, H, W)
    @common_utils.parametrize(
        "sizes",
        [
            (4, 16, 64, 32, 32, 32),
        ],
    )
    def test_fp8_conv_variants(
        self,
        dtype: torch.dtype,
        compile: bool,
        granularity,
        inference_mode: bool,
        kernel_preference: KernelPreference,
        sizes: Tuple,
    ):
        if (not _is_fbgemm_gpu_genai_available()) or (not is_sm_at_least_100()):
            return unittest.skip(
                "Requires fbgemm_gpu_genai and sm version >= 10.0 to run "
                "fbgemm kernel preference test"
            )

        dim = 3
        N, C_in, C_out, D, H, W = sizes
        kernel_size = 3

        # Note: this is channel last memory format
        input_tensor = torch.randn(N, C_in, D, H, W, dtype=dtype, device="cuda")
        input_tensor = input_tensor.to(memory_format=torch.channels_last_3d)

        # Create a linear layer with bfloat16 dtype
        model = ToyConvModel(
            dim,
            C_in,
            C_out,
            kernel_size,
            bias=False,
            padding=0,
            dtype=dtype,
            device="cuda",
        ).eval()

        quantized_model = copy.deepcopy(model)

        config = Float8DynamicActivationFloat8WeightConfig(
            granularity=granularity,
            kernel_preference=kernel_preference,
        )

        _is_conv3d = lambda m, fqn: isinstance(m, torch.nn.Conv3d)

        quantize_(quantized_model, config, filter_fn=_is_conv3d)

        if compile:
            quantized_model = torch.compile(quantized_model, fullgraph=True)

        inference_mode_ctx = torch.inference_mode() if inference_mode else nullcontext()
        with inference_mode_ctx:
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
        model = ToyLinearModel(K, N, bias=False).eval().to(dtype).to("cuda")

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
        if (
            _is_fbgemm_gpu_genai_available()
            and is_sm_at_least_90()
            and not isinstance(granularity, PerTensor)
        ):
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
    @unittest.skipIf(not _is_fbgemm_gpu_genai_available(), "Need fbgemm_gpu_genai")
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
    @unittest.skipIf(not _is_fbgemm_gpu_genai_available(), "Need fbgemm_gpu_genai")
    def test_moe_weight_reshape_ops(self):
        # only per row quantization is supported for bmm
        granularity = PerRow()
        config = Float8DynamicActivationFloat8WeightConfig(granularity=granularity)
        self._test_moe_weight_reshape_ops(config)

    # TODO: we have some other tests living in https://github.com/pytorch/ao/blob/4ecc89edd7b5cfc12e6f80854c85d04c472a0eb0/test/dtypes/test_affine_quantized_float.py#L743
    # that should be moved here after v1 config is deprecated:
    # https://github.com/pytorch/ao/issues/2649
    @unittest.skipIf(not is_sm_at_least_90(), "Nedd sm90+")
    @unittest.skipIf(not _is_fbgemm_gpu_genai_available(), "Need fbgemm_gpu_genai")
    def test_expected_gpu_kernel_fbgemm(self):
        """Making sure KernelPreference.FBGEMM calls correct quantize and gemm kernels
        and the bias add happens in the gemm kernel for per row quantization
        """
        torch.compiler.reset()

        M, K, N = 128, 256, 512
        m = torch.nn.Sequential(
            torch.nn.Linear(K, N, device="cuda", dtype=torch.bfloat16)
        )
        config = Float8DynamicActivationFloat8WeightConfig(
            granularity=PerRow(),
            kernel_preference=KernelPreference.FBGEMM,
        )
        quantize_(m, config)
        m = torch.compile(m)
        x = torch.randn(M, K, device="cuda", dtype=torch.bfloat16)
        out, code = run_and_get_code(m, x)

        # 1. check at least one occurrence of the quantize op and rowwise gemm op
        # 2. check that there are no additional kernels like `triton_poi_fused_add_0`
        # are run, since the bias add should happen in the `f8f8bf16_rowwise.default`
        # op instead of separately
        FileCheck().check_count(
            "torch.ops.triton.quantize_fp8_row.default(", 1
        ).check_count("torch.ops.fbgemm.f8f8bf16_rowwise.default(", 1).check_not(
            ".run("
        ).run(code[0])

    @unittest.skipIf(not is_sm_at_least_90(), "Nedd sm90+")
    def test_index_select(self):
        """
        test that `x_0 = x[0]` works when `x` is a 3D `Float8Tensor`. This is
        useful when stitching checkpoints of `num_experts` 2D parameters into
        a single 3D parameter when converting between model definitions that
        use 2D and 3D parameters for their expert weights.
        """

        E, K, N = 128, 256, 512
        x = torch.randn(E, N, K, device="cuda", dtype=torch.bfloat16)
        x_fp8 = Float8Tensor.from_hp(x)
        x_fp8_1 = x_fp8[1]
        torch.testing.assert_close(
            x_fp8.dequantize()[1], x_fp8_1.dequantize(), atol=0, rtol=0
        )

    @common_utils.parametrize("granularity", [PerTensor(), PerRow()])
    @common_utils.parametrize(
        "sizes",
        [
            ((128,), 256, 128),
            ((32, 128), 64, 256),
        ],
    )
    def test_unsqueeze_operation(self, granularity, sizes):
        """Test aten.unsqueeze.default operation on Float8Tensor"""
        config = Float8DynamicActivationFloat8WeightConfig(granularity=granularity)
        dtype = torch.bfloat16
        device = "cuda"
        M, N, K = sizes

        # Create a linear layer and quantize it
        linear = torch.nn.Linear(K, N, bias=False, dtype=dtype, device=device)
        quantize_(linear, config)

        original_weight = linear.weight
        original_shape = original_weight.shape

        # Test unsqueeze operation at dim=0 (only supported dimension)
        unsqueezed_weight = original_weight.unsqueeze(0)

        # Verify the unsqueezed tensor has correct shape
        expected_shape = [1] + list(original_shape)
        self.assertEqual(unsqueezed_weight.shape, torch.Size(expected_shape))

        # Verify qdata and scale shapes
        expected_qdata_shape = [1] + list(original_weight.qdata.shape)
        expected_scale_shape = [1] + list(original_weight.scale.shape)

        self.assertEqual(
            unsqueezed_weight.qdata.shape, torch.Size(expected_qdata_shape)
        )
        self.assertEqual(
            unsqueezed_weight.scale.shape, torch.Size(expected_scale_shape)
        )

        # Verify block_size is correctly updated
        expected_block_size = []
        for i in range(len(expected_shape)):
            expected_block_size.append(expected_shape[i] // expected_scale_shape[i])

        self.assertEqual(unsqueezed_weight.block_size, expected_block_size)

        # Test that metadata is preserved
        self.assertEqual(unsqueezed_weight.mm_config, original_weight.mm_config)
        self.assertEqual(
            unsqueezed_weight.act_quant_kwargs, original_weight.act_quant_kwargs
        )
        self.assertEqual(
            unsqueezed_weight.kernel_preference, original_weight.kernel_preference
        )
        self.assertEqual(unsqueezed_weight.dtype, original_weight.dtype)

        # Test numerical correctness
        original_dequant = original_weight.dequantize()
        unsqueezed_dequant = unsqueezed_weight.dequantize()
        expected_dequant = original_dequant.unsqueeze(0)

        self.assertEqual(unsqueezed_dequant, expected_dequant)

    @common_utils.parametrize("granularity", [PerTensor(), PerRow()])
    def test_unsqueeze_error_cases(self, granularity):
        """Test error cases for aten.unsqueeze.default operation"""
        config = Float8DynamicActivationFloat8WeightConfig(granularity=granularity)
        dtype = torch.bfloat16
        device = "cuda"

        # Create a linear layer and quantize it
        linear = torch.nn.Linear(128, 256, bias=False, dtype=dtype, device=device)
        quantize_(linear, config)

        weight = linear.weight

        # Test that unsqueezing on unsupported dimensions raises an error
        with self.assertRaisesRegex(AssertionError, "Only dim == 0 is supported"):
            weight.unsqueeze(1)  # dim=1 should not be supported

    @common_utils.parametrize("granularity", [PerTensor(), PerRow()])
    @common_utils.parametrize("slice_dim", [0, 1, 2])
    @common_utils.parametrize(
        "tensor_shape",
        [
            (8, 128, 256),  # 3D tensor: batch, seq_len, hidden_dim
            (4, 64, 128),  # smaller 3D tensor
        ],
    )
    def test_slice_3d_operation(self, granularity, slice_dim, tensor_shape):
        """Test slicing operations on 3D Float8Tensor across all dimensions"""
        config = Float8DynamicActivationFloat8WeightConfig(granularity=granularity)
        dtype = torch.bfloat16
        device = "cuda"

        B, S, H = tensor_shape

        # Create a 3D tensor and quantize it (simulating a batched weight tensor)
        original_tensor = torch.randn(B, S, H, dtype=dtype, device=device)

        # Create Float8Tensor from the 3D high-precision tensor
        float8_tensor = Float8Tensor.from_hp(
            original_tensor,
            granularity=granularity,
            mm_config=config.mm_config,
        )

        slice_size = tensor_shape[slice_dim]
        start_idx = 1
        end_idx = slice_size - 1

        # Perform slicing on the specified dimension
        if slice_dim == 0:
            sliced_tensor = float8_tensor[start_idx:end_idx, :, :]
            expected_qdata = float8_tensor.qdata[start_idx:end_idx, :, :]
            expected_scale = float8_tensor.scale[start_idx:end_idx, :]
        elif slice_dim == 1:
            sliced_tensor = float8_tensor[:, start_idx:end_idx, :]
            expected_qdata = float8_tensor.qdata[:, start_idx:end_idx, :]
            expected_scale = float8_tensor.scale[:, start_idx:end_idx]
        elif slice_dim == 2:
            sliced_tensor = float8_tensor[:, :, start_idx:end_idx]
            expected_qdata = float8_tensor.qdata[:, :, start_idx:end_idx]
            expected_scale = float8_tensor.scale[:, :]

        if isinstance(granularity, PerTensor):
            # Per-tensor quantization: scale should remain scalar
            expected_scale = float8_tensor.scale

        # Verify the sliced tensor shape
        expected_shape = list(tensor_shape)
        expected_shape[slice_dim] = end_idx - start_idx
        self.assertEqual(sliced_tensor.shape, torch.Size(expected_shape))

        # Verify qdata shape matches
        self.assertEqual(sliced_tensor.qdata.shape, torch.Size(expected_shape))
        self.assertEqual(sliced_tensor.qdata, expected_qdata)

        # Verify scale shape is correct based on granularity and slice dimension
        if isinstance(granularity, PerTensor):
            # Per-tensor quantization: scale should remain scalar
            self.assertEqual(sliced_tensor.scale.numel(), 1)
        else:
            # Per-row quantization: scale shape depends on which dimension we sliced
            if slice_dim == 0:
                # Slicing batch dimension affects scale
                expected_scale_shape = list(float8_tensor.scale.shape)
                expected_scale_shape[0] = end_idx - start_idx
                self.assertEqual(
                    sliced_tensor.scale.shape, torch.Size(expected_scale_shape)
                )
            elif slice_dim == 1:
                # Slicing sequence dimension affects scale
                expected_scale_shape = list(float8_tensor.scale.shape)
                expected_scale_shape[1] = end_idx - start_idx
                self.assertEqual(
                    sliced_tensor.scale.shape, torch.Size(expected_scale_shape)
                )
            else:
                # Slicing hidden dimension (dim=2) typically doesn't affect scale in per-row quantization
                self.assertEqual(sliced_tensor.scale.shape, float8_tensor.scale.shape)

        self.assertEqual(sliced_tensor.scale, expected_scale)

        # Verify block_size is correctly updated
        self.assertEqual(len(sliced_tensor.block_size), len(expected_shape))
        for i in range(len(expected_shape)):
            expected_block_dim = min(float8_tensor.block_size[i], expected_shape[i])
            self.assertEqual(sliced_tensor.block_size[i], expected_block_dim)

        # Test that metadata is preserved
        self.assertEqual(sliced_tensor.mm_config, float8_tensor.mm_config)
        self.assertEqual(sliced_tensor.act_quant_kwargs, float8_tensor.act_quant_kwargs)
        self.assertEqual(
            sliced_tensor.kernel_preference, float8_tensor.kernel_preference
        )
        self.assertEqual(sliced_tensor.dtype, float8_tensor.dtype)

        # Test numerical correctness by comparing dequantized results
        original_dequantized = float8_tensor.dequantize()
        if slice_dim == 0:
            sliced_original = original_dequantized[start_idx:end_idx, :, :]
        elif slice_dim == 1:
            sliced_original = original_dequantized[:, start_idx:end_idx, :]
        elif slice_dim == 2:
            sliced_original = original_dequantized[:, :, start_idx:end_idx]
        sliced_dequantized = sliced_tensor.dequantize()

        self.assertEqual(sliced_dequantized, sliced_original)


common_utils.instantiate_parametrized_tests(TestFloat8Tensor)

if __name__ == "__main__":
    run_tests()
