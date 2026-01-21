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

from torchao.prototype.quantization.float8_static_quant.prototype_float8_tensor import (
    PrototypeFloat8Tensor,
    _choose_quant_func_and_quantize_tensor,
)
from torchao.prototype.quantization.quant_api import (
    Float8ObservedLinear,
    Float8StaticActivationFloat8WeightConfig,
)
from torchao.quantization import (
    Float8DynamicActivationFloat8WeightConfig,
    quantize_,
)
from torchao.quantization.granularity import PerRow, PerTensor
from torchao.quantization.utils import compute_error
from torchao.testing.model_architectures import ToyTwoLinearModel
from torchao.testing.utils import TorchAOIntegrationTestCase
from torchao.utils import (
    is_sm_at_least_90,
)


# copied from test/quantization/quantize_/workflows/float8/test_float8_tensor.py
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

    def forward(self, x):
        return self.conv(x)


@unittest.skipIf(not torch.cuda.is_available(), "Need CUDA available")
@unittest.skipIf(not is_sm_at_least_90(), "Need sm90+")
@common_utils.instantiate_parametrized_tests
class TestFloat8StaticActivation(TorchAOIntegrationTestCase):
    def setUp(self):
        super().setUp()
        self.dtype = torch.bfloat16
        torch.manual_seed(42)

    @common_utils.parametrize("granularity", [PerRow(), PerTensor()])
    def test_static_activation_float8_weight(self, granularity):
        """Test that static quantization matches dynamic quantization when using the same scale"""
        torch.compiler.reset()

        dtype = torch.bfloat16

        M, N, K = 32, 32, 32
        input_tensor = torch.randn(M, K, dtype=dtype, device="cuda")

        model = torch.nn.Linear(K, N, bias=False).eval().to(device="cuda", dtype=dtype)
        model_static_quant = copy.deepcopy(model)
        model_dynamic_quant = copy.deepcopy(model)

        # Apply dynamic quantization
        dynamic_config = Float8DynamicActivationFloat8WeightConfig(
            granularity=granularity,
        )
        quantize_(model_dynamic_quant, dynamic_config)

        dynamic_out_eager = model_dynamic_quant(input_tensor)
        model_dynamic_quant = torch.compile(model_dynamic_quant, fullgraph=True)
        dynamic_out_compile = model_dynamic_quant(input_tensor)

        # Get activation scale from dynamic quantization
        float8_input = _choose_quant_func_and_quantize_tensor(
            input_tensor, model_dynamic_quant.weight.act_quant_kwargs
        )
        static_config = Float8StaticActivationFloat8WeightConfig(
            act_quant_scale=float8_input.scale.detach().clone(),
            granularity=granularity,
        )
        quantize_(model_static_quant, static_config)

        # Verify weight is PrototypeFloat8Tensor
        self.assertIsInstance(model_static_quant.weight, PrototypeFloat8Tensor)
        self.assertIsNotNone(model_static_quant.weight.act_quant_scale)
        self.assertIsNotNone(model_static_quant.weight.act_quant_kwargs)

        static_out_eager = model_static_quant(input_tensor)
        model_static_quant = torch.compile(model_static_quant, fullgraph=True)
        static_out_compile = model_static_quant(input_tensor)

        sqnr_static_vs_dynamic_eager = compute_error(
            dynamic_out_eager, static_out_eager
        )
        sqnr_static_vs_dynamic_compile = compute_error(
            dynamic_out_compile, static_out_compile
        )
        self.assertGreater(
            sqnr_static_vs_dynamic_eager,
            40,
            "SQNR of static v.s. dynamic (eager) should be > 40 dB",
        )
        self.assertGreater(
            sqnr_static_vs_dynamic_compile,
            40,
            "SQNR of static v.s. dynamic (compile) should be > 40 dB",
        )

    @common_utils.parametrize("granularity", [PerRow(), PerTensor()])
    def test_creation_and_attributes(self, granularity):
        """Test tensor creation, dtypes, and attributes"""
        M, N, K = 32, 32, 32
        dtype = torch.bfloat16

        input_tensor = torch.randn(M, K, dtype=dtype, device="cuda")
        linear = torch.nn.Linear(K, N, bias=False, dtype=dtype, device="cuda")

        # First get a scale from dynamic quantization
        dynamic_config = Float8DynamicActivationFloat8WeightConfig(
            granularity=granularity,
        )
        model_dynamic = copy.deepcopy(linear)
        quantize_(model_dynamic, dynamic_config)

        quantized_input = _choose_quant_func_and_quantize_tensor(
            input_tensor, model_dynamic.weight.act_quant_kwargs
        )

        static_config = Float8StaticActivationFloat8WeightConfig(
            act_quant_scale=quantized_input.scale.detach().clone(),
            granularity=granularity,
        )
        quantize_(linear, static_config)

        w = linear.weight

        # Verify attributes
        self.assertEqual(w.shape, (N, K))
        self.assertEqual(w.qdata.dtype, torch.float8_e4m3fn)
        self.assertIsInstance(w, PrototypeFloat8Tensor)
        self.assertIsNotNone(w.act_quant_kwargs)
        self.assertIsNotNone(w.act_quant_scale)

        # Check scale shape based on granularity
        if isinstance(granularity, PerRow):
            self.assertEqual(w.scale.shape, (N, 1))
        elif isinstance(granularity, PerTensor):
            self.assertEqual(w.scale.shape, (1, 1))

    @common_utils.parametrize("dtype", [torch.bfloat16, torch.float32])
    @common_utils.parametrize("compile", [True, False])
    @common_utils.parametrize("inference_mode", [True, False])
    # test for 2D/3D conv
    # Inputs are (N, C_in, C_out, (D, H, W), kernel_size or
    # (N, C_in, C_out, (H, W), kernel_size
    @common_utils.parametrize(
        "sizes",
        [
            (1, 160, 320, (3, 194, 130), 3),
            # Note: kernel_size can't be 1, otherwise
            # the weight will be channels_last even though
            # it's contiguous because of the value of
            # stride
            (1, 320, 640, (96, 64), 3),
        ],
    )
    def test_fp8_conv_variants(
        self,
        dtype: torch.dtype,
        compile: bool,
        inference_mode: bool,
        sizes: Tuple,
    ):
        torch.compiler.reset()
        granularity = PerTensor()
        N, C_in, C_out, spatial_dims, kernel_size = sizes
        dim = len(spatial_dims)
        convs = {1: torch.nn.Conv1d, 2: torch.nn.Conv2d, 3: torch.nn.Conv3d}
        assert dim in convs, f"Unsupported dim: {dim}"
        conv_class = convs[dim]
        _is_conv = lambda m, fqn: isinstance(m, conv_class)

        input_tensor = torch.randn(N, C_in, *spatial_dims, dtype=dtype, device="cuda")

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

        channels_last_memory_format = (
            torch.channels_last_3d if dim == 3 else torch.channels_last
        )
        input_tensor = input_tensor.to(memory_format=channels_last_memory_format)
        model = model.to(memory_format=channels_last_memory_format)

        quantized_model = copy.deepcopy(model)

        dynamic_config = Float8DynamicActivationFloat8WeightConfig(
            granularity=granularity,
        )
        model_dynamic_quant = copy.deepcopy(model)
        quantize_(model_dynamic_quant, dynamic_config, filter_fn=_is_conv)
        # Get activation scale from dynamic quantization
        tmp_input_tensor = _choose_quant_func_and_quantize_tensor(
            input_tensor.clone(), model_dynamic_quant.conv.weight.act_quant_kwargs
        )
        config = Float8StaticActivationFloat8WeightConfig(
            act_quant_scale=tmp_input_tensor.scale.detach().clone(),
            granularity=granularity,
        )
        quantize_(quantized_model, config, filter_fn=_is_conv)

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

    def test_static_quant_flow_with_observers(self):
        """
        Test the full static quantization flow following the AWQ-style API.

        This follows the AWQ pattern:
        1. Prepare model by inserting observers (step="prepare")
        2. Calibrate with representative data
        3. Convert observed model to quantized model (step="convert")
        """
        torch.compiler.reset()
        torch.manual_seed(42)

        dtype = torch.bfloat16

        # Create model
        model = ToyTwoLinearModel(
            input_dim=64, hidden_dim=64, output_dim=32, dtype=dtype, device="cuda"
        ).eval()
        example_inputs = model.example_inputs(batch_size=4)

        # Get reference output before quantization
        before_quant = model(*example_inputs)

        # Step 1: Prepare model by inserting observers
        quantize_(model, Float8StaticActivationFloat8WeightConfig(step="prepare"))

        # Verify observers were inserted
        self.assertIsInstance(model.linear1, Float8ObservedLinear)
        self.assertIsInstance(model.linear2, Float8ObservedLinear)

        # Step 2: Calibrate with representative data
        for _ in range(10):
            model(*example_inputs)

        # Step 3: Convert observed model to quantized model
        quantize_(model, Float8StaticActivationFloat8WeightConfig(step="convert"))

        # Verify quantization was applied
        self.assertIsInstance(model.linear1.weight, PrototypeFloat8Tensor)
        self.assertIsInstance(model.linear2.weight, PrototypeFloat8Tensor)
        self.assertIsNotNone(model.linear1.weight.act_quant_scale)
        self.assertIsNotNone(model.linear2.weight.act_quant_scale)

        # Test inference
        after_quant = model(*example_inputs)

        # Verify quantization quality
        error = compute_error(before_quant, after_quant)
        self.assertGreater(
            error,
            20,
            f"SQNR of quantized vs original should be > 20 dB, got {error}",
        )

        # Test with torch.compile
        model_compiled = torch.compile(model, fullgraph=True)
        after_quant_compiled = model_compiled(*example_inputs)

        error_compiled = compute_error(before_quant, after_quant_compiled)
        self.assertGreater(
            error_compiled,
            20,
            f"SQNR of compiled quantized vs original should be > 20 dB, got {error_compiled}",
        )


if __name__ == "__main__":
    common_utils.run_tests()
