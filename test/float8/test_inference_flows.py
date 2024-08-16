# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD 3-Clause license found in the
# LICENSE file in the root directory of this source tree.
import copy
import io
import os
import random
import unittest

import pytest
from unittest.mock import patch
from torchao.utils import (
    TORCH_VERSION_AT_LEAST_2_5,
    unwrap_tensor_subclass,
)

if not TORCH_VERSION_AT_LEAST_2_5:
    pytest.skip("Unsupported PyTorch version", allow_module_level=True)

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.export._trace import _export as _export_private
from torchao.float8.float8_linear_utils import convert_to_float8_training
from torchao.float8.float8_tensor import Float8Tensor
from torchao.float8.float8_utils import compute_error
from torchao.float8.inference import (
    ActivationCasting,
    Float8InferenceLinear,
    QuantConfig,
    quantize_to_float8,
)


random.seed(0)
torch.manual_seed(0)

is_H100 = torch.cuda.is_available() and torch.cuda.get_device_capability() >= (9, 0)
is_cuda_8_9 = torch.cuda.is_available() and torch.cuda.get_device_capability() >= (8, 9)

class FeedForward(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.w1 = nn.Linear(4096, 14336, bias=False)
        self.w3 = nn.Linear(4096, 14336, bias=False)
        self.w2 = nn.Linear(14336, 4096, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.w2(F.silu(self.w1(x)) * self.w3(x))

    def reset_parameters(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                m.reset_parameters()


class TestHPTrainToFP8LinearInference:
    @pytest.fixture(autouse=True)
    def setup_mock(self):
        with patch("torch._dynamo.config.cache_size_limit", 20):
            yield

    def base_test_mlp_transform(self, base_mlp, quantized_mlp, input_tensor):
        with torch.no_grad():
            base_output = base_mlp(input_tensor)
            transformed_output = quantized_mlp(input_tensor)

        # Compute and check SQNR
        sqnr = compute_error(base_output, transformed_output)
        assert sqnr.item() > 20, f"SQNR is too low: {sqnr.item()} dB"

    @pytest.mark.parametrize("compile_backend", ["eager", "inductor"])
    @pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float32])
    @unittest.skipIf(
        not torch.cuda.is_available() or not is_cuda_8_9,
        "CUDA not available or machine does not support SM89",
    )
    def test_dynamic_fp8_mlp(self, compile_backend, dtype):
        original_mlp = FeedForward().to("cuda", dtype=dtype)
        original_mlp.reset_parameters()

        dynamic_fp8_mlp = copy.deepcopy(original_mlp)

        quant_config = QuantConfig(ActivationCasting.DYNAMIC)
        quantize_to_float8(dynamic_fp8_mlp, quant_config)

        batch_size = 4
        num_tokens = 1024
        embedding_dim = 4096

        input_tensor = torch.randn(
            batch_size, num_tokens, embedding_dim, device="cuda", dtype=dtype
        )

        # Compile the models
        compiled_original_mlp = torch.compile(
            original_mlp, backend=compile_backend, fullgraph=True
        )
        compiled_dynamic_fp8_mlp = torch.compile(
            dynamic_fp8_mlp, backend=compile_backend, fullgraph=True
        )

        self.base_test_mlp_transform(
            compiled_original_mlp, compiled_dynamic_fp8_mlp, input_tensor
        )

    @pytest.mark.parametrize("compile_backend", ["eager", "inductor"])
    @pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float32])
    @unittest.skipIf(
        not torch.cuda.is_available() or not is_cuda_8_9,
        "CUDA not available or machine does not support SM89",
    )
    def test_static_fp8_mlp(self, compile_backend, dtype):
        original_mlp = FeedForward().to("cuda", dtype=dtype)
        original_mlp.reset_parameters()

        static_fp8_mlp = copy.deepcopy(original_mlp)
        quant_config = QuantConfig(
            ActivationCasting.STATIC,
            static_quantization_scale=torch.tensor(
                [1.0], device="cuda", dtype=torch.float32
            ),
        )
        quantize_to_float8(static_fp8_mlp, quant_config)

        batch_size = 4
        num_tokens = 1024
        embedding_dim = 4096

        input_tensor = torch.randn(
            batch_size, num_tokens, embedding_dim, device="cuda", dtype=dtype
        )

        # Compile the models
        compiled_original_mlp = torch.compile(
            original_mlp,
            backend=compile_backend,
            fullgraph=True,
        )
        compiled_static_fp8_mlp = torch.compile(
            static_fp8_mlp, backend=compile_backend, fullgraph=True
        )

        self.base_test_mlp_transform(
            compiled_original_mlp, compiled_static_fp8_mlp, input_tensor
        )

    @pytest.mark.parametrize("compile_backend", ["eager", "inductor"])
    @pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float32])
    @unittest.skipIf(
        not torch.cuda.is_available() or not is_cuda_8_9,
        "CUDA not available or machine does not support SM89",
    )
    def test_weight_only_fp8_mlp(self, compile_backend, dtype):
        original_mlp = FeedForward().to("cuda", dtype=dtype)
        original_mlp.reset_parameters()

        static_fp8_mlp = copy.deepcopy(original_mlp)
        quant_config = QuantConfig(ActivationCasting.WEIGHT_ONLY)
        quantize_to_float8(static_fp8_mlp, quant_config)

        batch_size = 4
        num_tokens = 1024
        embedding_dim = 4096

        input_tensor = torch.randn(
            batch_size, num_tokens, embedding_dim, device="cuda", dtype=dtype
        )

        # Compile the models
        compiled_original_mlp = torch.compile(
            original_mlp, backend=compile_backend, fullgraph=True
        )
        compiled_static_fp8_mlp = torch.compile(
            static_fp8_mlp, backend=compile_backend, fullgraph=True
        )

        self.base_test_mlp_transform(
            compiled_original_mlp, compiled_static_fp8_mlp, input_tensor
        )


class TestFP8TrainToFP8LinearInference:
    @pytest.fixture(autouse=True)
    def setup_mock(self):
        with patch("torch._dynamo.config.cache_size_limit", 20):
            yield

    def train(self, model: nn.Module, dtype: torch.dtype):
        model.train()
        optimizer = torch.optim.SGD(model.parameters(), lr=0.001)
        criterion = nn.MSELoss()
        target_tensor = torch.randn(4, 1024, 4096, device="cuda", dtype=dtype)
        for _ in range(10):
            input_tensor = torch.randn(4, 1024, 4096, device="cuda", dtype=dtype)
            optimizer.zero_grad()
            output = model(input_tensor)
            loss = criterion(output, target_tensor)
            loss.backward()
            optimizer.step()
        model.eval()
        return model

    @pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float32])
    @unittest.skipIf(
        not torch.cuda.is_available() or not is_cuda_8_9,
        "CUDA not available or machine does not support SM89",
    )
    def test_fp8_save_and_load(self, dtype: torch.dtype):
        # Initialize FP8 model
        fp8_mlp = FeedForward().to("cuda", dtype=torch.float32)
        fp8_mlp.reset_parameters()
        convert_to_float8_training(fp8_mlp)

        # Train the model
        self.train(fp8_mlp, dtype)

        # Generate input tensor and original out
        input_tensor = torch.randn(4, 1024, 4096, device="cuda", dtype=dtype)
        og_out = fp8_mlp(input_tensor)

        # Save model state dict
        buffer = io.BytesIO()
        torch.save(fp8_mlp.state_dict(), buffer)

        # Reset buffer position to the beginning
        buffer.seek(0)

        # Later on you load the model, will be w/ Float8Linear on meta device
        with torch.device("meta"):
            new_fp8_mlp = FeedForward().to(dtype=dtype)
            convert_to_float8_training(new_fp8_mlp)

        # Load the actual data
        new_fp8_mlp.load_state_dict(
            torch.load(buffer, weights_only=True), strict=True, assign=True
        )

        quant_config = QuantConfig(ActivationCasting.DYNAMIC)
        quantize_to_float8(new_fp8_mlp, quant_config)

        fp8_mod_count = 0
        for module in new_fp8_mlp.modules():
            if isinstance(module, Float8InferenceLinear):
                assert isinstance(module.weight, Float8Tensor)
                assert module.weight.requires_grad is False
                fp8_mod_count += 1
        assert fp8_mod_count == 3, "Expected 3 FP8 modules, got {}".format(
            fp8_mod_count
        )

        new_out = new_fp8_mlp(input_tensor)

        # Assert exact equality
        assert torch.all(og_out == new_out).item()


class TestFP8Export:
    @pytest.fixture(autouse=True)
    def setup_mock(self):
        with patch("torch._dynamo.config.cache_size_limit", 20):
            yield

    @unittest.skipIf(
        not torch.cuda.is_available() or not is_H100,
        "CUDA not available or on non H100 machine",
    )
    @unittest.skip("Pytorch needs a fix to ensure codegen maintains stride order")
    def test_fp8_export(self):
        export_model = FeedForward().to("cuda")
        quant_config = QuantConfig(ActivationCasting.DYNAMIC)
        quantize_to_float8(export_model, quant_config)
        batch_size = 4
        num_tokens = 1024
        embedding_dim = 4096

        inp = torch.randn(
            batch_size, num_tokens, embedding_dim, device="cuda", dtype=torch.float32
        )
        example_args = (inp,)

        fp8_compile_model = copy.deepcopy(export_model)
        fp8_compile_model = torch.compile(fp8_compile_model)
        fp8_compile_out = fp8_compile_model(*example_args)

        # Export model with subclass weights

        export_model = unwrap_tensor_subclass(export_model)

        # Export the model
        exported_model = _export_private(
            export_model,
            example_args,
            strict=False,
            pre_dispatch=False,
        )

        so_path = None
        try:
            # Compile the exported program to a .so using AOTInductor
            with torch.no_grad():
                so_path = torch._inductor.aot_compile(
                    exported_model.module(), example_args
                )

            # Load and run the .so file in Python
            res = torch._export.aot_load(so_path, device="cuda")(example_args)
            torch.testing.assert_close(fp8_compile_out, res)

        finally:
            # Cleanup: remove the .so file
            if so_path and os.path.exists(so_path):
                os.remove(so_path)


if __name__ == "__main__":
    pytest.main([__file__])
