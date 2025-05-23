# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD 3-Clause license found in the
# LICENSE file in the root directory of this source tree.

import importlib.util
import os
import random
from pathlib import Path

import numpy as np
import pytest
import torch

from torchao.utils import TORCH_VERSION_AT_LEAST_2_7

if not TORCH_VERSION_AT_LEAST_2_7:
    pytest.skip("Requires PyTorch 2.7 or higher", allow_module_level=True)


VLLM_AVAILABLE = importlib.util.find_spec("vllm") is not None
TRANSFORMERS_AVAILABLE = importlib.util.find_spec("transformers") is not None

if not VLLM_AVAILABLE:
    pytest.skip("vLLM not installed", allow_module_level=True)

if not TRANSFORMERS_AVAILABLE:
    pytest.skip("transformers not installed", allow_module_level=True)

from transformers import AutoModelForCausalLM, AutoTokenizer, TorchAoConfig
from vllm import LLM, SamplingParams

from torchao.prototype.mx_formats import MXGemmKernelChoice
from torchao.prototype.mx_formats.mx_subclass import MXFPInferenceConfig
from torchao.quantization.granularity import PerRow, PerTensor
from torchao.quantization.quant_api import (
    CutlassInt4PackedLayout,
    Float8DynamicActivationFloat8WeightConfig,
    GemliteUIntXWeightOnlyConfig,
    Int4DynamicActivationInt4WeightConfig,
    Int4WeightOnlyConfig,
    Int8DynamicActivationInt4WeightConfig,
    Int8DynamicActivationInt8WeightConfig,
    Int8WeightOnlyConfig,
)


class TestVLLMIntegration:
    """Integration tests for vLLM with quantized models."""

    @classmethod
    def setup_class(cls):
        """Set up test environment."""
        # Set seeds for reproducibility
        cls.set_seed(42)

        # Set vLLM environment variables
        os.environ["VLLM_USE_V1"] = "1"
        os.environ["VLLM_ENABLE_V1_MULTIPROCESSING"] = "0"
        os.environ["VLLM_TEST_STANDALONE_COMPILE"] = "1"

    @classmethod
    def teardown_class(cls):
        """Clean up after all tests."""
        torch.cuda.empty_cache()
        import gc

        gc.collect()

    def setup_method(self, method):
        """Clean up before each test method."""
        torch.cuda.empty_cache()
        import gc

        gc.collect()

    def teardown_method(self, method):
        """Clean up after each test method."""
        torch.cuda.empty_cache()
        import gc

        gc.collect()

    @staticmethod
    def set_seed(seed):
        """Set random seeds for reproducibility."""
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    def get_quantization_config(self, quant_type: str, granularity: str = "per_tensor"):
        """Create TorchAo quantization config based on provided parameters."""
        granularity_mapping = {
            "per_row": PerRow(),
            "per_tensor": PerTensor(),
        }

        gran = granularity_mapping[granularity]

        match quant_type:
            case "autoquant":
                return TorchAoConfig("autoquant", min_sqnr=40.0)
            case "fp8":
                return TorchAoConfig(
                    Float8DynamicActivationFloat8WeightConfig(granularity=gran)
                )
            case "int4_weight_only":
                return TorchAoConfig(Int4WeightOnlyConfig(group_size=128))
            case "int8_weight_only":
                return TorchAoConfig(Int8WeightOnlyConfig())
            case "int8_dynamic_act_int8_weight":
                return TorchAoConfig(Int8DynamicActivationInt8WeightConfig())
            case "gemlite":
                return TorchAoConfig(GemliteUIntXWeightOnlyConfig())
            case "A4W4":
                return TorchAoConfig(Int4DynamicActivationInt4WeightConfig())
            case "A8W4":
                return TorchAoConfig(
                    Int8DynamicActivationInt4WeightConfig(
                        layout=CutlassInt4PackedLayout()
                    )
                )
            case "mxfp8":
                return TorchAoConfig(MXFPInferenceConfig())
            case "mxfp4":
                return TorchAoConfig(
                    MXFPInferenceConfig(
                        activation_dtype=torch.float4_e2m1fn_x2,
                        weight_dtype=torch.float4_e2m1fn_x2,
                        block_size=32,
                        gemm_kernel_choice=MXGemmKernelChoice.CUTLASS,
                    )
                )
            case _:
                raise ValueError(f"Unsupported quantization type: {quant_type}")

    def quantize_and_save_model(
        self,
        model_name: str,
        quant_type: str,
        output_dir: Path,
        granularity: str = "per_tensor",
    ):
        """Quantize a model and save it to disk."""
        # Get quantization config
        quantization_config = self.get_quantization_config(quant_type, granularity)

        # Load and quantize model
        print(f"Loading and quantizing model with {quant_type}...")
        quantized_model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype="bfloat16",
            device_map="cuda",
            quantization_config=quantization_config,
        )

        # Load tokenizer
        tokenizer = AutoTokenizer.from_pretrained(model_name)

        # Quick test generation to verify model works
        test_input = "Hello, world!"
        input_ids = tokenizer(test_input, return_tensors="pt").to(
            quantized_model.device
        )

        with torch.no_grad():
            output = quantized_model.generate(**input_ids, max_new_tokens=5)
            decoded = tokenizer.decode(output[0], skip_special_tokens=True)
            print(f"Quick test - Input: {test_input}, Output: {decoded}")

        # Save quantized model
        print(f"Saving quantized model to {output_dir}...")
        quantized_model.save_pretrained(output_dir, safe_serialization=False)
        tokenizer.save_pretrained(output_dir)

        # Clean up to free memory
        del quantized_model
        torch.cuda.empty_cache()

        return output_dir

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    @pytest.mark.skipif(not VLLM_AVAILABLE, reason="vLLM not installed")
    @pytest.mark.parametrize(
        "quant_type,granularity",
        [
            # ("fp8", "per_tensor"),
            ("fp8", "per_row"),
            # ("int8_weight_only", "per_tensor"),
            # ("int4_weight_only", "per_tensor"),
            # ("A8W4", "per_tensor"),
        ],
    )
    @pytest.mark.parametrize("compile", [True, False])
    @pytest.mark.parametrize(
        "tp_size", [1, 2] if torch.cuda.device_count() > 1 else [1]
    )
    def test_vllm_smoke_test(self, tmp_path, quant_type, granularity, compile, tp_size):
        """Test vLLM generation with quantized models."""
        # Skip per_row tests if not supported
        torch._dynamo.reset()
        if granularity == "per_row" and not torch.cuda.get_device_capability()[0] >= 9:
            pytest.skip("Per-row quantization requires SM90+")

        # Use a small model for testing
        base_model = "facebook/opt-125m"

        # Quantize the model
        output_dir = tmp_path / f"{quant_type}-{granularity}-opt-125m"
        quantized_model_path = self.quantize_and_save_model(
            base_model, quant_type, output_dir, granularity
        )

        # Test generation with vLLM
        sampling_params = SamplingParams(
            temperature=0.8,
            top_p=0.95,
            seed=42,
            max_tokens=16,  # Small for testing
        )

        # Create LLM instance
        llm = LLM(
            model=str(quantized_model_path),
            tensor_parallel_size=tp_size,
            enforce_eager=not compile,
            dtype="bfloat16",
        )

        # Test prompts
        prompts = [
            "Hello, my name is",
            "The capital of France is",
        ]

        # Generate outputs
        outputs = llm.generate(prompts, sampling_params)

        # Verify outputs
        assert len(outputs) == len(prompts)
        for output in outputs:
            assert output.prompt in prompts
            assert len(output.outputs) > 0
            generated_text = output.outputs[0].text
            assert isinstance(generated_text, str)
            assert len(generated_text) > 0

        # Clean up
        del llm
        torch.cuda.empty_cache()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
