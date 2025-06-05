# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD 3-Clause license found in the
# LICENSE file in the root directory of this source tree.

import importlib.metadata
import importlib.util
import os
import random
import shutil
from pathlib import Path
from typing import List

import numpy as np
import pytest
import torch

from packaging import version
from torchao.utils import TORCH_VERSION_AT_LEAST_2_7

if not TORCH_VERSION_AT_LEAST_2_7:
    pytest.skip("Requires PyTorch 2.7 or higher", allow_module_level=True)


VLLM_AVAILABLE = importlib.util.find_spec("vllm") is not None
TRANSFORMERS_AVAILABLE = importlib.util.find_spec("transformers") is not None

if not VLLM_AVAILABLE:
    pytest.skip("vLLM not installed", allow_module_level=True)

if not TRANSFORMERS_AVAILABLE:
    pytest.skip("transformers not installed", allow_module_level=True)

if VLLM_AVAILABLE:
    vllm_version = importlib.metadata.version("vllm")
    # Bad vLLM version due to adding AOPerModuleConfig
    if version.parse(vllm_version) == version.parse("0.9.0"):
        pytest.skip("vLLM version must be greater than 0.9.0", allow_module_level=True)

from transformers import AutoModelForCausalLM, AutoTokenizer, TorchAoConfig
from vllm import LLM, SamplingParams

from torchao.quantization.granularity import PerRow, PerTensor
from torchao.quantization.quant_api import (
    CutlassInt4PackedLayout,
    Float8DynamicActivationFloat8WeightConfig,
    Int8DynamicActivationInt4WeightConfig,
    Int8WeightOnlyConfig,
)


def get_tests() -> List[TorchAoConfig]:
    """Get all the tests based off of device info"""

    # Helper objects for granularity
    per_tensor = PerTensor()
    per_row = PerRow()

    BASE_TESTS = [TorchAoConfig(Int8WeightOnlyConfig())]
    SM89_TESTS = [
        TorchAoConfig(
            Float8DynamicActivationFloat8WeightConfig(granularity=per_tensor)
        ),
        TorchAoConfig(Float8DynamicActivationFloat8WeightConfig(granularity=per_row)),
    ]
    SM90_ONLY_TESTS = [
        TorchAoConfig(
            Int8DynamicActivationInt4WeightConfig(layout=CutlassInt4PackedLayout())
        )
    ]
    SM100_TESTS = [
        # TorchAoConfig(MXFPInferenceConfig())
    ]  # Failing for : https://github.com/pytorch/ao/issues/2239

    # Check CUDA availability first
    if not torch.cuda.is_available():
        return []  # No CUDA, no tests

    major, minor = torch.cuda.get_device_capability()

    # Build test list based on compute capability
    all_tests = []

    # Always include base tests if we have CUDA
    all_tests.extend(BASE_TESTS)

    # Add SM89+ tests
    if major > 8 or (major == 8 and minor >= 9):
        all_tests.extend(SM89_TESTS)

    # Add SM100+ tests
    if major >= 10:
        all_tests.extend(SM100_TESTS)

    # Only work for sm 90
    if major == 9:
        all_tests.extend(SM90_ONLY_TESTS)

    return all_tests


class TestVLLMIntegration:
    """Integration tests for vLLM with quantized models."""

    @classmethod
    def setup_class(cls):
        """Set up test environment."""
        # Set seeds for reproducibility
        cls.set_seed(42)

        # See https://github.com/pytorch/ao/issues/2239 for details
        os.environ["VLLM_TEST_STANDALONE_COMPILE"] = "1"
        # For Small testing this makes it faster
        os.environ["VLLM_ENABLE_V1_MULTIPROCESSING"] = "0"

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

    def quantize_and_save_model(
        self,
        model_name: str,
        quantization_config: TorchAoConfig,
        output_dir: Path,
    ):
        """Quantize a model and save it to disk."""
        # Load and quantize model
        quantized_model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype="bfloat16",
            device_map="cuda",
            quantization_config=quantization_config,
        )
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        # Save quantized model
        quantized_model.save_pretrained(output_dir, safe_serialization=False)
        tokenizer.save_pretrained(output_dir)

        # Clean up to free memory
        del quantized_model
        torch.cuda.empty_cache()

        return output_dir

    def cleanup_model_directory(self, model_path: Path):
        """Clean up the model directory safely."""
        try:
            if model_path.exists() and model_path.is_dir():
                shutil.rmtree(model_path)
        except (OSError, PermissionError) as e:
            # Log the error but don't fail the test
            print(f"Warning: Failed to clean up {model_path}: {e}")

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    @pytest.mark.skipif(not VLLM_AVAILABLE, reason="vLLM not installed")
    @pytest.mark.parametrize(
        "quantization_config", get_tests(), ids=lambda config: f"{config.quant_type}"
    )
    @pytest.mark.parametrize("compile", [True, False])
    @pytest.mark.parametrize(
        "tp_size", [1, 2] if torch.cuda.device_count() > 1 else [1]
    )
    def test_vllm_smoke_test(self, tmp_path, quantization_config, compile, tp_size):
        """Test vLLM generation with quantized models."""
        # Skip per_row tests if not supported
        torch._dynamo.reset()

        # Use a small model for testing
        base_model = "facebook/opt-125m"

        # Create a descriptive name for the output directory
        config_name = str(quantization_config).replace("/", "_").replace(" ", "_")[:50]
        output_dir = tmp_path / f"{config_name}-opt-125m"

        llm = None
        quantized_model_path = None

        try:
            # Quantize the model
            quantized_model_path = self.quantize_and_save_model(
                base_model, quantization_config, output_dir
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
                num_gpu_blocks_override=128,
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

        finally:
            # Clean up resources
            if llm is not None:
                del llm

            # Clean up CUDA memory
            torch.cuda.empty_cache()

            # Clean up the saved model directory
            if quantized_model_path is not None:
                self.cleanup_model_directory(quantized_model_path)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
