# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD 3-Clause license found in the
# LICENSE file in the root directory of this source tree.
import unittest
import warnings

import torch
from torch.testing._internal import common_utils
from torch.testing._internal.common_utils import (
    TestCase,
    run_tests,
)

from torchao.utils import is_fbcode, is_sm_at_least_90

if is_fbcode():
    # don't import from transformer internally, since some imports might be missing
    pass
else:
    from transformers import AutoModelForCausalLM, AutoTokenizer, TorchAoConfig


# please check model card for how to generate these models

# high precision model, used for testing config deprecation warning
_HIGH_PRECISION_MODEL = "facebook/opt-125m"

_DEPRECATED_SINGLE_LINEAR_MODEL_INFO = [
    # model card: https://huggingface.co/torchao-testing/single-linear-Float8DynamicActivationFloat8WeightConfig-v1-0.13.dev
    (
        "torchao-testing/single-linear-Float8DynamicActivationFloat8WeightConfig-v1-0.13.dev",
        1,
        "Float8DynamicActivationFloat8WeightConfig",
    ),
    # model card: https://huggingface.co/torchao-testing/single-linear-Int4WeightOnlyConfig-v1-0.14.dev
    (
        "torchao-testing/single-linear-Int4WeightOnlyConfig-v1-0.14.dev",
        1,
        "Int4WeightOnlyConfig",
    ),
    # model card: https://huggingface.co/torchao-testing/single-linear-IntxWeightOnlyConfig-v1-0.14.dev
    (
        "torchao-testing/single-linear-IntxWeightOnlyConfig-v1-0.14.dev",
        1,
        "IntxWeightOnlyConfig",
    ),
    # model card: https://huggingface.co/torchao-testing/single-linear-Int8DynamicActivationIntxWeightConfig-v1-0.14.dev
    (
        "torchao-testing/single-linear-Int8DynamicActivationIntxWeightConfig-v1-0.14.dev",
        1,
        "Int8DynamicActivationIntxWeightConfig",
    ),
]

_DEPRECATED_MODEL_INFO = [
    # model card: https://huggingface.co/torchao-testing/opt-125m-Float8DynamicActivationFloat8WeightConfig-v1-0.13.dev
    (
        "torchao-testing/opt-125m-Float8DynamicActivationFloat8WeightConfig-v1-0.13.dev",
        1,
        "Float8DynamicActivationFloat8WeightConfig",
    ),
    # model card: https://huggingface.co/torchao-testing/opt-125m-Int4WeightOnlyConfig-v1-0.14.dev
    (
        "torchao-testing/opt-125m-Int4WeightOnlyConfig-v1-0.14.dev",
        1,
        "Int4WeightOnlyConfig",
    ),
    # https://huggingface.co/torchao-testing/opt-125m-IntxWeightOnlyConfig-v1-0.14.0.dev
    (
        "torchao-testing/opt-125m-IntxWeightOnlyConfig-v1-0.14.0.dev",
        1,
        "IntxWeightOnlyConfig",
    ),
    # https://huggingface.co/torchao-testing/opt-125m-Int8DynamicActivationIntxWeightConfig-v1-0.14.0.dev
    (
        "torchao-testing/opt-125m-Int8DynamicActivationIntxWeightConfig-v1-0.14.0.dev",
        1,
        "Int8DynamicActivationIntxWeightConfig",
    ),
]

_SINGLE_LINEAR_MODEL_INFO = [
    # model card: https://huggingface.co/torchao-testing/single-linear-Float8DynamicActivationFloat8WeightConfig-v2-0.13.dev
    (
        "torchao-testing/single-linear-Float8DynamicActivationFloat8WeightConfig-v2-0.13.dev",
        2,
        "Float8DynamicActivationFloat8WeightConfig",
    ),
    # model card: https://huggingface.co/torchao-testing/single-linear-Int4WeightOnlyConfig-v2-0.13.dev
    (
        "torchao-testing/single-linear-Int4WeightOnlyConfig-v2-0.13.dev",
        2,
        "Int4WeightOnlyConfig",
    ),
    # model card: https://huggingface.co/torchao-testing/single-linear-Int4WeightOnlyConfig-preshuffled-v2-0.13.dev
    (
        "torchao-testing/single-linear-Int4WeightOnlyConfig-preshuffled-v2-0.13.dev",
        2,
        "Int4WeightOnlyConfig",
    ),
    # model card: https://huggingface.co/torchao-testing/single-linear-IntxWeightOnlyConfig-v2-0.14.dev
    (
        "torchao-testing/single-linear-IntxWeightOnlyConfig-v2-0.14.dev",
        2,
        "IntxWeightOnlyConfig",
    ),
    # model card: https://huggingface.co/torchao-testing/single-linear-Int8DynamicActivationIntxWeightConfig-v2-0.14.dev
    (
        "torchao-testing/single-linear-Int8DynamicActivationIntxWeightConfig-v2-0.14.dev",
        2,
        "Int8DynamicActivationIntxWeightConfig",
    ),
]


@unittest.skipIf(not torch.cuda.is_available(), "Need CUDA available")
@unittest.skipIf(not is_sm_at_least_90(), "Checkpoints are produced in SM90+")
@unittest.skipIf(
    is_fbcode(),
    "Skipping the test in fbcode for now, not sure how to download from transformers",
)
class TestLoadAndRunCheckpoint(TestCase):
    def _test_single_linear_helper(
        self, model_name, version, config_name, is_deprecated
    ):
        from huggingface_hub import hf_hub_download

        downloaded_model = hf_hub_download(model_name, filename="model.pt")
        # Load model weights, example inputs and reference output,
        # run the loaded model and make sure the result matches reference output

        with torch.device("meta"):
            # 32 and 256 are the args we used when we save the model, see
            # model card:
            # https://huggingface.co/torchao-testing/single-linear-FP8-v2-0.13-dev
            model = torch.nn.Sequential(
                torch.nn.Linear(32, 256, dtype=torch.bfloat16, device="cuda")
            )

        with (
            open(downloaded_model, "rb") as f,
            warnings.catch_warnings(record=True) as caught_warnings,
        ):
            model.load_state_dict(torch.load(f), assign=True)
            if is_deprecated:
                assert any(
                    f"Models quantized with version {version} of {config_name} is deprecated"
                    in str(w.message)
                    for w in caught_warnings
                ), (
                    f"Didn't get expected warning message for deprecation for model: {model_name}"
                )

        downloaded_example_inputs = hf_hub_download(
            model_name, filename="model_inputs.pt"
        )
        with open(downloaded_example_inputs, "rb") as f:
            example_inputs = torch.load(f)
        downloaded_output = hf_hub_download(model_name, filename="model_output.pt")
        with open(downloaded_output, "rb") as f:
            ref_output = torch.load(f)

        output = model(*example_inputs)
        self.assertTrue(torch.equal(output, ref_output))

    @common_utils.parametrize("model_info", _DEPRECATED_SINGLE_LINEAR_MODEL_INFO)
    def test_deprecated_single_linear(self, model_info):
        model_name, version, config_name = model_info
        self._test_single_linear_helper(
            model_name, version, config_name, is_deprecated=True
        )

    @common_utils.parametrize("model_info", _SINGLE_LINEAR_MODEL_INFO)
    def test_single_linear(self, model_info):
        """Test that we can load and run the quantized linear checkpoint with saved sample input
        and match the saved output, to make sure there is no BC breaking changes
        when we make changes to tensor subclass implementations
        """
        model_name, version, config_name = model_info
        self._test_single_linear_helper(
            model_name, version, config_name, is_deprecated=False
        )

    @common_utils.parametrize("model_info", _DEPRECATED_MODEL_INFO)
    def test_deprecated_hf_models(self, model_info):
        """Test that we print correct warning message when loading a deprecated checkpoint
        and making sure the deprecated checkpoints can still be loaded
        """
        # Load and quantize model
        model_name, version, config_name = model_info
        with warnings.catch_warnings(record=True) as caught_warnings:
            quantized_model = AutoModelForCausalLM.from_pretrained(
                model_name,
                torch_dtype="bfloat16",
                device_map="cuda:0",
            )
            # version mismatch check in config.py
            assert any(
                "Stored version is not the same as current default version of the config"
                in str(w.message)
                for w in caught_warnings
            ), (
                f"Didn't get expected warning message for version mismatch for config {config_name}, model {model_name}"
            )

            # checkpoint deprecation
            assert any(
                f"Models quantized with version {version} of {config_name} is deprecated"
                in str(w.message)
                for w in caught_warnings
            ), (
                f"Didn't get expected warning message for deprecation for model {model_name}"
            )
            assert isinstance(quantized_model.config.quantization_config, TorchAoConfig)
            assert (
                quantized_model.config.quantization_config.quant_type.version == version
            )

        tokenizer = AutoTokenizer.from_pretrained(model_name)
        from huggingface_hub import hf_hub_download

        downloaded_example_inputs = hf_hub_download(
            model_name, filename="model_prompt.pt"
        )
        with open(downloaded_example_inputs, "rb") as f:
            prompt = torch.load(f)

        inputs = tokenizer(
            prompt,
            return_tensors="pt",
        ).to("cuda")
        generated_ids = quantized_model.generate(
            **inputs,
            max_new_tokens=128,
        )

        downloaded_output = hf_hub_download(model_name, filename="model_output.pt")
        with open(downloaded_output, "rb") as f:
            ref_generated_ids = torch.load(f)

        self.assertTrue(torch.equal(generated_ids, ref_generated_ids))

        # make sure can successfully decode
        _ = tokenizer.batch_decode(
            generated_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )

        # make sure we throw warning for config deprecation
        with warnings.catch_warnings(record=True) as caught_warnings:
            _ = AutoModelForCausalLM.from_pretrained(
                _HIGH_PRECISION_MODEL,
                torch_dtype="bfloat16",
                device_map="cuda:0",
                quantization_config=quantized_model.config.quantization_config,
            )
            # config version deprecation in quant_api.py
            assert any(
                f"Config Deprecation: version {version} of {config_name} is deprecated and will no longer be supported in a future release"
                in str(w.message)
                for w in caught_warnings
            ), (
                f"Didn't get expected warning message for version deprecation for config {config_name}, model {model_name}"
            )


common_utils.instantiate_parametrized_tests(TestLoadAndRunCheckpoint)

if __name__ == "__main__":
    run_tests()
