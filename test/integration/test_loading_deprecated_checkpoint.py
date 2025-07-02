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
from transformers import AutoModelForCausalLM, AutoTokenizer, TorchAoConfig

from torchao.utils import is_sm_at_least_89

_MODEL_NAME_AND_VERSIONS = [
    ("torchao-testing/opt-125m-float8dq-row-v1-0.13-dev", 1),
]


@unittest.skipIf(not torch.cuda.is_available(), "Need CUDA available")
@unittest.skipIf(not is_sm_at_least_89(), "Nedd sm89+")
class TestLoadingDeprecatedCheckpoint(TestCase):
    @common_utils.parametrize("model_name_and_version", _MODEL_NAME_AND_VERSIONS)
    def test_load_model_and_run(self, model_name_and_version):
        """Test that we print correct warning message when loading a deprecated checkpoint
        and making sure the deprecated checkpoints can still be loaded
        """
        # Load and quantize model
        model_name, version = model_name_and_version
        with warnings.catch_warnings(record=True) as caught_warnings:
            quantized_model = AutoModelForCausalLM.from_pretrained(
                model_name,
                torch_dtype="bfloat16",
                device_map="cuda",
            )
            assert any(
                "Stored version is not the same as current default version of the config"
                in str(w.message)
                for w in caught_warnings
            ), "Didn't get expected warning message for version mismatch"

            # TODO: generalize when we test more checkpoints
            assert any(
                "Models quantized with version 1 of Float8DynamicActivationFloat8WeightConfig is deprecated"
                in str(w.message)
                for w in caught_warnings
            ), "Didn't get expected warning message for deprecation"
            assert isinstance(quantized_model.config.quantization_config, TorchAoConfig)
            assert (
                quantized_model.config.quantization_config.quant_type.version == version
            )

        tokenizer = AutoTokenizer.from_pretrained(model_name)
        prompt = ("Hello, my name is",)
        inputs = tokenizer(
            prompt,
            return_tensors="pt",
        ).to("cuda")
        generated_ids = quantized_model.generate(**inputs, max_new_tokens=128)
        # make sure it runs
        _ = tokenizer.batch_decode(
            generated_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )


common_utils.instantiate_parametrized_tests(TestLoadingDeprecatedCheckpoint)

if __name__ == "__main__":
    run_tests()
