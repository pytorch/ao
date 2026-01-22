# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD 3-Clause license found in the
# LICENSE file in the root directory of this source tree.
import unittest

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
    pass


# please check model card for how to generate these models

# high precision model, used for testing config deprecation warning
_HIGH_PRECISION_MODEL = "facebook/opt-125m"

_DEPRECATED_MODEL_INFO = []

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
        self,
        model_name,
        version,
        config_name,
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
                torch.nn.Linear(32, 256, dtype=torch.bfloat16)  # , device="cuda")
            )

        with open(downloaded_model, "rb") as f:
            model.load_state_dict(torch.load(f), assign=True)

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

    @common_utils.parametrize("model_info", _SINGLE_LINEAR_MODEL_INFO)
    def test_single_linear(self, model_info):
        """Test that we can load and run the quantized linear checkpoint with saved sample input
        and match the saved output, to make sure there is no BC breaking changes
        when we make changes to tensor subclass implementations
        """
        model_name, version, config_name = model_info
        self._test_single_linear_helper(
            model_name,
            version,
            config_name,
        )


common_utils.instantiate_parametrized_tests(TestLoadAndRunCheckpoint)

if __name__ == "__main__":
    run_tests()
