# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD 3-Clause license found in the
# LICENSE file in the root directory of this source tree.

import unittest

import torch
from parameterized import parameterized

from torchao.testing.model_architectures import (
    create_model_and_input_data,
)
from torchao.utils import get_available_devices


class TestModels(unittest.TestCase):
    @parameterized.expand([(device,) for device in get_available_devices()])
    def test_toy_linear_model(self, device):
        # Skip if device is not available
        if device == "cuda" and not torch.cuda.is_available():
            self.skipTest("CUDA not available")

        model, input_data = create_model_and_input_data(
            "linear", 64, 32, 16, device=device
        )
        output = model(input_data)
        self.assertEqual(output.shape, (1, 16))

    @parameterized.expand([(device,) for device in get_available_devices()])
    def test_ln_linear_activation_model(self, device):
        # Skip if device is not available
        if device == "cuda" and not torch.cuda.is_available():
            self.skipTest("CUDA not available")

        model, input_data = create_model_and_input_data(
            "ln_linear_sigmoid", 10, 64, 32, device=device
        )
        output = model(input_data)
        self.assertEqual(output.shape, (10, 32))

    @parameterized.expand([(device,) for device in get_available_devices()])
    def test_transformer_block(self, device):
        # Skip if device is not available
        if device == "cuda" and not torch.cuda.is_available():
            self.skipTest("CUDA not available")

        model, input_data = create_model_and_input_data("transformer_block", 10, 64, 32)
        output = model(input_data)
        self.assertEqual(output.shape, (10, 16, 64))


if __name__ == "__main__":
    unittest.main()
