# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD 3-Clause license found in the
# LICENSE file in the root directory of this source tree.

import unittest

from torchao.testing.model_architectures import create_model_and_input_data


class TestModels(unittest.TestCase):
    def test_toy_linear_model(self):
        model, input_data = create_model_and_input_data("linear", 10, 64, 32)
        output = model(input_data)
        self.assertEqual(output.shape, (10, 32))

    def test_ln_linear_activation_model(self):
        model, input_data = create_model_and_input_data("ln_linear_sigmoid", 10, 64, 32)
        output = model(input_data)
        self.assertEqual(output.shape, (10, 32))

    def test_transformer_block(self):
        model, input_data = create_model_and_input_data("transformer_block", 10, 64, 32)
        output = model(input_data)
        self.assertEqual(output.shape, (10, 16, 64))


if __name__ == "__main__":
    unittest.main()
