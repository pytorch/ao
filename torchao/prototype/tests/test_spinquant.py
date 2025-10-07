# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD 3-Clause license found in the
# LICENSE file in the root directory of this source tree.

import unittest

import torch
import torch.nn as nn

from torchao.prototype.spinquant.hadamard_utils import apply_exact_had_to_linear


class TestSpinQuant(unittest.TestCase):
    def test_rotate_in_and_out(self):
        """Perform rotation to output of linear layer and inverse rotation to input of next layer; test that the output is the same."""
        with torch.no_grad():
            layer1 = nn.Linear(256, 256, bias=True)
            layer2 = nn.Linear(256, 256, bias=True)
            model = nn.Sequential(layer1, layer2)
            input = torch.rand(256)
            output = model(input)
            apply_exact_had_to_linear(layer1, output=True)
            apply_exact_had_to_linear(layer2, output=False)
            new_output = model(input)
            torch.testing.assert_allclose(output, new_output)
