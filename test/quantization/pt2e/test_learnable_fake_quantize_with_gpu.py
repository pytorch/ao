# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD 3-Clause license found in the
# LICENSE file in the root directory of this source tree.

import unittest

import numpy as np
import torch

from test.quantization.pt2e.test_learnable_fake_quantize import (
    NP_RANDOM_SEED,
    _LearnableFakeQuantizeAcceleratorAwareTestCase,
)


class TestLearnableFakeQuantizeWithGPU(_LearnableFakeQuantizeAcceleratorAwareTestCase):
    def setUp(self):
        torch.manual_seed(42)
        np.random.seed(NP_RANDOM_SEED)

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA unavailable")
    def test_device_compatibility(self):
        self._test_device_compatibility(torch.device("cuda"))

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA unavailable")
    def test_numerical_consistency_per_tensor(self):
        self._test_numerical_consistency_per_tensor(torch.device("cuda"))
