# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD 3-Clause license found in the
# LICENSE file in the root directory of this source tree.

# Owner(s): ["oncall: quantization"]
import unittest

import torch
from torch.testing._internal.common_quantization import skipIfNoQNNPACK
from torch.testing._internal.common_utils import TEST_CUDA, TEST_HPU, TEST_XPU

from test.quantization.pt2e.test_quantize_pt2e import (
    _TestQuantizePT2EAcceleratorAware,
)
from torchao.utils import get_current_accelerator_device


@skipIfNoQNNPACK
class TestQuantizePT2EWithGPU(_TestQuantizePT2EAcceleratorAware):
    @unittest.skipUnless(TEST_CUDA or TEST_XPU or TEST_HPU, "accelerator unavailable")
    def test_move_exported_model_bn(self):
        if TEST_HPU:
            device = torch.device("hpu")
        else:
            device = get_current_accelerator_device()
        self._test_move_exported_model_bn(device)

    @unittest.skipUnless(TEST_CUDA or TEST_XPU, "CUDA or XPU unavailable")
    def test_allow_exported_model_train_eval(self):
        self._test_allow_exported_model_train_eval(get_current_accelerator_device())
