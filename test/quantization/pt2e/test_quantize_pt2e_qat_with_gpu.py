# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD 3-Clause license found in the
# LICENSE file in the root directory of this source tree.

# Owner(s): ["oncall: quantization"]
import unittest

import torch
from torch.testing._internal.common_cuda import TEST_CUDA
from torch.testing._internal.common_quantization import skipIfNoQNNPACK
from torch.testing._internal.common_utils import TEST_XPU

from test.quantization.pt2e.test_quantize_pt2e_qat import PT2EQATTestCase
from torchao.utils import get_current_accelerator_device


class _TestQuantizePT2EQATConvBnWithGPU:
    __test__ = False

    @unittest.skipUnless(TEST_CUDA or TEST_XPU, "GPU unavailable")
    def test_qat_conv_bn_fusion_cuda(self):
        self._test_qat_conv_bn_fusion(get_current_accelerator_device(), has_relu=False)

    @unittest.skipUnless(TEST_CUDA or TEST_XPU, "GPU unavailable")
    def test_qat_conv_bn_relu_fusion_cuda(self):
        self._test_qat_conv_bn_fusion(get_current_accelerator_device(), has_relu=True)


@skipIfNoQNNPACK
class TestQuantizePT2EQATConvBn1dWithGPU(
    _TestQuantizePT2EQATConvBnWithGPU, PT2EQATTestCase
):
    __test__ = True
    example_inputs = (torch.randn(1, 3, 5),)
    conv_class = torch.nn.Conv1d
    conv_transpose_class = torch.nn.ConvTranspose1d
    bn_class = torch.nn.BatchNorm1d


@skipIfNoQNNPACK
class TestQuantizePT2EQATConvBn2dWithGPU(
    _TestQuantizePT2EQATConvBnWithGPU, PT2EQATTestCase
):
    __test__ = True
    example_inputs = (torch.randn(1, 3, 5, 5),)
    conv_class = torch.nn.Conv2d
    conv_transpose_class = torch.nn.ConvTranspose2d
    bn_class = torch.nn.BatchNorm2d
