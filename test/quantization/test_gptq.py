# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD 3-Clause license found in the
# LICENSE file in the root directory of this source tree.

import unittest

import torch
from torch.testing._internal.common_utils import TestCase

from torchao.utils import get_current_accelerator_device

torch.manual_seed(0)

_DEVICE = get_current_accelerator_device()


class TestMultiTensorFlow(TestCase):
    @unittest.skipIf(not torch.accelerator.is_available(), "Need GPU available")
    def test_multitensor_add_tensors(self):
        from torchao.quantization.GPTQ import MultiTensor

        tensor1 = torch.randn(3, 3)
        tensor2 = torch.randn(3, 3)
        mt = MultiTensor(tensor1)
        mt.add_tensors(tensor2)
        self.assertEqual(mt.count, 2)
        self.assertTrue(torch.equal(mt.values[0], tensor1))
        self.assertTrue(torch.equal(mt.values[1], tensor2))

    @unittest.skipIf(not torch.accelerator.is_available(), "Need GPU available")
    def test_multitensor_pad_unpad(self):
        from torchao.quantization.GPTQ import MultiTensor

        tensor1 = torch.randn(3, 3)
        mt = MultiTensor(tensor1)
        mt.pad_to_length(3)
        self.assertEqual(mt.count, 3)
        mt.unpad()
        self.assertEqual(mt.count, 1)

    @unittest.skipIf(not torch.accelerator.is_available(), "Need GPU available")
    def test_multitensor_inplace_operation(self):
        from torchao.quantization.GPTQ import MultiTensor

        tensor1 = torch.ones(3, 3)
        mt = MultiTensor(tensor1)
        mt += 1  # In-place addition
        self.assertTrue(torch.equal(mt.values[0], torch.full((3, 3), 2)))


class TestMultiTensorInputRecorder(TestCase):
    @unittest.skipIf(not torch.accelerator.is_available(), "Need GPU available")
    def test_multitensor_input_recorder(self):
        from torchao.quantization.GPTQ import MultiTensor, MultiTensorInputRecorder

        input_recorder = MultiTensorInputRecorder()
        in1 = ([1], torch.randn(3, 3), (1, "dog", torch.randn(3, 3)), torch.float)
        in2 = ([1], torch.randn(3, 3), (1, "dog", torch.randn(3, 3)), torch.float)

        input_recorder(*in1)
        input_recorder(*in2)

        MT_input = input_recorder.get_recorded_inputs()

        self.assertEqual(MT_input[0], [1])
        self.assertTrue(isinstance(MT_input[1], MultiTensor))
        self.assertTrue(isinstance(MT_input[2], tuple))
        self.assertEqual(MT_input[2][0], 1)
        self.assertEqual(MT_input[2][1], "dog")
        self.assertTrue(isinstance(MT_input[2][2], MultiTensor))
        self.assertEqual(MT_input[3], torch.float)


if __name__ == "__main__":
    unittest.main()
