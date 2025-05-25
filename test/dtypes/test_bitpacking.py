# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD 3-Clause license found in the
# LICENSE file in the root directory of this source tree.
import unittest

import torch
from torch.testing._internal import common_utils
from torch.utils._triton import has_triton

from torchao.dtypes.uintx.bitpacking import pack, pack_cpu, unpack, unpack_cpu

bit_widths = (1, 2, 3, 4, 5, 6, 7)
dimensions = (0, -1, 1)


class TestBitpacking(unittest.TestCase):
    def setUp(self):
        """Set up test fixtures before each test method."""
        # Set random seeds for reproducibility
        torch.manual_seed(0)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(0)

        # Initialize any common test data
        self.test_shape = (32, 32, 32)

        # Set default device
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Reset any cached state
        torch._dynamo.reset()

    def tearDown(self):
        """Clean up after each test method."""
        # Reset torch._dynamo cache
        torch._dynamo.reset()

        # Clear CUDA cache if available
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        # Reset any other state if needed
        torch._dynamo.config.specialize_int = False

    @common_utils.parametrize("bit_width", bit_widths)
    @common_utils.parametrize("dim", dimensions)
    def test_CPU(self, bit_width, dim):
        test_tensor = torch.randint(
            0, 2**bit_width, (32, 32, 32), dtype=torch.uint8, device="cpu"
        )
        packed = pack_cpu(test_tensor, bit_width, dim=dim)
        unpacked = unpack_cpu(packed, bit_width, dim=dim)
        self.assertTrue(unpacked.allclose(test_tensor))

    @unittest.skipIf(not torch.cuda.is_available(), "CUDA not available")
    @common_utils.parametrize("bit_width", bit_widths)
    @common_utils.parametrize("dim", dimensions)
    def test_GPU(self, bit_width, dim):
        test_tensor = torch.randint(
            0, 2**bit_width, (32, 32, 32), dtype=torch.uint8
        ).cuda()
        packed = pack(test_tensor, bit_width, dim=dim)
        unpacked = unpack(packed, bit_width, dim=dim)
        self.assertTrue(unpacked.allclose(test_tensor))

    @unittest.skipIf(not torch.cuda.is_available(), "CUDA not available")
    @unittest.skipIf(not has_triton(), "unsupported without triton")
    @common_utils.parametrize("bit_width", bit_widths)
    @common_utils.parametrize("dim", dimensions)
    def test_compile(self, bit_width, dim):
        torch._dynamo.config.specialize_int = True
        torch.compile(pack, fullgraph=True)
        torch.compile(unpack, fullgraph=True)
        test_tensor = torch.randint(
            0, 2**bit_width, (32, 32, 32), dtype=torch.uint8
        ).cuda()
        packed = pack(test_tensor, bit_width, dim=dim)
        unpacked = unpack(packed, bit_width, dim=dim)
        self.assertTrue(unpacked.allclose(test_tensor))

    @unittest.skipIf(not torch.cuda.is_available(), "CUDA not available")
    def test_pack_example(self):
        test_tensor = torch.tensor(
            [0x30, 0x29, 0x17, 0x5, 0x20, 0x16, 0x9, 0x22], dtype=torch.uint8
        ).cuda()
        shard_4, shard_2 = pack(test_tensor, 6)
        print(shard_4, shard_2)
        self.assertTrue(
            torch.tensor([0, 105, 151, 37], dtype=torch.uint8).cuda().allclose(shard_4)
        )
        self.assertTrue(
            torch.tensor([39, 146], dtype=torch.uint8).cuda().allclose(shard_2)
        )
        unpacked = unpack([shard_4, shard_2], 6)
        self.assertTrue(unpacked.allclose(test_tensor))

    def test_pack_example_CPU(self):
        test_tensor = torch.tensor(
            [0x30, 0x29, 0x17, 0x5, 0x20, 0x16, 0x9, 0x22], dtype=torch.uint8
        )
        shard_4, shard_2 = pack(test_tensor, 6)
        print(shard_4, shard_2)
        self.assertTrue(
            torch.tensor([0, 105, 151, 37], dtype=torch.uint8).allclose(shard_4)
        )
        self.assertTrue(torch.tensor([39, 146], dtype=torch.uint8).allclose(shard_2))
        unpacked = unpack([shard_4, shard_2], 6)
        self.assertTrue(unpacked.allclose(test_tensor))


common_utils.instantiate_parametrized_tests(TestBitpacking)

if __name__ == "__main__":
    unittest.main()
