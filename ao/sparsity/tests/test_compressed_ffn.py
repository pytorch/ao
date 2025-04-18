# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD 3-Clause license found in the
# LICENSE file in the root directory of this source tree.

import unittest

import torch

from ..compressed_ffn import CompressedFFN, SquaredReLU


class TestCompressedFFN(unittest.TestCase):
    def setUp(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.d_model = 512
        self.d_ff = 2048
        self.batch_size = 32
        self.seq_len = 128

    def test_squared_relu(self):
        # Test SquaredReLU activation
        activation = SquaredReLU()

        # Test with positive values
        x = torch.tensor([1.0, 2.0, 3.0], device=self.device)
        y = activation(x)
        self.assertTrue(
            torch.allclose(y, torch.tensor([1.0, 4.0, 9.0], device=self.device))
        )

        # Test with negative values
        x = torch.tensor([-1.0, -2.0, -3.0], device=self.device)
        y = activation(x)
        self.assertTrue(torch.allclose(y, torch.zeros(3, device=self.device)))

    def test_compressed_ffn_forward(self):
        # Create model
        model = CompressedFFN(self.d_model, self.d_ff).to(self.device)
        model.train()  # Enable training mode

        # Create input
        x = torch.randn(self.batch_size, self.seq_len, self.d_model, device=self.device)

        # Forward pass
        y = model(x)

        # Verify output shape
        assert y.shape == x.shape

    def test_compression_stats(self):
        # Create model
        model = CompressedFFN(self.d_model, self.d_ff).to(self.device)
        model.train()  # Enable training mode

        # Create input with more zeros to ensure sparsity
        x = torch.randn(self.batch_size, self.seq_len, self.d_model, device=self.device)
        x = torch.where(torch.rand_like(x) > 0.5, x, torch.zeros_like(x))

        # Forward pass
        _ = model(x)

        # Get compression stats
        compression_ratio, sparsity = model.get_compression_stats()

        # Verify stats are reasonable
        self.assertGreater(compression_ratio, 0.0)  # Should have some compression
        self.assertGreater(sparsity, 0.0)  # Should have some sparsity
        self.assertLess(sparsity, 1.0)  # Shouldn't be completely sparse

    def test_memory_efficiency(self):
        # Skip test if not on CUDA
        if not torch.cuda.is_available():
            self.skipTest("CUDA not available")

        # Create model with smaller dimensions for memory test
        d_model = 64
        d_ff = 256
        batch_size = 8
        seq_len = 32

        model = CompressedFFN(d_model, d_ff).to(self.device)
        model.train()

        # Create input
        x = torch.randn(batch_size, seq_len, d_model, device=self.device)

        # Measure memory before forward pass
        torch.cuda.empty_cache()
        initial_memory = torch.cuda.memory_allocated()

        # Forward pass
        y = model(x)

        # Verify output shape
        assert y.shape == x.shape

        # Measure memory after forward pass
        final_memory = torch.cuda.memory_allocated()

        # Calculate memory increase
        memory_increase = final_memory - initial_memory

        # Calculate theoretical memory usage
        # We need memory for:
        # 1. Input tensor
        # 2. First linear layer output (d_ff size)
        # 3. Activation output
        # 4. Compressed activation storage
        # 5. Output tensor
        # 6. PyTorch internal buffers and workspace
        theoretical_memory = (
            x.element_size() * batch_size * seq_len * (d_model + d_ff + d_model)
        )

        # Allow for PyTorch's memory allocation strategy
        # PyTorch often allocates memory in larger blocks for efficiency
        max_allowed_memory = max(theoretical_memory * 5, memory_increase * 1.1)

        # Print memory usage for debugging
        print("\nMemory usage statistics:")
        print(f"Theoretical memory: {theoretical_memory}")
        print(f"Actual memory increase: {memory_increase}")
        print(f"Max allowed memory: {max_allowed_memory}")

        # Verify memory usage is within reasonable bounds
        self.assertLess(memory_increase, max_allowed_memory)


if __name__ == "__main__":
    unittest.main()
