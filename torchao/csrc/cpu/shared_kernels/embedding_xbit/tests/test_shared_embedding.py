# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""
Tests for the _shared_embedding_Xbit ops.

These tests verify that the shared embedding kernels correctly extract
embeddings from linear-packed weights, matching the reference dequantization
logic on both ARM and x86 platforms.
"""

import unittest

import torch

# Import torchao to register the ops
import torchao  # noqa: F401


class TestSharedEmbedding(unittest.TestCase):
    """Tests for the _shared_embedding_Xbit ops."""

    def _create_linear_packed_weights(
        self,
        bit_width: int,
        n: int,
        k: int,
        group_size: int,
        has_weight_zeros: bool = True,
        has_bias: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Create quantized weights and pack them using the linear packing op.

        The linear packing format is different from the embedding packing format.
        Shared embedding uses linear-packed weights to extract embeddings.

        Args:
            bit_width: Number of bits per weight (1-8)
            n: Number of output features (num_embeddings for shared embedding)
            k: Number of input features (embedding_dim for shared embedding)
            group_size: Group size for quantization
            has_weight_zeros: Whether to include weight zeros
            has_bias: Whether to include bias

        Returns:
            Tuple of (packed_weights, weight_qvals, weight_scales, weight_zeros, bias)
        """
        # Create quantized weights in the expected signed range
        min_val = -(1 << (bit_width - 1))
        max_val = (1 << (bit_width - 1)) - 1

        torch.manual_seed(42 + bit_width)  # Reproducible per bit_width
        weight_qvals = torch.randint(min_val, max_val + 1, (n, k), dtype=torch.int8)

        num_groups = k // group_size
        # Scales need to be flattened for the pack op
        weight_scales = (torch.rand(n, num_groups, dtype=torch.float32) + 0.1).reshape(
            -1
        )
        weight_zeros = torch.zeros(n * num_groups, dtype=torch.int8)
        if has_weight_zeros:
            weight_zeros = torch.randint(
                min_val, max_val + 1, (n * num_groups,), dtype=torch.int8
            )

        bias = torch.zeros(n, dtype=torch.float32)
        if has_bias:
            bias = torch.randn(n, dtype=torch.float32)

        # Pack using the linear packing op
        pack_op = getattr(torch.ops.torchao, f"_pack_8bit_act_{bit_width}bit_weight")

        packed_weights = pack_op(
            weight_qvals,
            weight_scales,
            weight_zeros if has_weight_zeros else None,
            group_size,
            bias if has_bias else None,
            None,  # target
        )

        # Reshape scales and zeros back for expected output computation
        weight_scales_2d = weight_scales.reshape(n, num_groups)
        weight_zeros_2d = (
            weight_zeros.reshape(n, num_groups)
            if has_weight_zeros
            else torch.zeros(n, num_groups, dtype=torch.int8)
        )

        return packed_weights, weight_qvals, weight_scales_2d, weight_zeros_2d, bias

    def _compute_expected_embedding(
        self,
        weight_qvals: torch.Tensor,
        weight_scales: torch.Tensor,
        weight_zeros: torch.Tensor,
        indices: torch.Tensor,
        group_size: int,
        has_weight_zeros: bool,
    ) -> torch.Tensor:
        """
        Compute expected embedding output using reference dequantization.

        For each index, we dequantize the corresponding row:
            output[i] = (weight_qvals[index] - weight_zeros) * weight_scales

        Args:
            weight_qvals: Quantized weight values (n, k)
            weight_scales: Scale factors per group (n, num_groups)
            weight_zeros: Zero points per group (n, num_groups)
            indices: Indices to look up
            group_size: Group size for quantization
            has_weight_zeros: Whether weight zeros are used

        Returns:
            Dequantized embeddings (num_indices, k)
        """
        n, k = weight_qvals.shape
        num_groups = k // group_size
        num_indices = indices.shape[0]

        expected = torch.zeros(num_indices, k, dtype=torch.float32)

        for i, idx in enumerate(indices):
            for g in range(num_groups):
                start = g * group_size
                end = start + group_size
                scale = weight_scales[idx, g]
                zero = weight_zeros[idx, g].float() if has_weight_zeros else 0.0
                expected[i, start:end] = (
                    weight_qvals[idx, start:end].float() - zero
                ) * scale

        return expected

    def test_shared_embedding_basic(self) -> None:
        """Test basic shared embedding functionality.

        This test verifies that:
        1. Pack quantized weights using linear packing
        2. Extract embeddings using shared_embedding op
        3. The dequantized output matches: (qval - zero) * scale
        """
        n = 32  # num_embeddings
        k = 64  # embedding_dim
        group_size = 32

        for bit_width in range(1, 8):
            with self.subTest(bit_width=bit_width):
                (
                    packed_weights,
                    weight_qvals,
                    weight_scales,
                    weight_zeros,
                    _,
                ) = self._create_linear_packed_weights(
                    bit_width=bit_width,
                    n=n,
                    k=k,
                    group_size=group_size,
                    has_weight_zeros=True,
                    has_bias=False,
                )

                # Test with a few indices
                indices = torch.tensor([0, 5, 15, 31], dtype=torch.int64)

                shared_embedding_op = getattr(
                    torch.ops.torchao, f"_shared_embedding_{bit_width}bit"
                )
                output = shared_embedding_op(
                    packed_weights,
                    group_size,
                    n,
                    k,
                    indices,
                )

                self.assertEqual(output.shape, (len(indices), k))
                self.assertEqual(output.dtype, torch.float32)

                expected = self._compute_expected_embedding(
                    weight_qvals,
                    weight_scales,
                    weight_zeros,
                    indices,
                    group_size,
                    has_weight_zeros=True,
                )

                torch.testing.assert_close(output, expected, rtol=1e-5, atol=1e-5)

    def test_shared_embedding_without_zeros(self) -> None:
        """Test shared embedding without weight zeros.

        When weight zeros are not present, dequantization is:
            output = qval * scale
        """
        n = 32
        k = 64
        group_size = 32

        for bit_width in range(1, 8):
            with self.subTest(bit_width=bit_width):
                (
                    packed_weights,
                    weight_qvals,
                    weight_scales,
                    weight_zeros,
                    _,
                ) = self._create_linear_packed_weights(
                    bit_width=bit_width,
                    n=n,
                    k=k,
                    group_size=group_size,
                    has_weight_zeros=False,
                    has_bias=False,
                )

                indices = torch.tensor([0, 10, 20, 31], dtype=torch.int64)

                shared_embedding_op = getattr(
                    torch.ops.torchao, f"_shared_embedding_{bit_width}bit"
                )
                output = shared_embedding_op(
                    packed_weights,
                    group_size,
                    n,
                    k,
                    indices,
                )

                self.assertEqual(output.shape, (len(indices), k))

                expected = self._compute_expected_embedding(
                    weight_qvals,
                    weight_scales,
                    weight_zeros,
                    indices,
                    group_size,
                    has_weight_zeros=False,
                )

                torch.testing.assert_close(output, expected, rtol=1e-5, atol=1e-5)

    def test_shared_embedding_all_indices(self) -> None:
        """Test shared embedding looking up all embeddings.

        This verifies the output matches when looking up every embedding.
        """
        n = 16  # Smaller for faster test
        k = 32
        group_size = 32

        for bit_width in [2, 4]:  # Test a couple bit widths
            with self.subTest(bit_width=bit_width):
                (
                    packed_weights,
                    weight_qvals,
                    weight_scales,
                    weight_zeros,
                    _,
                ) = self._create_linear_packed_weights(
                    bit_width=bit_width,
                    n=n,
                    k=k,
                    group_size=group_size,
                    has_weight_zeros=True,
                    has_bias=False,
                )

                # Look up all embeddings
                indices = torch.arange(n, dtype=torch.int64)

                shared_embedding_op = getattr(
                    torch.ops.torchao, f"_shared_embedding_{bit_width}bit"
                )
                output = shared_embedding_op(
                    packed_weights,
                    group_size,
                    n,
                    k,
                    indices,
                )

                self.assertEqual(output.shape, (n, k))

                expected = self._compute_expected_embedding(
                    weight_qvals,
                    weight_scales,
                    weight_zeros,
                    indices,
                    group_size,
                    has_weight_zeros=True,
                )

                torch.testing.assert_close(output, expected, rtol=1e-5, atol=1e-5)

    def test_shared_embedding_single_index(self) -> None:
        """Test shared embedding with a single index."""
        n = 32
        k = 64
        group_size = 32
        bit_width = 4

        (
            packed_weights,
            weight_qvals,
            weight_scales,
            weight_zeros,
            _,
        ) = self._create_linear_packed_weights(
            bit_width=bit_width,
            n=n,
            k=k,
            group_size=group_size,
            has_weight_zeros=True,
            has_bias=False,
        )

        # Single index
        indices = torch.tensor([7], dtype=torch.int64)

        shared_embedding_op = getattr(
            torch.ops.torchao, f"_shared_embedding_{bit_width}bit"
        )
        output = shared_embedding_op(
            packed_weights,
            group_size,
            n,
            k,
            indices,
        )

        self.assertEqual(output.shape, (1, k))

        expected = self._compute_expected_embedding(
            weight_qvals,
            weight_scales,
            weight_zeros,
            indices,
            group_size,
            has_weight_zeros=True,
        )

        torch.testing.assert_close(output, expected, rtol=1e-5, atol=1e-5)

    def test_shared_embedding_repeated_indices(self) -> None:
        """Test shared embedding with repeated indices."""
        n = 32
        k = 64
        group_size = 32
        bit_width = 4

        (
            packed_weights,
            weight_qvals,
            weight_scales,
            weight_zeros,
            _,
        ) = self._create_linear_packed_weights(
            bit_width=bit_width,
            n=n,
            k=k,
            group_size=group_size,
            has_weight_zeros=True,
            has_bias=False,
        )

        # Repeated indices
        indices = torch.tensor([3, 3, 7, 3, 7], dtype=torch.int64)

        shared_embedding_op = getattr(
            torch.ops.torchao, f"_shared_embedding_{bit_width}bit"
        )
        output = shared_embedding_op(
            packed_weights,
            group_size,
            n,
            k,
            indices,
        )

        self.assertEqual(output.shape, (5, k))

        # Verify that repeated indices produce the same output
        torch.testing.assert_close(output[0], output[1], rtol=1e-5, atol=1e-5)
        torch.testing.assert_close(output[0], output[3], rtol=1e-5, atol=1e-5)
        torch.testing.assert_close(output[2], output[4], rtol=1e-5, atol=1e-5)

        expected = self._compute_expected_embedding(
            weight_qvals,
            weight_scales,
            weight_zeros,
            indices,
            group_size,
            has_weight_zeros=True,
        )

        torch.testing.assert_close(output, expected, rtol=1e-5, atol=1e-5)


if __name__ == "__main__":
    unittest.main()
