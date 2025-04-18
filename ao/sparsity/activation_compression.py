# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD 3-Clause license found in the
# LICENSE file in the root directory of this source tree.

import warnings
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn


class ActivationCompressor:
    """Handles compression of sparse activation tensors."""

    def __init__(self, compression_method: str = "simple"):
        """
        Initialize the compressor.

        Args:
            compression_method (str): The compression method to use.
                                    Options: 'simple', 'block', 'run_length'
        """
        if compression_method not in ["simple", "block", "run_length"]:
            warnings.warn(
                f"Unsupported compression method: {compression_method}. Using 'simple'."
            )
            compression_method = "simple"
        self.compression_method = compression_method
        self._memory_usage = 0

    def get_memory_usage(self) -> int:
        """Get the current memory usage in bytes."""
        return self._memory_usage

    def compress_tensor(self, tensor: torch.Tensor) -> Dict:
        """
        Compress a sparse tensor using the specified method.

        Args:
            tensor (torch.Tensor): Input tensor to compress

        Returns:
            Dict containing compressed tensor data

        Raises:
            ValueError: If tensor is not sparse enough to benefit from compression
        """
        if not isinstance(tensor, torch.Tensor):
            raise TypeError("Input must be a torch.Tensor")

        # Ensure tensor is contiguous
        tensor = tensor.contiguous()

        # Calculate sparsity
        sparsity = (tensor == 0).float().mean()
        if sparsity < 0.5:
            warnings.warn(
                f"Tensor sparsity ({sparsity:.2%}) is low. Compression may not be beneficial."
            )

        if self.compression_method == "simple":
            return self._compress_simple(tensor)
        elif self.compression_method == "block":
            return self._compress_block(tensor)
        else:  # run_length
            return self._compress_run_length(tensor)

    def _compress_simple(self, tensor: torch.Tensor) -> Dict:
        """Simple compression storing non-zero values and indices."""
        mask = tensor != 0
        values = tensor[mask]
        indices = torch.nonzero(mask)

        compressed = {
            "values": values,
            "indices": indices,
            "shape": tensor.shape,
            "dtype": tensor.dtype,
            "device": tensor.device,
            "method": "simple",
        }

        # Update memory usage
        self._memory_usage = (
            values.element_size() * values.numel()
            + indices.element_size() * indices.numel()
        )
        return compressed

    def _compress_block(self, tensor: torch.Tensor, block_size: int = 4) -> Dict:
        """Block-based compression for better cache efficiency."""
        # Reshape into blocks
        shape = tensor.shape
        blocks = tensor.unfold(0, block_size, block_size).unfold(
            1, block_size, block_size
        )
        block_mask = (blocks != 0).any(dim=(-1, -2))

        # Store non-zero blocks
        values = blocks[block_mask]
        indices = torch.nonzero(block_mask)

        compressed = {
            "values": values,
            "indices": indices,
            "shape": shape,
            "dtype": tensor.dtype,
            "device": tensor.device,
            "method": "block",
            "block_size": block_size,
        }

        # Update memory usage
        self._memory_usage = (
            values.element_size() * values.numel()
            + indices.element_size() * indices.numel()
        )
        return compressed

    def _compress_run_length(self, tensor: torch.Tensor) -> Dict:
        """Run-length encoding for sequences of zeros."""
        # Flatten tensor
        flat = tensor.flatten()
        changes = torch.cat(
            [torch.tensor([True], device=tensor.device), flat[1:] != flat[:-1]]
        )
        values = flat[changes]
        lengths = torch.diff(
            torch.cat(
                [
                    torch.tensor([0], device=tensor.device),
                    torch.nonzero(changes).squeeze(),
                ]
            )
        )

        compressed = {
            "values": values,
            "lengths": lengths,
            "shape": tensor.shape,
            "dtype": tensor.dtype,
            "device": tensor.device,
            "method": "run_length",
        }

        # Update memory usage
        self._memory_usage = (
            values.element_size() * values.numel()
            + lengths.element_size() * lengths.numel()
        )
        return compressed

    def decompress_tensor(self, compressed_data: Dict) -> torch.Tensor:
        """
        Decompress a tensor from its compressed form.

        Args:
            compressed_data (Dict): Compressed tensor data

        Returns:
            torch.Tensor: Reconstructed tensor

        Raises:
            ValueError: If compressed data is invalid or method is unsupported
        """
        if not isinstance(compressed_data, dict):
            raise TypeError("Compressed data must be a dictionary")

        method = compressed_data.get("method", "simple")

        if method == "simple":
            return self._decompress_simple(compressed_data)
        elif method == "block":
            return self._decompress_block(compressed_data)
        elif method == "run_length":
            return self._decompress_run_length(compressed_data)
        else:
            raise ValueError(f"Unsupported compression method: {method}")

    def _decompress_simple(self, compressed_data: Dict) -> torch.Tensor:
        """Decompress simple compressed tensor."""
        tensor = torch.zeros(
            compressed_data["shape"],
            dtype=compressed_data["dtype"],
            device=compressed_data["device"],
        )
        tensor.index_put_(
            tuple(compressed_data["indices"].t()), compressed_data["values"]
        )
        return tensor

    def _decompress_block(self, compressed_data: Dict) -> torch.Tensor:
        """Decompress block compressed tensor."""
        tensor = torch.zeros(
            compressed_data["shape"],
            dtype=compressed_data["dtype"],
            device=compressed_data["device"],
        )
        block_size = compressed_data["block_size"]

        # Reconstruct blocks
        for idx, block in zip(compressed_data["indices"], compressed_data["values"]):
            i, j = idx * block_size
            tensor[i : i + block_size, j : j + block_size] = block

        return tensor

    def _decompress_run_length(self, compressed_data: Dict) -> torch.Tensor:
        """Decompress run-length encoded tensor."""
        # Reconstruct flat array
        flat = torch.zeros(
            compressed_data["shape"].numel(),
            dtype=compressed_data["dtype"],
            device=compressed_data["device"],
        )

        pos = 0
        for val, length in zip(compressed_data["values"], compressed_data["lengths"]):
            flat[pos : pos + length] = val
            pos += length

        return flat.reshape(compressed_data["shape"])


class CompressedActivation(nn.Module):
    """Module that handles activation compression during training."""

    def __init__(self, compression_method: str = "simple"):
        """
        Initialize the compressed activation module.

        Args:
            compression_method (str): The compression method to use
        """
        super().__init__()
        self.compressor = ActivationCompressor(compression_method)
        self.compressed_data: Optional[Dict] = None
        self._original_shape: Optional[Tuple[int, ...]] = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass with optional compression during training.

        Args:
            x (torch.Tensor): Input tensor

        Returns:
            torch.Tensor: Output tensor
        """
        if self.training:
            # Store compressed version for backward pass
            self.compressed_data = self.compressor.compress_tensor(x)
            self._original_shape = x.shape
        return x

    def backward(self, grad_output: torch.Tensor) -> torch.Tensor:
        """
        Backward pass with decompression if needed.

        Args:
            grad_output (torch.Tensor): Gradient from next layer

        Returns:
            torch.Tensor: Gradient for previous layer
        """
        if self.compressed_data is not None:
            # Decompress for gradient computation
            original = self.compressor.decompress_tensor(self.compressed_data)
            self.compressed_data = None

            # Ensure shapes match
            if grad_output.shape != self._original_shape:
                grad_output = grad_output.reshape(self._original_shape)

            # Compute gradient with respect to decompressed tensor
            return grad_output * (original != 0).float()
        return grad_output
