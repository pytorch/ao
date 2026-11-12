# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD 3-Clause license found in the
# LICENSE file in the root directory of this source tree.
"""Save compressed activations for backward.

During training the autograd engine keeps every tensor a backward op will
need. For a Squared-ReLU FFN that includes the sparse activation between the
nonlinearity and the down projection. Packing that tensor, and unpacking it
on the way back, is the memory win described in pytorch/ao#1920. Inference
does not keep those tensors, so this does not change inference memory.
"""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass

import torch

from torchao.prototype.activation_compression.codec import (
    CompressedActivation,
    _check_method,
    _check_zlib_level,
    compress_activation,
    decompress_activation,
)

# Stride sample used to decide whether a saved tensor is worth packing.
# At the default 0.5 cutoff, an 85% sparse activation is not a close call.
_SPARSITY_SAMPLE = 8192


@dataclass
class CompressionStats:
    """Bytes the autograd engine kept during the forward wrapped by the hook.

    ``dense_bytes`` is what would have been saved with no compression.
    ``stored_bytes`` is what was actually retained. Weights and other dense
    tensors are counted in both and left unpacked.
    """

    dense_bytes: int = 0
    stored_bytes: int = 0
    n_saved: int = 0
    n_compressed: int = 0
    n_skipped: int = 0

    @property
    def ratio(self) -> float:
        if self.stored_bytes == 0:
            return float("inf") if self.dense_bytes else 1.0
        return self.dense_bytes / self.stored_bytes


@contextmanager
def compress_saved_activations(
    method: str = "bitpack",
    min_numel: int = 4096,
    min_sparsity: float = 0.5,
    zlib_level: int = 1,
) -> Iterator[CompressionStats]:
    """Pack sparse floating tensors saved for backward.

    Wrap the forward, then call ``backward`` as usual. The unpack hook is
    stored with each saved tensor, so backward may run outside this block.
    Only the innermost pair of saved-tensor hooks is applied if this is
    nested inside another ``saved_tensors_hooks`` context.

    ``min_numel`` skips tiny tensors, where the launch overhead dominates
    the bytes saved. ``min_sparsity`` skips tensors that are not sparse
    enough for the packed form to be smaller (dense weights included).
    """
    _check_method(method)
    _check_zlib_level(zlib_level)
    if min_numel < 0:
        raise ValueError(f"min_numel must be >= 0, got {min_numel}")
    if not 0.0 <= min_sparsity <= 1.0:
        raise ValueError(f"min_sparsity must be in [0, 1], got {min_sparsity}")

    stats = CompressionStats()

    def pack(tensor: torch.Tensor):
        dense = tensor.numel() * tensor.element_size()
        stats.dense_bytes += dense
        stats.n_saved += 1
        if not _eligible(tensor, min_numel):
            stats.stored_bytes += dense
            return tensor
        if _estimate_sparsity(tensor) < min_sparsity:
            stats.stored_bytes += dense
            stats.n_skipped += 1
            return tensor
        packed = compress_activation(tensor, method, zlib_level=zlib_level)
        if packed.nbytes >= dense:
            stats.stored_bytes += dense
            stats.n_skipped += 1
            return tensor
        stats.stored_bytes += packed.nbytes
        stats.n_compressed += 1
        return packed

    def unpack(obj):
        if isinstance(obj, CompressedActivation):
            return decompress_activation(obj)
        return obj

    with torch.autograd.graph.saved_tensors_hooks(pack, unpack):
        yield stats


def _eligible(tensor: torch.Tensor, min_numel: int) -> bool:
    return (
        tensor.is_floating_point()
        and not tensor.is_complex()
        and tensor.layout is torch.strided
        and not getattr(tensor, "is_quantized", False)
        and tensor.numel() >= min_numel
        and tensor.device.type != "meta"
    )


def _estimate_sparsity(tensor: torch.Tensor) -> float:
    flat = tensor.detach().reshape(-1)
    n = flat.numel()
    if n <= _SPARSITY_SAMPLE:
        sample = flat
    else:
        step = max(n // _SPARSITY_SAMPLE, 1)
        sample = flat[::step][:_SPARSITY_SAMPLE]
    return float((sample == 0).sum().item()) / sample.numel()
