# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD 3-Clause license found in the
# LICENSE file in the root directory of this source tree.
from dataclasses import dataclass, field
from typing import List, Tuple

import numpy
import torch

from torchao.quantization.granularity import (
    Granularity,
    PerAxis,
)


@dataclass(frozen=True)
class MarlinQQQConstants:
    TILE: int = 16
    MIN_THREAD_N: int = 64
    MAX_PARALLEL: int = 16

    SUPPORTED_NUM_BITS: List[int] = field(default_factory=lambda: [4])
    SUPPORTED_GROUP_SIZES: List[int] = field(default_factory=lambda: [-1, 128])


const = MarlinQQQConstants()


def get_pack_factor(num_bits: int) -> int:
    """Compute the packing factor for a given number of bits.

    Args:
        num_bits (int): Number of bits to pack.
    Returns:
        int: The packing factor.
    """
    assert 32 % num_bits == 0, f"Unsupported num_bits = {num_bits}"
    return 32 // num_bits


def marlin_permute_weights(
    q_w: torch.Tensor,
    size_k: int,
    size_n: int,
    perm: torch.Tensor,
    tile: int = const.TILE,
) -> torch.Tensor:
    """Permute weights to 16x64 Marlin tiles.

    Args:
        q_w (torch.Tensor): Quantized weights.
        size_k (int): Number of input features.
        size_n (int): Number of output features.
        perm (torch.Tensor): The computed permutation tensor to be applied.
        tile (int, optional): Tile size. Defaults to `TILE`.
    Returns:
        torch.Tensor: Weight tensor permuted to Marlin tiles.
    """
    assert q_w.shape == (size_k, size_n)
    assert size_k % tile == 0, f"size_k = {size_k}, tile = {tile}"
    assert size_n % tile == 0, f"size_k = {size_n}, tile = {tile}"

    # Permute weights to 16x64 marlin tiles
    q_w = q_w.reshape((size_k // tile, tile, size_n // tile, tile))
    q_w = q_w.permute((0, 2, 1, 3))
    q_w = q_w.reshape((size_k // tile, size_n * tile))

    q_w = q_w.reshape((-1, perm.numel()))[:, perm].reshape(q_w.shape)

    return q_w


def reverse_marlin_permute_weights(
    q_w_unpacked: torch.Tensor,
    size_k: int,
    size_n: int,
    reverse_perm: torch.Tensor,
    tile: int = const.TILE,
) -> torch.Tensor:
    """Reverse permute weights from 16x64 Marlin tiles.
    Args:
        q_w_unpacked (torch.Tensor): Unpacked quantized weights.
        size_k (int): Number of input features.
        size_n (int): Number of output features.
        reverse_perm (torch.Tensor): The computed reverse permutation tensor to be applied.
        tile (int, optional): Tile size. Defaults to `TILE`.
    Returns:
        torch.Tensor: Weight tensor reverse permuted from Marlin tiles.
    """

    assert (q_w_unpacked.shape[0], size_n) == (
        size_k // tile,
        q_w_unpacked.shape[1] // tile,
    )
    assert size_k % tile == 0, f"size_k = {size_k}, tile = {tile}"
    assert size_n % tile == 0, f"size_k = {size_n}, tile = {tile}"

    # Reverse permute weights to original shape
    q_w_comp = q_w_unpacked.reshape((-1, reverse_perm.numel()))[
        :, reverse_perm
    ].reshape(q_w_unpacked.shape)
    q_w_comp = q_w_comp.reshape((size_k // tile, size_n // tile, tile, tile))
    q_w_comp = q_w_comp.permute((0, 2, 1, 3))
    q_w_comp = q_w_comp.reshape((size_k, size_n))

    return q_w_comp


# NOTE(HandH1998): QQQ employs different perms for per-group and per-channel weight quantization. # noqa: E501
def get_qqq_weight_perm(num_bits: int, granularity: Granularity) -> torch.Tensor:
    """Precompute permutations for the marlin weight shuffling.

    Args:
        num_bits (int): Number of bits to pack.
        granularity (Granularity): The weight quantization granularity.
    Returns:
        torch.Tensor: The weight permutation tensor.
    """
    perm_list: List[int] = []
    for i in range(32):
        perm1: List[int] = []
        col = i // 4
        for block in [0, 1]:
            for row in [
                4 * (i % 4),
                4 * (i % 4) + 1,
                4 * (i % 4) + 2,
                4 * (i % 4) + 3,
            ]:
                perm1.append(16 * row + col + 8 * block)
        for j in range(4):
            perm_list.extend([p + 256 * j for p in perm1])

    perm = numpy.array(perm_list)

    if num_bits == 4:
        if isinstance(granularity, PerAxis):
            interleave = numpy.array([4, 0, 5, 1, 6, 2, 7, 3])
        else:
            interleave = numpy.array([0, 2, 4, 6, 1, 3, 5, 7])
    else:
        raise Exception("num_bits must be 4, got {}".format(num_bits))

    perm = perm.reshape((-1, len(interleave)))[:, interleave].ravel()
    perm = torch.from_numpy(perm)
    return perm


def get_qqq_scale_perms(num_bits: int) -> Tuple[List[int], List[int]]:
    """Precompute permutations for the marlin scale shuffling.
    Args:
        num_bits (int): Number of bits to pack.
    Returns:
        Tuple[List[int], List[int]]: Scale permutation list and
        scale permutation list for a single group.
    """
    if num_bits != 4:
        raise Exception("num_bits must be 4, got {}".format(num_bits))
    scale_perm: List[int] = []
    for i in range(8):
        scale_perm.extend([i + 8 * j for j in range(8)])
    scale_perm_single: List[int] = []
    for i in range(4):
        scale_perm_single.extend([2 * i + j for j in [0, 1, 8, 9, 16, 17, 24, 25]])
    return scale_perm, scale_perm_single


def get_qqq_weight_reverse_perm(
    num_bits: int, granularity: Granularity
) -> torch.Tensor:
    """Reverse permutation for Marlin weight shuffling from `get_qqq_weight_perm`.
    Args:
        num_bits (int): Number of bits to pack.
        granularity (Granularity): The weight quantization granularity.
    Returns:
        torch.Tensor: The reversed weight permutation tensor.
    """
    perm = get_qqq_weight_perm(num_bits, granularity)
    perm = perm.argsort()

    return perm


def get_qqq_scale_reverse_perms(num_bits: int) -> Tuple[torch.Tensor, torch.Tensor]:
    """Reverse permutation for Marlin scale shuffling from `get_qqq_scale_perms`.
    Args:
        num_bits (int): Number of bits to pack.
    Returns:
        Tuple[List[int], List[int]]: The reversed scale permutation list and
        the reversed scale permutation list for a single group.
    """
    scale_perm, scale_perm_single = get_qqq_scale_perms(num_bits)
    scale_perm = torch.tensor(scale_perm).argsort()
    scale_perm_single = torch.tensor(scale_perm_single).argsort()

    return scale_perm, scale_perm_single
