# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD 3-Clause license found in the
# LICENSE file in the root directory of this source tree.

from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class DeepGemmKGroupedQuantMetadata:
    dim: int
    group_sizes: list[int]
    q_offset_by_block: torch.Tensor
    group_size_by_block: torch.Tensor

    @property
    def valid_tokens(self) -> int:
        return sum(self.group_sizes)

    @property
    def valid_blocks(self) -> int:
        return self.q_offset_by_block.numel()


def group_sizes_from_offsets(group_end_offsets: torch.Tensor) -> list[int]:
    assert group_end_offsets is not None and group_end_offsets.dtype == torch.int32, (
        "group_end_offsets must be int32"
    )
    group_sizes = []
    start = 0
    for group_idx in range(group_end_offsets.numel()):
        end = int(group_end_offsets[group_idx].item())
        group_sizes.append(end - start)
        start = end
    return group_sizes


def build_deepgemm_k_grouped_quant_metadata(
    group_end_offsets: torch.Tensor,
    group_sizes: list[int],
    block_size: int,
    dim: int,
) -> DeepGemmKGroupedQuantMetadata:
    q_offset_by_block = []
    group_size_by_block = []
    group_start = 0
    for group_size in group_sizes:
        assert group_size % block_size == 0, (
            "DeepGEMM K-grouped activation quantization requires every group size "
            f"to be divisible by block_size={block_size}"
        )
        for block_idx in range(group_size // block_size):
            q_offset_by_block.append(group_start * dim + block_idx * block_size)
            group_size_by_block.append(group_size)
        group_start += group_size

    device = group_end_offsets.device
    return DeepGemmKGroupedQuantMetadata(
        dim=dim,
        group_sizes=group_sizes,
        q_offset_by_block=torch.tensor(
            q_offset_by_block,
            dtype=torch.int32,
            device=device,
        ),
        group_size_by_block=torch.tensor(
            group_size_by_block,
            dtype=torch.int32,
            device=device,
        ),
    )
