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


@dataclass(frozen=True)
class DeepGemmGroupedOffsetPlan:
    group_end_offsets: torch.Tensor
    group_sizes: list[int]
    ks_tensor: torch.Tensor
    grouped_layout: torch.Tensor
    original_group_sizes: list[int] | None = None
    padded_group_start_offsets: torch.Tensor | None = None

    @property
    def total_tokens(self) -> int:
        return sum(self.group_sizes)

    def groups_are_block_aligned(self, block_size: int) -> bool:
        return all(group_size % block_size == 0 for group_size in self.group_sizes)

    def valid_scale_blocks(self, block_size: int) -> int:
        assert self.groups_are_block_aligned(block_size), (
            "valid scale blocks require block-aligned group sizes"
        )
        return sum(group_size // block_size for group_size in self.group_sizes)

    def k_quant_metadata(
        self,
        block_size: int,
        dim: int,
    ) -> DeepGemmKGroupedQuantMetadata:
        return build_deepgemm_k_grouped_quant_metadata(
            self.group_end_offsets,
            self.group_sizes,
            block_size,
            dim,
        )

    def m_grouped_layout(self, num_rows: int) -> torch.Tensor:
        assert self.grouped_layout.numel() == num_rows, (
            "cached DeepGEMM grouped layout does not match operand rows"
        )
        return self.grouped_layout


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


def build_deepgemm_grouped_offset_plan(
    group_end_offsets: torch.Tensor,
    *,
    original_group_end_offsets: torch.Tensor | None = None,
    padded_group_start_offsets: torch.Tensor | None = None,
    num_rows: int | None = None,
) -> DeepGemmGroupedOffsetPlan:
    group_sizes = group_sizes_from_offsets(group_end_offsets)
    original_group_sizes = None
    padded_group_starts = None
    if original_group_end_offsets is not None:
        assert original_group_end_offsets.dtype == torch.int32, (
            "original_group_end_offsets must be int32"
        )
        assert padded_group_start_offsets is not None, (
            "padded_group_start_offsets must be provided with original_group_end_offsets"
        )
        assert padded_group_start_offsets.dtype == torch.int32, (
            "padded_group_start_offsets must be int32"
        )
        assert original_group_end_offsets.numel() == group_end_offsets.numel(), (
            "original and padded group offsets must have the same number of groups"
        )
        original_group_sizes = group_sizes_from_offsets(original_group_end_offsets)
        padded_group_starts = padded_group_start_offsets.tolist()

    grouped_layout = _build_deepgemm_m_grouped_layout(
        group_end_offsets.device,
        group_sizes,
        original_group_sizes=original_group_sizes,
        padded_group_starts=padded_group_starts,
        num_rows=num_rows,
    )

    return DeepGemmGroupedOffsetPlan(
        group_end_offsets=group_end_offsets,
        group_sizes=group_sizes,
        ks_tensor=torch.tensor(
            group_sizes,
            dtype=torch.int32,
            device=group_end_offsets.device,
        ),
        grouped_layout=grouped_layout,
        original_group_sizes=original_group_sizes,
        padded_group_start_offsets=padded_group_start_offsets,
    )


def _build_deepgemm_m_grouped_layout(
    device: torch.device,
    group_sizes: list[int],
    *,
    original_group_sizes: list[int] | None = None,
    padded_group_starts: list[int] | None = None,
    num_rows: int | None = None,
) -> torch.Tensor:
    # DeepGEMM's contiguous M-grouped kernel wants one int32 entry per row:
    # grouped_layout[row] = expert_idx. Padded rows are marked -1 so the kernel
    # skips them instead of routing them to an expert.
    min_rows = sum(group_sizes)
    if num_rows is None:
        num_rows = min_rows
    assert num_rows >= min_rows, "grouped layout must cover all padded groups"
    grouped_layout = torch.full(
        (num_rows,),
        -1,
        dtype=torch.int32,
        device=device,
    )

    if original_group_sizes is None:
        start = 0
        for group_idx, group_size in enumerate(group_sizes):
            end = start + group_size
            grouped_layout[start:end] = group_idx
            start = end
        return grouped_layout.contiguous()

    assert padded_group_starts is not None, (
        "padded_group_starts must be provided with original_group_sizes"
    )
    for group_idx, (group_size, padded_start) in enumerate(
        zip(original_group_sizes, padded_group_starts)
    ):
        grouped_layout[padded_start : padded_start + group_size] = group_idx
    return grouped_layout.contiguous()


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
