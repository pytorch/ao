# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD 3-Clause license found in the
# LICENSE file in the root directory of this source tree.
from __future__ import annotations

from dataclasses import dataclass
from functools import cached_property

import torch


def _group_sizes_tensor(group_end_offsets: torch.Tensor) -> torch.Tensor:
    assert group_end_offsets.dtype == torch.int32, "group_end_offsets must be int32"
    return torch.diff(
        group_end_offsets,
        prepend=group_end_offsets.new_zeros(1),
    )


@dataclass(frozen=True)
class DeepGemmKGroupedQuantMetadata:
    """Launch metadata for DeepGEMM K-grouped activation quantization.

    Args:
        dim: Logical feature dimension of the activation being quantized.
        group_sizes: Host-side token count for each expert group.
        q_offset_by_block: Per-token-block base offset into DeepGEMM's flat
            per-expert ``(dim, expert_tokens)`` output buffer.
        group_size_by_block: Token count for the expert owning each token block.
    """

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
    """Offset metadata shared by DeepGEMM grouped GEMM calls.

    Args:
        group_end_offsets: Cumulative end offsets for the padded expert groups.
        grouped_layout: Int32 row-to-expert mapping consumed by DeepGEMM's
            M-grouped kernel. Padding rows are marked as ``-1``.
        groups_block_aligned_by_construction: True when the caller has already
            padded every expert group to the block size, so alignment checks can
            skip a device-to-host reduction.
    """

    group_end_offsets: torch.Tensor
    grouped_layout: torch.Tensor
    groups_block_aligned_by_construction: bool = False

    @cached_property
    def group_sizes(self) -> list[int]:
        # DeepGEMM's K-grouped wgrad kernel requires the per-group sizes as a
        # host-side `ks` sequence, so this lazy property reads them back once.
        return _group_sizes_tensor(self.group_end_offsets).tolist()

    @cached_property
    def ks_tensor(self) -> torch.Tensor:
        return _group_sizes_tensor(self.group_end_offsets)

    @property
    def total_tokens(self) -> int:
        return sum(self.group_sizes)

    def groups_are_block_aligned(self, block_size: int) -> bool:
        if self.groups_block_aligned_by_construction:
            return True
        sizes = _group_sizes_tensor(self.group_end_offsets)
        return bool((sizes % block_size == 0).all().item())

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
    """Convert cumulative int32 group end offsets to host-side group sizes."""

    assert group_end_offsets is not None and group_end_offsets.dtype == torch.int32, (
        "group_end_offsets must be int32"
    )
    return _group_sizes_tensor(group_end_offsets).tolist()


def build_deepgemm_grouped_offset_plan(
    group_end_offsets: torch.Tensor,
    *,
    original_group_end_offsets: torch.Tensor | None = None,
    padded_group_start_offsets: torch.Tensor | None = None,
    num_rows: int | None = None,
    groups_block_aligned_by_construction: bool = False,
) -> DeepGemmGroupedOffsetPlan:
    """Build the row layout and offsets required by DeepGEMM grouped kernels.

    ``group_end_offsets`` describes the padded groups passed to grouped GEMM.
    When padding was applied, ``original_group_end_offsets`` and
    ``padded_group_start_offsets`` identify the live token ranges inside each
    padded group so ``grouped_layout`` can route only real rows and mark padding
    rows as ``-1``. ``num_rows`` controls the layout length; if omitted, the
    last padded offset is used.
    """

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

    grouped_layout = _build_deepgemm_m_grouped_layout(
        group_end_offsets,
        original_group_end_offsets=original_group_end_offsets,
        padded_group_start_offsets=padded_group_start_offsets,
        num_rows=num_rows,
    )

    return DeepGemmGroupedOffsetPlan(
        group_end_offsets=group_end_offsets,
        grouped_layout=grouped_layout,
        groups_block_aligned_by_construction=groups_block_aligned_by_construction,
    )


def _build_deepgemm_m_grouped_layout(
    group_end_offsets: torch.Tensor,
    *,
    original_group_end_offsets: torch.Tensor | None = None,
    padded_group_start_offsets: torch.Tensor | None = None,
    num_rows: int | None = None,
) -> torch.Tensor:
    """Build DeepGEMM's per-row expert map for M-grouped GEMM."""

    # Padded rows are marked -1 so the kernel skips them instead of routing
    # them to an expert.
    assert group_end_offsets.dtype == torch.int32, "group_end_offsets must be int32"
    device = group_end_offsets.device

    if original_group_end_offsets is None:
        total = group_end_offsets[-1]
        if num_rows is None:
            num_rows = int(total.item())
        row_ids = torch.arange(num_rows, dtype=torch.int32, device=device)
        layout = torch.bucketize(row_ids, group_end_offsets, right=True).to(torch.int32)
        layout[row_ids >= total] = -1
        return layout.contiguous()

    assert padded_group_start_offsets is not None, (
        "padded_group_start_offsets must be provided with original_group_end_offsets"
    )
    original_sizes = _group_sizes_tensor(original_group_end_offsets)
    padded_valid_ends = padded_group_start_offsets + original_sizes
    if num_rows is None:
        num_rows = int(group_end_offsets[-1].item())
    row_ids = torch.arange(num_rows, dtype=torch.int32, device=device)
    expert = torch.bucketize(row_ids, padded_group_start_offsets, right=True) - 1
    expert_clamped = expert.clamp(min=0)
    valid = (expert >= 0) & (row_ids < padded_valid_ends[expert_clamped])
    layout = expert_clamped.to(torch.int32)
    layout[~valid] = -1
    return layout.contiguous()


def build_deepgemm_k_grouped_quant_metadata(
    group_end_offsets: torch.Tensor,
    group_sizes: list[int],
    block_size: int,
    dim: int,
) -> DeepGemmKGroupedQuantMetadata:
    """Build dense per-block metadata for compact K-grouped quantization.

    DeepGEMM's K-grouped wgrad operands concatenate each expert as a flat
    ``(dim, expert_tokens)`` slice. This helper maps each valid token block to
    the base offset and expert size needed by the compact Triton quantizer.
    """

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
