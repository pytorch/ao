# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD 3-Clause license found in the
# LICENSE file in the root directory of this source tree.
"""
Syncless Expert Parallel: expert parallelism using MXFP8 token dispatch and
combine via symmetric memory push writes.  Zero device-to-host syncs.

This mirrors the ``ExpertParallel`` style in torchtitan
(``torchtitan/distributed/expert_parallel.py``) but replaces the standard
all-to-all + permute/unpermute pipeline with the syncless kernels in
``token_dispatch.py`` and ``token_combine.py``.

Dispatch (input_fn):
    1. Dynamically quantise bf16 tokens to MXFP8 (e4m3 + e8m0 scales).
    2. Push tokens to expert-major padded layout on destination ranks via
       symmetric memory – no NCCL all-to-all, no D2H syncs.
    3. Return the MXFP8 output tensors + expert padded offsets so the
       module can feed them into an MXFP8 grouped GEMM.

Combine (output_fn):
    1. Push bf16 expert outputs from expert-major layout back to source
       ranks in their original rank-major order via symmetric memory.
    2. Return rank-major bf16 output.
"""

import torch
import torch.distributed as dist
import torch.nn as nn
from torch.distributed.device_mesh import DeviceMesh
from torch.distributed.tensor import Shard
from torch.distributed.tensor.parallel import ParallelStyle

from torchao.prototype.moe_training.ep.syncless.sym_mem_buffer_manager import (
    SymmetricMemoryBufferManager,
    get_sym_mem_buffer_manager,
)
from torchao.prototype.moe_training.ep.syncless.token_combine import token_combine
from torchao.prototype.moe_training.ep.syncless.token_dispatch import (
    mxfp8_token_dispatch,
)


class SynclessExpertParallel(ParallelStyle):
    """Expert parallelism using syncless MXFP8 token dispatch and combine.

    Drop-in replacement for torchtitan's ``ExpertParallel`` that eliminates
    all device-to-host synchronisations in the dispatch/combine path.

    The wrapped module's ``forward`` must accept::

        forward(output_e4m3, output_scales_e8m0,
                num_tokens_per_expert, expert_padded_offsets)

    and return a **bf16** tensor in the same expert-major padded layout.
    ``token_combine`` then routes that tensor back to source ranks.

    Args:
        sym_mem_buffer_manager: optional pre-allocated ``SymmetricMemoryBufferManager``.
            If *None*, the module-level singleton from ``get_sym_mem_buffer_manager()``
            is used.
        token_alignment: pad each expert's token group to a multiple of this
            value (default 128, required by MXFP8 grouped GEMM).
    """

    def __init__(
        self,
        sym_mem_buffer_manager: SymmetricMemoryBufferManager | None = None,
        token_alignment: int = 128,
    ):
        super().__init__()
        self.sym_mem_buffer_manager = sym_mem_buffer_manager
        self.token_alignment = token_alignment

        # Metadata saved during dispatch for use in combine.
        self._all_expert_splits: torch.Tensor | None = None
        self._expert_padded_offsets: torch.Tensor | None = None
        self._num_input_tokens: int | None = None

    def _partition_fn(self, name: str, mod: nn.Module, device_mesh: DeviceMesh) -> None:
        from torch.distributed.tensor import distribute_tensor

        for param_name, param in mod.named_parameters(recurse=False):
            dist_param = nn.Parameter(distribute_tensor(param, device_mesh, [Shard(0)]))
            mod.register_parameter(param_name, dist_param)

    def _token_dispatch(
        self, mod: nn.Module, inputs: tuple, device_mesh: DeviceMesh
    ) -> tuple:
        """MXFP8 quantize + syncless push-based all-to-all dispatch, writing directly to padded expert major layout.

        Args:
            inputs: ``(routed_input, num_tokens_per_expert)`` where
                *routed_input* is bf16 ``(num_tokens, dim)`` in router-
                assigned order and *num_tokens_per_expert* is a 1-D int
                tensor of length ``world_size * num_local_experts``.

        Returns:
            Tuple of ``(output_e4m3, output_scales_e8m0,
            num_tokens_per_expert_group, expert_padded_offsets)`` ready
            for the module's grouped GEMM.
        """
        routed_input, num_tokens_per_expert = inputs
        group = device_mesh.get_group()
        world_size = dist.get_world_size(group)
        num_local_experts = num_tokens_per_expert.shape[0] // world_size

        # Remember how many tokens this rank owns (needed by combine).
        self._num_input_tokens = routed_input.shape[0]

        # Build per-(dst_rank, expert) split matrix from the flat vector.
        # input_expert_splits[dst_rank, expert_idx] = tokens this rank
        # sends to expert_idx on dst_rank.
        input_expert_splits = num_tokens_per_expert.view(
            world_size, num_local_experts
        ).to(torch.int64)
        input_rank_splits = input_expert_splits.sum(dim=1)

        sym_mem_buffers = self.sym_mem_buffer_manager or get_sym_mem_buffer_manager()

        (
            output_e4m3,
            output_scales_e8m0,
            _output_rank_splits,
            _output_expert_splits,
            expert_padded_offsets,
            all_expert_splits,
            padded_tokens_per_expert,
        ) = mxfp8_token_dispatch(
            routed_input,
            input_rank_splits,
            input_expert_splits,
            group,
            sym_mem_buffers,
            self.token_alignment,
        )

        # Save metadata for combine.
        self._expert_padded_offsets = expert_padded_offsets
        self._all_expert_splits = all_expert_splits

        return (
            output_e4m3,
            output_scales_e8m0,
            padded_tokens_per_expert,
            expert_padded_offsets,
        )

    def _token_combine(
        self, mod: nn.Module, routed_output: torch.Tensor, device_mesh: DeviceMesh
    ) -> torch.Tensor:
        """Syncless push-based combine: expert-major -> rank-major.

        Args:
            routed_output: bf16 tensor in expert-major padded layout
                (same layout as the dispatch output).

        Returns:
            bf16 tensor ``(num_input_tokens, dim)`` in the original
            rank-major order.
        """
        group = device_mesh.get_group()
        sym_mem_buffers = self.sym_mem_buffer_manager or get_sym_mem_buffer_manager()

        return token_combine(
            routed_output,
            self._all_expert_splits,
            self._expert_padded_offsets,
            self._num_input_tokens,
            group,
            sym_mem_buffers,
            self.token_alignment,
        )

    def _apply(self, module: nn.Module, device_mesh: DeviceMesh) -> nn.Module:
        from torch.distributed.tensor import distribute_module

        return distribute_module(
            module,
            device_mesh,
            partition_fn=self._partition_fn,
            input_fn=self._token_dispatch,
            output_fn=self._token_combine,
        )
