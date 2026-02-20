#!/usr/bin/env python3
"""
Standalone example of using MXFP8 Expert Parallel with a simplified MoE layer.

This example demonstrates:
1. Setting up a simple MoE layer with GroupedExperts
2. Applying MXFP8ExpertParallel sharding to the experts
3. Running forward and backward passes with MXFP8 all-to-all communication

Usage:
    torchrun --nproc_per_node=2 mxfp8_expert_parallel_example.py
"""

import os

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributed import destroy_process_group, init_process_group
from torch.distributed._functional_collectives import all_to_all_single
from torch.distributed.device_mesh import DeviceMesh
from torch.distributed.tensor import (
    DTensor,
    Shard,
    distribute_module,
    distribute_tensor,
)
from torch.distributed.tensor.parallel import ParallelStyle, parallelize_module


# ============================================================================
# MoE Components (simplified from torchtitan/models/moe/moe.py)
# ============================================================================
class SimplifiedMoE(nn.Module):
    """
    Simplified MoE layer for demonstration purposes.

    This version assumes the input is already routed (i.e., we skip the router
    and reorderer components) to focus on the expert parallel features.
    """

    def __init__(self, dim: int, hidden_dim: int, num_experts: int):
        super().__init__()
        self.num_experts = num_experts
        self.experts = GroupedExperts(
            dim=dim,
            hidden_dim=hidden_dim,
            num_experts=num_experts,
        )

    def forward(
        self,
        routed_input: torch.torch.Tensor,
        num_tokens_per_expert: torch.torch.Tensor,
    ) -> torch.torch.Tensor:
        """
        Forward pass with already routed inputs.

        Args:
            routed_input: Routed input tensor of shape (num_tokens, dim)
            num_tokens_per_expert: Number of tokens for each expert, shape (num_experts,)

        Returns:
            Output tensor of shape (num_tokens, dim)
        """
        return self.experts(routed_input, num_tokens_per_expert)


class GroupedExperts(nn.Module):
    """Grouped experts module that processes tokens with grouped matrix multiplication."""

    def __init__(
        self,
        dim: int,
        hidden_dim: int,
        num_experts: int,
    ):
        super().__init__()
        self.num_experts = num_experts
        self.w1 = nn.Parameter(torch.empty(num_experts, hidden_dim, dim))
        self.w2 = nn.Parameter(torch.empty(num_experts, dim, hidden_dim))
        self.w3 = nn.Parameter(torch.empty(num_experts, hidden_dim, dim))

    def forward(
        self,
        x: torch.torch.Tensor,
        num_tokens_per_expert: torch.torch.Tensor,
    ) -> torch.torch.Tensor:
        """
        Args:
            x: Input tensor of shape (num_tokens, dim)
            num_tokens_per_expert: Number of tokens for each expert, shape (num_experts,)

        Returns:
            Output tensor of shape (num_tokens, dim)
        """

        from torchao.prototype.mx_formats.grouped_mm import (
            _to_mxfp8_then_scaled_grouped_mm as mxfp8_gmm,
        )

        # Convert from DTensor to local tensor if needed (for EP)
        if isinstance(self.w1, DTensor):
            w1 = self.w1.to_local()
            w2 = self.w2.to_local()
            w3 = self.w3.to_local()
        else:
            w1 = self.w1
            w2 = self.w2
            w3 = self.w3

        # offsets: end index of each token group along dim 0
        offsets = torch.cumsum(num_tokens_per_expert, dim=0, dtype=torch.int32)

        # wgrad_with_hp recipe required for MXFP8 expert parallelism in this v0 prototype.
        h = F.silu(mxfp8_gmm(x, w1.transpose(-2, -1), offs=offsets, wgrad_with_hp=True))
        h = h * mxfp8_gmm(x, w3.transpose(-2, -1), offs=offsets, wgrad_with_hp=True)
        output = mxfp8_gmm(
            h, w2.transpose(-2, -1), offs=offsets, wgrad_with_hp=True
        ).type_as(x)
        return output

    def init_weights(self, init_std: float = 0.02):
        """Initialize weights with truncated normal distribution."""
        nn.init.trunc_normal_(self.w1, mean=0.0, std=init_std)
        nn.init.trunc_normal_(self.w2, mean=0.0, std=init_std)
        nn.init.trunc_normal_(self.w3, mean=0.0, std=init_std)


# ============================================================================
# MXFP8 Expert Parallel Setup (from torchtitan/distributed/expert_parallel.py)
# ============================================================================

# requires torchao nightly build for CUDA 12.8+
from torchao.prototype.mx_formats.expert_parallel import (
    a2a_combine_hp_fwd_mxfp8_bwd,
    a2a_dispatch_mxfp8_fwd_hp_bwd,
    permute_mxfp8_fwd_hp_bwd,
    unpermute_hp_fwd_mxfp8_bwd,
)


class MXFP8ExpertParallel(ParallelStyle):
    def __init__(self):
        super().__init__()
        self.input_splits = None
        self.output_splits = None
        self.input_shape = None
        self.permuted_indices = None

        # use torchao mxfp8 EP autograd functions as building blocks here
        self.a2a_dispatch_mxfp8_fwd_hp_bwd = a2a_dispatch_mxfp8_fwd_hp_bwd
        self.permute_mxfp8_fwd_hp_bwd = permute_mxfp8_fwd_hp_bwd
        self.unpermute_hp_fwd_mxfp8_bwd = unpermute_hp_fwd_mxfp8_bwd
        self.a2a_combine_hp_fwd_mxfp8_bwd = a2a_combine_hp_fwd_mxfp8_bwd

    def _partition_fn(self, name: str, mod: nn.Module, device_mesh: DeviceMesh) -> None:
        for param_name, param in mod.named_parameters(recurse=False):
            # experts are 3d nn.Parameters of shape (num_experts, ...,  ...).
            # shard along the experts dim.
            dist_param = nn.Parameter(distribute_tensor(param, device_mesh, [Shard(0)]))
            mod.register_parameter(param_name, dist_param)

    def _token_dispatch(
        self, mod: nn.Module, inputs: tuple, device_mesh: DeviceMesh
    ) -> tuple[torch.Tensor, torch.Tensor]:
        # annotate module input placements/sharding with input_layouts
        routed_input, num_tokens_per_expert = inputs
        ep_degree = device_mesh.shape[0]
        num_local_experts = num_tokens_per_expert.shape[0] // ep_degree

        # first all-to-all to calculate output splits from input splits.
        # note: this will incur a d2h sync
        with torch.no_grad():
            num_tokens_per_expert_group = all_to_all_single(
                num_tokens_per_expert,
                None,
                None,
                group=device_mesh.get_group(),
            )
            # Need to wait explicitly because it is used by a triton kernel later
            # which doesn't realize that AsyncCollectiveTensor needs unwrapping
            num_tokens_per_expert_group = torch.ops._c10d_functional.wait_tensor(
                num_tokens_per_expert_group
            )
            input_splits = (
                num_tokens_per_expert.view(ep_degree, -1)
                .sum(dim=1)
                .to(torch.device("cpu"), non_blocking=True)
            )
            # NOTE: this would incur a device-to-host sync
            output_splits = (
                num_tokens_per_expert_group.view(ep_degree, -1)
                .sum(dim=1)
                .to(torch.device("cpu"), non_blocking=False)
            )

            # torch dist.all_to_all_single requires input/output splits be python lists on the host
            self.input_splits, self.output_splits = (
                input_splits.tolist(),
                output_splits.tolist(),
            )

        # perform all-to-all token dispatch
        routed_input = self.a2a_dispatch_mxfp8_fwd_hp_bwd(
            routed_input,
            output_splits=self.output_splits,
            input_splits=self.input_splits,
            group_name=device_mesh.get_group().group_name,
        )

        # NOTE: After this all-to-all, the routed input is put on proper EP rank.
        #
        # However, the num_tokens_per_expert_group is not of the final target format
        # [#tokens for local expert 0, #tokens for local expert 1, ...]
        #
        # Rather, it is of the format
        # [#tokens for local expert 0 from EP rank 0, #tokens for local expert 1 from EP rank 0, ...,
        #  #tokens for local expert 0 from EP rank 1, #tokens for local expert 1 from EP rank 1, ...]
        #
        # We need to perform another shuffle to get the correct layout, via the _permute function
        # below, which also does padding to make sure the number of tokens each expert gets locally
        # is a multiple of 32 (scaling block size for MXFP8).
        (
            self.input_shape,
            routed_input,
            self.permuted_indices,
            num_tokens_per_expert_group,
            _,
        ) = self.permute_mxfp8_fwd_hp_bwd(
            routed_input, num_tokens_per_expert_group, ep_degree, num_local_experts
        )
        return routed_input, num_tokens_per_expert_group

    def _token_combine(
        self, mod: nn.Module, routed_output: torch.Tensor, device_mesh: DeviceMesh
    ) -> torch.Tensor:
        # unpermute tokens back to the original position they were received in
        routed_output = self.unpermute_hp_fwd_mxfp8_bwd(
            routed_output, self.permuted_indices, self.input_shape
        )

        # now reverse original input/output splits to route tokens back to where they came from
        routed_output = self.a2a_combine_hp_fwd_mxfp8_bwd(
            routed_output,
            output_splits=self.input_splits,  # swap input/output splits to reverse all-to-all dispatch
            input_splits=self.output_splits,
            group_name=device_mesh.get_group().group_name,
        )
        return routed_output

    def _apply(self, module: nn.Module, device_mesh: DeviceMesh) -> nn.Module:
        return distribute_module(
            module,
            device_mesh,
            partition_fn=self._partition_fn,
            input_fn=self._token_dispatch,
            output_fn=self._token_combine,
        )


def apply_mxfp8_expert_parallel(
    moe_layer: SimplifiedMoE,
    ep_mesh: DeviceMesh,
    use_mxfp8_a2a_dispatch_fwd: bool = True,
    use_mxfp8_a2a_combine_bwd: bool = True,
):
    """
    Apply MXFP8ExpertParallel to the MoE layer.
    """

    experts_plan = MXFP8ExpertParallel()

    # Apply the parallelization to the experts module
    parallelize_module(
        module=moe_layer.experts,
        device_mesh=ep_mesh,
        parallelize_plan=experts_plan,
    )


# ============================================================================
# Main Example
# ============================================================================


def main():
    # Initialize distributed process group
    init_process_group(backend="nccl")

    local_rank = int(os.environ["LOCAL_RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    device = torch.device(f"cuda:{local_rank}")
    torch.cuda.set_device(device)

    # Model configuration
    dim = 7168
    hidden_dim = 2048
    num_experts = 32
    num_local_experts = num_experts // world_size
    batch_size = 16
    seq_len = 8192

    # Create device mesh for expert parallelism
    ep_mesh = DeviceMesh("cuda", list(range(world_size)), mesh_dim_names=("ep",))

    # Create MoE layer
    moe = (
        SimplifiedMoE(
            dim=dim,
            hidden_dim=hidden_dim,
            num_experts=num_local_experts,
        )
        .to(device)
        .to(torch.bfloat16)
    )

    # Initialize weights
    moe.experts.init_weights(init_std=0.02)

    # Apply MXFP8 expert parallel
    print(f"[Rank {local_rank}] Applying MXFP8 Expert Parallelism to MoE layer...")
    apply_mxfp8_expert_parallel(
        moe,
        ep_mesh,
    )

    # Create sample routed inputs
    # In a real scenario, these would come from the router and reorderer
    num_tokens = batch_size * seq_len
    routed_input = torch.randn(num_tokens, dim, device=device, dtype=torch.bfloat16)

    # Simulate token distribution across experts
    # Each rank will have different tokens for its local experts
    num_tokens_per_local_expert = num_tokens // num_local_experts

    num_tokens_per_expert = torch.full(
        (num_local_experts,),
        num_tokens_per_local_expert,
        device=device,
        dtype=torch.int64,
    )

    # Forward pass
    print(f"[Rank {local_rank}] Running forward pass...")
    output = moe(routed_input, num_tokens_per_expert)

    # Backward pass
    print(f"[Rank {local_rank}] Running backward pass...")
    grad_output = torch.randn_like(output)
    output.backward(grad_output)

    # Cleanup
    destroy_process_group()
    print(f"[Rank {local_rank}] Done!")


if __name__ == "__main__":
    main()
