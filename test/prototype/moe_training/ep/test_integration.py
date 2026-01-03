import sys
from pathlib import Path

# Get the repo root (5 levels up from this file: ep -> moe_training -> prototype -> test -> ao)
repo_root = Path(__file__).resolve().parent.parent.parent.parent.parent
sys.path.insert(0, str(repo_root))

import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch.distributed._functional_collectives import (
    all_to_all_single,
)
from torch.testing._internal.common_distributed import MultiProcessTestCase
from torch.testing._internal.common_utils import run_tests

from test.prototype.moe_training.testing_utils import generate_split_sizes
from torchao.prototype.moe_training.ep import (
    a2a_combine,
    a2a_dispatch,
    permute,
    unpermute,
)
from torchao.prototype.moe_training.scaled_grouped_mm import (
    _to_mxfp8_then_scaled_grouped_mm,
)
from torchao.prototype.mx_formats.mx_tensor import MXTensor


class TestIntegration(MultiProcessTestCase):
    def setUp(self):
        super().setUp()
        self._spawn_processes()

    @property
    def world_size(self):
        return 2

    @property
    def device(self):
        return torch.device(f"cuda:{self.rank}")

    def _init_process(self):
        torch.cuda.set_device(self.device)
        store = dist.FileStore(self.file_name, self.world_size)
        dist.init_process_group(
            backend="nccl",
            world_size=self.world_size,
            rank=self.rank,
            store=store,
        )
        torch.manual_seed(42 + self.rank)

    def test_full_pipeline(self):
        self._init_process()
        try:
            tokens = 64
            hidden_dim = 128
            num_experts = 4

            # Create input activations and expert weights
            input_tensor = torch.randn(
                tokens,
                hidden_dim,
                device=self.device,
                dtype=torch.bfloat16,
                requires_grad=True,
            )

            expert_weights = torch.randn(
                num_experts,
                hidden_dim,
                hidden_dim,
                device=self.device,
                dtype=torch.bfloat16,
                requires_grad=True,
            )
            expert_weights_t = (
                expert_weights.transpose(-2, -1).contiguous().transpose(-2, -1)
            )

            # Generate random tokens per expert that sum to total tokens
            # Shape should be [ep_degree * num_experts] where each rank has num_experts values
            ep_degree = self.world_size
            num_tokens_per_expert = generate_split_sizes(
                ep_degree * num_experts, tokens, self.device
            )

            # Compute tokens per expert group using tokens per expert
            with torch.no_grad():
                num_tokens_per_expert_group = all_to_all_single(
                    num_tokens_per_expert,
                    None,
                    None,
                    group=dist.group.WORLD,
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

            group = dist.group.WORLD

            # === Stage 1: a2a dispatch
            # - inputs: bf16
            # - outputs: MXTensor (mxfp8)
            mx_dispatched = a2a_dispatch(
                input_tensor,
                output_splits.tolist(),
                input_splits.tolist(),
                group=group,
            )
            assert isinstance(mx_dispatched, MXTensor)

            # Use deterministic permutation based on rank to avoid randomness issues
            torch.manual_seed(42)  # Same seed on all ranks

            # === Stage 2: permute
            # - inputs: MXTensor (mxfp8)
            # - outputs: MXTensor (mxfp8)
            block_size = 32
            (
                padded_mx_shape,
                mx_permuted,
                permuted_indices,
                num_tokens_per_expert_padded,
            ) = permute(
                mx_dispatched,
                num_tokens_per_expert_group,
                ep_degree,
                num_experts,
                group_size_multiple_of=block_size,
            )
            assert isinstance(mx_permuted, MXTensor)

            # === Stage 3: mxfp8 grouped mm
            # - inputs: MXTensor (mxfp8), bf16 weight
            # - outputs: bf16
            gemm_output = _to_mxfp8_then_scaled_grouped_mm(
                mx_permuted,
                expert_weights_t,
                offs=num_tokens_per_expert_padded,
                block_size=block_size,
                use_cuda_for_blocked_layout=False,
            )
            assert gemm_output.dtype == torch.bfloat16

            # === Stage 4: unpermute
            # - inputs: bf16
            # - outputs: bf16
            unpermuted = unpermute(gemm_output, permuted_indices, padded_mx_shape)
            assert unpermuted.dtype == torch.bfloat16

            # === Stage 5: a2a combine
            # - inputs: bf16
            # - outputs: bf16
            final_output = a2a_combine(
                unpermuted,
                output_splits=input_splits.tolist(),
                input_splits=output_splits.tolist(),
                group=group,
            )
            assert final_output.dtype == torch.bfloat16

            # === Stage 6: backward
            # bf16 grad -> a2a combine backward produes MXTensor grad -> unpermute backward -> ...
            # ... mxfp8 grouped mm backward -> permute backward -> a2a
            labels = torch.ones_like(final_output)
            loss = F.mse_loss(final_output, labels)
            loss.backward()

            assert input_tensor.grad is not None
            assert input_tensor.grad.dtype == torch.bfloat16
            assert expert_weights.grad is not None
            assert expert_weights.grad.dtype == torch.bfloat16
        finally:
            dist.destroy_process_group()


if __name__ == "__main__":
    run_tests()
