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
            num_tokens_per_expert = generate_split_sizes(
                self.world_size, tokens, self.device
            )
            ep_degree = self.world_size

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

            print("Starting A2A dispatch...")
            mx_dispatched = a2a_dispatch(
                input_tensor,
                output_splits.tolist(),
                input_splits.tolist(),
                group=group,
            )
            assert isinstance(mx_dispatched, MXTensor)
            print(f"A2A dispatch complete. Output shape: {mx_dispatched.shape}")

            # Permutation should be based on the size AFTER a2a
            tokens_after_a2a = mx_dispatched.shape[0]

            # Use deterministic permutation based on rank to avoid randomness issues
            torch.manual_seed(42)  # Same seed on all ranks
            permuted_indices = torch.randperm(tokens_after_a2a, device=self.device)
            padded_shape = torch.Size([tokens_after_a2a + 1, hidden_dim])

            print(
                f"Rank {self.rank}: tokens_after_a2a={tokens_after_a2a}, permuted_indices size={permuted_indices.shape}"
            )

            print("Starting permute...")
            mx_permuted = permute(mx_dispatched, permuted_indices, padded_shape)
            assert isinstance(mx_permuted, MXTensor)
            print("Permute complete.")

            print("Starting MXFP8 scaled grouped MM...")
            # Create group offsets that match the actual token distribution
            # Each expert gets tokens_after_a2a / num_experts tokens
            tokens_per_expert = tokens_after_a2a // num_experts
            group_offsets = torch.tensor(
                [tokens_per_expert * (i + 1) for i in range(num_experts)],
                device=self.device,
                dtype=torch.int32,
            )
            print(f"Group offsets: {group_offsets.tolist()}")
            gemm_output = _to_mxfp8_then_scaled_grouped_mm(
                mx_permuted, expert_weights_t, offs=group_offsets, block_size=32
            )
            assert gemm_output.dtype == torch.bfloat16
            print("MXFP8 scaled grouped MM complete.")

            print("Starting unpermute...")
            unpermuted = unpermute(gemm_output, permuted_indices, padded_shape)
            assert unpermuted.dtype == torch.bfloat16
            print("Unpermute complete.")

            print("Starting A2A combine...")
            final_output = a2a_combine(
                unpermuted,
                output_splits.tolist(),
                input_splits.tolist(),
                group=group,
            )
            assert final_output.dtype == torch.bfloat16
            print("A2A combine complete.")

            print("Starting backward pass...")
            # Create toy labels (all ones)
            labels = torch.ones_like(final_output)
            loss = F.mse_loss(final_output, labels)
            loss.backward()
            print("Backward pass complete.")

            assert input_tensor.grad is not None
            assert input_tensor.grad.dtype == torch.bfloat16
            assert expert_weights.grad is not None
            assert expert_weights.grad.dtype == torch.bfloat16
        finally:
            dist.destroy_process_group()


if __name__ == "__main__":
    run_tests()
