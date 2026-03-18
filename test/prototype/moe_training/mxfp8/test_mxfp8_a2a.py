import pytest
import torch

if not torch.cuda.is_available() or torch.cuda.get_device_capability() != (10, 0):
    pytest.skip("Test requires CUDA build on SM100", allow_module_level=True)

import torch.distributed as dist
import torch.distributed._symmetric_memory as symm_mem
from torch.distributed._functional_collectives import (
    all_to_all_single,
    all_to_all_single_autograd,
)
from torch.nn import functional as F
from torch.testing._internal.common_distributed import (
    MultiProcessTestCase,
)
from torch.testing._internal.common_utils import (
    instantiate_parametrized_tests,
    run_tests,
)

from torchao.float8.float8_utils import (
    compute_error,
)
from torchao.prototype.moe_training.kernels.mxfp8.comms import (
    mxfp8_syncless_all_to_all_expert_major,
)
from torchao.prototype.moe_training.ep.permute import permute_and_pad

from test.prototype.moe_training.testing_utils import generate_split_sizes


@instantiate_parametrized_tests
class MXFP8SynclessAllToAllExpertMajorTest(MultiProcessTestCase):
    def setUp(self) -> None:
        super().setUp()
        self._spawn_processes()

    @property
    def world_size(self) -> int:
        return 2

    @property
    def device(self) -> torch.device:
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

    def _init_device(self):
        symm_mem.set_backend("NVSHMEM")

    def test_a2a_fwd_bwd(self):
        self._init_process()
        try:
            torch.manual_seed(42 + self.rank)
            self._init_device()

            group_name = dist.group.WORLD.group_name
            symm_mem.enable_symm_mem_for_group(group_name)

            tokens_per_expert = 2
            experts_per_rank = 2
            total_local_tokens = tokens_per_expert * experts_per_rank
            dim = 32

            # Create input tensor with uniform distinct data per token
            # Each rank starts at different values:
            # Rank 0: rows of [1, 1, ...], [2, 2, ...], [3, 3, ...], [4, 4, ...]
            # Rank 1: rows of [5, 5, ...], [6, 6, ...], [7, 7, ...], [8, 8, ...]
            start_value = self.rank * total_local_tokens + 1
            input_tensor = torch.stack([
                torch.full((dim,), float(start_value + i), dtype=torch.bfloat16, device=self.device)
                for i in range(total_local_tokens)
            ])
            input_tensor.requires_grad_(True)
            ref_input_tensor = input_tensor.detach().clone().requires_grad_(True)

            # generate splits for our local tokens to define which are assigned to each expert GLOBALLY
            total_experts_global = self.world_size * experts_per_rank
            num_tokens_per_expert_global = generate_split_sizes(
                total_experts_global, total_local_tokens, self.device
            )

            # tokens per expert per rank. shape (world_size, experts_per_rank)
            # [r0e0, r0e1, r1e0, r1e1, ...]
            expert_splits_per_rank = num_tokens_per_expert_global.view(
                self.world_size, -1
            )

            # tokens per rank: sum of tokens per expert per rank. shape (world_size,)
            input_rank_level_splits = expert_splits_per_rank.sum(dim=1)

            # Max output tokens per rank is worst case where one rank receives all tokens.
            # Use upper bound padding, assuming 32 padding tokens per expert globally.
            total_global_tokens = total_local_tokens * self.world_size
            max_output_tokens_per_rank = (
                total_global_tokens + 32 * experts_per_rank * self.world_size
            )

            # Test forward
            (
                output_e4m3,
                output_scales_e8m0,
                output_rank_level_splits,
                output_expert_splits,
                expert_padded_offsets,
            ) = mxfp8_syncless_all_to_all_expert_major(
                input_tensor,
                input_rank_level_splits,
                expert_splits_per_rank,
                max_output_tokens_per_rank,
                group_name,
            )

            # Reference torch.all_to_all_single to compare against
            output_rank_level_splits_ref = torch.empty_like(output_rank_level_splits)

            # Compute output splits from input splits
            dist.all_to_all_single(
                output_rank_level_splits_ref, input_rank_level_splits
            )

            # Pre-allocate output buffer for reference a2a
            total_tokens_on_rank_after_a2a = output_rank_level_splits_ref.sum()
            ref_output = torch.empty(
                total_tokens_on_rank_after_a2a,
                dim,
                device=self.device,
                dtype=torch.bfloat16,
            )

            # Do the actual all_to_all_single
            ref_output = all_to_all_single_autograd(
                ref_input_tensor,
                output_rank_level_splits_ref.tolist(),
                input_rank_level_splits.tolist(),
                dist.group.WORLD,
            )

            # Compute reference expert splits using all-gather.
            # Gather expert_splits_per_rank from all ranks into a
            # (world_size, world_size, num_experts_per_rank) tensor where
            # gathered[source_rank, dest_rank, expert_idx] = tokens source_rank sends to expert_idx on dest_rank.
            gathered_expert_splits = torch.empty(
                self.world_size * self.world_size,
                experts_per_rank,
                dtype=torch.int64,
                device=self.device,
            )
            dist.all_gather_into_tensor(
                gathered_expert_splits,  # [world_size * world_size, num_experts_per_rank]
                expert_splits_per_rank,  # [world_size, num_experts_per_rank]
                group=dist.group.WORLD,
            )
            gathered_expert_splits = gathered_expert_splits.view(
                self.world_size, self.world_size, experts_per_rank
            )
            # output_expert_splits_ref[remote_rank, :] = tokens remote_rank sends to each expert on this rank
            # shape: (world_size, experts_per_rank)
            output_expert_splits_per_rank_ref = torch.empty_like(expert_splits_per_rank)
            for remote_rank in range(self.world_size):
                output_expert_splits_per_rank_ref[remote_rank, :] = (
                    gathered_expert_splits[remote_rank, self.rank, :]
                )

            # Reduce to get total tokens received per rank
            output_expert_splits_ref = output_expert_splits_per_rank_ref

            # Reorder reference output from rank-major to expert-major using permute_bf16
            # output_expert_splits_ref is [world_size, num_experts_per_rank]
            # permute_and_pad expects num_tokens_per_expert in [e0r0, e0r1, e1r0, e1r1, ...] order
            # (grouped by expert first, then by remote rank)
            # So we need to transpose first, then flatten
            num_tokens_per_expert_flat = output_expert_splits_ref.T.flatten()

            # Debug: print the shapes and first few values
            print(f"\n=== Debug info (rank {self.rank}) ===")
            print(f"[rank {self.rank}] world_size: {self.world_size}")
            print(f"[rank {self.rank}] num_experts_per_rank: {experts_per_rank}")
            print()
            print(f"[rank {self.rank}] expert_splits_per_rank shape: {expert_splits_per_rank.shape}")
            print(f"[rank {self.rank}] expert_splits_per_rank:\n{expert_splits_per_rank}")
            print(
                f"[rank {self.rank}] expert_splits (local):",
                expert_splits_per_rank[dist.get_rank(), :],
            )
            print()
            print(f"[rank {self.rank}] output_expert_splits_per_rank:\n{output_expert_splits}")
            print(f"[rank {self.rank}] output_expert_splits_ref:\n{output_expert_splits_ref}")
            print(f"[rank {self.rank}] output_expert_splits (local): {output_expert_splits}")
            print()
            print(f"[rank {self.rank}] expert_padded_offsets: {expert_padded_offsets}")
            print(f"[rank {self.rank}] num_tokens_per_expert_flat: {num_tokens_per_expert_flat}")

            # Debug: Print expected data layout
            print(f"\n[rank {self.rank}] === Expected Data Layout ===")
            for expert_idx in range(experts_per_rank):
                start_offset = expert_padded_offsets[expert_idx].item()
                tokens_from_all_ranks = output_expert_splits[:, expert_idx].sum().item()
                print(f"[rank {self.rank}] Expert {expert_idx}:")
                print(f"  - Starts at offset: {start_offset}")
                print(f"  - Total tokens: {tokens_from_all_ranks}")
                for remote_rank in range(self.world_size):
                    tokens_from_rank = output_expert_splits[remote_rank, expert_idx].item()
                    if tokens_from_rank > 0:
                        print(f"  - From rank {remote_rank}: {tokens_from_rank} tokens")
            print()

            _, ref_output_expert_major, _, _, _ = permute_and_pad(
                ref_output,
                num_tokens_per_expert_flat,
                ep_degree=self.world_size,
                num_local_experts=experts_per_rank,
                alignment=32,
            )

            # Quantize reference output to mxfp8 for comparison
            from torchao.prototype.mx_formats.mx_tensor import to_mx
            block_size = 32
            ref_output_scales_e8m0, ref_output_e4m3 = to_mx(
                ref_output_expert_major,
                elem_dtype=torch.float8_e4m3fn,
                block_size=block_size,
            )

            # Compare output splits
            assert torch.equal(
                output_rank_level_splits, output_rank_level_splits_ref
            ), "output_rank_level_splits mismatch"

            assert torch.equal(
                output_expert_splits, output_expert_splits_ref
            ), f"output_expert_splits mismatch: got {output_expert_splits}, expected {output_expert_splits_ref}"

            # Slice to permute_and_pad size for comparison.
            # - Triton kernel is "EP aware" and overallocates for case where all tokens go to one EP rank.
            #   It returns this overallocated buffer with offsets for use by grouped_mm directly.
            # - permute_and_pad is intended for local use, and overallocates based on case where
            compare_size = ref_output_e4m3.shape[0]

            # Compare quantized output data (fp8 e4m3)
            print(f"\n[rank {self.rank}] === Comparing Outputs ===")
            print(f"[rank {self.rank}] compare_size: {compare_size}")
            print(f"[rank {self.rank}] output_e4m3 shape: {output_e4m3.shape}")
            print(f"[rank {self.rank}] ref_output_e4m3 shape: {ref_output_e4m3.shape}")

            # Print first few rows and key positions
            print(f"\n[rank {self.rank}] Kernel output - First 5 rows:")
            for i in range(min(5, output_e4m3.shape[0])):
                print(f"  Row {i}: {output_e4m3[i, :8]}")  # First 8 elements

            print(f"\n[rank {self.rank}] Reference output - First 5 rows:")
            for i in range(min(5, ref_output_e4m3.shape[0])):
                print(f"  Row {i}: {ref_output_e4m3[i, :8]}")

            # For rank 1, also check around offset 32 where expert 1 should be
            if self.rank == 1 and output_e4m3.shape[0] > 34:
                print(f"\n[rank {self.rank}] Kernel output - Rows around offset 32:")
                for i in range(30, min(35, output_e4m3.shape[0])):
                    print(f"  Row {i}: {output_e4m3[i, :8]}")

                print(f"\n[rank {self.rank}] Reference output - Rows around offset 32:")
                for i in range(30, min(35, ref_output_e4m3.shape[0])):
                    print(f"  Row {i}: {ref_output_e4m3[i, :8]}")

            torch.testing.assert_close(
                output_e4m3[:compare_size].view(torch.float32),
                ref_output_e4m3.view(torch.float32),
                atol=0,
                rtol=0,
            )

            # Compare scales
            print(f"[rank {self.rank}] output_scales_e8m0 shape: {output_scales_e8m0.shape}")
            print(f"[rank {self.rank}] output_scales_e8m0[:compare_size]:\n{output_scales_e8m0[:compare_size]}")
            print(f"[rank {self.rank}] ref_output_scales_e8m0 shape: {ref_output_scales_e8m0.shape}")
            print(f"[rank {self.rank}] ref_output_scales_e8m0:\n{ref_output_scales_e8m0}")

            torch.testing.assert_close(
                output_scales_e8m0[:compare_size].view(torch.uint8),
                ref_output_scales_e8m0.view(torch.uint8),
                atol=0,
                rtol=0,
            )

            # Test backwards
            # labels = torch.ones_like(out_no_padding)
            # loss = F.mse_loss(out_no_padding, labels)
            # ref_loss = F.mse_loss(ref_output_no_padding, labels)
            # loss.backward()
            # ref_loss.backward()

            # # Compare grads
            # grad_sqnr = compute_error(ref_input_tensor.grad, input_tensor.grad)
            # min_grad_sqnr = 28.0
            # assert (
            #     grad_sqnr > min_grad_sqnr
            # ), f"grad_sqnr={grad_sqnr} is less than min_grad_sqnr={min_grad_sqnr}"

        finally:
            dist.destroy_process_group()


if __name__ == "__main__":
    run_tests()
