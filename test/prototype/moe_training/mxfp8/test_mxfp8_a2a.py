import pytest
import torch

if not torch.cuda.is_available() or torch.cuda.get_device_capability() != (10, 0):
    pytest.skip("Test requires CUDA build on SM100", allow_module_level=True)

import torch.distributed as dist
import torch.distributed._symmetric_memory as symm_mem
from torch.distributed._functional_collectives import (
    all_to_all_single_autograd,
)
from torch.testing._internal.common_distributed import (
    MultiProcessTestCase,
)
from torch.testing._internal.common_utils import (
    instantiate_parametrized_tests,
    run_tests,
)

from test.prototype.moe_training.testing_utils import generate_split_sizes
from torchao.prototype.moe_training.kernels.mxfp8.comms import (
    mxfp8_syncless_all_to_all_expert_major,
)


@instantiate_parametrized_tests
class MXFP8SynclessAllToAllExpertMajorTest(MultiProcessTestCase):
    def setUp(self) -> None:
        super().setUp()
        self._spawn_processes()

    @property
    def world_size(self) -> int:
        return 4

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
            input_tensor = torch.stack(
                [
                    torch.full(
                        (dim,),
                        float(start_value + i),
                        dtype=torch.bfloat16,
                        device=self.device,
                    )
                    for i in range(total_local_tokens)
                ]
            )
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
                mx_output,
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

            # Extract fp8 data and scales from MXTensor for comparison
            from torchao.prototype.mx_formats.mx_tensor import MXTensor

            assert isinstance(mx_output, MXTensor)
            output_e4m3 = mx_output.qdata
            output_scales_e8m0 = mx_output.scale.view(torch.uint8)

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

            # Build reference expert-major layout with padding
            ref_output_expert_major = build_reference_expert_major_output(
                ref_output,
                output_expert_splits,
                output_rank_level_splits_ref,
                experts_per_rank,
                self.world_size,
                self.device,
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

            assert torch.equal(output_expert_splits, output_expert_splits_ref), (
                f"output_expert_splits mismatch: got {output_expert_splits}, expected {output_expert_splits_ref}"
            )

            # Slice to permute_and_pad size for comparison.
            # - Triton kernel is "EP aware" and overallocates for case where all tokens go to one EP rank.
            #   It returns this overallocated buffer with offsets for use by grouped_mm directly.
            # - permute_and_pad is intended for local use, and overallocates based on case where
            compare_size = ref_output_e4m3.shape[0]

            # Compare quantized output data and scales
            torch.testing.assert_close(
                output_e4m3[:compare_size].view(torch.float32),
                ref_output_e4m3.view(torch.float32),
                atol=0,
                rtol=0,
            )
            torch.testing.assert_close(
                output_scales_e8m0[:compare_size].view(
                    torch.uint8
                ),  # output has full overallocated buffer, so only compare relevant part
                ref_output_scales_e8m0.view(torch.uint8),
                atol=0,
                rtol=0,
            )

            print(f"\n[rank {self.rank}] === Testing Backward Pass ===")

            # Create dummy gradients only for actual data (not padded/overallocated portion)
            grad = torch.ones(
                compare_size, dim, dtype=torch.bfloat16, device=self.device
            )
            ref_grad = torch.ones_like(ref_output)

            print(f"[rank {self.rank}] grad shape: {grad.shape}")
            print(f"[rank {self.rank}] ref_grad shape: {ref_grad.shape}")
            print(f"[rank {self.rank}] mx_output shape: {mx_output.shape}")
            print(f"[rank {self.rank}] compare_size: {compare_size}")

            # Backward pass - slice mx_output to actual data size
            mx_output[:compare_size].backward(grad)
            ref_output.backward(ref_grad)

            print(
                f"[rank {self.rank}] input_tensor.grad dtype: {input_tensor.grad.dtype}"
            )
            print(
                f"[rank {self.rank}] ref_input_tensor.grad dtype: {ref_input_tensor.grad.dtype}"
            )
            print(
                f"[rank {self.rank}] input_tensor.grad shape: {input_tensor.grad.shape}"
            )
            print(
                f"[rank {self.rank}] ref_input_tensor.grad shape: {ref_input_tensor.grad.shape}"
            )

            # Verify gradients are in bf16 (no quantization in backward)
            assert input_tensor.grad.dtype == torch.bfloat16, (
                f"Expected bf16 gradients, got {input_tensor.grad.dtype}"
            )

            # Compare gradients - should match exactly since no quantization in backward
            torch.testing.assert_close(
                input_tensor.grad,
                ref_input_tensor.grad,
                atol=0,
                rtol=0,
                msg="Backward gradients should match exactly (no quantization)",
            )

            print(f"[rank {self.rank}] Backward pass test PASSED!")
            print(f"[rank {self.rank}] Gradients match exactly (bf16, no quantization)")

        finally:
            dist.destroy_process_group()


def build_reference_expert_major_output(
    ref_output: torch.Tensor,
    output_expert_splits: torch.Tensor,
    output_rank_level_splits_ref: torch.Tensor,
    experts_per_rank: int,
    world_size: int,
    device: torch.device,
) -> torch.Tensor:
    """
    Build reference output in expert-major layout with padding.

    Args:
        ref_output: rank-major output from torch all_to_all
        output_expert_splits: shape (world_size, experts_per_rank) - tokens per expert from each rank
        output_rank_level_splits_ref: shape (world_size,) - total tokens from each rank
        experts_per_rank: number of experts per rank
        world_size: number of ranks
        device: torch device

    Returns:
        ref_output_expert_major: expert-major layout with padding (zeros in padded positions)
    """
    dim = ref_output.shape[1]

    # Calculate size: sum of padded expert sizes
    total_size = 0
    for expert_idx in range(experts_per_rank):
        tokens_for_expert = output_expert_splits[:, expert_idx].sum().item()
        padded_size = ((tokens_for_expert + 31) // 32) * 32
        total_size += padded_size

    # Create padded output buffer
    ref_output_expert_major = torch.zeros(
        total_size, dim, dtype=ref_output.dtype, device=device
    )

    # Compute starting offset for each remote rank's data in ref_output
    rank_start_offsets = torch.cat(
        [
            torch.tensor([0], device=device),
            output_rank_level_splits_ref.cumsum(0)[:-1],
        ]
    )

    # Fill in the data expert by expert
    ref_write_offset = 0  # where to write in ref_output_expert_major

    for expert_idx in range(experts_per_rank):
        # Write tokens from each remote rank for this expert
        for remote_rank in range(world_size):
            tokens_from_rank = output_expert_splits[remote_rank, expert_idx].item()
            if tokens_from_rank > 0:
                # Compute offset into ref_output for this (remote_rank, expert_idx) pair
                # Start at this rank's base offset
                rank_base = rank_start_offsets[remote_rank].item()
                # Add offset for previous experts from this rank
                expert_offset = (
                    output_expert_splits[remote_rank, :expert_idx].sum().item()
                )
                ref_read_offset = rank_base + expert_offset

                # Copy tokens
                ref_output_expert_major[
                    ref_write_offset : ref_write_offset + tokens_from_rank
                ] = ref_output[ref_read_offset : ref_read_offset + tokens_from_rank]
                ref_write_offset += tokens_from_rank

        # Pad to next multiple of 32
        tokens_for_expert = output_expert_splits[:, expert_idx].sum().item()
        padded_size = ((tokens_for_expert + 31) // 32) * 32
        ref_write_offset = ref_write_offset - tokens_for_expert + padded_size

    return ref_output_expert_major


if __name__ == "__main__":
    run_tests()
