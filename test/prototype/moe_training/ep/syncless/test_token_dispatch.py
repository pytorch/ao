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
from torchao.prototype.moe_training.ep.permute import permute_and_pad
from torchao.prototype.moe_training.ep.syncless.token_dispatch import (
    mxfp8_token_dispatch,
)


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

    def test_mxfp8_a2a(self):
        self._init_process()
        try:
            torch.manual_seed(42 + self.rank)
            self._init_device()

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

            # Preallocate symmetric memory buffers
            from torchao.prototype.moe_training.ep.syncless.sym_mem_buffer_manager import (
                get_sym_mem_buffer_manager,
            )

            sym_mem_buffer_manager = get_sym_mem_buffer_manager()
            sym_mem_buffer_manager.preallocate_sym_mem_buffers(
                max_output_rows_per_rank=max_output_tokens_per_rank,
                data_shape=(dim,),
                scales_shape=(dim // 32,),
                data_dtype=torch.float8_e4m3fn,
                scales_dtype=torch.uint8,
                device=self.device,
            )

            # Test forward
            (
                output_e4m3,
                output_scales_e8m0,
                output_rank_level_splits,
                output_expert_splits,
                expert_padded_offsets,
                all_expert_splits,
                _padded_tokens_per_expert,
            ) = mxfp8_token_dispatch(
                input_tensor,
                input_rank_level_splits,
                expert_splits_per_rank,
                dist.group.WORLD,
            )

            # Convert scales to uint8 for comparison
            output_scales_e8m0 = output_scales_e8m0.view(torch.uint8)

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

            # Exchange expert split information using all_to_all_single
            # shape: [world_size * num_experts_per_rank]
            num_tokens_per_expert = expert_splits_per_rank.flatten()

            # Use all_to_all_single to exchange expert splits
            tokens_per_expert_group = (
                torch.distributed._functional_collectives.all_to_all_single(
                    num_tokens_per_expert,
                    None,
                    None,
                    group=dist.group.WORLD,
                )
            )
            tokens_per_expert_group = torch.ops._c10d_functional.wait_tensor(
                tokens_per_expert_group
            )

            # For comparison with the MXFP8 kernel output, we need output_expert_splits
            # Reshape tokens_per_expert_group back to match expected format
            output_expert_splits_ref = tokens_per_expert_group.view(
                self.world_size, experts_per_rank
            )

            _, ref_output_expert_major, _, _, _ = permute_and_pad(
                ref_output,
                tokens_per_expert_group,  # Use flattened version
                ep_degree=self.world_size,
                num_local_experts=experts_per_rank,
                alignment=32,
            )

            # Quantize reference output to mxfp8 for comparison
            from torchao.prototype.mx_formats.kernels import triton_to_mxfp8_dim0

            block_size = 32
            ref_output_e4m3, ref_output_scales_e8m0 = triton_to_mxfp8_dim0(
                ref_output_expert_major,
                inner_block_size=block_size,
                scaling_mode="rceil",
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

        finally:
            dist.destroy_process_group()


if __name__ == "__main__":
    run_tests()
