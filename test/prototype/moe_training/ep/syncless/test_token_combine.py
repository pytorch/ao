import pytest
import torch

if not torch.cuda.is_available() or torch.cuda.get_device_capability() != (10, 0):
    pytest.skip("Test requires CUDA build on SM100", allow_module_level=True)

import torch.distributed as dist
import torch.distributed._symmetric_memory as symm_mem
from torch.distributed._functional_collectives import all_to_all_single_autograd
from torch.testing._internal.common_distributed import MultiProcessTestCase
from torch.testing._internal.common_utils import (
    instantiate_parametrized_tests,
    run_tests,
)

from test.prototype.moe_training.testing_utils import generate_split_sizes
from torchao.prototype.moe_training.ep.syncless.token_combine import token_combine
from torchao.prototype.moe_training.ep.syncless.token_dispatch import (
    mxfp8_token_dispatch,
)


@instantiate_parametrized_tests
class SynclessTokenCombineTest(MultiProcessTestCase):
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

    def test_token_combine(self):
        """Dispatch tokens, then combine them back. Verify the round-trip
        recovers the original token values on each source rank.

        Uses small integer values that are exactly representable in fp8 e4m3,
        so the dispatch (bf16 -> fp8) -> combine (bf16) round-trip is exact.
        """
        self._init_process()
        try:
            torch.manual_seed(42 + self.rank)
            self._init_device()

            tokens_per_expert = 2
            experts_per_rank = 2
            total_local_tokens = tokens_per_expert * experts_per_rank
            dim = 32

            # Create input with distinct per-token values (small integers for fp8 exactness)
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

            # --- Dispatch: route tokens to expert-major layout ---
            total_experts_global = self.world_size * experts_per_rank
            num_tokens_per_expert_global = generate_split_sizes(
                total_experts_global, total_local_tokens, self.device
            )
            expert_splits_per_rank = num_tokens_per_expert_global.view(
                self.world_size, -1
            )
            input_rank_level_splits = expert_splits_per_rank.sum(dim=1)

            total_global_tokens = total_local_tokens * self.world_size
            max_output_tokens_per_rank = (
                total_global_tokens + 128 * experts_per_rank * self.world_size
            )

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

            # --- Create bf16 expert-major data for combine input ---
            # Convert to bf16 here in a real model we'd go through the mxfp8 grouped_mm which produces bf16 outputs.
            combine_input = output_e4m3.to(torch.bfloat16)

            # --- Combine: route tokens back to source ranks ---
            combine_output = token_combine(
                combine_input,
                all_expert_splits,
                expert_padded_offsets,
                total_local_tokens,
                dist.group.WORLD,
            )

            # Build reference using standard all_to_all / NCCL APIs
            # Extract actual (non-padded) tokens from expert-major layout,
            # reordered from expert-major to source-rank-major for all_to_all.
            rank = self.rank
            num_experts = experts_per_rank
            recv_expert_splits = all_expert_splits[
                :, rank, :
            ]  # (world_size, num_experts)

            # Collect tokens grouped by source rank (for all_to_all input)
            tokens_by_src_rank = []
            for src_rank in range(self.world_size):
                for expert_idx in range(num_experts):
                    n = recv_expert_splits[src_rank, expert_idx].item()
                    if n > 0:
                        # Compute read position in expert-major buffer
                        expert_start = expert_padded_offsets[expert_idx].item()
                        within_expert_off = 0
                        for prev_src in range(src_rank):
                            within_expert_off += recv_expert_splits[
                                prev_src, expert_idx
                            ].item()
                        read_pos = expert_start + within_expert_off
                        tokens_by_src_rank.append(
                            combine_input[read_pos : read_pos + n]
                        )

            if tokens_by_src_rank:
                rank_major_on_this_rank = torch.cat(tokens_by_src_rank, dim=0)
            else:
                rank_major_on_this_rank = torch.empty(
                    0, dim, dtype=torch.bfloat16, device=self.device
                )

            # Compute per-source-rank token counts for the all_to_all
            combine_input_splits = [
                recv_expert_splits[src, :].sum().item()
                for src in range(self.world_size)
            ]
            # Output splits = what this rank originally sent to each dst_rank
            combine_output_splits = [
                expert_splits_per_rank[dst, :].sum().item()
                for dst in range(self.world_size)
            ]

            ref_output = all_to_all_single_autograd(
                rank_major_on_this_rank,
                combine_output_splits,
                combine_input_splits,
                dist.group.WORLD,
            )

            assert combine_output.shape == ref_output.shape, (
                f"Shape mismatch: combine={combine_output.shape}, ref={ref_output.shape}"
            )

            torch.testing.assert_close(
                combine_output,
                ref_output,
                atol=0,
                rtol=0,
                msg="Token combine output does not match reference all_to_all",
            )

        finally:
            dist.destroy_process_group()


if __name__ == "__main__":
    run_tests()
