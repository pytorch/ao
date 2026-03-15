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

from test.prototype.moe_training.testing_utils import generate_split_sizes
from torchao.float8.float8_utils import (
    compute_error,
)
from torchao.prototype.moe_training.kernels.mxfp8.comms import (
    mxfp8_on_device_all_to_all_v,
    to_mxfp8_a2a_dequant,
)


def _build_padded_expert_group_reference(
    dispatched: torch.Tensor,
    output_expert_splits: torch.Tensor,
    alignment_size: int = 32,
) -> tuple[torch.Tensor, torch.Tensor]:
    num_remote_ranks, num_experts_per_rank = output_expert_splits.shape
    output_chunks = []
    padded_group_end_offsets = torch.empty(
        num_experts_per_rank, dtype=torch.int64, device=dispatched.device
    )

    remote_rank_offsets = torch.cumsum(
        output_expert_splits.sum(dim=1, dtype=torch.int64),
        dim=0,
        dtype=torch.int64,
    )
    remote_rank_starts = torch.cat(
        (
            output_expert_splits.new_zeros(1, dtype=torch.int64),
            remote_rank_offsets[:-1],
        )
    )

    cumulative_padded_rows = 0
    for expert_idx in range(num_experts_per_rank):
        per_expert_chunks = []
        for remote_rank in range(num_remote_ranks):
            expert_counts_for_remote_rank = output_expert_splits[remote_rank]
            expert_start = remote_rank_starts[
                remote_rank
            ] + expert_counts_for_remote_rank[:expert_idx].sum(dtype=torch.int64)
            expert_tokens = expert_counts_for_remote_rank[expert_idx]
            expert_end = expert_start + expert_tokens
            per_expert_chunks.append(dispatched[expert_start:expert_end])

        expert_chunk = torch.cat(per_expert_chunks, dim=0)
        padded_rows = (-expert_chunk.shape[0]) % alignment_size
        if padded_rows > 0:
            expert_chunk = torch.cat(
                (
                    expert_chunk,
                    expert_chunk.new_zeros((padded_rows, expert_chunk.shape[1])),
                ),
                dim=0,
            )
        output_chunks.append(expert_chunk)
        cumulative_padded_rows += expert_chunk.shape[0]
        padded_group_end_offsets[expert_idx] = cumulative_padded_rows

    return torch.cat(output_chunks, dim=0), padded_group_end_offsets


@instantiate_parametrized_tests
class MXFP8OnDeviceAllToAllVTest(MultiProcessTestCase):
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

            tokens_per_ep_rank = 8192
            dim = 2048
            num_experts_per_rank = 2
            input_tensor = torch.randn(
                tokens_per_ep_rank,
                dim,
                device=self.device,
                dtype=torch.bfloat16,
                requires_grad=True,
            )
            ref_input_tensor = input_tensor.detach().clone().requires_grad_(True)

            # Partition the sender's local tokens across all (destination rank, local expert)
            # buckets. Row i is what this rank sends to the experts hosted on rank i.
            expert_splits_per_rank = generate_split_sizes(
                self.world_size * num_experts_per_rank,
                tokens_per_ep_rank,
                self.device,
            ).view(self.world_size, num_experts_per_rank)

            # Compute input_splits from expert_splits_per_rank (sum across experts)
            input_splits = expert_splits_per_rank.sum(dim=1)

            # Max output tokens per rank is worst case where one rank receives all tokens,
            # padded independently for each expert.
            max_output_tokens_per_rank = (
                tokens_per_ep_rank * self.world_size + 32 * num_experts_per_rank
            )

            # Test forward
            output, output_splits, output_expert_splits, padded_group_end_offsets = (
                mxfp8_on_device_all_to_all_v(
                    input_tensor,
                    input_splits,
                    max_output_tokens_per_rank,
                    expert_splits_per_rank,
                    dist.group.WORLD,
                )
            )

            # Reference torch.all_to_all_single to compare against
            output_splits_ref = torch.empty_like(output_splits)

            # Compute output splits from input splits
            dist.all_to_all_single(output_splits_ref, input_splits)

            ref_dispatched = all_to_all_single_autograd(
                ref_input_tensor,
                output_splits_ref.tolist(),
                input_splits.tolist(),
                dist.group.WORLD,
            )

            # Compute reference expert splits: each row is what one remote rank sends to
            # our local experts.
            output_expert_splits_ref = torch.empty_like(output_expert_splits)
            gathered_expert_splits = [
                torch.empty_like(expert_splits_per_rank) for _ in range(self.world_size)
            ]
            dist.all_gather(gathered_expert_splits, expert_splits_per_rank)
            for remote_rank, remote_expert_splits in enumerate(gathered_expert_splits):
                output_expert_splits_ref[remote_rank] = remote_expert_splits[self.rank]

            ref_output, padded_group_end_offsets_ref = (
                _build_padded_expert_group_reference(
                    ref_dispatched,
                    output_expert_splits_ref,
                )
            )
            total_padded_rows = int(padded_group_end_offsets_ref[-1].item())

            # Compare output
            assert torch.equal(output_splits, output_splits_ref), (
                "output_splits mismatch"
            )
            assert torch.equal(output_expert_splits, output_expert_splits_ref), (
                f"output_expert_splits mismatch: got {output_expert_splits}, expected {output_expert_splits_ref}"
            )
            assert torch.equal(
                padded_group_end_offsets, padded_group_end_offsets_ref
            ), (
                f"padded_group_end_offsets mismatch: got {padded_group_end_offsets}, expected {padded_group_end_offsets_ref}"
            )

            output = output[:total_padded_rows]
            sqnr = compute_error(ref_output, output)
            min_sqnr = 30.0
            assert sqnr > min_sqnr, f"sqnr={sqnr} is less than min_sqnr={min_sqnr}"

            # Test backwards
            labels = torch.ones_like(output)
            loss = F.mse_loss(output, labels)
            ref_loss = F.mse_loss(ref_output, labels)
            loss.backward()
            ref_loss.backward()

            # Compare grads
            grad_sqnr = compute_error(ref_input_tensor.grad, input_tensor.grad)
            min_grad_sqnr = 28.0
            assert grad_sqnr > min_grad_sqnr, (
                f"grad_sqnr={grad_sqnr} is less than min_grad_sqnr={min_grad_sqnr}"
            )

        finally:
            dist.destroy_process_group()


@instantiate_parametrized_tests
class ToMXFP8AllToAllVDequantTest(MultiProcessTestCase):
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

            tokens_per_ep_rank = 8192
            dim = 2048
            input_tensor = torch.randn(
                tokens_per_ep_rank,
                dim,
                device=self.device,
                dtype=torch.bfloat16,
                requires_grad=True,
            )
            ref_input_tensor = input_tensor.detach().clone().requires_grad_(True)

            # Generate random input splits that sum to tokens_per_ep_rank
            experts_per_rank = 2
            num_splits = experts_per_rank * self.world_size
            num_tokens_per_expert = generate_split_sizes(
                num_splits, tokens_per_ep_rank, self.device
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

            # Compute reference a2a autograd
            ref_output = all_to_all_single_autograd(
                ref_input_tensor,
                output_splits.tolist(),
                input_splits.tolist(),
                dist.group.WORLD,
            )

            # Compute mxfp8 a2a sync
            output = to_mxfp8_a2a_dequant(
                input_tensor,
                output_splits.tolist(),
                input_splits.tolist(),
                group_name,
            )

            # Compare output
            sqnr = compute_error(ref_output, output)
            min_sqnr = 30.0
            assert sqnr > min_sqnr, f"sqnr={sqnr} is less than min_sqnr={min_sqnr}"

            # Test backwards
            labels = torch.ones_like(output)
            ref_loss = F.mse_loss(ref_output, labels)
            loss = F.mse_loss(output, labels)
            ref_loss.backward()
            loss.backward()

            # Compare grads
            grad_sqnr = compute_error(ref_input_tensor.grad, input_tensor.grad)
            min_grad_sqnr = 28.0
            assert grad_sqnr > min_grad_sqnr, (
                f"grad_sqnr={grad_sqnr} is less than min_grad_sqnr={min_grad_sqnr}"
            )

        finally:
            dist.destroy_process_group()


if __name__ == "__main__":
    run_tests()
