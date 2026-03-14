import pytest
import torch

from torchao.utils import is_cuda_version_at_least, is_sm_at_least_100

if not (
    torch.cuda.is_available()
    and is_sm_at_least_100()
    and is_cuda_version_at_least(12, 8)
):
    pytest.skip("Test requires CUDA 12.8+ with SM >= 100", allow_module_level=True)


import torch.distributed as dist
import torch.distributed._symmetric_memory as symm_mem
from torch.distributed._functional_collectives import (
    all_to_all_single,
)
from torch.testing._internal.common_distributed import MultiProcessTestCase
from torch.testing._internal.common_utils import run_tests

from test.prototype.moe_training.testing_utils import generate_split_sizes
from torchao.prototype.moe_training.ep import (
    a2a_dispatch_and_group_mxfp8_fwd_hp_bwd,
    a2a_dispatch_mxfp8_fwd_hp_bwd,
)
from torchao.prototype.moe_training.ep.permute import _permute_bf16
from torchao.prototype.mx_formats.mx_tensor import MXTensor
from torchao.quantization.utils import compute_error


class TestA2ADispatch(MultiProcessTestCase):
    def setUp(self):
        super().setUp()
        self._spawn_processes()

    @property
    def world_size(self):
        return 4

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

    def _init_device(self):
        symm_mem.set_backend("NVSHMEM")

    def test_dispatch(self):
        self._init_process()
        try:
            tokens = 64
            dim = 128
            input_tensor = torch.randn(
                tokens,
                dim,
                device=self.device,
                dtype=torch.bfloat16,
                requires_grad=False,
            )
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

            mx_output = a2a_dispatch_mxfp8_fwd_hp_bwd(
                input_tensor,
                output_splits.tolist(),
                input_splits.tolist(),
                group_name=dist.group.WORLD.group_name,
            )
            assert isinstance(mx_output, MXTensor)

            ref_output = all_to_all_single(
                input_tensor,
                output_splits.tolist(),
                input_splits.tolist(),
                group=dist.group.WORLD,
            )
            ref_output = torch.ops._c10d_functional.wait_tensor(ref_output)

            # Compare outputs
            output = mx_output.dequantize()
            sqnr = compute_error(output, ref_output).item()
            assert sqnr > 30.0, f"SQNR too low: {sqnr} dB"

            # Note: tests for backwards will be in an integration test
        finally:
            dist.destroy_process_group()

    def test_dispatch_and_group(self):
        self._init_process()
        try:
            self._init_device()
            symm_mem.enable_symm_mem_for_group(dist.group.WORLD.group_name)

            tokens = 64
            dim = 128
            input_tensor = torch.randn(
                tokens,
                dim,
                device=self.device,
                dtype=torch.bfloat16,
                requires_grad=False,
            )
            num_local_experts = 2
            num_tokens_per_expert = generate_split_sizes(
                self.world_size * num_local_experts, tokens, self.device
            )
            expert_splits_per_rank = num_tokens_per_expert.view(
                self.world_size, num_local_experts
            )

            with torch.no_grad():
                num_tokens_per_expert_group = all_to_all_single(
                    num_tokens_per_expert,
                    None,
                    None,
                    group=dist.group.WORLD,
                )
                num_tokens_per_expert_group = torch.ops._c10d_functional.wait_tensor(
                    num_tokens_per_expert_group
                )
                input_splits = expert_splits_per_rank.sum(dim=1)
                output_splits = num_tokens_per_expert_group.view(
                    self.world_size, num_local_experts
                ).sum(dim=1)

            bf16_dispatched = all_to_all_single(
                input_tensor,
                output_splits.tolist(),
                input_splits.tolist(),
                group=dist.group.WORLD,
            )
            bf16_dispatched = torch.ops._c10d_functional.wait_tensor(bf16_dispatched)
            (
                bf16_input_shape,
                bf16_permuted,
                bf16_permuted_indices,
                bf16_num_tokens_per_expert_padded,
                bf16_group_offsets,
            ) = _permute_bf16(
                bf16_dispatched,
                num_tokens_per_expert_group,
                self.world_size,
                num_local_experts,
                32,
            )
            active_rows = int(bf16_group_offsets[-1].item())
            bf16_permuted = bf16_permuted[:active_rows]
            bf16_permuted_indices = bf16_permuted_indices[:active_rows]

            (
                padded_input_shape,
                mx_grouped,
                permuted_indices,
                num_tokens_per_expert_padded,
                group_offsets,
            ) = a2a_dispatch_and_group_mxfp8_fwd_hp_bwd(
                input_tensor,
                output_splits.tolist(),
                input_splits.tolist(),
                num_tokens_per_expert,
                group_name=dist.group.WORLD.group_name,
            )
            assert isinstance(mx_grouped, MXTensor)
            assert padded_input_shape == bf16_input_shape
            assert torch.equal(permuted_indices, bf16_permuted_indices)
            assert torch.equal(
                num_tokens_per_expert_padded, bf16_num_tokens_per_expert_padded
            )
            assert torch.equal(group_offsets, bf16_group_offsets)

            grouped_sqnr = compute_error(mx_grouped.dequantize(), bf16_permuted).item()
            assert grouped_sqnr > 30.0, f"Grouped SQNR too low: {grouped_sqnr} dB"
        finally:
            dist.destroy_process_group()


if __name__ == "__main__":
    run_tests()
