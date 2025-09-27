import pytest
import torch

if not torch.cuda.is_available() or torch.cuda.get_device_capability() != (10, 0):
    pytest.skip("Test requires CUDA build on SM100", allow_module_level=True)

import torch.distributed as dist
import torch.distributed._symmetric_memory as symm_mem
from torch.distributed._functional_collectives import (
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
    mxfp8_on_device_all_to_all_v,
)

from ..testing_utils import generate_split_sizes


@instantiate_parametrized_tests
class MXFP8AllToAllVTest(MultiProcessTestCase):
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
                dtype=torch.float32,
                requires_grad=True,
            )
            ref_input_tensor = input_tensor.detach().clone().requires_grad_(True)

            # Generate random input splits that sum to tokens_per_ep_rank
            num_splits = self.world_size
            input_splits = generate_split_sizes(
                num_splits, tokens_per_ep_rank, self.device
            )

            # Max output tokens per rank is worst case where one rank receives all tokens
            max_output_tokens_per_rank = tokens_per_ep_rank * self.world_size

            # Test forward
            output, output_splits = mxfp8_on_device_all_to_all_v(
                input_tensor,
                input_splits,
                max_output_tokens_per_rank,
                group_name,
            )

            # Reference torch.all_to_all_single to compare against
            output_splits_ref = torch.empty_like(output_splits)

            # Compute output splits from input splits
            dist.all_to_all_single(output_splits_ref, input_splits)

            # Pre-allocate output buffer for reference a2a
            total_tokens_on_rank_after_a2a = output_splits_ref.sum()
            ref_output = torch.empty(
                total_tokens_on_rank_after_a2a,
                dim,
                device=self.device,
                dtype=torch.float32,
            )

            # Do the actual all_to_all_single
            ref_output = all_to_all_single_autograd(
                ref_input_tensor,
                output_splits_ref.tolist(),
                input_splits.tolist(),
                dist.group.WORLD,
            )

            # Compare output
            assert torch.equal(output_splits, output_splits_ref), (
                "output_splits mismatch"
            )
            out_no_padding = output[:total_tokens_on_rank_after_a2a]
            sqnr = compute_error(ref_output, out_no_padding)
            min_sqnr = 30.0
            assert sqnr > min_sqnr, f"sqnr={sqnr} is less than min_sqnr={min_sqnr}"

            # Test backwards
            labels = torch.ones_like(out_no_padding)
            loss = F.mse_loss(out_no_padding, labels)
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


if __name__ == "__main__":
    run_tests()
