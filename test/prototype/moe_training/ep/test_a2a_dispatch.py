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
from torch.distributed._functional_collectives import (
    all_to_all_single,
)
from torch.testing._internal.common_distributed import MultiProcessTestCase
from torch.testing._internal.common_utils import run_tests

from test.prototype.moe_training.testing_utils import generate_split_sizes
from torchao.prototype.mx_formats.expert_parallel import a2a_dispatch_mxfp8_fwd_hp_bwd
from torchao.prototype.mx_formats.mx_tensor import MXTensor
from torchao.quantization.utils import compute_error
from torchao.utils import is_sm_at_least_100


class TestA2ADispatch(MultiProcessTestCase):
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

    def test(self):
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
                group=dist.group.WORLD,
            )
            assert isinstance(mx_output, MXTensor)

            ref_output = all_to_all_single(
                input_tensor,
                output_splits.tolist(),
                input_splits.tolist(),
                group=dist.group.WORLD,
            )

            # Compare outputs
            output = mx_output.dequantize()
            sqnr = compute_error(output, ref_output)
            assert sqnr > 30.0, f"SQNR too low: {sqnr} dB"

            # Note: tests for backwards will be in an integration test
        finally:
            dist.destroy_process_group()


if __name__ == "__main__":
    run_tests()
