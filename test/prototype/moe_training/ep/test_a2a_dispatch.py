import pytest
import torch

from torchao.utils import is_cuda_version_at_least, is_sm_at_least_100, is_XPU

_is_xpu = is_XPU()
_is_compatible_cuda = (
    torch.cuda.is_available()
    and is_sm_at_least_100()
    and is_cuda_version_at_least(12, 8)
)

if not (_is_xpu or _is_compatible_cuda):
    pytest.skip(
        "Test requires XPU or CUDA 12.8+ with SM >= 100", allow_module_level=True
    )


import torch.distributed as dist
from torch.distributed._functional_collectives import (
    all_to_all_single,
)
from torch.testing._internal.common_distributed import MultiProcessTestCase
from torch.testing._internal.common_utils import (
    instantiate_parametrized_tests,
    parametrize,
    run_tests,
)

from test.prototype.moe_training.testing_utils import generate_split_sizes
from torchao.prototype.moe_training.ep import a2a_dispatch_mxfp8_fwd_hp_bwd
from torchao.prototype.mx_formats.mx_tensor import MXTensor
from torchao.quantization.utils import compute_error
from torchao.utils import get_available_devices

_DEVICES = get_available_devices()[1:]


@instantiate_parametrized_tests
class TestA2ADispatch(MultiProcessTestCase):
    def __init__(self, methodName="runTest", device_type="cuda"):
        super().__init__(methodName)
        self._device_type = device_type

    def setUp(self):
        super().setUp()
        self._spawn_processes()

    @property
    def world_size(self):
        return 2

    @property
    def device(self):
        return torch.device(self._device_type, self.rank)

    @device.setter
    def device(self, device_type):
        self._device_type = device_type

    def set_device(self):
        if self._device_type == "cuda":
            torch.cuda.set_device(self.device)
        elif self._device_type == "xpu":
            torch.xpu.set_device(self.device)
        else:
            raise ValueError(f"Unsupported device type: {self._device_type}")

    @property
    def backend(self):
        if self._device_type == "cuda":
            return "nccl"
        elif self._device_type == "xpu":
            return "xccl"
        else:
            raise ValueError(f"Unsupported device type: {self._device_type}")

    def _init_process(self):
        self.set_device()
        store = dist.FileStore(self.file_name, self.world_size)
        dist.init_process_group(
            backend=self.backend,
            world_size=self.world_size,
            rank=self.rank,
            store=store,
        )
        torch.manual_seed(42 + self.rank)

    @parametrize("device_type", _DEVICES)
    def test(self, device_type: str):
        self.device = device_type
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

            # Compare outputs
            output = mx_output.dequantize()
            sqnr = compute_error(output, ref_output)
            assert sqnr > 30.0, f"SQNR too low: {sqnr} dB"

            # Note: tests for backwards will be in an integration test
        finally:
            dist.destroy_process_group()


if __name__ == "__main__":
    run_tests()
