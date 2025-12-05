import torch
import torch.distributed as dist
from torch.testing._internal.common_distributed import (
    MultiProcessTestCase,
)
from torch.testing._internal.common_utils import (
    instantiate_parametrized_tests,
)

from torchao.prototype.mx_formats.mx_tensor import MXTensor


@instantiate_parametrized_tests
class MXFP8OnDeviceAllGatherTest(MultiProcessTestCase):
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
        torch.manual_seed(42)

    def test_allgather(self):
        self._init_process()
        try:
            torch.manual_seed(42)
            golden_qdata = (
                torch.randint(0, 256, (256, 512), dtype=torch.uint8)
                .to(torch.float8_e5m2)
                .to(self.device)
            )

            # Random scale factors (typically float32 or uint8 for e8m0)
            golden_scale = (
                torch.randint(0, 256, (256, 16), dtype=torch.uint8)
                .view(torch.float8_e8m0fnu)
                .to(self.device)
            )

            # Create golden MXTensor
            golden_mx = MXTensor(
                golden_qdata,
                golden_scale,
                elem_dtype=torch.float8_e5m2,
                block_size=32,
                orig_dtype=torch.float32,
                kernel_preference=None,
                act_quant_kwargs=None,
                is_swizzled_scales=None,
            )

            world_size = self.world_size
            # Each rank gets its shard (split along dim 0)
            shard_size = golden_qdata.shape[0] // world_size  # 2 rows per rank
            start_idx = self.rank * shard_size
            end_idx = (self.rank + 1) * shard_size

            # Create local MXTensor from shard
            local_mx = MXTensor(
                golden_qdata[start_idx:end_idx].clone().to(self.device),
                golden_scale[start_idx:end_idx].clone().to(self.device),
                elem_dtype=torch.float8_e5m2,
                block_size=32,
                orig_dtype=torch.float32,
                kernel_preference=None,
                act_quant_kwargs=None,
                is_swizzled_scales=None,
            )

            # Perform all_gather
            gathered_mx = torch.ops._c10d_functional.all_gather_into_tensor.default(
                local_mx,
                world_size,
                "0",
            )
            gathered_mx = torch.ops._c10d_functional.wait_tensor.default(gathered_mx)

            # Verify type
            assert isinstance(gathered_mx, MXTensor), (
                f"Expected MXTensor, got {type(gathered_mx)}"
            )

            # Verify shape
            assert gathered_mx.shape == golden_mx.shape, (
                f"Shape mismatch: {gathered_mx.shape} vs {golden_mx.shape}"
            )

            # Verify qdata matches golden exactly
            if not torch.equal(gathered_mx.qdata, golden_qdata):
                assert False, "qdata mismatch"

            # Verify scale matches golden exactly
            if not torch.equal(
                gathered_mx.scale.view(torch.uint8),
                golden_scale.view(torch.uint8),
            ):
                assert False, "scale mismatch"

            assert gathered_mx.block_size == 32

        finally:
            dist.destroy_process_group()
