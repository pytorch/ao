# Owner(s): ["oncall: distributed"]
import contextlib
import copy
import functools
import unittest
from typing import Iterable, List, Tuple, Type, Union
import torch
import torch.distributed as dist
import torch.nn as nn
from torch.distributed._composable.fsdp import (
    fully_shard,
    OffloadPolicy,
)
from torch.testing._internal.common_cuda import TEST_CUDA
from torch.testing._internal.common_distributed import skip_if_lt_x_gpu
from torch.testing._internal.common_fsdp import (
    FSDPTest,
    FSDPTestMultiThread,
    MLP,
)
from torch.testing._internal.common_utils import (
    run_tests,
)
class TestFullyShard1DTrainingCore(FSDPTestMultiThread):
    @property
    def world_size(self) -> int:
        return 2
    @skip_if_lt_x_gpu(2)
    def test_train_parity_multi_group(self):
        """
        Tests train parity against DDP when using multiple parameter groups for
        communication (for communication and computation overlap plus memory
        reduction).
        """
        self.run_subtests(
            {
                "offload_policy": [OffloadPolicy()],
            },
            self._test_train_parity_multi_group,
        )
    def _test_train_parity_multi_group(
        self,
        offload_policy: OffloadPolicy,
    ):
        device_type = "cuda"
        torch.manual_seed(42)
        lin_dim = 32
        model = nn.Sequential(*[MLP(lin_dim, torch.device("cpu")) for _ in range(3)])
        fully_shard_fn = functools.partial(
            fully_shard,
            offload_policy=offload_policy,
        )
        for mlp in model:
            fully_shard_fn(mlp)
        fully_shard_fn(model)
        optim = torch.optim.Adam(model.parameters(), lr=1e-2)
        torch.manual_seed(42 + self.rank + 1)
        for iter_idx in range(10):
            inp = torch.randn((8, lin_dim), device=torch.device(device_type))
            optim.zero_grad()
            loss = model(inp).sum()
            loss.backward()
            optim.step()
if __name__ == "__main__":
    run_tests()
