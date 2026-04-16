# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD 3-Clause license found in the
# LICENSE file in the root directory of this source tree.

import copy

import pytest
import torch
import torch.distributed as dist
from packaging import version
from torch.distributed._composable.fsdp import fully_shard
from torch.distributed._tensor import DTensor
from torch.nn import functional as F
from torch.testing._internal.common_distributed import skip_if_lt_x_gpu
from torch.testing._internal.common_fsdp import FSDPTest
from torch.testing._internal.common_utils import run_tests

triton = pytest.importorskip("triton", reason="Triton required to run this test")

from torchao.prototype.blockwise_fp8_training.linear import (
    Float8BlockwiseLinear,
    Float8BlockwiseLinearConfig,
)
from torchao.quantization import quantize_
from torchao.testing.training.dtensor_utils import ToyModel
from torchao.utils import is_sm_at_least_90

if not torch.cuda.is_available():
    pytest.skip("CUDA not available", allow_module_level=True)

if not is_sm_at_least_90():
    pytest.skip(
        "Float8BlockwiseLinear currently requires CUDA SM90+",
        allow_module_level=True,
    )

if version.parse(triton.__version__) < version.parse("3.3.0"):
    pytest.skip("Triton version < 3.3.0", allow_module_level=True)


class TestBlockwiseFP8FSDP2(FSDPTest):
    @property
    def world_size(self) -> int:
        return 2

    def broadcast_module(self, module: torch.nn.Module) -> None:
        for param in module.parameters():
            dist.broadcast(param, src=0)

    def init_model(self, size: int = 128) -> torch.nn.Module:
        torch.manual_seed(42)
        model = ToyModel(size).cuda().to(torch.bfloat16)
        self.broadcast_module(model)
        return model

    def get_local_batch(
        self, iter_idx: int, size: int = 128
    ) -> tuple[torch.Tensor, torch.Tensor]:
        torch.manual_seed(100 + iter_idx)
        global_input = torch.randn(
            self.world_size,
            1,
            size,
            size,
            device="cuda",
            dtype=torch.bfloat16,
        )
        global_target = torch.randn_like(global_input)
        dist.broadcast(global_input, src=0)
        dist.broadcast(global_target, src=0)
        return global_input[self.rank].contiguous(), global_target[
            self.rank
        ].contiguous()

    def set_use_triton(self, model: torch.nn.Module, use_triton: bool) -> None:
        converted = 0
        for module in model.modules():
            if isinstance(module, Float8BlockwiseLinear):
                module.use_triton = use_triton
                converted += 1
        self.assertGreater(converted, 0)

    def full_tensor(self, tensor: torch.Tensor) -> torch.Tensor:
        return tensor.full_tensor() if isinstance(tensor, DTensor) else tensor

    def assert_close(
        self,
        actual: torch.Tensor,
        expected: torch.Tensor,
        *,
        atol: float = 2e-2,
        rtol: float = 2e-2,
    ) -> None:
        torch.testing.assert_close(
            actual.float(),
            expected.float(),
            atol=atol,
            rtol=rtol,
        )

    @skip_if_lt_x_gpu(2)
    def test_fsdp2_parity(self):
        for use_triton in (False, True):
            with self.subTest(use_triton=use_triton):
                ref_model = self.init_model()
                fsdp_model = copy.deepcopy(ref_model)

                quantize_(ref_model, Float8BlockwiseLinearConfig())
                quantize_(fsdp_model, Float8BlockwiseLinearConfig())
                self.set_use_triton(ref_model, use_triton)
                self.set_use_triton(fsdp_model, use_triton)

                fully_shard(fsdp_model.ffn.w1)
                fully_shard(fsdp_model.ffn.w2)
                fully_shard(fsdp_model.ffn.out_proj)
                fully_shard(fsdp_model)

                for param in fsdp_model.parameters():
                    self.assertIsInstance(param, DTensor)

                ref_optim = torch.optim.SGD(ref_model.parameters(), lr=1e-2)
                fsdp_optim = torch.optim.SGD(fsdp_model.parameters(), lr=1e-2)

                for iter_idx in range(2):
                    local_input, local_target = self.get_local_batch(iter_idx)

                    ref_optim.zero_grad(set_to_none=True)
                    fsdp_optim.zero_grad(set_to_none=True)

                    ref_out = ref_model(local_input)
                    fsdp_out = fsdp_model(local_input)
                    self.assert_close(fsdp_out, ref_out)

                    ref_loss = F.mse_loss(ref_out, local_target)
                    fsdp_loss = F.mse_loss(fsdp_out, local_target)
                    self.assert_close(fsdp_loss, ref_loss, atol=1e-3, rtol=1e-3)

                    ref_loss.backward()
                    fsdp_loss.backward()

                    for ref_param in ref_model.parameters():
                        self.assertIsNotNone(ref_param.grad)
                        dist.all_reduce(ref_param.grad)
                        ref_param.grad.div_(self.world_size)

                    for ref_param, fsdp_param in zip(
                        ref_model.parameters(), fsdp_model.parameters(), strict=True
                    ):
                        self.assertIsNotNone(fsdp_param.grad)
                        self.assertIsInstance(fsdp_param, DTensor)
                        self.assertIsInstance(fsdp_param.grad, DTensor)
                        self.assert_close(
                            self.full_tensor(fsdp_param.grad),
                            ref_param.grad,
                        )

                    ref_optim.step()
                    fsdp_optim.step()

                    for ref_param, fsdp_param in zip(
                        ref_model.parameters(), fsdp_model.parameters(), strict=True
                    ):
                        self.assert_close(self.full_tensor(fsdp_param), ref_param)


if __name__ == "__main__":
    run_tests()
