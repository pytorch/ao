# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD 3-Clause license found in the
# LICENSE file in the root directory of this source tree.

import copy
from collections.abc import Iterable

import torch
import torch.distributed as dist
from torch import nn
from torch.distributed._tensor import DTensor
from torch.distributed.device_mesh import DeviceMesh
from torch.distributed.tensor import Shard
from torch.nn import functional as F

from packaging import version
from torchao.prototype.blockwise_fp8_training.dtensor_utils import (
    replicate_like_dtensor,
)
from torchao.prototype.moe_training.config import (
    Float8TrainingOpConfig,
    Float8TrainingRecipe,
)
from torchao.prototype.moe_training.tensor import TrainingWeightWrapperBaseTensor
from torchao.quantization import quantize_
from torchao.quantization.quantize_.common import KernelPreference
from torchao.quantization.utils import compute_error
from torchao.utils import is_sm_at_least_90


def get_blockwise_moe_skip_reason(
    *,
    triton_module,
    min_cuda_devices: int,
) -> str | None:
    if not torch.cuda.is_available():
        return "CUDA not available"
    if torch.cuda.device_count() < min_cuda_devices:
        return f"Need at least {min_cuda_devices} CUDA devices"
    if not is_sm_at_least_90():
        return "Blockwise FP8 MoE grouped GEMM requires CUDA SM90+"
    if version.parse(triton_module.__version__) < version.parse("3.3.0"):
        return "Triton version < 3.3.0"
    return None


def full_tensor(tensor: torch.Tensor) -> torch.Tensor:
    if isinstance(tensor, DTensor):
        tensor = tensor.full_tensor()
    if isinstance(tensor, TrainingWeightWrapperBaseTensor):
        tensor = tensor._data
    return tensor


def assert_close(
    actual: torch.Tensor,
    expected: torch.Tensor,
    *,
    atol: float = 2e-2,
    rtol: float = 2e-2,
    min_sqnr: float | None = None,
) -> None:
    actual = full_tensor(actual).float()
    expected = full_tensor(expected).float()
    if min_sqnr is not None:
        sqnr = compute_error(expected, actual)
        assert sqnr >= min_sqnr, f"SQNR {sqnr.item()} must be >= {min_sqnr}"
        return
    torch.testing.assert_close(actual, expected, atol=atol, rtol=rtol)


def broadcast_module(module: nn.Module) -> None:
    for param in module.parameters():
        dist.broadcast(full_tensor(param), src=0)


class BlockwiseGroupedExperts(nn.Module):
    def __init__(
        self,
        experts: int = 2,
        in_features: int = 256,
        hidden_features: int = 256,
        *,
        device: torch.device | str = "cuda",
    ):
        super().__init__()
        in_scale = in_features**-0.5
        hidden_scale = hidden_features**-0.5
        self.w1 = nn.Parameter(
            torch.randn(
                experts,
                hidden_features,
                in_features,
                dtype=torch.bfloat16,
                device=device,
            )
            * in_scale
        )
        self.w2 = nn.Parameter(
            torch.randn(
                experts,
                in_features,
                hidden_features,
                dtype=torch.bfloat16,
                device=device,
            )
            * hidden_scale
        )
        self.w3 = nn.Parameter(
            torch.randn(
                experts,
                hidden_features,
                in_features,
                dtype=torch.bfloat16,
                device=device,
            )
            * in_scale
        )

    def forward(self, x: torch.Tensor, group_offsets: torch.Tensor) -> torch.Tensor:
        w1_t = self.w1.transpose(-2, -1)
        w2_t = self.w2.transpose(-2, -1)
        w3_t = self.w3.transpose(-2, -1)
        x = replicate_like_dtensor(x, w1_t, w2_t, w3_t)
        group_offsets = replicate_like_dtensor(group_offsets, x, w1_t, w2_t, w3_t)

        h = F.silu(torch._grouped_mm(x, w1_t, offs=group_offsets))
        h = h * torch._grouped_mm(x, w3_t, offs=group_offsets)
        return torch._grouped_mm(h, w2_t, offs=group_offsets)


def _is_blockwise_grouped_experts_module(module: nn.Module, cur_fqn: str) -> bool:
    return isinstance(module, BlockwiseGroupedExperts)


def make_blockwise_grouped_experts_pair(
    *,
    experts: int = 2,
    in_features: int = 256,
    hidden_features: int = 256,
    seed: int = 42,
    device: torch.device | str = "cuda",
    kernel_preference: KernelPreference = KernelPreference.EMULATED,
    pad_token_groups_for_grouped_mm: bool = True,
    broadcast_weights: bool = False,
    quantize_ref_model: bool = True,
) -> tuple[nn.Module, nn.Module]:
    torch.manual_seed(seed)
    ref_model = BlockwiseGroupedExperts(
        experts=experts,
        in_features=in_features,
        hidden_features=hidden_features,
        device=device,
    )
    if broadcast_weights:
        broadcast_module(ref_model)
    dist_model = copy.deepcopy(ref_model)

    config = Float8TrainingOpConfig.from_recipe(Float8TrainingRecipe.FP8_BLOCKWISE)
    config.kernel_preference = kernel_preference
    config.pad_token_groups_for_grouped_mm = pad_token_groups_for_grouped_mm
    models_to_quantize = (ref_model, dist_model)
    if not quantize_ref_model:
        models_to_quantize = (dist_model,)
    for model in models_to_quantize:
        quantize_(
            model,
            config,
            filter_fn=_is_blockwise_grouped_experts_module,
        )
    return ref_model, dist_model


def parallelize_blockwise_grouped_experts_tensor_parallel(
    module: nn.Module,
    device_mesh: DeviceMesh,
) -> nn.Module:
    from torch.distributed.tensor import distribute_module, distribute_tensor
    from torch.distributed.tensor.parallel import ParallelStyle, parallelize_module

    class BlockwiseGroupedExpertsTensorParallel(ParallelStyle):
        def _partition_fn(
            self,
            name: str,
            module: nn.Module,
            device_mesh: DeviceMesh,
        ) -> None:
            module.register_parameter(
                "w1",
                nn.Parameter(distribute_tensor(module.w1, device_mesh, [Shard(1)])),
            )
            module.register_parameter(
                "w2",
                nn.Parameter(distribute_tensor(module.w2, device_mesh, [Shard(2)])),
            )
            module.register_parameter(
                "w3",
                nn.Parameter(distribute_tensor(module.w3, device_mesh, [Shard(1)])),
            )

        def _apply(self, module: nn.Module, device_mesh: DeviceMesh) -> nn.Module:
            return distribute_module(
                module,
                device_mesh,
                partition_fn=self._partition_fn,
            )

    return parallelize_module(
        module, device_mesh, BlockwiseGroupedExpertsTensorParallel()
    )


def assert_blockwise_grouped_experts_tp_applied(module: nn.Module) -> None:
    assert isinstance(module.w1, DTensor)
    assert isinstance(module.w2, DTensor)
    assert isinstance(module.w3, DTensor)
    assert module.w1.placements == (Shard(1),)
    assert module.w2.placements == (Shard(2),)
    assert module.w3.placements == (Shard(1),)


def get_replicated_local_batch(
    *,
    replica_count: int,
    replica_index: int,
    iter_idx: int,
    tokens: int = 256,
    in_features: int = 256,
    device: str | torch.device = "cuda",
) -> tuple[torch.Tensor, torch.Tensor]:
    torch.manual_seed(100 + iter_idx)
    global_input = torch.randn(
        replica_count,
        tokens,
        in_features,
        device=device,
        dtype=torch.bfloat16,
    )
    global_target = torch.randn_like(global_input)
    dist.broadcast(global_input, src=0)
    dist.broadcast(global_target, src=0)
    return (
        global_input[replica_index].contiguous(),
        global_target[replica_index].contiguous(),
    )


def allreduce_reference_grads(
    model: nn.Module,
    *,
    world_size: int,
    group=None,
) -> None:
    for param in model.parameters():
        assert param.grad is not None
        dist.all_reduce(full_tensor(param.grad), group=group)
        full_tensor(param.grad).div_(world_size)


def assert_parameters_are_dtensors(parameters: Iterable[torch.Tensor]) -> None:
    for param in parameters:
        assert isinstance(param, DTensor)


def assert_parameters_are_training_wrappers(
    parameters: Iterable[torch.Tensor],
) -> None:
    for param in parameters:
        assert isinstance(param, TrainingWeightWrapperBaseTensor)


def assert_dtensor_parameter_grads_match(
    ref_parameters: Iterable[nn.Parameter],
    dist_parameters: Iterable[nn.Parameter],
    *,
    min_sqnr: float = 30.0,
) -> None:
    for ref_param, dist_param in zip(ref_parameters, dist_parameters, strict=True):
        assert ref_param.grad is not None
        assert dist_param.grad is not None
        assert isinstance(dist_param, DTensor)
        assert isinstance(dist_param.grad, DTensor)
        assert_close(dist_param.grad, ref_param.grad, min_sqnr=min_sqnr)


def assert_dtensor_parameter_values_match(
    ref_parameters: Iterable[nn.Parameter],
    dist_parameters: Iterable[nn.Parameter],
    *,
    min_sqnr: float = 30.0,
) -> None:
    for ref_param, dist_param in zip(ref_parameters, dist_parameters, strict=True):
        assert isinstance(dist_param, DTensor)
        assert_close(dist_param, ref_param, min_sqnr=min_sqnr)
