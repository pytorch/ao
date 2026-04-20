# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD 3-Clause license found in the
# LICENSE file in the root directory of this source tree.

import copy
from collections.abc import Iterable

import torch
import torch.distributed as dist
from torch.distributed._tensor import DTensor

from packaging import version
from torchao.prototype.blockwise_fp8_training.linear import (
    Float8BlockwiseLinear,
    Float8BlockwiseLinearConfig,
)
from torchao.quantization import quantize_
from torchao.testing.training.dtensor_utils import ToyModel
from torchao.utils import is_sm_at_least_90


def get_blockwise_linear_skip_reason(
    *,
    triton_module,
    min_cuda_devices: int,
) -> str | None:
    """Shared module-level gating for Float8BlockwiseLinear distributed tests.

    This is intentionally separate from the lower-level kernel test gating because
    the module swap currently requires SM90+ and the newer scaled_mm/Triton path.
    """
    if not torch.cuda.is_available():
        return "CUDA not available"
    if torch.cuda.device_count() < min_cuda_devices:
        return f"Need at least {min_cuda_devices} CUDA devices"
    if not is_sm_at_least_90():
        return "Float8BlockwiseLinear currently requires CUDA SM90+"
    if version.parse(triton_module.__version__) < version.parse("3.3.0"):
        return "Triton version < 3.3.0"
    return None


def full_tensor(tensor: torch.Tensor) -> torch.Tensor:
    """Materialize a DTensor for parity checks, otherwise return the tensor as-is."""
    return tensor.full_tensor() if isinstance(tensor, DTensor) else tensor


def assert_close(
    actual: torch.Tensor,
    expected: torch.Tensor,
    *,
    atol: float = 2e-2,
    rtol: float = 2e-2,
) -> None:
    """Compare eager tensors and DTensors using a common float32 tolerance path."""
    torch.testing.assert_close(
        full_tensor(actual).float(),
        full_tensor(expected).float(),
        atol=atol,
        rtol=rtol,
    )


def set_blockwise_linear_use_triton(
    model: torch.nn.Module,
    use_triton: bool,
) -> None:
    converted = 0
    for module in model.modules():
        if isinstance(module, Float8BlockwiseLinear):
            module.use_triton = use_triton
            converted += 1
    if converted == 0:
        raise AssertionError("Expected at least one Float8BlockwiseLinear module")


def broadcast_module(module: torch.nn.Module) -> None:
    for param in module.parameters():
        dist.broadcast(param, src=0)


def init_toy_model(
    *,
    size: int = 128,
    seed: int = 42,
    device: str | torch.device = "cuda",
    broadcast_weights: bool = False,
) -> torch.nn.Module:
    torch.manual_seed(seed)
    model = ToyModel(size).to(device=device, dtype=torch.bfloat16)
    if broadcast_weights:
        broadcast_module(model)
    return model


def make_quantized_toy_model_pair(
    *,
    size: int = 128,
    seed: int = 42,
    device: str | torch.device = "cuda",
    use_triton: bool,
    broadcast_weights: bool = False,
) -> tuple[torch.nn.Module, torch.nn.Module]:
    ref_model = init_toy_model(
        size=size,
        seed=seed,
        device=device,
        broadcast_weights=broadcast_weights,
    )
    dist_model = copy.deepcopy(ref_model)
    for model in (ref_model, dist_model):
        quantize_(model, Float8BlockwiseLinearConfig())
        set_blockwise_linear_use_triton(model, use_triton)
    return ref_model, dist_model


def get_replicated_local_batch(
    *,
    replica_count: int,
    replica_index: int,
    iter_idx: int,
    size: int = 128,
    device: str | torch.device = "cuda",
) -> tuple[torch.Tensor, torch.Tensor]:
    """Build one global batch and hand each replica its deterministic local slice.

    TP peers should see the same sample, while different data-parallel replicas
    should see different samples. Broadcasting from rank 0 keeps the reference
    and distributed models aligned across all ranks.
    """
    torch.manual_seed(100 + iter_idx)
    global_input = torch.randn(
        replica_count,
        1,
        size,
        size,
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


def assert_parameters_are_dtensors(parameters: Iterable[torch.Tensor]) -> None:
    for param in parameters:
        assert isinstance(param, DTensor)


def allreduce_reference_grads(
    model: torch.nn.Module,
    *,
    world_size: int,
    group=None,
) -> None:
    for param in model.parameters():
        assert param.grad is not None
        dist.all_reduce(param.grad, group=group)
        param.grad.div_(world_size)


def assert_dtensor_parameter_grads_match(
    ref_parameters: Iterable[torch.nn.Parameter],
    dist_parameters: Iterable[torch.nn.Parameter],
) -> None:
    for ref_param, dist_param in zip(ref_parameters, dist_parameters, strict=True):
        assert ref_param.grad is not None
        assert dist_param.grad is not None
        assert isinstance(dist_param, DTensor)
        assert isinstance(dist_param.grad, DTensor)
        assert_close(dist_param.grad, ref_param.grad)


def assert_dtensor_parameter_values_match(
    ref_parameters: Iterable[torch.nn.Parameter],
    dist_parameters: Iterable[torch.nn.Parameter],
) -> None:
    for ref_param, dist_param in zip(ref_parameters, dist_parameters, strict=True):
        assert isinstance(dist_param, DTensor)
        assert_close(dist_param, ref_param)
