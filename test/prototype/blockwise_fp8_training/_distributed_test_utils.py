# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD 3-Clause license found in the
# LICENSE file in the root directory of this source tree.

import copy

import torch

from torchao.prototype.blockwise_fp8_training.linear import (
    Float8BlockwiseLinear,
    Float8BlockwiseLinearConfig,
)
from torchao.quantization import quantize_
from torchao.testing.training import fp8_blockwise_distributed_utils as fp8_dist_utils
from torchao.testing.training.dtensor_utils import ToyModel

allreduce_reference_grads = fp8_dist_utils.allreduce_reference_grads
assert_close = fp8_dist_utils.assert_close
assert_dtensor_parameter_grads_match = (
    fp8_dist_utils.assert_dtensor_parameter_grads_match
)
assert_dtensor_parameter_values_match = (
    fp8_dist_utils.assert_dtensor_parameter_values_match
)
assert_parameters_are_dtensors = fp8_dist_utils.assert_parameters_are_dtensors
broadcast_module = fp8_dist_utils.broadcast_module
full_tensor = fp8_dist_utils.full_tensor


def get_blockwise_linear_skip_reason(
    *,
    triton_module,
    min_cuda_devices: int,
) -> str | None:
    """Shared module-level gating for Float8BlockwiseLinear distributed tests.

    This is intentionally separate from the lower-level kernel test gating because
    the module swap currently requires SM90+ and the newer scaled_mm/Triton path.
    """
    return fp8_dist_utils.get_blockwise_fp8_distributed_skip_reason(
        triton_module=triton_module,
        min_cuda_devices=min_cuda_devices,
        sm90_reason="Float8BlockwiseLinear currently requires CUDA SM90+",
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
    return fp8_dist_utils.get_replicated_local_batch(
        replica_count=replica_count,
        replica_index=replica_index,
        iter_idx=iter_idx,
        sample_shape=(1, size, size),
        device=device,
    )
