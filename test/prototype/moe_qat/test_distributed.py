"""Distributed tests for MoE QAT parallelism. Run with: torchrun --nproc_per_node=4 -m pytest test/prototype/moe_qat/test_distributed.py"""

import copy

import pytest
import torch
from torch import nn
from torch.distributed._tensor import DTensor
from torch.distributed.device_mesh import DeviceMesh
from torch.distributed.fsdp import MixedPrecisionPolicy
from torch.distributed.tensor import Partial, Replicate, Shard
from torch.nn import functional as F

from torchao.float8.float8_utils import compute_error
from torchao.prototype.moe_qat import MoEQATConfig
from torchao.prototype.moe_qat.wrapper_tensor import FakeQuantizedWeightWrapperBaseTensor, Float8FakeQuantizedWeightWrapperTensor
from torchao.quantization.qat.fake_quantize_config import Float8FakeQuantizeConfig
from torchao.quantization.granularity import PerRow
from torchao.quantization.quant_api import quantize_

from .reference_moe import MoE, MoEArgs
from .testing_utils import (
    _expert_weight_filter,
    _moe_input,
    apply_moe_ep_tp,
    consolidate_tensor_to_cpu,
    create_moe_model,
    distributed_env,
    ParallelStrategy,
    target_devices,
    apply_parallel_strategy,
)


@pytest.mark.parametrize("use_grouped_mm", [True, False])
@pytest.mark.parametrize("parallel_strategy", [
    ParallelStrategy.FSDP,
    ParallelStrategy.EXPERT_PARALLEL,
    ParallelStrategy.TENSOR_PARALLEL,
    ParallelStrategy.EXPERT_TENSOR_PARALLEL,
    ParallelStrategy.FSDP_TP,
])
@pytest.mark.parametrize("wrapper_cls,weight_config,activation_config, min_sqnr", [
    (Float8FakeQuantizedWeightWrapperTensor, Float8FakeQuantizeConfig(), None,                       {"out": 30, "input_grad": 31, "param_grad": 22}),
    (Float8FakeQuantizedWeightWrapperTensor, Float8FakeQuantizeConfig(), Float8FakeQuantizeConfig(), {"out": 28, "input_grad": 25, "param_grad": 16}),
])
def test_moe_qat_parallel(parallel_strategy, wrapper_cls, weight_config, activation_config, min_sqnr, use_grouped_mm, distributed_env):

    device = distributed_env["device_type"]
    base_model = create_moe_model(device, use_grouped_mm=use_grouped_mm)
    base_x = _moe_input(base_model, batch=8, seq=64)

    # Reference model (no fake quantization)
    ref_model = copy.deepcopy(base_model)
    model = copy.deepcopy(base_model)

    # Apply QAT to test model
    qat_config = MoEQATConfig(
        weight_config=weight_config,
        activation_config=activation_config,
        step="prepare",
        params_filter_fn=_expert_weight_filter,
    )
    quantize_(model, qat_config, filter_fn=lambda m, fqn: isinstance(m, MoE))

    # Create optimizers
    model_optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    ref_optimizer = torch.optim.SGD(ref_model.parameters(), lr=0.01)

    # Verify starting params are identical
    for ref_param, model_param in zip(ref_model.parameters(), model.parameters()):
        if isinstance(model_param.data, FakeQuantizedWeightWrapperBaseTensor):
            assert isinstance(model_param.data, wrapper_cls)
            assert torch.equal(ref_param, model_param.data.to_tensor())
        else:
            assert torch.equal(ref_param, model_param)

    model, ref_model = apply_parallel_strategy(
        model, ref_model, parallel_strategy, False, distributed_env
    )

    # Rough validation that parallelization was applied properly
    if parallel_strategy != ParallelStrategy.FSDP:
        for name in ("gate_proj", "down_proj", "up_proj"):
            assert isinstance(getattr(model.experts, name).data, DTensor), (
                f"model.experts.{name} is not a DTensor"
            )
            assert isinstance(getattr(ref_model.experts, name).data, DTensor), (
                f"ref_model.experts.{name} is not a DTensor"
            )

    # Inputs
    ref_x = base_x.clone().detach().requires_grad_(True)
    x = ref_x.clone().detach().requires_grad_(True)

    # Forward
    ref_out = ref_model(ref_x)
    out = model(x)

    # Compute loss
    labels = torch.ones_like(ref_out)
    ref_loss = F.mse_loss(ref_out, labels)
    out_loss = F.mse_loss(out, labels)

    # Backward
    ref_loss.backward()
    out_loss.backward()

    # Optimizer step
    model_optimizer.step()
    ref_optimizer.step()

    # --- All collectives (all ranks must participate) ---
    gathered_out = consolidate_tensor_to_cpu(out)
    gathered_ref_out = consolidate_tensor_to_cpu(ref_out)

    gathered_x_grad = consolidate_tensor_to_cpu(x.grad)
    gathered_ref_x_grad = consolidate_tensor_to_cpu(ref_x.grad)

    param_results = []
    for (name, param), (ref_name, ref_param) in zip(
        model.named_parameters(), ref_model.named_parameters()
    ):
        gathered = consolidate_tensor_to_cpu(param.grad)
        ref_gathered = consolidate_tensor_to_cpu(ref_param.grad)
        if torch.distributed.get_rank() == 0:
            param_results.append((name, ref_name, gathered, ref_gathered))

    # --- Barrier: sync before any rank-0 assertion ---
    torch.distributed.barrier()

    # --- Rank-0 assertions ---
    if torch.distributed.get_rank() == 0:
        assert torch.isfinite(gathered_out).all(), "Consolidated output has non-finite values"
        assert torch.isfinite(gathered_ref_out).all(), "Consolidated ref output has non-finite values"
        out_sqnr = compute_error(gathered_out, gathered_ref_out)
        assert out_sqnr.item() >= min_sqnr["out"], f"Output SQNR must be >= {min_sqnr['out']} dB, got {out_sqnr.item():.1f} dB"

        assert torch.isfinite(gathered_x_grad).all(), "Consolidated input grad has non-finite values"
        assert torch.isfinite(gathered_ref_x_grad).all(), "Consolidated ref input grad has non-finite values"
        input_grad_sqnr = compute_error(gathered_x_grad, gathered_ref_x_grad)
        assert input_grad_sqnr.item() >= min_sqnr["input_grad"], f"Input grad SQNR must be >= {min_sqnr['input_grad']} dB, got {input_grad_sqnr.item():.1f} dB"

        for name, ref_name, gathered, ref_gathered in param_results:
            assert gathered is not None
            assert ref_gathered is not None
            assert torch.isfinite(gathered).all(), f"Consolidated {name} grad has non-finite values"
            assert torch.isfinite(ref_gathered).all(), f"Consolidated {ref_name} grad has non-finite values"
            sqnr = compute_error(gathered, ref_gathered)
            assert sqnr.item() >= min_sqnr["param_grad"], (
                f"{name} grad SQNR must be >= {min_sqnr['param_grad']} dB, got {sqnr.item():.1f} dB"
            )




