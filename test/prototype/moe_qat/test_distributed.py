"""Distributed tests for MoE QAT parallelism. Run with: torchrun --nproc_per_node=2 -m pytest test/prototype/moe_qat/test_distributed.py"""

import copy
import os

import pytest
import torch
from torch import nn
from torch.distributed._composable.fsdp import fully_shard
from torch.distributed._tensor import DTensor
from torch.distributed.device_mesh import DeviceMesh, init_device_mesh
from torch.distributed.fsdp import MixedPrecisionPolicy
from torch.distributed.tensor import Partial, Replicate, Shard
from torch.distributed.tensor.parallel import (
    ColwiseParallel,
    PrepareModuleInputOutput,
    RowwiseParallel,
    parallelize_module,
)
from torch.nn import functional as F

from torchao.float8.float8_utils import compute_error
from torchao.prototype.moe_qat import MoEQATConfig
from torchao.prototype.moe_qat.tensor import FakeQuantizedWeightWrapperBaseTensor, Float8FakeQuantizedWeightWrapperTensor
from torchao.quantization.qat.fake_quantize_config import Float8FakeQuantizeConfig
from torchao.quantization.granularity import PerRow
from torchao.quantization.quant_api import quantize_

from .reference_moe import MoE, MoEArgs
from .testing_utils import _expert_weight_filter, create_moe_model, target_devices
from .reference_parallel_styles import ExpertParallel, ExpertTensorParallel, NoParallel, TensorParallel

@pytest.fixture(scope="module", params=target_devices)
def distributed_env(request):
    world_size = int(os.environ["WORLD_SIZE"])

    assert world_size == 4, (
        f"This test requires world_size=4, but got world_size={world_size}. "
        "Run with: torchrun --nproc_per_node=4 -m pytest test/prototype/moe_qat/test_distributed.py"
    )

    torch.manual_seed(42)
    device_type = request.param

    # Create meshes for different parallelization strategies:
    # - tp_mesh: 1D mesh for tensor parallel only
    # - ep_mesh: 1D mesh for expert parallel only
    # - dp_mesh: 1D mesh for FSDP only
    # - ep_tp_mesh: 2D mesh for combined EP and TP
    # - dp_tp_mesh: 2D mesh for combined FSDP and TP
    tp_mesh = init_device_mesh(device_type, (world_size,), mesh_dim_names=("tp",))
    ep_mesh = init_device_mesh(device_type, (world_size,), mesh_dim_names=("ep",))
    dp_mesh = init_device_mesh(device_type, (world_size,), mesh_dim_names=("dp",))
    ep_tp_mesh = init_device_mesh(device_type, (2, 2), mesh_dim_names=("ep", "tp"))
    dp_tp_mesh = init_device_mesh(device_type, (2, 2), mesh_dim_names=("dp", "tp"))

    yield {
        "tp_mesh": tp_mesh,
        "ep_mesh": ep_mesh,
        "dp_mesh": dp_mesh,
        "ep_tp_mesh": ep_tp_mesh,
        "dp_tp_mesh": dp_tp_mesh,
        "device_type": device_type,
    }

    torch.distributed.destroy_process_group()


@pytest.fixture(scope="module")
def fixed_model_and_input(distributed_env):
    """Create a fixed MoE model and input once per device, so all strategies use identical starting conditions."""
    device = torch.device(distributed_env["device_type"])
    model = create_moe_model(device, use_grouped_mm=(device.type == "cuda"))
    x = torch.randn(8, 64, model.experts.gate_proj.shape[-1], device=device)
    return model, x


class ParallelStrategy:
    """Enum-like class for parallelization strategies."""

    TENSOR_PARALLEL = "tensor_parallel"
    EXPERT_PARALLEL = "expert_parallel"
    EXPERT_TENSOR_PARALLEL = "expert_tensor_parallel"
    FSDP = "fsdp"
    FSDP_TP = "fsdp_tp"


@pytest.mark.parametrize("use_compile", [False, True])
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
def test_moe_qat_parallel(parallel_strategy, wrapper_cls, weight_config, activation_config, min_sqnr, use_compile, distributed_env, fixed_model_and_input):

    if use_compile:
        if parallel_strategy == ParallelStrategy.EXPERT_TENSOR_PARALLEL:
            pytest.skip("torch.compile does not support device_mesh._get_submesh")
        if activation_config is not None:
            pytest.skip("Currently activation fake-quant with DTensor is not compatible with torch.compile")

    base_model, base_x = fixed_model_and_input

    tp_mesh = distributed_env["tp_mesh"]
    ep_mesh = distributed_env["ep_mesh"]
    ep_tp_mesh = distributed_env["ep_tp_mesh"]
    dp_mesh = distributed_env["dp_mesh"]
    dp_tp_mesh = distributed_env["dp_tp_mesh"]

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

    # Apply parallelization based on strategy
    # Strategy      Compilation Order           Rationale
    # FSDP          Compile -> FSDP             Compiling after FSDP breaks tracing due to sharded dynamic weights.
    # EP            EP -> Compile               Compiling must trace the dynamic All-to-All routing kernels.
    # TP            TP -> Compile               Compiling must target the modified weight shapes and layouts.
    # EP + TP       EP + TP -> Compile          Both structurally alter the layers; compile the final local graph.
    # FSDP + TP     TP -> Compile -> FSDP       Standard "Sandwich Strategy".
    if parallel_strategy == ParallelStrategy.TENSOR_PARALLEL:
        apply_moe_ep_tp(model, tp_mesh=tp_mesh, ep_mesh=None, ep_tp_mesh=None)
        apply_moe_ep_tp(ref_model, tp_mesh=tp_mesh, ep_mesh=None, ep_tp_mesh=None)
        if use_compile:
            model = torch.compile(model)
            ref_model = torch.compile(ref_model)
    elif parallel_strategy == ParallelStrategy.EXPERT_PARALLEL:
        apply_moe_ep_tp(model, tp_mesh=None, ep_mesh=ep_mesh, ep_tp_mesh=None)
        apply_moe_ep_tp(ref_model, tp_mesh=None, ep_mesh=ep_mesh, ep_tp_mesh=None)
        if use_compile:
            model = torch.compile(model)
            ref_model = torch.compile(ref_model)
    elif parallel_strategy == ParallelStrategy.EXPERT_TENSOR_PARALLEL:
        tp_submesh = ep_tp_mesh["tp"]
        apply_moe_ep_tp(model, tp_mesh=tp_submesh, ep_mesh=ep_mesh, ep_tp_mesh=ep_tp_mesh)
        apply_moe_ep_tp(ref_model, tp_mesh=tp_submesh, ep_mesh=ep_mesh, ep_tp_mesh=ep_tp_mesh)
        if use_compile:
            model = torch.compile(model)
            ref_model = torch.compile(ref_model)
    elif parallel_strategy == ParallelStrategy.FSDP:
        if use_compile:
            model = torch.compile(model)
            ref_model = torch.compile(ref_model)
        
        # FSDP only - use 1D dp mesh
        fsdp_config = {"mesh": dp_mesh}
        fully_shard(model, **fsdp_config)
        fully_shard(ref_model, **fsdp_config)
    elif parallel_strategy == ParallelStrategy.FSDP_TP:
        # FSDP + TP - use 2D mesh with (dp, tp) dimensions
        tp_submesh = dp_tp_mesh["tp"]
        dp_submesh = dp_tp_mesh["dp"]

        # First apply TP
        apply_moe_ep_tp(model, tp_mesh=tp_submesh, ep_mesh=None, ep_tp_mesh=None)
        apply_moe_ep_tp(ref_model, tp_mesh=tp_submesh, ep_mesh=None, ep_tp_mesh=None)

        if use_compile:
            model = torch.compile(model)
            ref_model = torch.compile(ref_model)

        # Then apply FSDP
        fsdp_config = {"mesh": dp_submesh}
        fully_shard(model, **fsdp_config)
        fully_shard(ref_model, **fsdp_config)

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

    # Validate output (consolidated to full CPU tensors)
    gathered_out = consolidate_tensor_to_cpu(out)
    gathered_ref_out = consolidate_tensor_to_cpu(ref_out)
    if torch.distributed.get_rank() == 0:
        assert torch.isfinite(gathered_out).all(), "Consolidated output has non-finite values"
        assert torch.isfinite(gathered_ref_out).all(), "Consolidated ref output has non-finite values"
        out_sqnr = compute_error(gathered_out, gathered_ref_out)
        assert out_sqnr.item() >= min_sqnr["out"], f"Output SQNR must be >= {min_sqnr['out']} dB, got {out_sqnr.item():.1f} dB"

    # Validate input gradient (consolidated)
    gathered_x_grad = consolidate_tensor_to_cpu(x.grad)
    gathered_ref_x_grad = consolidate_tensor_to_cpu(ref_x.grad)
    if torch.distributed.get_rank() == 0:
        assert torch.isfinite(gathered_x_grad).all(), "Consolidated input grad has non-finite values"
        assert torch.isfinite(gathered_ref_x_grad).all(), "Consolidated ref input grad has non-finite values"
        input_grad_sqnr = compute_error(gathered_x_grad, gathered_ref_x_grad)
        assert input_grad_sqnr.item() >= min_sqnr["input_grad"], f"Input grad SQNR must be >= {min_sqnr['input_grad']} dB, got {input_grad_sqnr.item():.1f} dB"

    # Validate param gradients (consolidated)
    for (name, param), (ref_name, ref_param) in zip(
        model.named_parameters(), ref_model.named_parameters()
    ):
        gathered = consolidate_tensor_to_cpu(param.grad)
        ref_gathered = consolidate_tensor_to_cpu(ref_param.grad)
        if torch.distributed.get_rank() == 0:
            assert gathered is not None
            assert ref_gathered is not None
            assert torch.isfinite(gathered).all(), f"Consolidated {name} grad has non-finite values"
            assert torch.isfinite(ref_gathered).all(), f"Consolidated {ref_name} grad has non-finite values"
            sqnr = compute_error(gathered, ref_gathered)
            assert sqnr.item() >= min_sqnr["param_grad"], (
                f"{name} grad SQNR must be >= {min_sqnr['param_grad']} dB, got {sqnr.item():.1f} dB"
            )


def consolidate_tensor_to_cpu(tensor, target_rank=0):
    """Consolidate a potentially sharded tensor to a full CPU tensor on target_rank.

    DTensor: uses full_tensor() to reconstruct the complete tensor.
    Plain tensor: returns as-is on CPU (assumed already full, e.g., FSDP2 forward outputs).
    All ranks must call this with the same tensor argument.
    """
    if not isinstance(tensor, torch.Tensor):
        raise ValueError("`tensor` must be a `torch.Tensor` or its subclasses.")
    
    if isinstance(tensor, DTensor):
        # full_tensor() is a collective — all ranks must call it
        result = tensor.full_tensor().cpu()
    else:
        result = tensor.clone().cpu()
    
    return result if torch.distributed.get_rank() == target_rank else None



# Copied from test/prototype/moe_training/test_distributed.py
def apply_moe_ep_tp(
    model: nn.Module,
    tp_mesh: DeviceMesh | None,
    ep_mesh: DeviceMesh | None,
    ep_tp_mesh: DeviceMesh | None,
):
    # Modified version of moe parallelization from https://github.com/pytorch/torchtitan/pull/1324/
    # that supports single MoE layer independent of a transformer.
    if tp_mesh is not None:
        moe_layer_plan = {
            # input / output sharding on the seqlen dim
            # all-gather for input, reduce-scatter for output
            "": PrepareModuleInputOutput(
                input_layouts=(Shard(1),),
                desired_input_layouts=(Replicate(),),
                use_local_input=True,
                output_layouts=(Partial(),),
                desired_output_layouts=(Shard(1),),
            ),
            # replicate computation for the router
            "router.gate": NoParallel(),
            # shared_experts uses ColwiseParallel/RowwiseParallel for individual weights
            # (unlike GroupedExperts which uses TensorParallel for the whole module)
            "shared_experts.gate_proj": ColwiseParallel(),
            "shared_experts.down_proj": RowwiseParallel(output_layouts=Partial()),
            "shared_experts.up_proj": ColwiseParallel(),
        }
        parallelize_module(
            module=model,
            device_mesh=tp_mesh,
            parallelize_plan=moe_layer_plan,
        )

    # if ep_mesh is not None:
    experts_mesh, experts_plan = None, None
    if ep_mesh is None:
        experts_mesh = tp_mesh
        # input Replicate, output Partial
        experts_plan = TensorParallel()
    elif tp_mesh is None:
        experts_mesh = ep_mesh
        # input / output sharding on the batch / tokens dim
        experts_plan = ExpertParallel()
    else:
        experts_mesh = ep_tp_mesh
        experts_plan = ExpertTensorParallel()

    parallelize_module(
        module=model.experts,
        device_mesh=experts_mesh,
        parallelize_plan=experts_plan,
    )