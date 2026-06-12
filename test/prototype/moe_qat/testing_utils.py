"""Shared fixtures and utilities for MoE QAT tests."""

import os

import pytest
import torch
from torch import nn
from torch.distributed._tensor import DTensor
from torch.distributed.device_mesh import DeviceMesh, init_device_mesh
from torch.distributed.tensor import Partial, Replicate, Shard
from torch.distributed.tensor.parallel import (
    ColwiseParallel,
    PrepareModuleInputOutput,
    RowwiseParallel,
    parallelize_module,
)
from torch.distributed._composable.fsdp import fully_shard

from .reference_moe import MoE, MoEArgs
from .reference_parallel_styles import (
    ExpertParallel,
    ExpertTensorParallel,
    NoParallel,
    TensorParallel,
)

target_devices = ["cpu"]
if torch.cuda.is_available():
    target_devices.append("cuda")


@pytest.fixture(scope="module", params=target_devices)
def distributed_env(request):
    world_size = int(os.environ["WORLD_SIZE"])

    assert world_size == 4, (
        f"This test requires world_size=4, but got world_size={world_size}. "
        "Run with: torchrun --nproc_per_node=4 -m pytest"
    )

    torch.manual_seed(42)
    device_type = request.param

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


def create_moe_model(device, use_grouped_mm, dtype=torch.float32):
    dim, hidden_dim = 1024, 2048
    args = MoEArgs(
        num_experts=8,
        num_shared_experts=2,
        use_grouped_mm=use_grouped_mm,
        load_balance_coeff=None,
    )
    model = MoE(args, dim=dim, hidden_dim=hidden_dim)
    with torch.no_grad():
        for param in model.parameters():
            nn.init.trunc_normal_(param, std=0.5)
    return model.to(device=device, dtype=dtype)


def _expert_weight_filter(param, fqn):
    """A `params_filter_fn` for MoEQATConfig: wraps only 3D expert weight parameters,
    skipping 2D parameters (e.g., gate, router, bias)."""
    return param.ndim == 3


def _moe_input(model, batch=2, seq=8):
    """Create input tensor whose last dim matches the model's dim."""
    dim = model.experts.gate_proj.shape[-1]
    device=model.experts.gate_proj.device
    dtype=model.experts.gate_proj.dtype
    return torch.randn(batch, seq, dim, device=device, dtype=dtype)


class ParallelStrategy:
    """Enum-like class for parallelization strategies."""

    TENSOR_PARALLEL = "tensor_parallel"
    EXPERT_PARALLEL = "expert_parallel"
    EXPERT_TENSOR_PARALLEL = "expert_tensor_parallel"
    FSDP = "fsdp"
    FSDP_TP = "fsdp_tp"


def consolidate_tensor_to_cpu(tensor, target_rank=0):
    """Consolidate a potentially sharded tensor to a full CPU tensor on target_rank.

    DTensor: uses full_tensor() to reconstruct the complete tensor.
    Plain tensor: returns as-is on CPU (assumed already full, e.g., FSDP2 forward outputs).
    All ranks must call this with the same tensor argument.
    """
    if not isinstance(tensor, torch.Tensor):
        raise ValueError("`tensor` must be a `torch.Tensor` or its subclasses.")

    if isinstance(tensor, DTensor):
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


def apply_parallel_strategy(model, ref_model, parallel_strategy, use_compile, distributed_env):
    """Apply the given parallelization strategy to both model and ref_model."""
    tp_mesh = distributed_env["tp_mesh"]
    ep_mesh = distributed_env["ep_mesh"]
    ep_tp_mesh = distributed_env["ep_tp_mesh"]
    dp_mesh = distributed_env["dp_mesh"]
    dp_tp_mesh = distributed_env["dp_tp_mesh"]

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
        fsdp_config = {"mesh": dp_mesh}
        fully_shard(model, **fsdp_config)
        fully_shard(ref_model, **fsdp_config)
    elif parallel_strategy == ParallelStrategy.FSDP_TP:
        tp_submesh = dp_tp_mesh["tp"]
        dp_submesh = dp_tp_mesh["dp"]
        apply_moe_ep_tp(model, tp_mesh=tp_submesh, ep_mesh=None, ep_tp_mesh=None)
        apply_moe_ep_tp(ref_model, tp_mesh=tp_submesh, ep_mesh=None, ep_tp_mesh=None)
        if use_compile:
            model = torch.compile(model)
            ref_model = torch.compile(ref_model)
        fsdp_config = {"mesh": dp_submesh}
        fully_shard(model, **fsdp_config)
        fully_shard(ref_model, **fsdp_config)

    return model, ref_model