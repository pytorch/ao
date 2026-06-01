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
from .testing_utils import _expert_weight_filter, create_moe_model
from .reference_parallel_styles import ExpertParallel, ExpertTensorParallel, NoParallel, TensorParallel

def _get_device():
    if torch.cuda.is_available():
        return torch.device(f"cuda:{torch.cuda.current_device()}")
    return torch.device("cpu")


@pytest.fixture(scope="module")
def distributed_env():
    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])

    assert world_size == 4, (
        f"This test requires world_size=4, but got world_size={world_size}. "
        "Run with: torchrun --nproc_per_node=4 -m pytest test/prototype/moe_qat/test_distributed.py"
    )

    torch.manual_seed(42)
    torch.cuda.manual_seed_all(42)

    device = _get_device()
    if device.type == "cuda":
        torch.cuda.set_device(rank)

    # Create meshes for different parallelization strategies:
    # - tp_mesh: 1D mesh for tensor parallel only
    # - ep_mesh: 1D mesh for expert parallel only
    # - dp_mesh: 1D mesh for FSDP only
    # - ep_tp_mesh: 2D mesh for combined EP and TP
    # - dp_tp_mesh: 2D mesh for combined FSDP and TP
    tp_mesh = init_device_mesh(device.type, (world_size,), mesh_dim_names=("tp",))
    ep_mesh = init_device_mesh(device.type, (world_size,), mesh_dim_names=("ep",))
    dp_mesh = init_device_mesh(device.type, (world_size,), mesh_dim_names=("dp",))
    ep_tp_mesh = init_device_mesh(device.type, (2, 2), mesh_dim_names=("ep", "tp"))
    dp_tp_mesh = init_device_mesh(device.type, (2, 2), mesh_dim_names=("dp", "tp"))

    yield {
        "tp_mesh": tp_mesh,
        "ep_mesh": ep_mesh,
        "dp_mesh": dp_mesh,
        "ep_tp_mesh": ep_tp_mesh,
        "dp_tp_mesh": dp_tp_mesh,
    }

    torch.distributed.destroy_process_group()


class ParallelStrategy:
    """Enum-like class for parallelization strategies."""

    TENSOR_PARALLEL = "tensor_parallel"
    EXPERT_PARALLEL = "expert_parallel"
    EXPERT_TENSOR_PARALLEL = "expert_tensor_parallel"
    FSDP = "fsdp"
    FSDP_TP = "fsdp_tp"


@pytest.mark.parametrize("parallel_strategy", [
    ParallelStrategy.FSDP,
    ParallelStrategy.EXPERT_PARALLEL,
    ParallelStrategy.TENSOR_PARALLEL,
    ParallelStrategy.EXPERT_TENSOR_PARALLEL,
    ParallelStrategy.FSDP_TP,
])
@pytest.mark.parametrize("wrapper_cls,weight_config,activation_config, min_sqnr", [
    (Float8FakeQuantizedWeightWrapperTensor, Float8FakeQuantizeConfig(), None,                       {"out": 28, "input_grad": 31, "param_grad": 22}),
    (Float8FakeQuantizedWeightWrapperTensor, Float8FakeQuantizeConfig(), Float8FakeQuantizeConfig(), {"out": 25, "input_grad": 25, "param_grad": 16}),
])
def test_moe_qat_parallel(parallel_strategy, wrapper_cls, weight_config, activation_config, min_sqnr, distributed_env):

    device = _get_device()

    tp_mesh = distributed_env["tp_mesh"]
    ep_mesh = distributed_env["ep_mesh"]
    ep_tp_mesh = distributed_env["ep_tp_mesh"]
    dp_mesh = distributed_env["dp_mesh"]
    dp_tp_mesh = distributed_env["dp_tp_mesh"]

    # Reference model (no fake quantization)
    ref_model = create_moe_model(device)
    model = copy.deepcopy(ref_model)
    dim = model.experts.gate_proj.shape[-1]

    # Apply QAT to test model
    qat_config = MoEQATConfig(
        weight_config=weight_config,
        activation_config=activation_config,
        step="prepare",
        params_filter_fn=_expert_weight_filter,
    )
    quantize_(model, qat_config, filter_fn=lambda m, fqn: isinstance(m, MoE))

    # Verify starting params are identical
    for ref_param, model_param in zip(ref_model.parameters(), model.parameters()):
        if isinstance(model_param.data, FakeQuantizedWeightWrapperBaseTensor):
            assert torch.equal(ref_param, model_param.data.to_tensor())
        else:
            assert torch.equal(ref_param, model_param)

    # Apply parallelization based on strategy
    if parallel_strategy == ParallelStrategy.TENSOR_PARALLEL:
        apply_moe_ep_tp(model, tp_mesh=tp_mesh, ep_mesh=None, ep_tp_mesh=None)
        apply_moe_ep_tp(ref_model, tp_mesh=tp_mesh, ep_mesh=None, ep_tp_mesh=None)
    elif parallel_strategy == ParallelStrategy.EXPERT_PARALLEL:
        apply_moe_ep_tp(model, tp_mesh=None, ep_mesh=ep_mesh, ep_tp_mesh=None)
        apply_moe_ep_tp(ref_model, tp_mesh=None, ep_mesh=ep_mesh, ep_tp_mesh=None)
    elif parallel_strategy == ParallelStrategy.EXPERT_TENSOR_PARALLEL:
        apply_moe_ep_tp(model, tp_mesh=tp_mesh, ep_mesh=ep_mesh, ep_tp_mesh=ep_tp_mesh)
        apply_moe_ep_tp(ref_model, tp_mesh=tp_mesh, ep_mesh=ep_mesh, ep_tp_mesh=ep_tp_mesh)
    elif parallel_strategy == ParallelStrategy.FSDP:
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
    batch, seq = 8, 64
    ref_x = torch.randn(batch, seq, dim, device=device, requires_grad=True)
    x = ref_x.detach().clone().requires_grad_(True)

    # Forward
    ref_out = ref_model(ref_x)
    out = model(x)

    # Validate output
    assert torch.isfinite(ref_out).all(), "Reference output has non-finite values"
    assert torch.isfinite(out).all(), "Output has non-finite values"
    out_sqnr = compute_error(out, ref_out)
    print(f"  output SQNR: {out_sqnr.item():.1f} dB")
    assert out_sqnr.item() >= min_sqnr["out"], f"Output SQNR must be >= {min_sqnr['out']} dB, got {out_sqnr.item():.1f} dB"

    # Compute loss
    labels = torch.ones_like(ref_out)
    ref_loss = F.mse_loss(ref_out, labels)
    out_loss = F.mse_loss(out, labels)

    # Backward
    ref_loss.backward()
    out_loss.backward()

    # Validate input gradient
    assert torch.isfinite(ref_x.grad).all(), "Reference input grad has non-finite values"
    assert torch.isfinite(x.grad).all(), "Input grad has non-finite values"
    input_grad_sqnr = compute_error(x.grad, ref_x.grad)
    print(f"  input grad SQNR: {input_grad_sqnr.item():.1f} dB")
    assert input_grad_sqnr.item() >= min_sqnr["input_grad"], f"Input grad SQNR must be >= {min_sqnr['input_grad']} dB, got {input_grad_sqnr.item():.1f} dB"

    # Validate param gradients
    for (name, param), (ref_name, ref_param) in zip(
        model.named_parameters(), ref_model.named_parameters()
    ):
        assert torch.isfinite(ref_param.grad).all(), f"{ref_name} grad has non-finite values"
        assert torch.isfinite(param.grad).all(), f"{name} grad has non-finite values"
        sqnr = compute_error(param.grad, ref_param.grad)
        print(f"  {name} grad SQNR: {sqnr.item():.1f} dB")
        assert sqnr.item() >= min_sqnr["param_grad"], (
            f"{name} grad SQNR must be >= {min_sqnr['param_grad']} dB, got {sqnr.item():.1f} dB"
        )

    # # Convert: unwrap quantized weights
    # from torch.distributed._composable.fsdp import FSDPModule
    # convert_config = MoEQATConfig(step="convert")
    # if parallel_strategy in (ParallelStrategy.FSDP, ParallelStrategy.FSDP_TP):
    #     with FSDPModule.unsharded_params(model):
    #         quantize_(model, convert_config, filter_fn=lambda m, fqn: isinstance(m, MoE))
    # else:
    #     quantize_(model, convert_config, filter_fn=lambda m, fqn: isinstance(m, MoE))

    # for name, param in model.named_parameters():
    #     local_param = param.to_local() if isinstance(param, DTensor) else param
    #     assert isinstance(local_param.data, torch.Tensor), f"{name}.data must be a `torch.Tensor`"
    #     assert not isinstance(local_param.data, FakeQuantizedWeightWrapperBaseTensor), (
    #         f"{name}.data should not be wrapped after convert, got {type(local_param.data)}"
    #     )


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





def _make_moe(dim, hidden_dim, use_grouped_mm, device):
    args = MoEArgs(
        num_experts=4,
        num_shared_experts=1,
        use_grouped_mm=use_grouped_mm,
        load_balance_coeff=None,
    )
    model = MoE(args, dim=dim, hidden_dim=hidden_dim)
    with torch.no_grad():
        for param in model.parameters():
            nn.init.trunc_normal_(param, std=0.5)
    return model.to(device)




def test_fsdp2_mixed_precision_no_cast(distributed_env):
    """FSDP2 with same dtype (no-cast path, storage-sharing)."""
    device = _get_device()

    use_grouped_mm = torch.cuda.is_available()
    model = _make_moe(dim=2048, hidden_dim=4096, use_grouped_mm=use_grouped_mm, device=device)
    weight_config = Float8FakeQuantizeConfig(dtype=torch.float8_e4m3fn, granularity=PerRow())

    qat_config = MoEQATConfig(weight_config=weight_config, step="prepare")
    quantize_(model, qat_config, filter_fn=lambda m, fqn: isinstance(m, MoE))

    mp_policy = MixedPrecisionPolicy(param_dtype=torch.float32)
    fully_shard(model, mesh=distributed_env["dp_mesh"], mp_policy=mp_policy)

    x = torch.randn(2, 4, 2048, device=device)
    out = model(x)
    loss = out.sum()
    loss.backward()
    for name in ("gate_proj", "down_proj", "up_proj"):
        p = getattr(model.experts, name)
        assert p.grad is not None, f"{name} grad is None"
        assert p.grad.abs().sum() > 0, f"{name} gradient is zero"


def test_fsdp2_mixed_precision_cast(distributed_env):
    """FSDP2 with different dtype (cast + copy_ path)."""
    device = _get_device()

    use_grouped_mm = torch.cuda.is_available()
    model = _make_moe(dim=2048, hidden_dim=4096, use_grouped_mm=use_grouped_mm, device=device)
    weight_config = Float8FakeQuantizeConfig(dtype=torch.float8_e4m3fn, granularity=PerRow())

    qat_config = MoEQATConfig(weight_config=weight_config, step="prepare")
    quantize_(model, qat_config, filter_fn=lambda m, fqn: isinstance(m, MoE))

    mp_policy = MixedPrecisionPolicy(param_dtype=torch.bfloat16)
    fully_shard(model, mesh=distributed_env["dp_mesh"], mp_policy=mp_policy)

    x = torch.randn(2, 4, 2048, device=device)
    out = model(x)
    loss = out.sum()
    loss.backward()
    for name in ("gate_proj", "down_proj", "up_proj"):
        p = getattr(model.experts, name)
        assert p.grad is not None, f"{name} grad is None"
        assert p.grad.abs().sum() > 0, f"{name} gradient is zero"