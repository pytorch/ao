# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD 3-Clause license found in the
# LICENSE file in the root directory of this source tree.
######################################################################
#
# To run these unit tests, use the following command:
#
# torchrun --nproc_per_node=${NUM_GPUS} -m pytest test_fsdp_tp.py
#
#######################################################################

import copy
import os

import pytest
import torch
from torch import distributed as dist
from torch import nn
from torch.distributed._composable.fsdp import fully_shard
from torch.distributed._tensor import DTensor
from torch.distributed.device_mesh import DeviceMesh, init_device_mesh
from torch.distributed.tensor import Partial, Replicate, Shard
from torch.nn import functional as F

try:
    from torch.distributed.tensor.parallel import (
        PrepareModuleInputOutput,
        parallelize_module,
    )
except ImportError:
    import warnings

    warnings.warn(
        "torch version is too old, these tests require nightly build. Skipping MoE training tests."
    )
    pytest.skip(allow_module_level=True)

# this feature requires CUDA and SM89+
if not torch.cuda.is_available() or torch.cuda.get_device_capability() < (8, 9):
    pytest.skip(
        "CUDA not available or compute capability < 8.9", allow_module_level=True
    )

from torchao.float8.float8_utils import compute_error
from torchao.prototype.moe_training.conversion_utils import MoETrainingConfig
from torchao.quantization.quant_api import quantize_

from .testing_utils import _validate_model_conversion

# this test requires torchtitan
try:
    from torchtitan.experiments.llama4.infra.expert_parallel import (
        ExpertParallel,
        ExpertTensorParallel,
        NoParallel,
        TensorParallel,
    )
    from torchtitan.experiments.llama4.model.args import TransformerModelArgs
    from torchtitan.experiments.llama4.model.moe import MoE
except ImportError:
    import warnings

    warnings.warn("torchtitan not installed, skipping MoE tests.")
    pytest.skip(allow_module_level=True)


@pytest.mark.parametrize(
    "target_fqns",
    [
        ["experts"],
        # TODO: investigate hang when shared_expert is converted
        # ["experts,shared_expert"],
    ],
)
def test_moe_float8_training_fsdp_tp(target_fqns: list[str]):
    assert torch.cuda.is_available()

    # setup distributed for tp
    mesh = setup_distributed()

    # define model args
    model_args = TransformerModelArgs(
        moe_enabled=True,
        num_experts=8,
        dim=256,
        vocab_size=1024,
    )
    init_std = 0.02
    device = torch.device("cuda")

    # reference bf16 MoE
    ref_model = MoE(model_args).to(torch.bfloat16).cuda()
    torch.manual_seed(1)
    ref_model.init_weights(init_std, device)

    # target MoE for testing conversion
    model = copy.deepcopy(ref_model)

    # assert starting params are identical for both models
    for param1, param2 in zip(model.parameters(), ref_model.parameters()):
        assert torch.equal(param1, param2)

    # convert MoE to float8 training
    def moe_module_filter_fn(mod: nn.Module, cur_fqn: str) -> bool:
        for target_fqn in target_fqns:
            if target_fqn in cur_fqn:
                return True
        return False

    # quantize test model
    config = MoETrainingConfig()
    quantize_(model, config=config, filter_fn=moe_module_filter_fn)

    # validate that only the experts were converted
    _validate_model_conversion(
        model,
        target_fqns=target_fqns,
    )

    # apply TP
    apply_moe_ep_tp(model, tp_mesh=mesh["tp"], ep_mesh=None, ep_tp_mesh=None)
    apply_moe_ep_tp(ref_model, tp_mesh=mesh["tp"], ep_mesh=None, ep_tp_mesh=None)

    # apply FSDP2
    fsdp_config = {"mesh": mesh["dp"]}
    fully_shard(model, **fsdp_config)
    fully_shard(ref_model, **fsdp_config)

    # Rough validation that parallelization was applied properly.
    assert isinstance(model.experts.w1.data, DTensor), (
        "test model experts.w1 is not a DTensor"
    )
    assert isinstance(model.experts.w2.data, DTensor), (
        "test model experts.w2 is not a DTensor"
    )
    assert isinstance(model.experts.w3.data, DTensor), (
        "test model experts.w3 is not a DTensor"
    )
    assert isinstance(ref_model.experts.w1.data, DTensor), (
        "ref model experts.w1 is not a DTensor"
    )
    assert isinstance(ref_model.experts.w2.data, DTensor), (
        "ref model experts.w2 is not a DTensor"
    )
    assert isinstance(ref_model.experts.w3.data, DTensor), (
        "ref model experts.w3 is not a DTensor"
    )

    # inputs
    batch, seq, dim = 8, 2048, 256
    ref_x = torch.randn(
        batch, seq, dim, dtype=torch.bfloat16, requires_grad=True, device=device
    )
    x = ref_x.detach().clone().requires_grad_(True)

    # forward pass
    ref_out = ref_model(ref_x)
    out = model(x)

    # validate output
    out_sqnr = compute_error(out, ref_out)
    assert out_sqnr.item() >= 30.0, f"SQNR must be >= 30.0, got {out_sqnr.item()}."

    # compute loss
    labels = torch.ones_like(ref_out)
    ref_loss = F.mse_loss(ref_out, labels)
    out_loss = F.mse_loss(out, labels)

    # backward pass
    ref_loss.backward()
    out_loss.backward()

    # validate input gradient
    input_grad_sqnr = compute_error(x.grad, ref_x.grad)
    assert input_grad_sqnr.item() >= 28.0, (
        f"SQNR must be >= 28.0, got {input_grad_sqnr.item()}."
    )

    # validate param gradients
    for param1, param2 in zip(model.parameters(), ref_model.parameters()):
        param_grad_sqnr = compute_error(param1.grad, param2.grad)
        assert param_grad_sqnr.item() >= 25.0, (
            f"SQNR must be >= 25.0, got {param_grad_sqnr.item()}."
        )

    dist.destroy_process_group()


def setup_distributed():
    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    dist.init_process_group("nccl", rank=rank, world_size=world_size)

    # https://pytorch.org/tutorials/recipes/distributed_device_mesh.html
    device_mesh = init_device_mesh(
        "cuda",
        (world_size // 2, 2),
        mesh_dim_names=("dp", "tp"),
    )

    # seed must be the same in all processes
    torch.manual_seed(1)
    torch.cuda.set_device(rank)
    return device_mesh


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
            "moe": PrepareModuleInputOutput(
                input_layouts=(Shard(1),),
                desired_input_layouts=(Replicate(),),
                use_local_input=True,
                output_layouts=(Partial(),),
                desired_output_layouts=(Shard(1),),
            ),
            # replicate computation for the router
            "moe.router.gate": NoParallel(),
            # input Replicate, output Partial
            "moe.shared_expert": TensorParallel(),
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
        experts_plan = ExpertTensorParallel(tp_mesh=tp_mesh, ep_mesh=ep_mesh)

    parallelize_module(
        module=model.experts,
        device_mesh=experts_mesh,
        parallelize_plan=experts_plan,
    )
