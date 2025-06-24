# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD 3-Clause license found in the
# LICENSE file in the root directory of this source tree.
######################################################################
#
# To run these unit tests, use the following command:
#
# torchrun --nproc_per_node=${NUM_GPUS} -m pytest test_tp.py
#
#######################################################################

import copy
import os

import pytest
import torch
from torch import distributed as dist
from torch import nn
from torch.distributed.device_mesh import DeviceMesh, init_device_mesh
from torch.nn import functional as F

# this feature requires CUDA and SM89+
if not torch.cuda.is_available() or torch.cuda.get_device_capability() < (8, 9):
    pytest.skip(
        "CUDA not available or compute capability < 8.9", allow_module_level=True
    )

from torchao.float8.float8_utils import compute_error
from torchao.prototype.moe_training.conversion_utils import MoETrainingConfig
from torchao.prototype.moe_training.tensor import ScaledGroupedMMTensor
from torchao.quantization.quant_api import quantize_

# this test requires torchtitan
try:
    from torchtitan.experiments.llama4.infra.parallelize import apply_moe_tp
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
        ["experts,shared_expert"],
    ],
)
def test_moe_float8_training_tp_sp(target_fqns: list[str]):
    assert torch.cuda.is_available()

    # setup distributed for fsdp
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
    apply_moe_tp(model, mesh)
    apply_moe_tp(ref_model, mesh)

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
    assert input_grad_sqnr.item() >= 30.0, (
        f"SQNR must be >= 30.0, got {input_grad_sqnr.item()}."
    )

    # validate param gradients
    for param1, param2 in zip(model.parameters(), ref_model.parameters()):
        param_grad_sqnr = compute_error(param1.grad, param2.grad)
        assert param_grad_sqnr.item() >= 25.0, (
            f"SQNR must be >= 25.0, got {param_grad_sqnr.item()}."
        )

    dist.destroy_process_group()


def _validate_model_conversion(
    root_module: nn.Module,
    target_fqns: list[str],
):
    def _recursive_validate(
        module: nn.Module,
        cur_fqn: str,
    ):
        is_allowed_module = any([target_fqn in cur_fqn for target_fqn in target_fqns])

        # check current module params
        for param_name, param in module.named_parameters(recurse=False):
            is_converted_type = isinstance(param, ScaledGroupedMMTensor)
            if is_converted_type:
                assert is_allowed_module, (
                    f"Module {cur_fqn} is not in target_fqns, but has converted param {param_name}."
                )
            if not is_allowed_module:
                assert not is_converted_type, (
                    f"Module {cur_fqn} is not in target_fqns, but has converted param {param_name}."
                )

        # recursively check child modules
        for child_name, child_module in module.named_children():
            child_fqn = f"{cur_fqn}.{child_name}" if cur_fqn else child_name
            _recursive_validate(child_module, child_fqn)

    _recursive_validate(root_module, "")


def setup_distributed():
    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    dist.init_process_group("nccl", rank=rank, world_size=world_size)
    device_mesh = init_device_mesh("cuda", (world_size,))
    # seed must be the same in all processes
    torch.manual_seed(1)
    torch.cuda.set_device(rank)
    return device_mesh


def apply_moe_tp(
    model: nn.Module,
    tp_mesh: DeviceMesh,
):
    # base on llama4 MoE TP implementation here: https://github.com/pytorch/torchtitan/blob/d9cc6b4df341eec27768b5ab9cead87ef595dbc2/torchtitan/experiments/llama4/infra/parallelize.py#L147C1-L180C10
    from torch.distributed.tensor import Partial, Replicate, Shard
    from torch.distributed.tensor.parallel import (
        PrepareModuleInputOutput,
        parallelize_module,
    )
    from torchtitan.experiments.llama4.infra.expert_parallel import (
        NoParallel,
        TensorParallel,
    )

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
        "moe.experts": TensorParallel(output_layout=Partial()),
        "moe.shared_expert": TensorParallel(output_layout=Partial()),
    }
    parallelize_module(
        module=model,
        device_mesh=tp_mesh,
        parallelize_plan=moe_layer_plan,
    )
