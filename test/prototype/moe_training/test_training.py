import copy

import pytest
import torch
from torch import nn
from torch.nn import functional as F

# this feature requires CUDA and SM89+
if not torch.cuda.is_available() or torch.cuda.get_device_capability() < (8, 9):
    pytest.skip(
        "CUDA not available or compute capability < 8.9", allow_module_level=True
    )

from torchao.float8.float8_utils import compute_error
from torchao.prototype.moe_training.conversion_utils import (
    MoEScalingType,
    MoETrainingConfig,
)
from torchao.quantization.quant_api import quantize_

from .testing_utils import _validate_model_conversion

# this test requires torchtitan
try:
    from torchtitan.distributed.expert_parallel import (
        set_token_group_alignment_size_m,
    )
    from torchtitan.models.moe import MoE, MoEArgs
except ImportError:
    pytest.skip(
        "torchtitan not installed, skipping MoE tests.", allow_module_level=True
    )


@pytest.mark.parametrize(
    "target_fqns",
    [
        ["experts"],
        ["does.not.exist"],
    ],
)
@pytest.mark.parametrize("compile", [False, True])
@pytest.mark.parametrize(
    "recipe_config",
    [
        # {"recipe": MoEScalingType.FP8_ROWWISE, "group_alignment_size": 16, "min_out_sqnr": 29.0, "min_input_grad_sqnr": 29.0, "min_param_grad_sqnr": 23.0},
        {
            "recipe": MoEScalingType.MXFP8,
            "group_alignment_size": 32,
            "min_out_sqnr": 28.0,
            "min_input_grad_sqnr": 29.0,
            "min_param_grad_sqnr": 21.0,
        },
    ],
)
def test_moe_training(target_fqns: list[str], compile: bool, recipe_config: dict):
    (
        recipe,
        group_alignment_size,
        min_out_sqnr,
        min_input_grad_sqnr,
        min_param_grad_sqnr,
    ) = (
        recipe_config["recipe"],
        recipe_config["group_alignment_size"],
        recipe_config["min_out_sqnr"],
        recipe_config["min_input_grad_sqnr"],
        recipe_config["min_param_grad_sqnr"],
    )
    # Set token group alignment size. This is required so that
    # each logically distinct gemm in the grouped gemm `grad_weight = grad_output_t @ input`
    # has the contraction dim be divisible by 16. 16 byte alignment is required
    # for the slowest moving dim (stride 1).
    set_token_group_alignment_size_m(group_alignment_size)
    model_args = MoEArgs(
        num_experts=8,
    )
    init_std = 0.02
    device = torch.device("cuda")

    # reference bf16 MoE
    dim, hidden_dim = 5120, 8192
    ref_model = MoE(model_args, dim, hidden_dim).to(torch.bfloat16).cuda()
    torch.manual_seed(42)
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
    config = MoETrainingConfig(scaling_type=recipe)
    quantize_(model, config=config, filter_fn=moe_module_filter_fn)

    # validate that only the experts were converted
    _validate_model_conversion(
        model,
        target_fqns=target_fqns,
    )
    if compile:
        # TODO: compile with fullgraph=True when torchtitan llama4 moe supports it
        model = torch.compile(model, fullgraph=False)
        ref_model = torch.compile(ref_model, fullgraph=False)

    # inputs
    batch, seq = 8, 2048
    ref_x = torch.randn(
        batch, seq, dim, dtype=torch.bfloat16, requires_grad=True, device=device
    )
    x = ref_x.detach().clone().requires_grad_(True)

    # forward pass
    ref_out = ref_model(ref_x)
    out = model(x)

    # validate output
    out_sqnr = compute_error(out, ref_out)
    assert out_sqnr.item() >= min_out_sqnr, (
        f"SQNR must be >= {min_out_sqnr}, got {out_sqnr.item()}."
    )

    # compute loss
    labels = torch.ones_like(ref_out)
    ref_loss = F.mse_loss(ref_out, labels)
    out_loss = F.mse_loss(out, labels)

    # backward pass
    ref_loss.backward()
    out_loss.backward()

    # validate input gradient
    input_grad_sqnr = compute_error(x.grad, ref_x.grad)
    assert input_grad_sqnr.item() >= min_input_grad_sqnr, (
        f"SQNR must be >= {min_input_grad_sqnr}, got {input_grad_sqnr.item()}."
    )

    # validate param gradients
    for param1, param2 in zip(model.parameters(), ref_model.parameters()):
        param_grad_sqnr = compute_error(param1.grad, param2.grad)
        assert param_grad_sqnr.item() >= min_param_grad_sqnr, (
            f"SQNR must be >= {min_param_grad_sqnr}, got {param_grad_sqnr.item()}."
        )
