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
from torchao.prototype.moe_training.config import (
    FP8GroupedMMConfig,
    MXFP8TrainingConfig,
    MXFP8TrainingRecipe,
)
from torchao.quantization.quant_api import quantize_
from torchao.quantization.quantize_.common import KernelPreference

# Reference MoE implementation (copied from torchtitan to avoid external dependency)
from .reference_moe import MoE, MoEArgs, set_token_group_alignment_size_m
from .testing_utils import _validate_model_conversion

# Needed since changing args to function causes recompiles
torch._dynamo.config.cache_size_limit = 1000


@pytest.mark.parametrize(
    "target_fqns", [["experts"], ["shared_experts"], ["experts", "shared_experts"]]
)
@pytest.mark.parametrize("compile", [False, True])
@pytest.mark.parametrize(
    "kernel_preference", [KernelPreference.AUTO, KernelPreference.EMULATED]
)
@pytest.mark.parametrize(
    "recipe_config",
    [
        {
            "recipe": MXFP8TrainingRecipe.MXFP8_RCEIL,
            "group_alignment_size": 32,
            "min_out_sqnr": 26.5,
            "min_input_grad_sqnr": 29.0,
            "min_param_grad_sqnr": 21.0,
        },
        {
            "recipe": MXFP8TrainingRecipe.MXFP8_RCEIL_WGRAD_WITH_HP,
            "group_alignment_size": 32,
            "min_out_sqnr": 26.5,
            "min_input_grad_sqnr": 29.0,
            "min_param_grad_sqnr": 23.0,
        },
        {
            "recipe": MXFP8TrainingRecipe.MXFP8_EMULATED_RCEIL,
            "group_alignment_size": 32,
            "min_out_sqnr": 26.5,
            "min_input_grad_sqnr": 29.0,
            "min_param_grad_sqnr": 21.0,
        },
    ],
)
def test_moe_training(
    target_fqns: list[str],
    compile: bool,
    kernel_preference: KernelPreference,
    recipe_config: dict,
):
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
    assert torch.cuda.is_available()

    # Emulated mode with compile is not supported
    if recipe == MXFP8TrainingRecipe.MXFP8_EMULATED_RCEIL and compile:
        pytest.skip(
            "Skipping compile=True with kernel_preference=EMULATED, not currently supported"
        )

    # MXFP8 hardware path requires SM100
    if recipe in (
        MXFP8TrainingRecipe.MXFP8_RCEIL,
        MXFP8TrainingRecipe.MXFP8_RCEIL_WGRAD_WITH_HP,
    ) and torch.cuda.get_device_capability() != (
        10,
        0,
    ):
        pytest.skip(
            f"Skipping MXFP8 hardware mode tests, only supported on compute capability 10.0 and found {torch.cuda.get_device_capability()}"
        )

    # Set token group alignment size. This is required so that
    # each logically distinct gemm in the grouped gemm `grad_weight = grad_output_t @ input`
    # has the contraction dim be divisible by 16. 16 byte alignment is required
    # for the slowest moving dim (stride 1).
    set_token_group_alignment_size_m(group_alignment_size)
    model_args = MoEArgs(
        num_experts=8,
        num_shared_experts=1,
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
    config_cls = (
        MXFP8TrainingConfig
        if isinstance(recipe, MXFP8TrainingRecipe)
        else FP8GroupedMMConfig
    )
    config = config_cls.from_recipe(recipe)
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
