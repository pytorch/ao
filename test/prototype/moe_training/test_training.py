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
from torchao.prototype.moe_training.conversion_utils import MoETrainingConfig
from torchao.prototype.moe_training.tensor import ScaledGroupedMMTensor
from torchao.quantization.quant_api import quantize_

# this test requires torchtitan
try:
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
        ["does.not.exist"],
    ],
)
def test_moe_float8_training(target_fqns: list[str]):
    model_args = TransformerModelArgs(
        moe_enabled=True,
        num_experts=8,
        dim=256,
    )
    init_std = 0.02
    device = torch.device("cuda")

    # reference bf16 MoE
    ref_model = MoE(model_args).to(torch.bfloat16).cuda()
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
    config = MoETrainingConfig()
    quantize_(model, config=config, filter_fn=moe_module_filter_fn)

    # validate that only the experts were converted
    _validate_model_conversion(
        model,
        target_fqns=target_fqns,
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
    assert input_grad_sqnr.item() >= 30.0, (
        f"SQNR must be >= 30.0, got {input_grad_sqnr.item()}."
    )

    # validate param gradients
    for param1, param2 in zip(model.parameters(), ref_model.parameters()):
        param_grad_sqnr = compute_error(param1.grad, param2.grad)
        assert param_grad_sqnr.item() >= 25.0, (
            f"SQNR must be >= 25.0, got {param_grad_sqnr.item()}."
        )


def _validate_model_conversion(
    root_module: nn.Module,
    target_fqns: list[str],
):
    def _recursive_validate(
        module: nn.Module,
        cur_fqn: str,
    ):
        is_allowed_module = cur_fqn in target_fqns

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
