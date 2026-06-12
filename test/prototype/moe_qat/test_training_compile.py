import copy
import pytest
import torch
import torch.nn.functional as F

from torchao.prototype.moe_qat import MoEQATConfig
from torchao.quantization.qat.fake_quantize_config import Float8FakeQuantizeConfig
from torchao.quantization.quant_api import quantize_
from torchao.float8.float8_utils import compute_error

from .reference_moe import MoE
from .testing_utils import _moe_input, _expert_weight_filter, create_moe_model, target_devices


@pytest.mark.parametrize("device", target_devices)
@pytest.mark.parametrize("weight_config, act_config, sqnr_threshold", [
    (Float8FakeQuantizeConfig(), None,                      {"out": 48, "input_grad": 45, "param_grad": 45, "weight": 45}),
    (Float8FakeQuantizeConfig(), Float8FakeQuantizeConfig(), {"out": 38, "input_grad": 38, "param_grad": 33, "weight": 33}),
])
def test_torch_compile_model(device, weight_config, act_config, sqnr_threshold):
    """torch.compile on the full QAT model should match eager output."""

    eager_model = create_moe_model(device, use_grouped_mm=True, dtype=torch.bfloat16)
    compiled_model = copy.deepcopy(eager_model)

    qat_config = MoEQATConfig(
        activation_config=act_config,
        weight_config=weight_config,
        step="prepare",
        params_filter_fn=_expert_weight_filter,
    )
    quantize_(eager_model, qat_config, filter_fn=lambda m, fqn: isinstance(m, MoE))
    quantize_(compiled_model, qat_config, filter_fn=lambda m, fqn: isinstance(m, MoE))
    compiled_model = torch.compile(compiled_model, fullgraph=False, backend="inductor")

    learning_rate = 1
    eager_optimizer = torch.optim.SGD(eager_model.parameters(), lr=learning_rate)
    compiled_optimizer = torch.optim.SGD(compiled_model.parameters(), lr=learning_rate)

    # Generate input randomly
    eager_x = _moe_input(eager_model).requires_grad_(True)
    compiled_x = eager_x.clone().detach().requires_grad_()

    # Propagate forward
    eager_out = eager_model(eager_x)
    compiled_out = compiled_model(compiled_x)

    out_sqnr = compute_error(compiled_out, eager_out)
    assert out_sqnr > sqnr_threshold["out"], f"Compiled vs eager output SQNR too low ({out_sqnr:.1f} dB)"

    # Set up target
    target = torch.ones_like(eager_out)

    # Compute loss and propagate backward
    eager_loss = F.mse_loss(eager_out, target)
    eager_loss.backward()

    compiled_loss = F.mse_loss(compiled_out, target)
    compiled_loss.backward()

    loss_rel_diff = abs(eager_loss.item() - compiled_loss.item()) / eager_loss.item()
    assert loss_rel_diff < 1e-6, f"Compiled vs eager loss should align (rel diff: {loss_rel_diff:.4f})"

    # Check gradients
    x_grad_sqnr = compute_error(eager_x.grad, compiled_x.grad)
    assert x_grad_sqnr > sqnr_threshold["input_grad"], f"Compiled vs eager x.grad SQNR too low ({x_grad_sqnr:.1f} dB)"

    for (eager_name, eager_param), (compiled_name, compiled_param) in zip(
        eager_model.named_parameters(), compiled_model.named_parameters(),
    ):
        if eager_param.requires_grad:
            sqnr = compute_error(eager_param.grad, compiled_param.grad)
            assert sqnr > sqnr_threshold["param_grad"], \
                f"Compiled vs eager {eager_name}.grad SQNR too low ({sqnr:.1f} dB)"
        else:
            assert compiled_param.requires_grad is False, f"Compiled {compiled_name} should not require gradients"

    # Update weights
    eager_optimizer.step()
    compiled_optimizer.step()

    # Check weights
    for (eager_name, eager_param), (compiled_name, compiled_param) in zip(
        eager_model.named_parameters(), compiled_model.named_parameters(),
    ):
        sqnr = compute_error(compiled_param, eager_param)
        assert sqnr > sqnr_threshold["weight"], \
            f"Compiled vs eager {eager_name} weight SQNR too low ({sqnr:.1f} dB)"

    eager_optimizer.zero_grad()
    compiled_optimizer.zero_grad()
