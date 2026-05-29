import copy
import pytest
import warnings
import torch
import torch.nn.functional as F
from torch import nn

from torchao.prototype.moe_qat import MoEQATConfig
from torchao.prototype.moe_qat.tensor import FakeQuantizedWeightWrapperBaseTensor, Float8FakeQuantizedWeightWrapperTensor
from torchao.quantization.qat.fake_quantize_config import Float8FakeQuantizeConfig
from torchao.quantization.granularity import PerRow, PerTensor
from torchao.quantization.quant_api import quantize_

from .reference_moe import MoE
from .testing_utils import _moe_input, _expert_weight_filter, _set_seed, create_moe_model, target_devices
from torchao.float8.float8_utils import compute_error


@pytest.mark.parametrize("device", target_devices)
@pytest.mark.parametrize("dtype", [torch.float32])
@pytest.mark.parametrize("learning_rate", [0.00001, 0.0001])
@pytest.mark.parametrize("weight_config, act_config, sqnr_threshold", [
    (Float8FakeQuantizeConfig(), None,                      15),
    (Float8FakeQuantizeConfig(), Float8FakeQuantizeConfig(), 15),
])
def test_moe_qat(device, dtype, learning_rate, weight_config, act_config, sqnr_threshold):
    """Forward and gradient SQNR vs FP32 reference for the QAT model."""
    qat_model = create_moe_model(device)
    ref_model = copy.deepcopy(qat_model)

    quantize_(
        qat_model,
        MoEQATConfig(
            activation_config=act_config,
            weight_config=weight_config,
            step="prepare",
            params_filter_fn=_expert_weight_filter,
        ),
        filter_fn=lambda m, fqn: isinstance(m, MoE),
    )

    qat_optimizer = torch.optim.SGD(qat_model.parameters(), lr=learning_rate)
    ref_optimizer = torch.optim.SGD(ref_model.parameters(), lr=learning_rate)


    class EarlyPassException(Exception):
        pass

    def check_finite(qat_name, qat_param, ref_name, ref_param):
        qat_is_finite = torch.isfinite(qat_param).all()
        ref_is_finite = torch.isfinite(ref_param).all()
        assert qat_is_finite == ref_is_finite, (
            f"{qat_name} in the QAT model and {ref_name} in the reference model do not become non-finite at the same time. "
            f"There are non-finite values in the {"QAT" if qat_is_finite is False else "reference"} model."
        )

        if ref_is_finite is False:
            warnings.warn("Exit early due to NaN in the reference model, an issue of the model used.")
            raise EarlyPassException()

    def check_all_zero(qat_name, qat_param, ref_name, ref_param):
        qat_is_zero = qat_param.norm() == 0
        ref_is_zero = ref_param.norm() == 0
        assert qat_is_zero == ref_is_zero, (
            f"{qat_name} in the QAT model and {ref_name} in the reference model do not become complete zero at the same time. "
            f"There are non-zero values in the {"QAT" if qat_is_zero is False else "reference"} model."
        )

        if ref_is_zero:
            warnings.warn(f"Exit early becuase {ref_name} in the reference model becomes zero, an issue of the model used.")
            raise EarlyPassException(0)

    try:
        for i in range(5):
            qat_prev = copy.deepcopy(qat_model)

            # Generate input randomly
            qat_x = _moe_input(qat_model).requires_grad_(True)
            ref_x = qat_x.clone().detach().requires_grad_(True)

            # Propagate forward
            qat_out = qat_model(qat_x)
            ref_out = ref_model(ref_x)

            # Set up target
            target = torch.ones_like(qat_out)

            # Compute loss and propagate backward
            qat_loss = F.mse_loss(qat_out, target)
            qat_loss.backward()

            ref_loss = F.mse_loss(ref_out, target)
            ref_loss.backward()

            # Update weights
            qat_optimizer.step()
            ref_optimizer.step()

            # Check loss alignment
            loss_rel_diff = abs(qat_loss.item() - ref_loss.item()) / ref_loss.item()
            assert loss_rel_diff < 0.03, f"Loss of the QAT and reference models should align."

            # Check SQNR of output
            check_finite("out", qat_out, "out", ref_out)
            check_all_zero("out", qat_out, "out", ref_out)
            out_sqnr = compute_error(qat_out, ref_out)
            assert out_sqnr != float("inf"), "SQNR should be finite (fake quant was applied)"
            assert out_sqnr > sqnr_threshold, f"The output SQNR too low ({out_sqnr:.1f} dB), fake quant may be degrading output"

            # Check SQNR of the input's gradient
            check_finite("qat_x.grad", qat_x.grad, "ref_x.grad", ref_x.grad)
            check_all_zero("qat_x.grad", qat_x.grad, "ref_x.grad", ref_x.grad)
            x_grad_sqnr = compute_error(qat_x.grad, ref_x.grad)
            assert x_grad_sqnr > sqnr_threshold, f"Input grad SQNR too low ({x_grad_sqnr:.1f} dB)"

            # Check SQNR of gradients of all wrapped paramters to be updated
            # Skip the 2D gate weight — it is not wrapped and accumulates routing noise
            for (qat_name, qat_param), (ref_name, ref_param) in zip(
                qat_model.named_parameters(), ref_model.named_parameters()
            ):
                if ref_param.requires_grad:
                    if ".gate" in qat_name:
                        continue
                    assert qat_param.requires_grad, f"{qat_name} should require gradients"
                    assert qat_param.grad is not None, f"{qat_name} has no gradient"
                    check_finite(f"{qat_name}.grad", qat_param.grad, f"{ref_name}.grad", ref_param.grad)
                    check_all_zero(f"{qat_name}.grad", qat_param.grad, f"{ref_name}.grad", ref_param.grad)
                    sqnr = compute_error(qat_param.grad, ref_param.grad)
                    assert sqnr > sqnr_threshold, f"Weight grad SQNR too low for {qat_name} ({sqnr:.1f} dB)"
                else:
                    assert qat_param.requires_grad is False, f"{qat_name} should not require gradients"

            # Check the change of weights
            for (cur_name, cur_param), (prev_name, prev_param) in zip(
                qat_model.named_parameters(), qat_prev.named_parameters()
            ):
                assert type(cur_param) == type(prev_param), \
                    f"The type of {cur_name} changed from {type(cur_param)} to {type(prev_param)}"

                assert cur_param.requires_grad == prev_param.requires_grad, \
                    f"{cur_name}.requires_grad changed from {prev_param.requires_grad} to {cur_param.requires_grad}"

                assert torch.isfinite(cur_param).all(), f"Elements of {cur_name} in the QAT model should all be finite."

                if cur_param.requires_grad and ".gate" not in cur_name:
                    data = cur_param.data.to_tensor() \
                        if isinstance(cur_param.data, FakeQuantizedWeightWrapperBaseTensor) \
                        else cur_param.data
                    assert not torch.equal(data, prev_param.data), \
                        f"Weight {cur_name} should be updated after optimizer step."

            # Clear gradients
            qat_optimizer.zero_grad()
            ref_optimizer.zero_grad()

    except EarlyPassException:
        pass


@pytest.mark.parametrize("device", target_devices)
@pytest.mark.parametrize("dtype", [torch.float32])
@pytest.mark.parametrize("fullgraph", [False, True])
def test_torch_compile_model(device, dtype, fullgraph):
    """torch.compile on the full QAT model should match eager output."""
    eager_model = create_moe_model(device)
    compiled_model = copy.deepcopy(eager_model)

    quantize_(
        eager_model,
        MoEQATConfig(
            weight_config=Float8FakeQuantizeConfig(),
            step="prepare",
            params_filter_fn=_expert_weight_filter,
        ),
        filter_fn=lambda m, fqn: isinstance(m, MoE)
    )

    quantize_(
        compiled_model,
        MoEQATConfig(
            weight_config=Float8FakeQuantizeConfig(),
            step="prepare",
            params_filter_fn=_expert_weight_filter,
        ),
        filter_fn=lambda m, fqn: isinstance(m, MoE)
    )
    compiled_model = torch.compile(compiled_model, fullgraph=fullgraph)

    learning_rate = 0.00001
    eager_optimizer = torch.optim.SGD(eager_model.parameters(), lr=learning_rate)
    compiled_optimizer = torch.optim.SGD(compiled_model.parameters(), lr=learning_rate)

    for i in range(5):
        print(f"step {i}")

        # Generate input randomly
        eager_x = _moe_input(eager_model).requires_grad_(True)
        compiled_x = eager_x.clone().detach().requires_grad_()

        # Propagate forward
        eager_out = eager_model(eager_x)
        compiled_out = compiled_model(compiled_x)

        out_sqnr = compute_error(compiled_out, eager_out)
        print(f"  output SQNR: {out_sqnr:.1f} dB")
        assert out_sqnr > 30, f"Compiled vs eager output SQNR too low ({out_sqnr:.1f} dB)"

        # Set up target
        target = torch.ones_like(eager_out)

        # Compute loss and propagate backward
        eager_loss = F.mse_loss(eager_out, target)
        eager_loss.backward()

        compiled_loss = F.mse_loss(compiled_out, target)
        compiled_loss.backward()

        loss_rel_diff = abs(eager_loss.item() - compiled_loss.item()) / eager_loss.item()
        print(f"  qat_loss: {eager_loss.item():.4f}, compiled_loss: {compiled_loss.item():.4f}, rel_diff: {loss_rel_diff:.8f}")
        assert loss_rel_diff < 1e-6, f"Compiled vs eager loss should align (rel diff: {loss_rel_diff:.4f})"

        # Check gradients
        x_grad_sqnr = compute_error(eager_x.grad, compiled_x.grad)
        print(f"  input grad SQNR: {x_grad_sqnr:.1f} dB")
        assert x_grad_sqnr > 94, f"Compiled vs eager x.grad SQNR too low ({x_grad_sqnr:.1f} dB)"

        for (eager_name, eager_param), (compiled_name, compiled_param) in zip(
            eager_model.named_parameters(), compiled_model.named_parameters(),
        ):
            if eager_param.requires_grad:
                sqnr = compute_error(eager_param.grad, compiled_param.grad)
                print(f"  {eager_name}.grad SQNR: {sqnr:.1f} dB")
                assert sqnr > 65, \
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
            print(f"  {eager_name} weight SQNR: {sqnr:.1f} dB")
            assert sqnr > 108, \
                f"Compiled vs eager {eager_name} weight SQNR too low ({sqnr:.1f} dB)"
        
        eager_optimizer.zero_grad()
        compiled_optimizer.zero_grad()