import copy

import pytest
import torch
import torch.nn.functional as F
from torch import nn

from torchao.prototype.moe_qat import MoEQATConfig
from torchao.prototype.moe_qat.tensor import Float8FakeQuantizedWeightWrapperTensor
from torchao.quantization.qat.fake_quantize_config import Float8FakeQuantizeConfig
from torchao.quantization.granularity import PerTensor
from torchao.quantization.quant_api import quantize_

from .reference_moe import MoE
from .testing_utils import _moe_input, _expert_weight_filter, _set_seed, device, moe_model, use_grouped_mm, weight_config


class TestFloat8MoEQAT:
    """Tests for FP8 row-wise MoE QAT — end-to-end training."""

    def _prepare(self, model, config, activation_config=None):
        qat_config = MoEQATConfig(
            activation_config=activation_config,
            weight_config=config,
            step="prepare",
            params_filter_fn=_expert_weight_filter,
        )
        quantize_(model, qat_config, filter_fn=lambda m, fqn: isinstance(m, MoE))
        return model

    # =========================================================================
    # Forward / backward SQNR (single pass)
    # =========================================================================

    def test_forward_differs_after_prepare(self, moe_model, weight_config, device):
        """Quantized model SQNR should meet threshold against FP32 reference."""
        from torchao.float8.float8_utils import compute_error

        ref_model = copy.deepcopy(moe_model)
        for p1, p2 in zip(moe_model.parameters(), ref_model.parameters()):
            assert torch.equal(p1, p2)

        self._prepare(moe_model, weight_config)

        x = _moe_input(moe_model)
        ref_out = ref_model(x)
        qat_out = moe_model(x)

        sqnr = compute_error(qat_out, ref_out)
        assert sqnr != float("inf"), "SQNR should be finite (fake quant was applied)"
        assert sqnr > 10, f"SQNR too low ({sqnr:.1f} dB), fake quant may be degrading output"

    def test_gradient_sqnr(self, moe_model, weight_config, device):
        """Gradient SQNR should meet threshold against FP32 reference."""
        from torchao.float8.float8_utils import compute_error

        ref_model = copy.deepcopy(moe_model)
        for p1, p2 in zip(moe_model.parameters(), ref_model.parameters()):
            assert torch.equal(p1, p2)

        self._prepare(moe_model, weight_config)

        x = _moe_input(moe_model)
        ref_x = x.detach().clone().requires_grad_(True)
        test_x = x.detach().clone().requires_grad_(True)

        ref_out = ref_model(ref_x)
        out = moe_model(test_x)

        ref_loss = F.mse_loss(ref_out, torch.ones_like(ref_out))
        loss = F.mse_loss(out, torch.ones_like(out))
        ref_loss.backward()
        loss.backward()

        input_sqnr = compute_error(test_x.grad, ref_x.grad)
        assert input_sqnr > 10, f"Input grad SQNR too low ({input_sqnr:.1f} dB)"

        for (name, param), (ref_name, ref_param) in zip(
            moe_model.named_parameters(), ref_model.named_parameters()
        ):
            if param.requires_grad and param.ndim == 3:
                sqnr = compute_error(param.grad, ref_param.grad)
                assert sqnr > 10, f"Weight grad SQNR too low for {name} ({sqnr:.1f} dB)"

    # =========================================================================
    # Multi-step training
    # =========================================================================

    def test_training_runs_without_nan(self, moe_model, weight_config, device):
        """10-step QAT training: verify weights change and output stays finite."""
        self._prepare(moe_model, weight_config)
        optimizer = torch.optim.SGD(moe_model.parameters(), lr=0.001)

        initial_weights = {}
        for name, param in moe_model.named_parameters():
            if param.ndim == 3:
                initial_weights[name] = param.data.to_tensor().clone()

        num_steps = 10
        for _ in range(num_steps):
            optimizer.zero_grad()
            x = _moe_input(moe_model)
            out = moe_model(x)
            out.sum().backward()
            optimizer.step()

        for name, param in moe_model.named_parameters():
            if param.ndim == 3:
                assert not torch.equal(
                    param.data.to_tensor(), initial_weights[name]
                ), f"{name} should have changed after {num_steps} steps"

        x = _moe_input(moe_model)
        with torch.no_grad():
            out = moe_model(x)
        assert torch.isfinite(out).all(), "Output should be finite after training"

    # =========================================================================
    # Multi-step gradient flow
    # =========================================================================

    def test_backward_flows(self, moe_model, weight_config):
        w = torch.randn(64, 128, dtype=torch.float32)
        param = nn.Parameter(
            Float8FakeQuantizedWeightWrapperTensor(w, weight_config=weight_config)
        )
        optimizer = torch.optim.SGD([param], lr=0.001)
        A = torch.randn(16, 64)

        num_steps = 10
        prev_grad = None
        for step in range(num_steps):
            optimizer.zero_grad()
            out = torch.mm(A, param)
            loss = out.sum()
            loss.backward()
            assert param.grad is not None, f"Step {step}: gradient is None"
            assert param.grad.abs().sum() > 0, f"Step {step}: gradient is zero"
            if step > 0:
                assert not torch.equal(prev_grad, param.grad), (
                    f"Step {step}: gradient unchanged"
                )
            prev_grad = param.grad.clone()
            optimizer.step()

    def test_backward_flows_model(self, moe_model, weight_config, device):
        """Multi-step gradient test through the full model."""
        self._prepare(moe_model, weight_config)
        optimizer = torch.optim.SGD(moe_model.parameters(), lr=0.001)
        x = _moe_input(moe_model)

        num_steps = 10
        prev_grads = {}
        for step in range(num_steps):
            optimizer.zero_grad()
            out = moe_model(x)
            loss = out.sum()
            loss.backward()
            for name, param in moe_model.named_parameters():
                if param.requires_grad and param.ndim == 3:
                    assert param.grad is not None, f"Step {step}: {name} has no gradient"
                    assert param.grad.abs().sum() > 0, f"Step {step}: {name} gradient is zero"
                    if step > 0:
                        assert not torch.equal(prev_grads[name], param.grad), (
                            f"Step {step}: {name} gradient unchanged"
                        )
                    prev_grads[name] = param.grad.clone()
            optimizer.step()

    # =========================================================================
    # torch.compile
    # =========================================================================

    def test_torch_compile_mm(self, weight_config):
        """torch.compile on torch.mm with wrapped weight should match eager."""
        w = torch.randn(64, 128)
        wrapper = Float8FakeQuantizedWeightWrapperTensor(w, weight_config=weight_config)
        A = torch.randn(16, 64)
        compiled = torch.compile(torch.mm, fullgraph=True)
        out = compiled(A, wrapper)
        eager_out = torch.mm(A, wrapper)
        assert out.shape == (16, 128)
        assert torch.allclose(out, eager_out), "Compiled output should match eager output"

    def test_torch_compile_model(self, moe_model, weight_config, device):
        """torch.compile on the full MoE model should match eager output."""
        self._prepare(moe_model, weight_config)
        x = _moe_input(moe_model)

        with torch.no_grad():
            eager_out = moe_model(x)
            compiled_out = torch.compile(moe_model)(x)

        assert torch.allclose(compiled_out, eager_out, atol=1e-3, rtol=1e-2), (
            "Compiled model output should match eager"
        )

    # =========================================================================
    # Activation QAT
    # =========================================================================

    def test_activation_qat(self, moe_model, weight_config, device):
        """Activation QAT should meet SQNR threshold vs weight-only QAT baseline."""
        from torchao.float8.float8_utils import compute_error

        act_config = Float8FakeQuantizeConfig(dtype=torch.float8_e4m3fn, granularity=PerTensor())

        ref_model = copy.deepcopy(moe_model)
        self._prepare(ref_model, weight_config)

        self._prepare(moe_model, weight_config, activation_config=act_config)

        x = _moe_input(moe_model)
        with torch.no_grad():
            ref_out = ref_model(x)
            out = moe_model(x)
        sqnr = compute_error(ref_out, out)
        assert sqnr != float("inf"), "SQNR should be finite (activation fake quant was applied)"
        assert sqnr > 10, f"Activation QAT SQNR too low vs weight-only QAT ({sqnr:.1f} dB)"

    def test_activation_qat_backward_mm(self, weight_config):
        """Forward/gradient SQNR and multi-step backward for activation+weight QAT."""
        from torchao.float8.float8_utils import compute_error

        act_config = Float8FakeQuantizeConfig(dtype=torch.float8_e4m3fn, granularity=PerTensor())
        w = torch.randn(64, 128, dtype=torch.float32)
        A = torch.randn(16, 64)

        w_only = Float8FakeQuantizedWeightWrapperTensor(w.clone(), weight_config=weight_config)
        w_param = nn.Parameter(w_only)

        wrapper = Float8FakeQuantizedWeightWrapperTensor(
            w, activation_config=act_config, weight_config=weight_config
        )
        param = nn.Parameter(wrapper)

        with torch.no_grad():
            ref_out = torch.mm(A, w_param)
            out = torch.mm(A, param)
        fwd_sqnr = compute_error(ref_out, out)
        assert fwd_sqnr != float("inf"), "Activation QAT should change forward output"
        assert fwd_sqnr > 10, f"Activation QAT forward SQNR too low vs weight-only ({fwd_sqnr:.1f} dB)"

        torch.mm(A.clone(), w_param).sum().backward()
        torch.mm(A.clone(), param).sum().backward()
        grad_sqnr = compute_error(w_param.grad, param.grad)
        assert grad_sqnr > 10, f"Activation QAT grad SQNR too low vs weight-only ({grad_sqnr:.1f} dB)"

        optimizer = torch.optim.SGD([param], lr=0.001)
        num_steps = 10
        prev_grad = None
        for step in range(num_steps):
            optimizer.zero_grad()
            out = torch.mm(A, param)
            loss = out.sum()
            loss.backward()
            assert param.grad is not None, f"Step {step}: gradient is None"
            assert param.grad.abs().sum() > 0, f"Step {step}: gradient is zero"
            if step > 0:
                assert not torch.equal(prev_grad, param.grad), (
                    f"Step {step}: gradient unchanged"
                )
            prev_grad = param.grad.clone()
            optimizer.step()

    # =========================================================================
    # Weight update verification
    # =========================================================================

    def test_weight_update_after_step(self, weight_config):
        """One optimizer step should change the wrapped weight (torch.mm path)."""
        w = torch.randn(64, 128, dtype=torch.float32)
        param = nn.Parameter(
            Float8FakeQuantizedWeightWrapperTensor(w, weight_config=weight_config)
        )
        prev_w = param.data.to_tensor().clone()
        optimizer = torch.optim.SGD([param], lr=0.001)
        A = torch.randn(16, 64)
        out = torch.mm(A, param)
        loss = out.sum()
        loss.backward()
        optimizer.step()
        assert not torch.equal(param.data.to_tensor(), prev_w), "Weight should change after optimizer step"

    def test_weight_update_after_step_model(self, moe_model, weight_config, device):
        """One optimizer step should change expert weights through the model."""
        self._prepare(moe_model, weight_config)
        optimizer = torch.optim.SGD(moe_model.parameters(), lr=0.001)
        prev_w1 = moe_model.experts.w1.data.to_tensor().clone()
        prev_w2 = moe_model.experts.w2.data.to_tensor().clone()
        prev_w3 = moe_model.experts.w3.data.to_tensor().clone()
        x = _moe_input(moe_model)
        out = moe_model(x)
        loss = out.sum()
        loss.backward()
        optimizer.step()
        assert not torch.equal(moe_model.experts.w1.data.to_tensor(), prev_w1), "w1 should change"
        assert not torch.equal(moe_model.experts.w2.data.to_tensor(), prev_w2), "w2 should change"
        assert not torch.equal(moe_model.experts.w3.data.to_tensor(), prev_w3), "w3 should change"
