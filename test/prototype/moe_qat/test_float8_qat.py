import copy

import pytest
import torch
from torch import nn

from torchao.prototype.moe_qat import MoEQATConfig
from torchao.prototype.moe_qat.tensor import Float8FakeQuantizedWeightWrapperTensor
from torchao.quantization.qat.fake_quantize_config import Float8FakeQuantizeConfig
from torchao.quantization.qat import QATStep
from torchao.quantization.granularity import PerRow, PerTensor
from torchao.quantization.quant_api import quantize_


class SimpleMoE(nn.Module):
    """Minimal MoE model with 3D expert weights routed through _grouped_mm."""

    def __init__(self, num_experts=4, hidden_dim=64, intermediate_dim=128):
        super().__init__()
        self.num_experts = num_experts
        self.hidden_dim = hidden_dim
        self.intermediate_dim = intermediate_dim
        # Expert weights: [E, hidden_dim, intermediate_dim]
        # _grouped_mm expects weight [E, K, N], K matches input last dim
        # So w1 is [E, inter_dim, hidden]; transpose(-2,-1) → [E, hidden, inter_dim]
        self.w1 = nn.Parameter(torch.randn(num_experts, intermediate_dim, hidden_dim))
        self.w2 = nn.Parameter(torch.randn(num_experts, hidden_dim, intermediate_dim))
        # Non-expert (gate) — should NOT be wrapped
        self.gate = nn.Linear(hidden_dim, num_experts)

    def forward(self, x):
        # x: [tokens, hidden_dim=64]
        num_tokens = x.shape[0]
        expert_ids = torch.arange(num_tokens) % self.num_experts
        # Sort by expert and reorder tokens for contiguous-by-expert layout.
        _, perm = expert_ids.sort()
        expert_ids = expert_ids[perm]
        x_sorted = x[perm]
        tokens_per_expert = torch.bincount(expert_ids, minlength=self.num_experts)
        offsets = torch.cumsum(tokens_per_expert, dim=0, dtype=torch.int32)

        # w1: [E, inter_dim, hidden] .T → [E, hidden, inter_dim], K=hidden matches x
        w1_t = self.w1.transpose(-2, -1)
        h = torch._grouped_mm(x_sorted, w1_t, offs=offsets)
        h = torch.nn.functional.silu(h)
        # w2: [E, hidden, inter_dim] .T → [E, inter_dim, hidden], K=inter matches h
        w2_t = self.w2.transpose(-2, -1)
        out = torch._grouped_mm(h, w2_t, offs=offsets)
        return out


@pytest.fixture
def moe_model():
    return SimpleMoE()


@pytest.fixture
def weight_config():
    return Float8FakeQuantizeConfig(dtype=torch.float8_e4m3fn, granularity=PerRow())


class TestFloat8MoEQAT:
    """Tests for FP8 row-wise MoE QAT."""

    @pytest.fixture(autouse=True)
    def _set_seed(self):
        torch.manual_seed(42)

    @staticmethod
    def _expert_weight_filter(param, fqn):
        return param.ndim == 3

    def _prepare(self, model, config, activation_config=None):
        qat_config = MoEQATConfig(
            activation_config=activation_config,
            weight_config=config,
            step="prepare",
            params_filter_fn=TestFloat8MoEQAT._expert_weight_filter,
        )
        quantize_(model, qat_config, filter_fn=lambda m, fqn: isinstance(m, SimpleMoE))
        return model

    def _convert(self, model):
        qat_config = MoEQATConfig(step="convert")
        quantize_(model, qat_config, filter_fn=lambda m, fqn: isinstance(m, SimpleMoE))
        return model

    def test_prepare_wraps_expert_weights(self, moe_model, weight_config):
        model = self._prepare(moe_model, weight_config)
        from torchao.prototype.moe_qat.tensor import Float8FakeQuantizedWeightWrapperTensor

        wrapped_count = 0
        for name, param in model.named_parameters():
            if isinstance(param.data, Float8FakeQuantizedWeightWrapperTensor):
                wrapped_count += 1
        # SimpleMoE has w1, w2 — 2 expert params
        assert wrapped_count == 2, f"Expected 2 wrapped params, got {wrapped_count}"

    def test_prepare_skips_non_expert_params(self, moe_model, weight_config):
        model = self._prepare(moe_model, weight_config)
        from torchao.prototype.moe_qat.tensor import Float8FakeQuantizedWeightWrapperTensor

        # Gate (router) weight should not be wrapped
        for name, param in model.named_parameters():
            if "router" in name.lower() or "gate" in name.lower():
                assert not isinstance(param.data, Float8FakeQuantizedWeightWrapperTensor), (
                    f"Router param {name} should not be wrapped"
                )

    def test_forward_differs_after_prepare(self, moe_model, weight_config):
        """Quantized model SQNR should meet threshold against FP32 reference."""
        from torchao.float8.float8_utils import compute_error

        ref_model = copy.deepcopy(moe_model)

        # Assert starting weights identical
        for p1, p2 in zip(moe_model.parameters(), ref_model.parameters()):
            assert torch.equal(p1, p2)

        self._prepare(moe_model, weight_config)

        x = torch.randn(16, 64)
        ref_out = ref_model(x)
        qat_out = moe_model(x)

        sqnr = compute_error(ref_out, qat_out)
        assert sqnr != float("inf"), "SQNR should be finite (fake quant was applied)"
        assert sqnr > 10, f"SQNR too low ({sqnr:.1f} dB), fake quant may be degrading output"

    def test_gradient_sqnr(self, moe_model, weight_config):
        """Gradient SQNR should meet threshold against FP32 reference."""
        from torchao.float8.float8_utils import compute_error

        ref_model = copy.deepcopy(moe_model)
        for p1, p2 in zip(moe_model.parameters(), ref_model.parameters()):
            assert torch.equal(p1, p2)

        self._prepare(moe_model, weight_config)

        ref_x = torch.randn(16, 64, requires_grad=True)
        x = ref_x.detach().clone().requires_grad_(True)

        ref_out = ref_model(ref_x)
        out = moe_model(x)

        ref_loss = torch.nn.functional.mse_loss(ref_out, torch.zeros_like(ref_out))
        loss = torch.nn.functional.mse_loss(out, torch.zeros_like(out))
        ref_loss.backward()
        loss.backward()

        # Validate input gradient
        input_sqnr = compute_error(x.grad, ref_x.grad)
        assert input_sqnr > 10, f"Input grad SQNR too low ({input_sqnr:.1f} dB)"

        # Validate weight gradients
        for (name, param), (ref_name, ref_param) in zip(
            moe_model.named_parameters(), ref_model.named_parameters()
        ):
            if param.requires_grad and param.ndim == 3:
                sqnr = compute_error(param.grad, ref_param.grad)
                assert sqnr > 10, (
                    f"Weight grad SQNR too low for {name} ({sqnr:.1f} dB)"
                )

    def test_backward_flows(self, moe_model, weight_config):
        # Multi-step gradient test following the dense QAT convention
        w = torch.randn(64, 128, dtype=torch.float32)
        param = nn.Parameter(
            Float8FakeQuantizedWeightWrapperTensor(w, weight_config=weight_config)
        )
        optimizer = torch.optim.SGD([param], lr=0.01)
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

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
    def test_backward_flows_grouped_mm(self, moe_model, weight_config):
        self._prepare(moe_model, weight_config)
        moe_model = moe_model.cuda()
        x = torch.randn(16, 64, device="cuda")
        optimizer = torch.optim.SGD(moe_model.parameters(), lr=0.01)

        num_steps = 10
        prev_grads = {}
        for step in range(num_steps):
            optimizer.zero_grad()
            out = moe_model(x)
            loss = out.sum()
            loss.backward()
            for name, param in moe_model.named_parameters():
                if param.requires_grad:
                    assert param.grad is not None, f"Step {step}: {name} has no gradient"
                    assert param.grad.abs().sum() > 0, f"Step {step}: {name} gradient is zero"
                    if step > 0:
                        assert not torch.equal(prev_grads[name], param.grad), (
                            f"Step {step}: {name} gradient unchanged"
                        )
                    prev_grads[name] = param.grad.clone()
            optimizer.step()

    def test_convert_unwraps(self, moe_model, weight_config):
        from torchao.prototype.moe_qat.tensor import FakeQuantizedWeightWrapperBaseTensor

        self._prepare(moe_model, weight_config)
        self._convert(moe_model)

        wrapped_count = 0
        for name, param in moe_model.named_parameters():
            if isinstance(param.data, FakeQuantizedWeightWrapperBaseTensor):
                wrapped_count += 1
        assert wrapped_count == 0, f"{wrapped_count} parameters should not be wrapped after convert"

    def test_slice_preserves_wrapper(self, moe_model, weight_config):
        from torchao.prototype.moe_qat.tensor import Float8FakeQuantizedWeightWrapperTensor

        self._prepare(moe_model, weight_config)

        checked = 0
        for name, param in moe_model.named_parameters():
            if isinstance(param.data, Float8FakeQuantizedWeightWrapperTensor):
                sliced = param.data[0]
                assert isinstance(sliced, Float8FakeQuantizedWeightWrapperTensor), (
                    f"Slice should preserve wrapper"
                )
                checked += 1
        assert checked > 0, "No wrapped parameters found"

    def test_torch_compile_mm(self, weight_config):
        """torch.compile should trace through fake-quantized mm without graph breaks."""
        w = torch.randn(64, 128)
        config = weight_config
        wrapper = Float8FakeQuantizedWeightWrapperTensor(w, weight_config=config)
        A = torch.randn(16, 64)
        compiled = torch.compile(torch.mm, fullgraph=True)
        out = compiled(A, wrapper)
        eager_out = torch.mm(A, wrapper)
        assert out.shape == (16, 128)
        assert torch.allclose(out, eager_out), "Compiled output should match eager output"

    # --- Op coverage ---

    def _make_wrapper(self, weight_config):
        w = torch.randn(64, 128)
        return Float8FakeQuantizedWeightWrapperTensor(w, weight_config=weight_config)

    def test_op_mm(self, weight_config):
        wrapper = self._make_wrapper(weight_config)
        A = torch.randn(16, 64)
        out = torch.mm(A, wrapper)
        assert out.shape == (16, 128)

    def test_op_bmm(self, weight_config):
        # Fake quantization is applied to the 3D weight with per-row granularity
        # along the last dim; for bmm([B,M,K], [B,K,N]) this quantizes per K position.
        w = torch.randn(4, 64, 128)
        wrapper = Float8FakeQuantizedWeightWrapperTensor(w, weight_config=weight_config)
        A = torch.randn(4, 16, 64)
        out = torch.bmm(A, wrapper)
        assert out.shape == (4, 16, 128)

    def test_op_linear(self, weight_config):
        w = torch.randn(128, 64)
        wrapper = Float8FakeQuantizedWeightWrapperTensor(w, weight_config=weight_config)
        A = torch.randn(16, 64)
        out = torch.nn.functional.linear(A, wrapper)
        assert out.shape == (16, 128)

    def test_op_matmul(self, weight_config):
        wrapper = self._make_wrapper(weight_config)
        A = torch.randn(16, 64)
        out = torch.matmul(A, wrapper)
        assert out.shape == (16, 128)

    def test_op_addmm(self, weight_config):
        wrapper = self._make_wrapper(weight_config)
        bias = torch.randn(128)
        A = torch.randn(16, 64)
        out = torch.addmm(bias, A, wrapper)
        assert out.shape == (16, 128)

    # --- Indexing ---

    def test_index_tensor_preserves_wrapper(self, weight_config):
        w = torch.randn(4, 64, 128)
        wrapper = Float8FakeQuantizedWeightWrapperTensor(w, weight_config=weight_config)
        ids = torch.tensor([0, 2])
        result = wrapper[ids]
        assert isinstance(result, Float8FakeQuantizedWeightWrapperTensor)
        assert result.shape == (2, 64, 128)

    # --- Activation QAT ---

    def test_activation_qat(self, moe_model, weight_config):
        act_config = Float8FakeQuantizeConfig(dtype=torch.float8_e4m3fn, granularity=PerTensor())
        self._prepare(moe_model, weight_config, activation_config=act_config)
        x = torch.randn(16, 64)
        with torch.no_grad():
            out = moe_model(x)
        assert out.shape == (16, 64)

    def test_activation_qat_backward_mm(self, weight_config):
        """Multi-step backward through activation+weight QAT via torch.mm."""
        act_config = Float8FakeQuantizeConfig(dtype=torch.float8_e4m3fn, granularity=PerTensor())
        w = torch.randn(64, 128, dtype=torch.float32)
        wrapper = Float8FakeQuantizedWeightWrapperTensor(w, activation_config=act_config, weight_config=weight_config)
        param = nn.Parameter(wrapper)
        optimizer = torch.optim.SGD([param], lr=0.01)
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

    # --- Config validation (following test_qat_config_init pattern) ---

    def test_config_step_validation(self):
        weight_config = Float8FakeQuantizeConfig(dtype=torch.float8_e4m3fn, granularity=PerRow())
        # OK
        MoEQATConfig(weight_config=weight_config, step="prepare")
        MoEQATConfig(step="convert")
        # Bad step
        with pytest.raises(ValueError, match="`step` must be one of"):
            MoEQATConfig(weight_config=weight_config, step="blah")

    def test_config_requires_weight_config(self):
        # Missing weight_config in prepare
        with pytest.raises(ValueError, match="Must specify"):
            MoEQATConfig(step="prepare")

    def test_config_rejects_base_config_in_convert(self):
        from torchao.quantization import Float8DynamicActivationFloat8WeightConfig
        weight_config = Float8FakeQuantizeConfig(dtype=torch.float8_e4m3fn, granularity=PerRow())
        # OK: prepare with weight_config
        MoEQATConfig(weight_config=weight_config, step="prepare")
        # Error: convert with base_config (not yet supported)
        base_config = Float8DynamicActivationFloat8WeightConfig()
        with pytest.raises(NotImplementedError, match="Applying PTQ"):
            MoEQATConfig(base_config=base_config, step="convert")

    def test_config_rejects_non_float8_config(self):
        from torchao.quantization.qat import IntxFakeQuantizeConfig
        intx_config = IntxFakeQuantizeConfig(torch.int8, "per_channel")
        with pytest.raises(ValueError, match="Only `Float8FakeQuantizeConfig`"):
            MoEQATConfig(weight_config=intx_config, step="prepare")

    # --- Convert round-trip (following test_qat_api_convert_no_quantization) ---

    def test_convert_round_trip(self, moe_model, weight_config):
        """Prepare → Convert should produce identical weights to the original model."""
        original = copy.deepcopy(moe_model)
        self._prepare(moe_model, weight_config)
        self._convert(moe_model)
        for (name, param), (orig_name, orig_param) in zip(
            moe_model.named_parameters(), original.named_parameters()
        ):
            assert torch.equal(param, orig_param), f"{name} should match after convert"

    # --- Weight update verification (following test_qat_fp8a4w_quantizer) ---

    def test_weight_update_after_step(self, weight_config):
        """One optimizer step should change the wrapped weight (torch.mm path, CPU-safe)."""
        w = torch.randn(64, 128, dtype=torch.float32)
        param = nn.Parameter(
            Float8FakeQuantizedWeightWrapperTensor(w, weight_config=weight_config)
        )
        prev_w = param.data.to_tensor().clone()
        optimizer = torch.optim.SGD([param], lr=0.01)
        A = torch.randn(16, 64)
        out = torch.mm(A, param)
        loss = out.sum()
        loss.backward()
        optimizer.step()
        assert not torch.equal(param.data.to_tensor(), prev_w), "Weight should change after optimizer step"

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
    def test_weight_update_after_step_grouped_mm(self, moe_model, weight_config):
        """One optimizer step should change expert weights via _grouped_mm."""
        self._prepare(moe_model, weight_config)
        moe_model = moe_model.cuda()
        optimizer = torch.optim.SGD(moe_model.parameters(), lr=0.01)
        prev_w1 = moe_model.w1.data.to_tensor().clone()
        prev_w2 = moe_model.w2.data.to_tensor().clone()
        x = torch.randn(16, 64, device="cuda")
        out = moe_model(x)
        loss = out.sum()
        loss.backward()
        optimizer.step()
        assert not torch.equal(moe_model.w1.data.to_tensor(), prev_w1), "w1 should change"
        assert not torch.equal(moe_model.w2.data.to_tensor(), prev_w2), "w2 should change"

    def test_default_params_filter(self, moe_model, weight_config):
        """Default filter (_is_parameter) wraps all parameters including 2D gate weights."""
        qat_config = MoEQATConfig(weight_config=weight_config, step="prepare")
        quantize_(moe_model, qat_config, filter_fn=lambda m, fqn: isinstance(m, SimpleMoE))
        from torchao.prototype.moe_qat.tensor import Float8FakeQuantizedWeightWrapperTensor

        wrapped_count = 0
        for name, param in moe_model.named_parameters():
            if isinstance(param.data, Float8FakeQuantizedWeightWrapperTensor):
                wrapped_count += 1
        # Gate (2D Linear) + w1, w2 (3D) + gate bias = 4 params total
        assert wrapped_count == 4, f"Expected 4 wrapped params, got {wrapped_count}"

    # --- detach and clone semantics (following dense QAT convention) ---

    def test_detach_preserves_wrapper(self, weight_config):
        w = torch.randn(64, 128)
        wrapper = Float8FakeQuantizedWeightWrapperTensor(w, weight_config=weight_config)
        detached = wrapper.detach()
        assert isinstance(detached, Float8FakeQuantizedWeightWrapperTensor)
        assert detached.requires_grad is False

    def test_clone_preserves_wrapper(self, weight_config):
        w = torch.randn(64, 128)
        wrapper = Float8FakeQuantizedWeightWrapperTensor(w, weight_config=weight_config)
        cloned = wrapper.clone()
        assert isinstance(cloned, Float8FakeQuantizedWeightWrapperTensor)

    def test_view_preserves_wrapper(self, weight_config):
        w = torch.randn(64, 128)
        wrapper = Float8FakeQuantizedWeightWrapperTensor(w, weight_config=weight_config)
        viewed = wrapper.view(64, 2, 64)
        assert isinstance(viewed, Float8FakeQuantizedWeightWrapperTensor)
        assert viewed.shape == (64, 2, 64)

    def test_permute_preserves_wrapper(self, weight_config):
        w = torch.randn(4, 64, 128)
        wrapper = Float8FakeQuantizedWeightWrapperTensor(w, weight_config=weight_config)
        permuted = wrapper.permute(1, 0, 2)
        assert isinstance(permuted, Float8FakeQuantizedWeightWrapperTensor)
        assert permuted.shape == (64, 4, 128)

    def test_copy_inplace_semantics(self, weight_config):
        """copy_ should return the original wrapper (x.copy_(y) is x)."""
        w1 = torch.randn(64, 128)
        w2 = torch.randn(64, 128)
        target = Float8FakeQuantizedWeightWrapperTensor(w1, weight_config=weight_config)
        src = Float8FakeQuantizedWeightWrapperTensor(w2, weight_config=weight_config)
        data_before = target.to_tensor().clone()
        result = target.copy_(src)
        assert result is target, "copy_ should return self"
        assert torch.equal(target.to_tensor(), src.to_tensor()), "copy_ should update data"
