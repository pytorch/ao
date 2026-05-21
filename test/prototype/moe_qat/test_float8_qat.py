import copy

import pytest
import torch
import torch.nn.functional as F
from torch import nn

from torchao.prototype.moe_qat import MoEQATConfig
from torchao.prototype.moe_qat.tensor import (
    FakeQuantizedWeightWrapperBaseTensor,
    Float8FakeQuantizedWeightWrapperTensor,
)
from torchao.quantization.qat.fake_quantize_config import Float8FakeQuantizeConfig
from torchao.quantization.qat import QATStep
from torchao.quantization.granularity import PerRow, PerTensor
from torchao.quantization.quant_api import quantize_

from .reference_moe import MoE, MoEArgs


@pytest.fixture
def device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


@pytest.fixture
def use_grouped_mm():
    return torch.cuda.is_available()


@pytest.fixture
def moe_model(device, use_grouped_mm):
    if device.type == "cuda":
        dim, hidden_dim = 5120, 8192
    else:
        dim, hidden_dim = 512, 1024
    args = MoEArgs(
        num_experts=8,
        num_shared_experts=0,
        use_grouped_mm=use_grouped_mm,
        load_balance_coeff=None,
    )
    model = MoE(args, dim=dim, hidden_dim=hidden_dim)
    with torch.no_grad():
        for param in model.parameters():
            nn.init.trunc_normal_(param, std=0.5)
    return model.to(device)


def _moe_input(model, batch=2, seq=8):
    """Create input tensor whose last dim matches the model's dim."""
    dim = model.experts.w1.shape[-1]
    return torch.randn(batch, seq, dim, device=model.experts.w1.device)


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
        quantize_(model, qat_config, filter_fn=lambda m, fqn: isinstance(m, MoE))
        return model

    def _convert(self, model):
        qat_config = MoEQATConfig(step="convert")
        quantize_(model, qat_config, filter_fn=lambda m, fqn: isinstance(m, MoE))
        return model

    # =========================================================================
    # Prepare / convert lifecycle
    # =========================================================================

    def test_prepare_wraps_expert_weights(self, moe_model, weight_config):
        model = self._prepare(moe_model, weight_config)
        from torchao.prototype.moe_qat.tensor import (
    FakeQuantizedWeightWrapperBaseTensor,
    Float8FakeQuantizedWeightWrapperTensor,
)

        wrapped_count = 0
        for name, param in model.named_parameters():
            if isinstance(param.data, Float8FakeQuantizedWeightWrapperTensor):
                wrapped_count += 1
        # MoE has 3 3D expert params (experts.w1, w2, w3 for SwiGLU)
        assert wrapped_count == 3, f"Expected 3 wrapped params, got {wrapped_count}"

    def test_prepare_skips_non_expert_params(self, moe_model, weight_config):
        model = self._prepare(moe_model, weight_config)
        from torchao.prototype.moe_qat.tensor import (
    FakeQuantizedWeightWrapperBaseTensor,
    Float8FakeQuantizedWeightWrapperTensor,
)

        wrapped = 0
        for name, param in model.named_parameters():
            if isinstance(param.data, Float8FakeQuantizedWeightWrapperTensor):
                wrapped += 1
                assert param.ndim == 3, f"Wrapped param {name} should be 3D, got {param.ndim}D"
        assert wrapped == 3, f"All 3D expert params should be wrapped, got {wrapped}"
        # Router gate.weight is 2D — should NOT be wrapped by the 3D filter
        assert not isinstance(
            model.router.gate.weight.data, Float8FakeQuantizedWeightWrapperTensor
        ), "router.gate.weight should not be wrapped"

    def test_convert_unwraps(self, moe_model, weight_config):
        from torchao.prototype.moe_qat.tensor import FakeQuantizedWeightWrapperBaseTensor

        self._prepare(moe_model, weight_config)
        self._convert(moe_model)

        wrapped_count = 0
        for name, param in moe_model.named_parameters():
            if isinstance(param.data, FakeQuantizedWeightWrapperBaseTensor):
                wrapped_count += 1
        assert wrapped_count == 0, f"{wrapped_count} parameters should not be wrapped after convert"

    def test_convert_round_trip(self, moe_model, weight_config):
        """Prepare → Convert should produce identical weights to the original model."""
        original = copy.deepcopy(moe_model)
        self._prepare(moe_model, weight_config)
        self._convert(moe_model)
        for (name, param), (orig_name, orig_param) in zip(
            moe_model.named_parameters(), original.named_parameters()
        ):
            assert torch.equal(param, orig_param), f"{name} should match after convert"

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

        # Use sum loss: the scale is detached so only the STE contributes to
        # the backward. A uniform upstream gradient isolates the STE effect.
        ref_out.sum().backward()
        out.sum().backward()

        input_sqnr = compute_error(test_x.grad, ref_x.grad)
        assert input_sqnr > 10, f"Input grad SQNR too low ({input_sqnr:.1f} dB)"

        for (name, param), (ref_name, ref_param) in zip(
            moe_model.named_parameters(), ref_model.named_parameters()
        ):
            if param.requires_grad and param.ndim == 3:
                sqnr = compute_error(param.grad, ref_param.grad)
                assert sqnr > 10, f"Weight grad SQNR too low for {name} ({sqnr:.1f} dB)"

    # =========================================================================
    # Multi-step training with post-training SQNR (following moe_training)
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
                # .to_tensor() implicitly validates that the wrapper survived all
                # optimizer steps: if __torch_dispatch__ had returned a plain tensor
                # instead of preserving the wrapper, param.data would no longer be a
                # Float8FakeQuantizedWeightWrapperTensor and .to_tensor() would fail.
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
        # Multi-step gradient test following the dense QAT convention
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
    # Op coverage — parametrized forward+backward SQNR (following moe_training)
    # =========================================================================

    @pytest.mark.parametrize("op_name", ["mm", "matmul", "linear", "addmm", "bmm"])
    @pytest.mark.parametrize("batch_size", [None, 2, 4])
    def test_op_fwd_bwd_sqnr(self, op_name, batch_size, weight_config, device):
        if op_name in ("mm", "addmm") and batch_size is not None:
            pytest.skip(f"{op_name} doesn't support batching")
        if op_name == "bmm" and batch_size is None:
            pytest.skip("bmm requires batch dimension (3D inputs)")

        from torchao.float8.float8_utils import compute_error

        M, K, N = 64, 64, 128

        if op_name == "addmm":
            A_shape = (M, K)
        elif batch_size is None:
            A_shape = (M, K)
        else:
            A_shape = (batch_size, M, K)

        A = torch.randn(*A_shape, device=device, requires_grad=True)

        # Weight shape
        if op_name == "linear":
            w = torch.randn(N, K, device=device)
        elif op_name == "bmm" and batch_size is not None:
            w = torch.randn(batch_size, K, N, device=device)
        elif op_name == "addmm":
            w = torch.randn(K, N, device=device)
        else:
            w = torch.randn(K, N, device=device)

        # Reference
        A_ref = A.clone().detach().requires_grad_(True)
        w_ref = w.clone().detach().requires_grad_(True)

        # Wrapped weight
        wrapper = Float8FakeQuantizedWeightWrapperTensor(w, weight_config=weight_config)
        param = nn.Parameter(wrapper)

        # Forward
        if op_name == "mm":
            ref_out = torch.mm(A_ref, w_ref)
            out = torch.mm(A, param)
        elif op_name == "matmul":
            ref_out = torch.matmul(A_ref, w_ref)
            out = torch.matmul(A, param)
        elif op_name == "bmm":
            ref_out = torch.bmm(A_ref, w_ref)
            out = torch.bmm(A, param)
        elif op_name == "addmm":
            bias = torch.randn(N, device=device)
            ref_out = torch.addmm(bias, A_ref, w_ref)
            out = torch.addmm(bias, A, param)
        else:  # linear
            ref_out = F.linear(A_ref, w_ref)
            out = F.linear(A, param)

        sqnr_fwd = compute_error(out, ref_out)
        assert sqnr_fwd > 15, (
            f"Forward SQNR too low for {op_name} batch={batch_size} ({sqnr_fwd:.1f} dB)"
        )

        # Use sum loss: the scale is detached so only the STE contributes to
        # the backward. A uniform upstream gradient isolates the STE effect.
        ref_out.sum().backward()
        out.sum().backward()

        assert A.grad is not None
        input_sqnr = compute_error(A.grad, A_ref.grad)
        assert input_sqnr > 20, (
            f"Input grad SQNR too low for {op_name} batch={batch_size} ({input_sqnr:.1f} dB)"
        )

        assert param.grad is not None
        weight_sqnr = compute_error(param.grad, w_ref.grad)
        assert weight_sqnr > 20, (
            f"Weight grad SQNR too low for {op_name} batch={batch_size} ({weight_sqnr:.1f} dB)"
        )

    @pytest.mark.parametrize("op_name", ["mm", "matmul", "linear", "addmm", "bmm"])
    @pytest.mark.parametrize("batch_size", [None, 2, 4])
    def test_op_fwd_bwd_sqnr_activation(self, op_name, batch_size, weight_config, device):
        """Activation+weight QAT SQNR vs weight-only QAT baseline (parametrized)."""
        if op_name in ("mm", "addmm") and batch_size is not None:
            pytest.skip(f"{op_name} doesn't support batching")
        if op_name == "bmm" and batch_size is None:
            pytest.skip("bmm requires batch dimension (3D inputs)")

        from torchao.float8.float8_utils import compute_error

        act_config = Float8FakeQuantizeConfig(dtype=torch.float8_e4m3fn, granularity=PerTensor())
        M, K, N = 64, 64, 128

        if op_name == "addmm":
            A_shape = (M, K)
        elif batch_size is None:
            A_shape = (M, K)
        else:
            A_shape = (batch_size, M, K)

        if op_name == "linear":
            w = torch.randn(N, K, device=device)
        elif op_name == "bmm" and batch_size is not None:
            w = torch.randn(batch_size, K, N, device=device)
        elif op_name == "addmm":
            w = torch.randn(K, N, device=device)
        else:
            w = torch.randn(K, N, device=device)

        # Weight-only QAT (baseline)
        w_only = Float8FakeQuantizedWeightWrapperTensor(w.clone(), weight_config=weight_config)
        w_param = nn.Parameter(w_only)

        # Activation+weight QAT
        aw = Float8FakeQuantizedWeightWrapperTensor(w, activation_config=act_config, weight_config=weight_config)
        aw_param = nn.Parameter(aw)

        # Forward SQNR
        A = torch.randn(*A_shape, device=device)
        if op_name == "mm":
            ref_out = torch.mm(A, w_param)
            out = torch.mm(A.clone(), aw_param)
        elif op_name == "matmul":
            ref_out = torch.matmul(A, w_param)
            out = torch.matmul(A.clone(), aw_param)
        elif op_name == "bmm":
            ref_out = torch.bmm(A, w_param)
            out = torch.bmm(A.clone(), aw_param)
        elif op_name == "addmm":
            bias = torch.randn(N, device=device)
            ref_out = torch.addmm(bias, A, w_param)
            out = torch.addmm(bias, A.clone(), aw_param)
        else:  # linear
            ref_out = F.linear(A, w_param)
            out = F.linear(A.clone(), aw_param)

        fwd_sqnr = compute_error(ref_out, out)
        assert fwd_sqnr != float("inf"), "Activation QAT should change forward output"
        assert fwd_sqnr > 15, (
            f"Activation QAT forward SQNR too low for {op_name} batch={batch_size} ({fwd_sqnr:.1f} dB)"
        )

        # Gradient SQNR (independent backward passes with cloned inputs)
        A1 = torch.randn(*A_shape, device=device)
        A2 = A1.clone()
        if op_name == "mm":
            torch.mm(A1, w_param).sum().backward()
            torch.mm(A2, aw_param).sum().backward()
        elif op_name == "matmul":
            torch.matmul(A1, w_param).sum().backward()
            torch.matmul(A2, aw_param).sum().backward()
        elif op_name == "bmm":
            torch.bmm(A1, w_param).sum().backward()
            torch.bmm(A2, aw_param).sum().backward()
        elif op_name == "addmm":
            bias1 = torch.randn(N, device=device)
            torch.addmm(bias1, A1, w_param).sum().backward()
            torch.addmm(bias1, A2, aw_param).sum().backward()
        else:
            F.linear(A1, w_param).sum().backward()
            F.linear(A2, aw_param).sum().backward()

        grad_sqnr = compute_error(w_param.grad, aw_param.grad)
        assert grad_sqnr > 20, (
            f"Activation QAT grad SQNR too low for {op_name} batch={batch_size} ({grad_sqnr:.1f} dB)"
        )

    # --- Meta weights ---

    def test_meta_weights(self, weight_config):
        """Prepare on meta device → copy_ real weights → forward works without extra code."""
        with torch.device("meta"):
            w = torch.randn(64, 128)
            wrapper = Float8FakeQuantizedWeightWrapperTensor(w, weight_config=weight_config)
        # load_state_dict / copy_ real weights into the meta wrapper
        real_w = torch.randn(64, 128)
        wrapper.copy_(real_w)
        # Forward should work after loading real data
        A = torch.randn(16, 64)
        out = torch.mm(A, wrapper)
        assert out.shape == (16, 128)

    # --- Standalone op shape tests ---

    def _make_wrapper(self, weight_config):
        w = torch.randn(64, 128)
        return Float8FakeQuantizedWeightWrapperTensor(w, weight_config=weight_config)

    def test_op_mm(self, weight_config):
        wrapper = self._make_wrapper(weight_config)
        A = torch.randn(16, 64)
        out = torch.mm(A, wrapper)
        assert out.shape == (16, 128)

    def test_op_bmm(self, weight_config):
        w = torch.randn(4, 64, 128)
        wrapper = Float8FakeQuantizedWeightWrapperTensor(w, weight_config=weight_config)
        A = torch.randn(4, 16, 64)
        out = torch.bmm(A, wrapper)
        assert out.shape == (4, 16, 128)

    def test_op_linear(self, weight_config):
        w = torch.randn(128, 64)
        wrapper = Float8FakeQuantizedWeightWrapperTensor(w, weight_config=weight_config)
        A = torch.randn(16, 64)
        out = F.linear(A, wrapper)
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

    def test_bias_bypass(self, weight_config):
        """Wrapped bias in addmm / F.linear is unconditionally bypassed."""
        w = torch.randn(64, 128)
        w_wrapped = Float8FakeQuantizedWeightWrapperTensor(w, weight_config=weight_config)
        A = torch.randn(16, 64)

        # torch.addmm: bias at args[0], weight at args[2] — bias passes through
        bias = torch.randn(128)
        bias_wrapped = Float8FakeQuantizedWeightWrapperTensor(bias, weight_config=weight_config)
        out_wrapped = torch.addmm(bias_wrapped, A, w_wrapped)
        out_ref = torch.addmm(bias, A, w_wrapped)
        assert torch.equal(out_wrapped, out_ref), "addmm bias should not be fake-quantized"

        # F.linear: bias at args[2], weight at args[1] — bias passes through
        w2 = torch.randn(128, 64)
        w2_wrapped = Float8FakeQuantizedWeightWrapperTensor(w2, weight_config=weight_config)
        bias2 = torch.randn(128)
        bias2_wrapped = Float8FakeQuantizedWeightWrapperTensor(bias2, weight_config=weight_config)
        out_wrapped2 = F.linear(A, w2_wrapped, bias2_wrapped)
        out_ref2 = F.linear(A, w2_wrapped, bias2)
        assert torch.equal(out_wrapped2, out_ref2), "linear bias should not be fake-quantized"

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
    # Indexing and wrapper semantics
    # =========================================================================

    def test_slice_preserves_wrapper(self, moe_model, weight_config):
        from torchao.prototype.moe_qat.tensor import (
    FakeQuantizedWeightWrapperBaseTensor,
    Float8FakeQuantizedWeightWrapperTensor,
)

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

    def test_index_tensor_preserves_wrapper(self, weight_config):
        w = torch.randn(4, 64, 128)
        wrapper = Float8FakeQuantizedWeightWrapperTensor(w, weight_config=weight_config)
        ids = torch.tensor([0, 2])
        result = wrapper[ids]
        assert isinstance(result, Float8FakeQuantizedWeightWrapperTensor)
        assert result.shape == (2, 64, 128)

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

    def test_t_preserves_wrapper(self, weight_config):
        w = torch.randn(64, 128)
        wrapper = Float8FakeQuantizedWeightWrapperTensor(w, weight_config=weight_config)
        transposed = wrapper.t()
        assert isinstance(transposed, Float8FakeQuantizedWeightWrapperTensor)
        assert transposed.shape == (128, 64)

    def test_wrapper_repr(self, weight_config):
        w = torch.randn(64, 128)
        # Base class
        base = FakeQuantizedWeightWrapperBaseTensor(w, weight_config=weight_config)
        r = repr(base)
        assert "FakeQuantizedWeightWrapperBaseTensor" in r
        assert "weight_config" in r
        # Subclass
        f8 = Float8FakeQuantizedWeightWrapperTensor(w, weight_config=weight_config)
        r = repr(f8)
        assert "Float8FakeQuantizedWeightWrapperTensor" in r
        assert "weight_config" in r

    def test_copy_inplace_semantics(self, weight_config):
        """copy_ should return the original wrapper (x.copy_(y) is x)."""
        w1 = torch.randn(64, 128)
        w2 = torch.randn(64, 128)
        target = Float8FakeQuantizedWeightWrapperTensor(w1, weight_config=weight_config)
        src = Float8FakeQuantizedWeightWrapperTensor(w2, weight_config=weight_config)
        result = target.copy_(src)
        assert result is target, "copy_ should return self"
        assert torch.equal(target.to_tensor(), src.to_tensor()), "copy_ should update data"

    # =========================================================================
    # Activation QAT
    # =========================================================================

    def test_activation_qat(self, moe_model, weight_config, device):
        """Activation QAT should meet SQNR threshold vs weight-only QAT baseline."""
        from torchao.float8.float8_utils import compute_error

        act_config = Float8FakeQuantizeConfig(dtype=torch.float8_e4m3fn, granularity=PerTensor())

        # Weight-only QAT baseline
        ref_model = copy.deepcopy(moe_model)
        self._prepare(ref_model, weight_config)

        # Activation+weight QAT
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

        # Weight-only QAT baseline
        w_only = Float8FakeQuantizedWeightWrapperTensor(w.clone(), weight_config=weight_config)
        w_param = nn.Parameter(w_only)

        # Activation+weight QAT
        wrapper = Float8FakeQuantizedWeightWrapperTensor(
            w, activation_config=act_config, weight_config=weight_config
        )
        param = nn.Parameter(wrapper)

        # Forward SQNR vs weight-only QAT
        with torch.no_grad():
            ref_out = torch.mm(A, w_param)
            out = torch.mm(A, param)
        fwd_sqnr = compute_error(ref_out, out)
        assert fwd_sqnr != float("inf"), "Activation QAT should change forward output"
        assert fwd_sqnr > 10, f"Activation QAT forward SQNR too low vs weight-only ({fwd_sqnr:.1f} dB)"

        # Gradient SQNR vs weight-only QAT via sum loss (recompute without no_grad).
        # Clone A to make the independence of the two backward passes explicit.
        torch.mm(A.clone(), w_param).sum().backward()
        torch.mm(A.clone(), param).sum().backward()
        grad_sqnr = compute_error(w_param.grad, param.grad)
        assert grad_sqnr > 10, f"Activation QAT grad SQNR too low vs weight-only ({grad_sqnr:.1f} dB)"

        # Multi-step gradient flow
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
    # Config validation
    # =========================================================================

    def test_config_step_validation(self):
        weight_config = Float8FakeQuantizeConfig(dtype=torch.float8_e4m3fn, granularity=PerRow())
        MoEQATConfig(weight_config=weight_config, step="prepare")
        MoEQATConfig(step="convert")
        with pytest.raises(ValueError, match="`step` must be one of"):
            MoEQATConfig(weight_config=weight_config, step="blah")

    def test_config_requires_weight_config(self):
        with pytest.raises(ValueError, match="Must specify"):
            MoEQATConfig(step="prepare")

    def test_config_rejects_base_config_in_convert(self):
        from torchao.quantization import Float8DynamicActivationFloat8WeightConfig
        weight_config = Float8FakeQuantizeConfig(dtype=torch.float8_e4m3fn, granularity=PerRow())
        MoEQATConfig(weight_config=weight_config, step="prepare")
        base_config = Float8DynamicActivationFloat8WeightConfig()
        with pytest.raises(NotImplementedError, match="Applying PTQ"):
            MoEQATConfig(base_config=base_config, step="convert")

    def test_config_infer_from_base_config(self):
        """MoEQATConfig can infer fake quantize configs from a PTQ base_config."""
        from torchao.quantization import Float8DynamicActivationFloat8WeightConfig
        base_config = Float8DynamicActivationFloat8WeightConfig()
        qat_config = MoEQATConfig(base_config=base_config, step="prepare")
        assert isinstance(qat_config.weight_config, Float8FakeQuantizeConfig)
        assert isinstance(qat_config.activation_config, Float8FakeQuantizeConfig)
        assert qat_config.base_config is None, "base_config should be cleared after inference"

    def test_config_prepare_with_base_config(self, moe_model):
        """Model can be prepared using base_config instead of explicit weight_config."""
        from torchao.quantization import Float8DynamicActivationFloat8WeightConfig
        base_config = Float8DynamicActivationFloat8WeightConfig()
        qat_config = MoEQATConfig(
            base_config=base_config,
            step="prepare",
            params_filter_fn=TestFloat8MoEQAT._expert_weight_filter,
        )
        model = copy.deepcopy(moe_model)
        quantize_(model, qat_config, filter_fn=lambda m, fqn: isinstance(m, MoE))
        from torchao.prototype.moe_qat.tensor import (
    FakeQuantizedWeightWrapperBaseTensor,
    Float8FakeQuantizedWeightWrapperTensor,
)
        wrapped = sum(
            1 for _, p in model.named_parameters()
            if isinstance(p.data, Float8FakeQuantizedWeightWrapperTensor)
        )
        assert wrapped == 3, f"Expected 3 wrapped params, got {wrapped}"

    def test_config_rejects_invalid_base_config(self):
        """Only Float8DynamicActivationFloat8WeightConfig is accepted as base_config."""
        from torchao.quantization import Int4WeightOnlyConfig
        base_config = Int4WeightOnlyConfig(group_size=32)
        with pytest.raises(ValueError, match="Only `Float8DynamicActivationFloat8WeightConfig`"):
            MoEQATConfig(base_config=base_config, step="prepare")

    def test_config_rejects_non_float8_config(self):
        from torchao.quantization.qat import IntxFakeQuantizeConfig
        intx_config = IntxFakeQuantizeConfig(torch.int8, "per_channel")
        with pytest.raises(ValueError, match="Only `Float8FakeQuantizeConfig`"):
            MoEQATConfig(weight_config=intx_config, step="prepare")

    @pytest.mark.parametrize("granularity", [PerRow(), PerTensor()])
    @pytest.mark.parametrize("dtype", [torch.float8_e4m3fn, torch.float8_e5m2])
    def test_weight_config_variants(self, granularity, dtype):
        """All Float8FakeQuantizeConfig variants should be accepted."""
        config = Float8FakeQuantizeConfig(dtype=dtype, granularity=granularity)
        qat_config = MoEQATConfig(weight_config=config, step="prepare")
        assert qat_config.step == QATStep.PREPARE

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

    # =========================================================================
    # Filter behavior
    # =========================================================================

    def test_default_params_filter(self, moe_model, weight_config):
        """Default filter (_is_parameter) wraps all parameters including 2D gate."""
        qat_config = MoEQATConfig(weight_config=weight_config, step="prepare")
        quantize_(moe_model, qat_config, filter_fn=lambda m, fqn: isinstance(m, MoE))
        from torchao.prototype.moe_qat.tensor import (
    FakeQuantizedWeightWrapperBaseTensor,
    Float8FakeQuantizedWeightWrapperTensor,
)

        wrapped_count = 0
        for name, param in moe_model.named_parameters():
            if isinstance(param.data, Float8FakeQuantizedWeightWrapperTensor):
                wrapped_count += 1
        # MoE has 4 params: experts.w1,w2,w3 (3D) + router.gate.weight (2D)
        assert wrapped_count == 4, f"Expected 4 wrapped params, got {wrapped_count}"
