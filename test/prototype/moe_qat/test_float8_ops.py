import pytest
import torch
import torch.nn.functional as F
from torch import nn

from torchao.prototype.moe_qat.tensor import (
    FakeQuantizedWeightWrapperBaseTensor,
    Float8FakeQuantizedWeightWrapperTensor,
)
from torchao.quantization.qat.fake_quantize_config import Float8FakeQuantizeConfig
from torchao.quantization.granularity import PerTensor

from .testing_utils import _moe_input, _set_seed, device, moe_model, use_grouped_mm, weight_config


class TestFloat8MoEQAT:
    """Tests for FP8 row-wise MoE QAT — standalone ops and wrapper semantics."""

    def _make_wrapper(self, weight_config):
        w = torch.randn(64, 128)
        return Float8FakeQuantizedWeightWrapperTensor(w, weight_config=weight_config)

    # =========================================================================
    # Standalone op shape tests
    # =========================================================================

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

    # =========================================================================
    # Parametrized forward+backward SQNR (following moe_training)
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

        if op_name == "linear":
            w = torch.randn(N, K, device=device)
        elif op_name == "bmm" and batch_size is not None:
            w = torch.randn(batch_size, K, N, device=device)
        elif op_name == "addmm":
            w = torch.randn(K, N, device=device)
        else:
            w = torch.randn(K, N, device=device)

        A_ref = A.clone().detach().requires_grad_(True)
        w_ref = w.clone().detach().requires_grad_(True)

        wrapper = Float8FakeQuantizedWeightWrapperTensor(w, weight_config=weight_config)
        param = nn.Parameter(wrapper)

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
        else:
            ref_out = F.linear(A_ref, w_ref)
            out = F.linear(A, param)

        sqnr_fwd = compute_error(out, ref_out)
        assert sqnr_fwd > 15, (
            f"Forward SQNR too low for {op_name} batch={batch_size} ({sqnr_fwd:.1f} dB)"
        )

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

        w_only = Float8FakeQuantizedWeightWrapperTensor(w.clone(), weight_config=weight_config)
        w_param = nn.Parameter(w_only)

        aw = Float8FakeQuantizedWeightWrapperTensor(w, activation_config=act_config, weight_config=weight_config)
        aw_param = nn.Parameter(aw)

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
        else:
            ref_out = F.linear(A, w_param)
            out = F.linear(A.clone(), aw_param)

        fwd_sqnr = compute_error(ref_out, out)
        assert fwd_sqnr != float("inf"), "Activation QAT should change forward output"
        assert fwd_sqnr > 15, (
            f"Activation QAT forward SQNR too low for {op_name} batch={batch_size} ({fwd_sqnr:.1f} dB)"
        )

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

    # =========================================================================
    # Indexing and wrapper semantics
    # =========================================================================

    def test_slice_preserves_wrapper(self, moe_model, weight_config):
        from torchao.prototype.moe_qat import MoEQATConfig
        from torchao.quantization.quant_api import quantize_
        from .reference_moe import MoE
        from .testing_utils import _expert_weight_filter

        qat_config = MoEQATConfig(
            weight_config=weight_config,
            step="prepare",
            params_filter_fn=_expert_weight_filter,
        )
        quantize_(moe_model, qat_config, filter_fn=lambda m, fqn: isinstance(m, MoE))

        checked = 0
        for name, param in moe_model.named_parameters():
            if isinstance(param.data, Float8FakeQuantizedWeightWrapperTensor):
                sliced = param.data[0]
                assert isinstance(sliced, Float8FakeQuantizedWeightWrapperTensor)
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
    # Activation QAT edge case
    # =========================================================================

    def test_activation_qat_empty_input(self, weight_config):
        """Activation fake quant is skipped for empty tensors (expert with 0 tokens)."""
        act_config = Float8FakeQuantizeConfig(dtype=torch.float8_e4m3fn, granularity=PerTensor())
        w = torch.randn(64, 128)
        wrapper = Float8FakeQuantizedWeightWrapperTensor(w, activation_config=act_config, weight_config=weight_config)
        param = nn.Parameter(wrapper)
        A = torch.randn(0, 64)
        out = torch.mm(A, param)
        assert out.numel() == 0, "Output should be empty when input is empty"
