# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD 3-Clause license found in the
# LICENSE file in the root directory of this source tree.

import importlib

import pytest
import torch
from torch._subclasses.fake_tensor import FakeTensorMode
from torch.fx.experimental.symbolic_shapes import ShapeEnv, has_free_symbols

pytest.importorskip("triton", reason="Triton required for blockwise FP8 modules")

from torchao.float8.config import e4m3_dtype
from torchao.prototype.blockwise_fp8_training import (
    deepgemm_grouped_kernels,
    grouped_kernels,
)
from torchao.prototype.blockwise_fp8_training.deepgemm_metadata import (
    build_deepgemm_grouped_offset_plan,
    group_sizes_from_offsets,
)
from torchao.quantization.quantize_.common import KernelPreference
from torchao.quantization.utils import compute_error
from torchao.utils import is_sm_at_least_90


@pytest.fixture(autouse=True)
def _clear_deepgemm_capabilities():
    deepgemm_grouped_kernels._clear_deepgemm_capability_cache()
    yield
    deepgemm_grouped_kernels._clear_deepgemm_capability_cache()


def _make_column_major_weight_t(E: int, N: int, K: int) -> torch.Tensor:
    weight = torch.randn(E, N, K, dtype=torch.bfloat16, device="cuda")
    return weight.contiguous().transpose(-2, -1)


def _hide_deep_gemm(monkeypatch):
    real_import_module = importlib.import_module

    def fake_import_module(name, *args, **kwargs):
        if name == "deep_gemm":
            raise ModuleNotFoundError(
                "No module named 'deep_gemm'",
                name="deep_gemm",
            )
        return real_import_module(name, *args, **kwargs)

    monkeypatch.setattr(importlib, "import_module", fake_import_module)


def test_deepgemm_backend_requires_optional_dependency(monkeypatch):
    _hide_deep_gemm(monkeypatch)

    with pytest.raises(ImportError, match="DeepGEMM backend selected"):
        deepgemm_grouped_kernels._require_deep_gemm()


@pytest.mark.parametrize(
    "exc",
    [
        ModuleNotFoundError("No module named 'cutlass'", name="cutlass"),
        ImportError("undefined symbol: _ZN3c104impl3cow23materialize_cow_storage"),
    ],
)
def test_deepgemm_backend_reports_broken_install(monkeypatch, exc):
    real_import_module = importlib.import_module

    def fake_import_module(name, *args, **kwargs):
        if name == "deep_gemm":
            raise exc
        return real_import_module(name, *args, **kwargs)

    monkeypatch.setattr(importlib, "import_module", fake_import_module)

    with pytest.raises(ImportError, match="active PyTorch and CUDA environment"):
        deepgemm_grouped_kernels._require_deep_gemm()


def test_auto_backend_selection_falls_back_without_deepgemm(monkeypatch):
    from torchao.prototype.moe_training.blockwise_fp8.grouped_mm_backend import (
        _GroupedMMBackendKind,
        _select_fp8_blockwise_grouped_mm_backend,
    )

    _hide_deep_gemm(monkeypatch)

    class CudaLikeTensor:
        is_cuda = True

    backend = _select_fp8_blockwise_grouped_mm_backend(
        KernelPreference.AUTO,
        CudaLikeTensor(),
        torch.bfloat16,
        128,
        torch.tensor([128], dtype=torch.int32),
    )

    assert backend.kind == _GroupedMMBackendKind.EMULATED


def test_auto_backend_selection_requires_full_deepgemm_training_symbols(monkeypatch):
    from torchao.prototype.moe_training.blockwise_fp8.grouped_mm_backend import (
        _GroupedMMBackendKind,
        _select_fp8_blockwise_grouped_mm_backend,
    )

    class DeepGemmWithoutKGroupedSymbol:
        def m_grouped_fp8_gemm_nt_contiguous(self):
            raise AssertionError("should not be called")

    real_import_module = importlib.import_module

    def fake_import_module(name, *args, **kwargs):
        if name == "deep_gemm":
            return DeepGemmWithoutKGroupedSymbol()
        return real_import_module(name, *args, **kwargs)

    monkeypatch.setattr(importlib, "import_module", fake_import_module)

    class CudaLikeTensor:
        is_cuda = True

    backend = _select_fp8_blockwise_grouped_mm_backend(
        KernelPreference.AUTO,
        CudaLikeTensor(),
        torch.bfloat16,
        128,
        torch.tensor([128], dtype=torch.int32),
    )

    assert backend.kind == _GroupedMMBackendKind.EMULATED


def test_auto_backend_selection_prefers_deepgemm_when_training_supported(monkeypatch):
    from torchao.prototype.moe_training.blockwise_fp8.grouped_mm_backend import (
        _GroupedMMBackendKind,
        _select_fp8_blockwise_grouped_mm_backend,
    )

    class DeepGemmWithTrainingSymbols:
        def m_grouped_fp8_gemm_nt_contiguous(self):
            raise AssertionError("should not be called")

        def k_grouped_fp8_gemm_nt_contiguous(self):
            raise AssertionError("should not be called")

    real_import_module = importlib.import_module

    def fake_import_module(name, *args, **kwargs):
        if name == "deep_gemm":
            return DeepGemmWithTrainingSymbols()
        return real_import_module(name, *args, **kwargs)

    monkeypatch.setattr(importlib, "import_module", fake_import_module)
    monkeypatch.setattr(
        deepgemm_grouped_kernels,
        "_is_cuda_sm90_or_newer",
        lambda _: True,
    )

    backend = _select_fp8_blockwise_grouped_mm_backend(
        KernelPreference.AUTO,
        torch.empty(1),
        torch.bfloat16,
        128,
        torch.tensor([128, 256], dtype=torch.int32),
        num_rows=384,
    )

    assert backend.kind == _GroupedMMBackendKind.DEEPGEMM
    layout = backend.offset_plan.grouped_layout
    assert torch.equal(layout[:128], torch.full((128,), 0, dtype=torch.int32))
    assert torch.equal(layout[128:256], torch.full((128,), 1, dtype=torch.int32))
    assert torch.equal(layout[256:], torch.full((128,), -1, dtype=torch.int32))


def test_deepgemm_grouped_layout_from_padded_offsets():
    original_group_end_offsets = torch.tensor([129, 384, 500], dtype=torch.int32)
    padded_group_start_offsets = torch.tensor([0, 256, 512], dtype=torch.int32)
    padded_group_end_offsets = torch.tensor([256, 512, 640], dtype=torch.int32)

    offset_plan = build_deepgemm_grouped_offset_plan(
        padded_group_end_offsets,
        original_group_end_offsets=original_group_end_offsets,
        padded_group_start_offsets=padded_group_start_offsets,
        num_rows=768,
    )
    layout = offset_plan.grouped_layout

    assert layout.dtype == torch.int32
    assert layout.is_contiguous()
    assert torch.equal(layout[:129], torch.full((129,), 0, dtype=torch.int32))
    assert torch.equal(layout[129:256], torch.full((127,), -1, dtype=torch.int32))
    assert torch.equal(layout[256:511], torch.full((255,), 1, dtype=torch.int32))
    assert torch.equal(layout[511:512], torch.full((1,), -1, dtype=torch.int32))
    assert torch.equal(layout[512:628], torch.full((116,), 2, dtype=torch.int32))
    assert torch.equal(layout[628:], torch.full((140,), -1, dtype=torch.int32))


@pytest.mark.skipif(
    not torch.cuda.is_available() or not is_sm_at_least_90(),
    reason="DeepGEMM FP8 kernels require CUDA SM90+",
)
def test_grouped_weight_quant_layouts_match_dense_per_expert_quantizers():
    from torchao.prototype.blockwise_fp8_training.grouped_weight_quant import (
        triton_fp8_blockwise_weight_quant_grouped_dgrad_rhs,
        triton_fp8_blockwise_weight_quant_grouped_forward_rhs,
    )
    from torchao.prototype.blockwise_fp8_training.kernels import (
        triton_fp8_blockwise_weight_quant_rhs,
        triton_fp8_blockwise_weight_quant_transposed_rhs,
    )

    def stack_per_expert_quant(quant_fn, weight):
        q_parts = []
        scale_parts = []
        for expert_weight in weight:
            q, scale = quant_fn(expert_weight)
            q_parts.append(q.transpose(-2, -1).contiguous())
            scale_parts.append(scale.transpose(-2, -1).contiguous())
        return torch.stack(q_parts), torch.stack(scale_parts)

    torch.manual_seed(123)
    E, N, K = 3, 256, 384
    B_t = _make_column_major_weight_t(E, N, K)
    weight = B_t.transpose(-2, -1).contiguous()

    grouped_fwd_q, grouped_fwd_s = (
        triton_fp8_blockwise_weight_quant_grouped_forward_rhs(B_t)
    )
    ref_fwd_q, ref_fwd_s = stack_per_expert_quant(
        triton_fp8_blockwise_weight_quant_transposed_rhs,
        weight,
    )

    assert grouped_fwd_q.is_contiguous()
    assert grouped_fwd_s.is_contiguous()
    assert torch.equal(grouped_fwd_q, ref_fwd_q)
    torch.testing.assert_close(
        grouped_fwd_s,
        ref_fwd_s,
        rtol=0,
        atol=0,
    )

    grouped_dgrad_q, grouped_dgrad_s = (
        triton_fp8_blockwise_weight_quant_grouped_dgrad_rhs(B_t)
    )
    ref_dgrad_q, ref_dgrad_s = stack_per_expert_quant(
        triton_fp8_blockwise_weight_quant_rhs,
        weight,
    )

    assert grouped_dgrad_q.is_contiguous()
    assert grouped_dgrad_s.is_contiguous()
    assert torch.equal(grouped_dgrad_q, ref_dgrad_q)
    torch.testing.assert_close(
        grouped_dgrad_s,
        ref_dgrad_s,
        rtol=0,
        atol=0,
    )


@pytest.mark.skipif(
    not torch.cuda.is_available() or not is_sm_at_least_90(),
    reason="DeepGEMM FP8 kernels require CUDA SM90+",
)
def test_grouped_weight_quant_supports_compile_fullgraph():
    from torch._dynamo.testing import CompileCounterWithBackend

    from torchao.prototype.blockwise_fp8_training.grouped_weight_quant import (
        triton_fp8_blockwise_weight_quant_grouped_forward_rhs,
    )

    torch._dynamo.reset()
    torch.manual_seed(123)
    B_t = _make_column_major_weight_t(E=1, N=128, K=128)
    compiled_frame_counter = CompileCounterWithBackend("inductor")

    def fn(weight_t):
        return triton_fp8_blockwise_weight_quant_grouped_forward_rhs(weight_t)

    compiled_fn = torch.compile(
        fn,
        backend=compiled_frame_counter,
        fullgraph=True,
    )

    q, scale = compiled_fn(B_t)
    assert q.shape == (1, 128, 128)
    assert q.dtype == e4m3_dtype
    assert scale.shape == (1, 1, 1)
    assert scale.dtype == torch.float32
    assert compiled_frame_counter.frame_count == 1


def test_deepgemm_k_grouped_activation_quant_fake_contract_tracks_valid_tokens():
    from torchao.prototype.blockwise_fp8_training.deepgemm_quant import (
        triton_fp8_blockwise_act_quant_k_grouped_deepgemm,
    )

    x = torch.empty((896, 4096), dtype=torch.bfloat16)
    offs = torch.tensor([256, 512, 640], dtype=torch.int32)
    shape_env = ShapeEnv(allow_dynamic_output_shape_ops=True)

    with FakeTensorMode(shape_env=shape_env) as mode:
        x_fake = mode.from_tensor(x)
        offs_fake = mode.from_tensor(offs)
        q, scale = triton_fp8_blockwise_act_quant_k_grouped_deepgemm(
            x_fake,
            offs_fake,
        )

    assert q.dtype == e4m3_dtype
    assert scale.dtype == torch.float32
    assert scale.shape[0] == x.shape[1]
    assert has_free_symbols(q.shape[0])
    assert has_free_symbols(scale.shape[1])


def test_prepare_fp8_blockwise_grouped_mm_plan_materializes_host_metadata():
    from torchao.prototype.moe_training.blockwise_fp8.grouped_mm import (
        prepare_fp8_blockwise_grouped_mm_plan,
    )

    offs = torch.tensor([128, 384, 512], dtype=torch.int32)
    plan = prepare_fp8_blockwise_grouped_mm_plan(offs)

    assert plan.groups_block_aligned_by_construction
    assert plan.__dict__["group_sizes"] == [128, 256, 128]
    assert torch.equal(
        plan.__dict__["ks_tensor"],
        torch.tensor([128, 256, 128], dtype=torch.int32),
    )


def test_deepgemm_grouped_custom_ops_fake_contracts():
    shape_env = ShapeEnv(allow_dynamic_output_shape_ops=True)
    with FakeTensorMode(shape_env=shape_env) as mode:
        a = mode.from_tensor(torch.empty((256, 128), dtype=e4m3_dtype))
        b = mode.from_tensor(torch.empty((2, 384, 128), dtype=e4m3_dtype))
        a_s = mode.from_tensor(torch.empty((2, 1), dtype=torch.float32))
        b_s = mode.from_tensor(torch.empty((2, 3, 1), dtype=torch.float32))
        grouped_layout = mode.from_tensor(torch.empty((256,), dtype=torch.int32))
        out = deepgemm_grouped_kernels._deepgemm_blockwise_scaled_grouped_mm_custom_op(
            a,
            b,
            a_s,
            b_s,
            grouped_layout,
            torch.bfloat16,
            128,
        )

        wgrad_a = mode.from_tensor(torch.empty((256 * 384,), dtype=e4m3_dtype))
        wgrad_a_s = mode.from_tensor(torch.empty((384, 2), dtype=torch.float32))
        wgrad_b = mode.from_tensor(torch.empty((256 * 512,), dtype=e4m3_dtype))
        wgrad_b_s = mode.from_tensor(torch.empty((512, 2), dtype=torch.float32))
        ks_tensor = mode.from_tensor(torch.tensor([128, 128], dtype=torch.int32))
        wgrad = deepgemm_grouped_kernels._deepgemm_blockwise_scaled_grouped_mm_wgrad_custom_op(
            wgrad_a,
            wgrad_a_s,
            wgrad_b,
            wgrad_b_s,
            ks_tensor,
            [128, 128],
            torch.bfloat16,
            128,
        )

    assert out.shape == (256, 384)
    assert out.dtype == torch.bfloat16
    assert wgrad.shape == (2, 384, 512)
    assert wgrad.dtype == torch.bfloat16


@pytest.mark.skipif(
    not torch.cuda.is_available() or not is_sm_at_least_90(),
    reason="DeepGEMM FP8 kernels require CUDA SM90+",
)
@pytest.mark.parametrize(
    "offsets",
    [
        [256, 640, 768],
        [256, 512, 768],
        [128, 384, 512, 768, 1152, 1280, 1664, 2048],
    ],
)
def test_deepgemm_k_grouped_activation_quant_matches_flattened_torchao_layouts(
    offsets,
):
    from torchao.prototype.blockwise_fp8_training.deepgemm_quant import (
        triton_fp8_blockwise_act_quant_k_grouped_deepgemm,
    )
    from torchao.prototype.blockwise_fp8_training.kernels import (
        triton_fp8_blockwise_act_quant_rhs,
        triton_fp8_blockwise_act_quant_transposed_lhs,
    )

    torch.manual_seed(123)
    offs = torch.tensor(offsets, dtype=torch.int32, device="cuda")
    group_sizes = group_sizes_from_offsets(offs)
    x = torch.randn(offsets[-1], 384, dtype=torch.bfloat16, device="cuda")

    direct_q, direct_s = triton_fp8_blockwise_act_quant_k_grouped_deepgemm(
        x,
        offs,
    )

    x_t_q, x_t_s = triton_fp8_blockwise_act_quant_transposed_lhs(x)
    expected_from_lhs = deepgemm_grouped_kernels._flatten_k_grouped_transposed_lhs(
        x_t_q,
        group_sizes,
    )
    assert torch.equal(direct_q, expected_from_lhs)
    torch.testing.assert_close(direct_s, x_t_s.contiguous(), rtol=0, atol=0)

    x_rhs_q, x_rhs_s = triton_fp8_blockwise_act_quant_rhs(x)
    expected_from_rhs = deepgemm_grouped_kernels._flatten_k_grouped_rhs(
        x_rhs_q,
        group_sizes,
    )
    assert torch.equal(direct_q, expected_from_rhs)
    torch.testing.assert_close(
        direct_s,
        x_rhs_s.transpose(-2, -1).contiguous(),
        rtol=0,
        atol=0,
    )


@pytest.mark.skipif(
    not torch.cuda.is_available() or not is_sm_at_least_90(),
    reason="DeepGEMM FP8 kernels require CUDA SM90+",
)
def test_deepgemm_k_grouped_wgrad_matches_emulated():
    pytest.importorskip("deep_gemm", reason="DeepGEMM is an optional dependency")

    from torchao.prototype.blockwise_fp8_training.kernels import (
        BLOCKWISE_1X128_SCALING_TYPE,
        _scaling_type_value,
        triton_fp8_blockwise_act_quant_rhs,
        triton_fp8_blockwise_act_quant_transposed_lhs,
    )

    torch.manual_seed(123)
    offs = torch.tensor([256, 640, 768], dtype=torch.int32, device="cuda")
    offset_plan = build_deepgemm_grouped_offset_plan(offs)
    M = int(offs[-1].item())
    N, K = 256, 384
    grad_output = torch.randn(M, N, dtype=torch.bfloat16, device="cuda")
    A = torch.randn(M, K, dtype=torch.bfloat16, device="cuda")

    grad_output_t_fp8, grad_output_t_scale = (
        triton_fp8_blockwise_act_quant_transposed_lhs(grad_output.contiguous())
    )
    A_rhs_fp8, A_rhs_scale = triton_fp8_blockwise_act_quant_rhs(A.contiguous())

    ref = grouped_kernels.emulated_blockwise_scaled_grouped_mm(
        grad_output_t_fp8,
        A_rhs_fp8,
        grad_output_t_scale,
        _scaling_type_value(BLOCKWISE_1X128_SCALING_TYPE),
        A_rhs_scale,
        _scaling_type_value(BLOCKWISE_1X128_SCALING_TYPE),
        offs,
        torch.bfloat16,
        128,
    )
    plan = deepgemm_grouped_kernels.prepare_deepgemm_wgrad_plan(
        grad_output,
        A,
        offset_plan,
        128,
        e4m3_dtype,
    )
    assert plan is not None
    out = deepgemm_grouped_kernels.deepgemm_blockwise_scaled_grouped_mm_wgrad(
        plan.lhs,
        plan.rhs,
        offset_plan,
        torch.bfloat16,
        128,
    )

    assert out.shape == ref.shape
    assert out.dtype == torch.bfloat16
    assert compute_error(ref, out) >= 35.0


@pytest.mark.skipif(
    not torch.cuda.is_available() or not is_sm_at_least_90(),
    reason="DeepGEMM FP8 kernels require CUDA SM90+",
)
@pytest.mark.parametrize(
    "offs,pad_token_groups_for_grouped_mm",
    [
        (torch.tensor([256, 512], dtype=torch.int32), False),
        (torch.tensor([129, 384, 500], dtype=torch.int32), True),
    ],
)
def test_deepgemm_matches_emulated_fp8_grouped_mm(
    offs, pad_token_groups_for_grouped_mm
):
    pytest.importorskip("deep_gemm", reason="DeepGEMM is an optional dependency")

    from torchao.prototype.moe_training.blockwise_fp8.grouped_mm import (
        _to_fp8_blockwise_then_emulated_scaled_grouped_mm,
        _to_fp8_blockwise_then_scaled_grouped_mm,
    )

    torch.manual_seed(123)
    offs = offs.cuda()
    E = offs.numel()
    M = int(offs[-1].item())
    K, N = 256, 256
    A = torch.randn(M, K, dtype=torch.bfloat16, device="cuda", requires_grad=True)
    B_t = _make_column_major_weight_t(E, N, K).requires_grad_(True)

    A_ref = A.detach().clone().requires_grad_(True)
    B_t_ref = B_t.detach().clone().requires_grad_(True)

    out = _to_fp8_blockwise_then_scaled_grouped_mm(
        A,
        B_t,
        offs,
        kernel_preference=KernelPreference.AUTO,
        pad_token_groups_for_grouped_mm=pad_token_groups_for_grouped_mm,
    )
    ref = _to_fp8_blockwise_then_emulated_scaled_grouped_mm(
        A_ref,
        B_t_ref,
        offs,
        pad_token_groups_for_grouped_mm=pad_token_groups_for_grouped_mm,
    )

    assert out.shape == ref.shape
    assert out.dtype == torch.bfloat16
    assert compute_error(ref, out) >= 35.0

    out.float().square().mean().backward()
    ref.float().square().mean().backward()

    assert compute_error(A_ref.grad, A.grad) >= 35.0
    assert compute_error(B_t_ref.grad, B_t.grad) >= 35.0
