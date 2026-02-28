# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD 3-Clause license found in the
# LICENSE file in the root directory of this source tree.

import pytest
import torch
from torch.nn import functional as F

from torchao.prototype.mx_formats.config import ScaleCalculationMode
from torchao.utils import is_sm_version, torch_version_at_least

# We need to skip before doing any imports which would use triton, since
# triton won't be available on CPU builds and torch < 2.5
if not (
    torch_version_at_least("2.7.0")
    and torch.cuda.is_available()
    and torch.cuda.get_device_capability()[0] >= 9
):
    pytest.skip("Unsupported PyTorch version", allow_module_level=True)

pytest.importorskip("triton", reason="Triton required to run this test")

from torchao.float8.config import (
    Float8LinearConfig,
    Float8LinearRecipeName,
)
from torchao.float8.float8_linear import matmul_with_hp_or_float8_args
from torchao.float8.float8_training_tensor import LinearMMConfig
from torchao.float8.float8_utils import compute_error, tensor_to_scale, to_fp8_saturated
from torchao.prototype.moe_training.config import (
    FP8GroupedMMConfig,
    FP8GroupedMMRecipe,
    MXFP8GroupedMMConfig,
    MXFP8GroupedMMRecipe,
)
from torchao.prototype.moe_training.mxfp8_grouped_mm import (
    _emulated_mxfp8_scaled_grouped_mm_2d_2d,
    _emulated_mxfp8_scaled_grouped_mm_2d_3d,
    _to_mxfp8_then_scaled_grouped_mm,
)
from torchao.prototype.moe_training.tensor import _quantize_then_scaled_grouped_mm
from torchao.prototype.moe_training.utils import (
    _to_mxfp8_per_group_colwise,
    _to_mxfp8_per_group_rowwise,
    generate_jagged_offs,
)
from torchao.prototype.mx_formats.mx_tensor import MXTensor, to_mx
from torchao.quantization.quantize_.common import KernelPreference
from torchao.testing.utils import skip_if_rocm
from torchao.utils import is_MI300, is_MI350, is_ROCM

# Needed since changing args to function causes recompiles
torch._dynamo.config.cache_size_limit = 1000


@pytest.mark.skipif(
    True,
    reason="Skipping FP8 rowwise test pending fix for https://github.com/pytorch/ao/issues/3957",
)
@pytest.mark.parametrize("m", [4096])
@pytest.mark.parametrize("n", [8192])
@pytest.mark.parametrize("k", [5120])
@pytest.mark.parametrize("n_groups", [1, 2, 4, 8])
def test_valid_scaled_grouped_mm_2d_3d(m, n, k, n_groups):
    if is_ROCM():
        if not (is_MI300() or is_MI350()):
            pytest.skip("FP8 rowwise test requires MI300 or MI350 on ROCm")
    else:
        if not is_sm_version(9, 0):
            pytest.skip("FP8 rowwise test requires SM 9.0 on CUDA")

    out_dtype = torch.bfloat16
    device = "cuda"
    a = torch.randn(
        m * n_groups,
        k,
        device=device,
        requires_grad=True,
        dtype=torch.bfloat16,
    )
    b = torch.randn(
        n_groups,
        n,
        k,
        device=device,
        dtype=torch.bfloat16,
    )
    offs = torch.arange(m, n_groups * m + 1, m, device="cuda", dtype=torch.int32)

    # b must be transposed and in column major format.
    b_t = b.contiguous().transpose(-2, -1).requires_grad_(True)

    # Compute output.
    config = FP8GroupedMMConfig.from_recipe(FP8GroupedMMRecipe.FP8_ROWWISE)
    out = _quantize_then_scaled_grouped_mm(
        a,
        b_t,
        offs=offs,
        config=config,
    )

    # Validate result.
    ref_a = a.detach().clone().requires_grad_(True)
    ref_b_t = b_t.detach().clone().requires_grad_(True)
    ref_out = compute_reference_forward(
        out,
        ref_a,
        ref_b_t,
        n_groups,
        out_dtype,
        offs,
    )

    # Run backward pass.
    out.sum().backward()
    ref_out.sum().backward()

    # Validate gradients.
    if is_ROCM():
        # ROCm: reference vs tested path use different backends:
        # - `torch._scaled_mm` uses hipBLASLt
        # - `_quantize_then_scaled_grouped_mm` uses CK
        # Different backends can use different kernel implementations / accumulation order, so the
        # outputs can differ slightly and we need tolerance.
        # On MI300/MI325 we need rtol=atol=1e-2 for this FP8 test to pass.
        assert torch.allclose(out, ref_out, rtol=1e-2, atol=1e-2)
        assert torch.allclose(a.grad, ref_a.grad, rtol=1e-2, atol=1e-2)
        assert torch.allclose(b_t.grad, ref_b_t.grad, rtol=1e-2, atol=1e-2)
    else:
        assert torch.equal(out, ref_out)
        assert torch.equal(a.grad, ref_a.grad)
        assert torch.equal(b_t.grad, ref_b_t.grad)


@skip_if_rocm("ROCm not supported")
@pytest.mark.parametrize("m", [16, 17])
@pytest.mark.parametrize("k", [16, 18])
@pytest.mark.parametrize("n", [32, 33])
def test_K_or_N_dim_not_multiple_of_16(m, n, k):
    # - Leading dim of A doesn't have to be divisible by 16, since it will be
    # divided up into groups based on offset anyway.
    # - Trailing dim of A must be divisible by 16.
    # - Leading dim of B (n_groups) doesn't need to be divisible by 16.
    # - Last 2 dims of B must be divisible by 16.
    if n % 16 == 0 and k % 16 == 0:
        return
    device = "cuda"
    n_groups = 4
    a = torch.randn(
        m * n_groups,
        k,
        device=device,
        requires_grad=True,
        dtype=torch.bfloat16,
    )
    b = torch.randn(
        n_groups,
        n,
        k,
        device=device,
        requires_grad=True,
        dtype=torch.bfloat16,
    )

    # b must be transposed and in column major format.
    b_t = b.transpose(-2, -1)
    b_t = b_t.transpose(-2, -1).contiguous().transpose(-2, -1)

    config = MXFP8GroupedMMConfig.from_recipe(MXFP8GroupedMMRecipe.MXFP8_EMULATED_RCEIL)
    offs = torch.arange(m, n_groups * m + 1, m, device="cuda", dtype=torch.int32)

    # Compute output.
    with pytest.raises(AssertionError):
        _quantize_then_scaled_grouped_mm(a, b_t, offs=offs, config=config)


def compute_reference_forward(
    result: torch.Tensor,
    A: torch.Tensor,
    B_t: torch.Tensor,
    n_groups: int,
    out_dtype: torch.dtype,
    offs: torch.Tensor,
):
    assert result.dtype == out_dtype

    # Use official rowwise recipe as reference to ensure implementation is correct.
    float8_config = Float8LinearConfig.from_recipe_name(Float8LinearRecipeName.ROWWISE)

    # Convert A to fp8.
    A_scales = tensor_to_scale(
        A,
        float8_config.cast_config_input.target_dtype,
        scaling_granularity=float8_config.cast_config_input.scaling_granularity,
        axiswise_dim=-1,
        round_scales_to_power_of_2=float8_config.round_scales_to_power_of_2,
    )
    A_scaled = A.to(torch.float32) * A_scales
    A_fp8 = to_fp8_saturated(A_scaled, float8_config.cast_config_input.target_dtype)

    # Convert B^t to fp8.
    B_t_scales = tensor_to_scale(
        B_t,
        float8_config.cast_config_weight.target_dtype,
        scaling_granularity=float8_config.cast_config_weight.scaling_granularity,
        axiswise_dim=-2,
        round_scales_to_power_of_2=float8_config.round_scales_to_power_of_2,
    )
    B_t_scaled = B_t.to(torch.float32) * B_t_scales
    B_t_fp8 = to_fp8_saturated(
        B_t_scaled,
        float8_config.cast_config_input.target_dtype,
    )

    # Split A and result into chunks, one for each group.
    offs_cpu = offs.cpu()
    A_list, A_list_fp8, A_scale_list, result_list = [], [], [], []
    start = 0
    for i in range(n_groups):
        A_list.append(A[start : offs_cpu[i]])
        A_list_fp8.append(A_fp8[start : offs_cpu[i]])
        A_scale_list.append(A_scales[start : offs_cpu[i]])
        result_list.append(result[start : offs_cpu[i]])
        start = offs_cpu[i]

    # Validate each actual result group from the _quantize_then_scaled_grouped_mm is equal to:
    # 1. A manual _scaled_mm for the group.
    # 2. A matmul_with_hp_or_float8_args for the group (which is differentiable, and thus used to validate gradients).
    outputs = []
    list1 = list(zip(A_list_fp8, B_t_fp8, A_scale_list, B_t_scales, result_list))
    list2 = list(zip(A_list, B_t, result_list))
    for i in range(len(list1)):
        a1, b1, a1scale, b1scale, result1 = list1[i]
        ref_group_result1 = torch._scaled_mm(
            a1,
            b1,
            a1scale.reciprocal(),
            b1scale.reciprocal(),
            out_dtype=out_dtype,
            bias=None,
            use_fast_accum=float8_config.gemm_config_output.use_fast_accum,
        )
        a2, b2, result2 = list2[i]
        ref_group_result2 = matmul_with_hp_or_float8_args.apply(
            a2,
            b2,
            LinearMMConfig(),
            float8_config,
        )
        if is_ROCM():
            assert torch.allclose(result1, ref_group_result1, rtol=1e-2, atol=1e-2)
            assert torch.allclose(result2, ref_group_result2, rtol=1e-2, atol=1e-2)
        else:
            assert torch.equal(result1, ref_group_result1)
            assert torch.equal(result2, ref_group_result2)
        outputs.append(ref_group_result2)

    # Concatenate the outputs and verify the full result is correct.
    output_ref = torch.cat(outputs, dim=0)
    return output_ref


@skip_if_rocm("ROCm not supported")
@pytest.mark.parametrize("M,K,N", [(1024, 1024, 1024), (1024, 2048, 4096)])
@pytest.mark.parametrize("num_experts", (1, 8, 16))
def test_emulate_mxfp8_grouped_gemm_2d_3d(M, K, N, num_experts):
    x = torch.randn(M, K, dtype=torch.bfloat16, device="cuda")
    w = torch.randn(num_experts, N, K, dtype=torch.bfloat16, device="cuda")
    offs = generate_jagged_offs(num_experts, M)
    x_ref, w_ref, offs_ref = x.clone(), w.clone(), offs.clone()

    # Quantize inputs to mxpf8 for emulated mxfp8 scaled grouped mm
    block_size = 32
    x_scale, x_fp8 = to_mx(x, elem_dtype=torch.float8_e4m3fn, block_size=block_size)

    # To cast B_t per-expert to mxfp8 across dim1, we transpose the experts, cast along dim -1, then untranspose.
    w_scale, w_fp8 = to_mx(
        w,
        elem_dtype=torch.float8_e4m3fn,
        block_size=block_size,
    )

    ref_out = torch._grouped_mm(
        x_ref, w_ref.transpose(-2, -1), offs=offs_ref, out_dtype=torch.bfloat16
    )
    out = _emulated_mxfp8_scaled_grouped_mm_2d_3d(
        x_fp8, x_scale, w_fp8, w_scale, offs=offs, out_dtype=torch.bfloat16
    )

    sqnr = compute_error(ref_out, out)
    min_sqnr = 27.0
    assert sqnr >= min_sqnr, f"sqnr {sqnr} is too low, must be >= {min_sqnr}"


@skip_if_rocm("ROCm not supported")
@pytest.mark.parametrize("M", (1024, 4096))
@pytest.mark.parametrize("N", (1024, 4096))
@pytest.mark.parametrize("num_experts", (8, 16))
def test_emulate_mxfp8_grouped_gemm_2d_2d(M, N, num_experts):
    # Simluate 2d-2d grouped gemm grad_weight = grad_output_t @ x
    block_size = 32
    grad_out = torch.randn(M, N, dtype=torch.bfloat16, device="cuda")
    grad_out_t = grad_out.t().contiguous()
    x = torch.randn(M, N, dtype=torch.bfloat16, device="cuda")
    offs = generate_jagged_offs(num_experts, M, multiple_of=block_size)
    x_ref, grad_out_t_ref, offs_ref = x.clone(), grad_out_t.clone(), offs.clone()

    # bf16 reference grouped gemm
    ref_out = torch._grouped_mm(
        grad_out_t_ref,
        x_ref,
        offs=offs_ref,
        out_dtype=torch.bfloat16,
    )

    # mxpf8 grouped gemm
    x_scale, x_mx = to_mx(x, elem_dtype=torch.float8_e4m3fn, block_size=block_size)
    grad_out_t_mx, grad_out_t_scale = _to_mxfp8_per_group_rowwise(
        grad_out_t,
        offs=offs,
        block_size=block_size,
    )
    x_mx, x_scale = _to_mxfp8_per_group_colwise(
        x,
        offs=offs,
        block_size=block_size,
    )
    out = _emulated_mxfp8_scaled_grouped_mm_2d_2d(
        grad_out_t_mx,
        grad_out_t_scale,
        x_mx.transpose(-2, -1),  # (K, N) -> (N, K)
        x_scale.transpose(-2, -1),  # (K//block_size, N) -> (N, K//block_size)
        offs=offs,
        out_dtype=torch.bfloat16,
        block_size=block_size,
    )

    sqnr = compute_error(ref_out, out)
    min_sqnr = 27.0
    assert sqnr >= min_sqnr, f"sqnr {sqnr} is too low, must be >= {min_sqnr}"


@skip_if_rocm("ROCm not supported")
@pytest.mark.parametrize("M,K,N", [(32768, 5120, 8192), (16640, 7168, 2048)])
@pytest.mark.parametrize("num_experts", (2, 4, 8, 16))
@pytest.mark.parametrize("wgrad_with_hp", (True, False))
@pytest.mark.parametrize("use_compile", (True, False))
@pytest.mark.parametrize(
    "kernel_preference", (KernelPreference.AUTO, KernelPreference.EMULATED)
)
@pytest.mark.parametrize(
    "scale_mode", (ScaleCalculationMode.FLOOR, ScaleCalculationMode.RCEIL)
)
def test_mxfp8_grouped_gemm_with_dq_fwd_bwd(
    M,
    K,
    N,
    num_experts,
    wgrad_with_hp,
    use_compile,
    kernel_preference,
    scale_mode,
):
    # MXFP8 hardware path requires SM100
    if kernel_preference != KernelPreference.EMULATED and not is_sm_version(10, 0):
        pytest.skip(
            f"Skipping MXFP8 hardware mode tests, only supported on compute capability 10.0 and found {torch.cuda.get_device_capability()}"
        )

    block_size = 32
    x = torch.randn(M, K, dtype=torch.bfloat16, device="cuda", requires_grad=True)
    w = torch.randn(
        num_experts,
        N,
        K,
        dtype=torch.bfloat16,
        device="cuda",
    )
    w_t = w.transpose(-2, -1).requires_grad_(True)
    offs = generate_jagged_offs(num_experts, M, multiple_of=block_size)
    x_ref, w_t_ref, offs_ref = (
        x.clone().detach().requires_grad_(True),
        w_t.clone().detach().requires_grad_(True),
        offs.clone(),
    )

    # Forward
    mxfp8_gmm = (
        torch.compile(_to_mxfp8_then_scaled_grouped_mm, fullgraph=True)
        if use_compile
        else _to_mxfp8_then_scaled_grouped_mm
    )
    out = mxfp8_gmm(
        x,
        w_t,
        offs=offs,
        block_size=block_size,
        kernel_preference=kernel_preference,
        wgrad_with_hp=wgrad_with_hp,
        scale_calculation_mode=scale_mode,
    )
    ref_out = torch._grouped_mm(x_ref, w_t_ref, offs=offs_ref, out_dtype=torch.bfloat16)
    sqnr = compute_error(ref_out, out)
    min_sqnr = 27.0
    assert sqnr >= min_sqnr, f"Output sqnr {sqnr} is too low, must be >= {min_sqnr}"

    # Backward
    labels = torch.ones_like(ref_out)
    ref_loss = F.mse_loss(ref_out, labels)
    out_loss = F.mse_loss(out, labels)
    ref_loss.backward()
    out_loss.backward()

    # Check input grads
    min_input_grad_sqnr = 25.0
    sqnr = compute_error(x_ref.grad, x.grad)
    assert sqnr >= min_input_grad_sqnr, (
        f"Input grad sqnr {sqnr} is too low, must be >= {min_input_grad_sqnr}"
    )

    # Check weight grads
    min_weight_grad_sqnr = 24.0
    sqnr = compute_error(w_t_ref.grad, w_t.grad)
    assert sqnr >= min_weight_grad_sqnr, (
        f"Weight grad sqnr {sqnr} is too low, must be >= {min_weight_grad_sqnr}"
    )


def _build_prequantized_a(
    A: torch.Tensor,
    block_size: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    A_scale, A_data = to_mx(
        A,
        elem_dtype=torch.float8_e4m3fn,
        block_size=block_size,
        scaling_mode=ScaleCalculationMode.RCEIL,
    )
    return A_data, A_scale


@skip_if_rocm("ROCm not supported")
@pytest.mark.parametrize("wgrad_with_hp", (True, False))
def test_mxfp8_grouped_gemm_prequantized_tuple_matches_dynamic(
    wgrad_with_hp: bool,
):
    block_size = 32
    M, K, N, num_experts = 4096, 1024, 2048, 8
    x = torch.randn(M, K, dtype=torch.bfloat16, device="cuda", requires_grad=True)
    w = torch.randn(
        num_experts,
        N,
        K,
        dtype=torch.bfloat16,
        device="cuda",
    )
    w_t = w.transpose(-2, -1).requires_grad_(True)
    offs = generate_jagged_offs(num_experts, M, multiple_of=block_size)

    x_ref = x.detach().clone().requires_grad_(True)
    w_t_ref = w_t.detach().clone().requires_grad_(True)

    A_data, A_scale = _build_prequantized_a(
        x.detach(),
        block_size,
    )
    out = _to_mxfp8_then_scaled_grouped_mm(
        x,
        w_t,
        offs=offs,
        block_size=block_size,
        out_dtype=torch.bfloat16,
        kernel_preference=KernelPreference.EMULATED,
        wgrad_with_hp=wgrad_with_hp,
        scale_calculation_mode=ScaleCalculationMode.RCEIL,
        prequantized_A=(A_data, A_scale),
    )
    out_ref = _to_mxfp8_then_scaled_grouped_mm(
        x_ref,
        w_t_ref,
        offs=offs,
        block_size=block_size,
        out_dtype=torch.bfloat16,
        kernel_preference=KernelPreference.EMULATED,
        wgrad_with_hp=wgrad_with_hp,
        scale_calculation_mode=ScaleCalculationMode.RCEIL,
    )

    output_sqnr = compute_error(out_ref, out)
    min_output_sqnr = 60.0
    assert output_sqnr >= min_output_sqnr, (
        f"Output sqnr {output_sqnr} is too low, must be >= {min_output_sqnr}"
    )

    labels = torch.ones_like(out_ref)
    F.mse_loss(out_ref, labels).backward()
    F.mse_loss(out, labels).backward()

    input_grad_sqnr = compute_error(x_ref.grad, x.grad)
    min_input_grad_sqnr = 40.0
    assert input_grad_sqnr >= min_input_grad_sqnr, (
        f"Input grad sqnr {input_grad_sqnr} is too low, must be >= {min_input_grad_sqnr}"
    )

    weight_grad_sqnr = compute_error(w_t_ref.grad, w_t.grad)
    min_weight_grad_sqnr = 40.0
    assert weight_grad_sqnr >= min_weight_grad_sqnr, (
        f"Weight grad sqnr {weight_grad_sqnr} is too low, must be >= {min_weight_grad_sqnr}"
    )


@skip_if_rocm("ROCm not supported")
def test_mxfp8_grouped_gemm_mxtensor_activation_forward():
    block_size = 32
    M, K, N, num_experts = 4096, 1024, 2048, 8
    x = torch.randn(M, K, dtype=torch.bfloat16, device="cuda")
    w = torch.randn(
        num_experts,
        N,
        K,
        dtype=torch.bfloat16,
        device="cuda",
    )
    w_t = w.transpose(-2, -1)
    offs = generate_jagged_offs(num_experts, M, multiple_of=block_size)

    x_mx = MXTensor.to_mx(
        x,
        elem_dtype=torch.float8_e4m3fn,
        block_size=block_size,
        scaling_mode=ScaleCalculationMode.RCEIL,
        is_swizzled_scales=False,
    )
    out_mx = _to_mxfp8_then_scaled_grouped_mm(
        x_mx,
        w_t,
        offs=offs,
        block_size=block_size,
        out_dtype=torch.bfloat16,
        kernel_preference=KernelPreference.EMULATED,
        wgrad_with_hp=True,
        scale_calculation_mode=ScaleCalculationMode.RCEIL,
    )
    out_ref = _to_mxfp8_then_scaled_grouped_mm(
        x,
        w_t,
        offs=offs,
        block_size=block_size,
        out_dtype=torch.bfloat16,
        kernel_preference=KernelPreference.EMULATED,
        wgrad_with_hp=True,
        scale_calculation_mode=ScaleCalculationMode.RCEIL,
    )

    output_sqnr = compute_error(out_ref, out_mx)
    min_output_sqnr = 60.0
    assert output_sqnr >= min_output_sqnr, (
        f"Output sqnr {output_sqnr} is too low, must be >= {min_output_sqnr}"
    )


@skip_if_rocm("ROCm not supported")
def test_mxfp8_grouped_gemm_mxtensor_requires_wgrad_with_hp():
    block_size = 32
    M, K, N, num_experts = 1024, 1024, 2048, 4
    x = torch.randn(M, K, dtype=torch.bfloat16, device="cuda")
    w = torch.randn(
        num_experts,
        N,
        K,
        dtype=torch.bfloat16,
        device="cuda",
    )
    w_t = w.transpose(-2, -1)
    offs = generate_jagged_offs(num_experts, M, multiple_of=block_size)

    x_mx = MXTensor.to_mx(
        x,
        elem_dtype=torch.float8_e4m3fn,
        block_size=block_size,
        scaling_mode=ScaleCalculationMode.RCEIL,
        is_swizzled_scales=False,
    )

    with pytest.raises(AssertionError, match="wgrad_with_hp"):
        _to_mxfp8_then_scaled_grouped_mm(
            x_mx,
            w_t,
            offs=offs,
            block_size=block_size,
            out_dtype=torch.bfloat16,
            kernel_preference=KernelPreference.EMULATED,
            wgrad_with_hp=False,
            scale_calculation_mode=ScaleCalculationMode.RCEIL,
        )
