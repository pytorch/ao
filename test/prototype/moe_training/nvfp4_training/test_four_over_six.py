# Copyright (c) Meta Platforms, Inc. and affiliates.
# Copyright (c) 2026, NVIDIA CORPORATION.
# All rights reserved.
#
# This source code is licensed under the BSD 3-Clause license found in the
# LICENSE file in the root directory of this source tree.


import pytest
import torch

import torchao.prototype.moe_training.nvfp4_training.four_over_six as four_over_six_module
from torchao.float8.float8_utils import compute_error
from torchao.prototype.moe_training.nvfp4_training.four_over_six import (
    NVFP4FourOverSixLinear,
    four_over_six_global_encode_scale,
    four_over_six_linear,
    four_over_six_quantize,
    nvfp4_dequantize,
)
from torchao.prototype.moe_training.nvfp4_training.hadamard_cutedsl_utils import (
    cutedsl_nvfp4_kernels_available,
)
from torchao.prototype.moe_training.nvfp4_training.nvfp4_training import (
    NVFP4Linear,
    NVFP4TrainingConfig,
)
from torchao.prototype.mx_formats.kernels import f4_unpacked_to_f32, unpack_uint4
from torchao.quantization import quantize_
from torchao.utils import is_sm_at_least_100, torch_version_at_least

_skip_no_cuda = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="requires CUDA"
)
_skip_no_sm100 = pytest.mark.skipif(
    not (
        torch.cuda.is_available()
        and is_sm_at_least_100()
        and torch_version_at_least("2.10.0")
    ),
    reason="requires SM100+ and PyTorch 2.10+ (FP4 scaled_mm)",
)
_cutedsl_available = torch.cuda.is_available() and cutedsl_nvfp4_kernels_available()
_skip_no_cutedsl = pytest.mark.skipif(
    not _cutedsl_available,
    reason="requires SM100+ and the CuTe DSL runtime packages",
)


def _reference_quantize(x, global_amax, **kwargs):
    """Run the pure-PyTorch four_over_six_quantize body (the bitwise oracle)
    by disabling the CuTe DSL dispatch gate for the duration of the call."""
    orig = four_over_six_module._cutedsl_quantize_eligible
    four_over_six_module._cutedsl_quantize_eligible = lambda t: False
    try:
        return four_over_six_quantize(x, global_amax, **kwargs)
    finally:
        four_over_six_module._cutedsl_quantize_eligible = orig


def _dequantize(codes, scales, global_amax, e4m3_scale_bound):
    """Reconstruct FP32 values from packed codes, block scales, and global amax."""
    rows = codes.shape[0]
    values = f4_unpacked_to_f32(unpack_uint4(codes)).view(rows, -1, 16)
    s_dec = 1.0 / four_over_six_global_encode_scale(global_amax, e4m3_scale_bound)
    if s_dec.dim() == 1:
        s_dec = s_dec.view(rows, 1, 1)
    return (values * scales.to(torch.float32).unsqueeze(-1) * s_dec).view(rows, -1)


def _map6_reference(x, global_amax, e4m3_scale_bound):
    """Standard (map-to-6 only) encoding with the four-over-six scale chain."""
    from torchao.prototype.moe_training.nvfp4_training.four_over_six import (
        _FP32_MAX,
        FP4_E2M1_MAX,
        FP8_E4M3_MAX,
        _fp4_rtne,
    )

    rows, cols = x.shape
    xf = x.float().view(rows, cols // 16, 16)
    s_enc = four_over_six_global_encode_scale(global_amax, e4m3_scale_bound)
    fp4_max = torch.full((), FP4_E2M1_MAX, dtype=torch.float32, device=x.device)
    base = (xf.abs().amax(dim=-1) / fp4_max) * s_enc
    scale6 = base.clamp(max=FP8_E4M3_MAX).to(torch.float8_e4m3fn)
    inv6 = (1.0 / (scale6.to(torch.float32) * (1.0 / s_enc))).clamp(max=_FP32_MAX)
    _, values6 = _fp4_rtne(xf * inv6.unsqueeze(-1))
    s_dec = (1.0 / s_enc).view(-1, 1, 1) if s_enc.dim() == 1 else 1.0 / s_enc
    dequant6 = values6 * scale6.to(torch.float32).unsqueeze(-1) * s_dec
    return dequant6.view(rows, cols)


def _make_test_data(init_data, shape, dtype):
    """Deterministic input constructions for the kernel-vs-oracle tests."""
    n = shape[0] * shape[1]
    if init_data == "random":
        return torch.randn(*shape, dtype=dtype, device="cuda")
    if init_data == "boundary":
        # A linspace across the FP4 range interleaved with the same values
        # nudged outward by 1e-3, straddling every rounding-decision boundary.
        base = torch.linspace(-12.0, 12.0, n // 2, dtype=torch.float32, device="cuda")
        nudge = torch.where(base < 0, base - 1e-3, base + 1e-3)
        return torch.stack((base, nudge), dim=1).view(shape).to(dtype)
    if init_data == "maxes":
        return torch.full(shape, torch.finfo(dtype).max, dtype=dtype, device="cuda")
    if init_data == "denormal":
        # bf16 subnormals with mixed signs (randn never generates them).
        tiny = torch.finfo(torch.bfloat16).smallest_normal
        x = (torch.rand(*shape, dtype=torch.float32, device="cuda") - 0.5) * tiny
        return x.to(dtype)
    assert init_data == "negzero"
    x = torch.randn(*shape, dtype=dtype, device="cuda")
    x[::2, ::3] = -0.0
    return x


@_skip_no_cuda
@pytest.mark.parametrize("err_mode", ["mae", "mse"])
@pytest.mark.parametrize("e4m3_scale_bound", [256, 448])
@pytest.mark.parametrize("block", ["1x16", "16x16"])
def test_scales_are_candidate_scales(err_mode, e4m3_scale_bound, block):
    """Every stored block scale is one of the two candidate scales."""
    torch.manual_seed(0)
    x = torch.randn(128, 256, dtype=torch.bfloat16, device="cuda")
    amax = x.abs().amax().to(torch.float32)
    _, scales = four_over_six_quantize(
        x, amax, block=block, err_mode=err_mode, e4m3_scale_bound=e4m3_scale_bound
    )

    xf = x.float().view(128, 16, 16)
    if block == "16x16":
        tiles = x.float().abs().view(8, 16, 16, 16)
        block_amax = tiles.amax(dim=(1, 3)).repeat_interleave(16, dim=0)
    else:
        block_amax = xf.abs().amax(dim=-1)
    s_enc = four_over_six_global_encode_scale(amax, e4m3_scale_bound)
    fp4_max = torch.full((), 6.0, dtype=torch.float32, device="cuda")
    base = (block_amax / fp4_max) * s_enc
    scale6 = base.clamp(max=448.0).to(torch.float8_e4m3fn).view(torch.uint8)
    scale4 = (base * 1.5).clamp(max=448.0).to(torch.float8_e4m3fn).view(torch.uint8)
    got = scales.view(torch.uint8)
    assert ((got == scale6) | (got == scale4)).all()


@_skip_no_cuda
@pytest.mark.parametrize("e4m3_scale_bound", [256, 448])
def test_selection_not_worse_than_map6(e4m3_scale_bound):
    """Per-block MAE of the stored encoding <= the map-to-6-only encoding."""
    torch.manual_seed(0)
    x = torch.randn(256, 512, dtype=torch.bfloat16, device="cuda")
    amax = x.abs().amax().to(torch.float32)
    codes, scales = four_over_six_quantize(
        x, amax, block="1x16", err_mode="mae", e4m3_scale_bound=e4m3_scale_bound
    )
    dq = _dequantize(codes, scales, amax, e4m3_scale_bound)
    dq6 = _map6_reference(x, amax, e4m3_scale_bound)
    xf = x.float()
    err = (dq - xf).abs().view(256, -1, 16).sum(dim=-1).double()
    err6 = (dq6 - xf).abs().view(256, -1, 16).sum(dim=-1).double()
    # Selection minimizes the FP32 sequential-sum error; allow FP32-vs-FP64
    # summation slack on ties.
    assert (err <= err6 + 1e-4).all()
    # And the recipe must actually engage: some blocks pick map-to-4.
    assert (err < err6 - 1e-4).any()


@_skip_no_cuda
@pytest.mark.parametrize("err_mode", ["mae", "mse"])
@pytest.mark.parametrize("e4m3_scale_bound", [256, 448])
def test_selection_minimizes_err_mode(err_mode, e4m3_scale_bound):
    """The stored encoding's per-block error is the minimum over both
    candidates under the configured metric (the map-to-6 comparison above
    only bounds the mae side)."""
    from torchao.prototype.moe_training.nvfp4_training.four_over_six import (
        _FP32_MAX,
        FP4_E2M1_MAX,
        FP8_E4M3_MAX,
        _candidate_error,
        _fp4_rtne,
    )

    torch.manual_seed(0)
    rows, cols = 256, 512
    x = torch.randn(rows, cols, dtype=torch.bfloat16, device="cuda")
    amax = x.abs().amax().to(torch.float32)
    _, scales = four_over_six_quantize(
        x, amax, block="1x16", err_mode=err_mode, e4m3_scale_bound=e4m3_scale_bound
    )

    # Recompute both candidate encodings with the quantizer's own chain.
    xf = x.float().view(rows, cols // 16, 16)
    s_enc = four_over_six_global_encode_scale(amax, e4m3_scale_bound)
    fp4_max = torch.full((), FP4_E2M1_MAX, dtype=torch.float32, device=x.device)
    base = (xf.abs().amax(dim=-1) / fp4_max) * s_enc
    scale6 = base.clamp(max=FP8_E4M3_MAX).to(torch.float8_e4m3fn)
    scale4 = (base * 1.5).clamp(max=FP8_E4M3_MAX).to(torch.float8_e4m3fn)
    s_dec = 1.0 / s_enc
    inv6 = (1.0 / (scale6.to(torch.float32) * s_dec)).clamp(max=_FP32_MAX)
    inv4 = (1.0 / (scale4.to(torch.float32) * s_dec)).clamp(max=_FP32_MAX)
    _, values6 = _fp4_rtne(xf * inv6.unsqueeze(-1))
    _, values4 = _fp4_rtne(xf * inv4.unsqueeze(-1))
    err6 = _candidate_error(
        values6, scale6.unsqueeze(-1), xf, amax, err_mode, e4m3_scale_bound
    )
    err4 = _candidate_error(
        values4, scale4.unsqueeze(-1), xf, amax, err_mode, e4m3_scale_bound
    )

    # Every stored scale is one of the candidates, and its error is the
    # candidate minimum (equal candidate bytes encode identically, so the
    # attribution below is unambiguous).
    stored = scales.view(torch.uint8)
    scale6_u8 = scale6.view(torch.uint8)
    scale4_u8 = scale4.view(torch.uint8)
    assert ((stored == scale6_u8) | (stored == scale4_u8)).all()
    stored_err = torch.where(stored == scale4_u8, err4, err6)
    min_err = torch.where(err4 < err6, err4, err6)
    torch.testing.assert_close(stored_err, min_err, atol=0, rtol=0)
    # Both candidates must win somewhere for the check to bite.
    assert (err4 < err6).any() and (err6 < err4).any()


@_skip_no_cuda
def test_row_scaled_matches_per_row_quantization():
    """Row-scaled output == each row quantized alone with its own scalar amax."""
    torch.manual_seed(0)
    x = torch.randn(64, 256, dtype=torch.bfloat16, device="cuda")
    row_amax = x.abs().amax(dim=1).to(torch.float32)
    codes, scales = four_over_six_quantize(x, row_amax, block="1x16")
    for r in range(0, 64, 17):
        codes_r, scales_r = four_over_six_quantize(x[r : r + 1], row_amax[r].view(()))
        torch.testing.assert_close(codes[r : r + 1], codes_r, atol=0, rtol=0)
        torch.testing.assert_close(
            scales[r : r + 1].view(torch.uint8),
            scales_r.view(torch.uint8),
            atol=0,
            rtol=0,
        )


@_skip_no_cuda
def test_row_scaled_rejects_16x16():
    x = torch.randn(64, 256, dtype=torch.bfloat16, device="cuda")
    row_amax = x.abs().amax(dim=1).to(torch.float32)
    with pytest.raises(ValueError, match="1x16 blocks only"):
        four_over_six_quantize(x, row_amax, block="16x16")


@_skip_no_cuda
@pytest.mark.parametrize("block", ["1x16", "16x16"])
def test_dequant_sqnr(block):
    torch.manual_seed(0)
    x = torch.randn(128, 512, dtype=torch.bfloat16, device="cuda")
    amax = x.abs().amax().to(torch.float32)
    codes, scales = four_over_six_quantize(x, amax, block=block)
    dq = _dequantize(codes, scales, amax, 256)
    assert compute_error(x.float(), dq).item() > 14.0


@_skip_no_cuda
@pytest.mark.parametrize("block", ["1x16", "16x16"])
@pytest.mark.parametrize("row_scaled", [False, True])
def test_dequantize_roundtrip(block, row_scaled):
    """nvfp4_dequantize reconstructs the quantized values."""
    if row_scaled and block == "16x16":
        pytest.skip("row-scaled is 1x16 only")
    torch.manual_seed(0)
    x = torch.randn(128, 512, dtype=torch.bfloat16, device="cuda")
    amax = (x.abs().amax(dim=1) if row_scaled else x.abs().amax()).to(torch.float32)
    codes, scales = four_over_six_quantize(x, amax, block=block)
    dq = nvfp4_dequantize(codes, scales, amax, out_dtype=torch.float32)
    assert compute_error(x.float(), dq).item() > 14.0
    # Zero blocks reconstruct exactly: scale byte 0x00 makes the decode
    # scale exactly zero regardless of the global amax.
    x[:, :16] = 0.0
    codes, scales = four_over_six_quantize(x, amax, block=block)
    dq = nvfp4_dequantize(codes, scales, amax, out_dtype=torch.float32)
    assert (dq[:, :16] == 0.0).all()


@_skip_no_cuda
def test_dequantize_validation():
    codes = torch.zeros(32, 128, dtype=torch.uint8, device="cuda")
    scales = torch.zeros(32, 16, dtype=torch.uint8, device="cuda").view(
        torch.float8_e4m3fn
    )
    amax = torch.ones((), dtype=torch.float32, device="cuda")
    with pytest.raises(ValueError, match="e4m3_scale_bound"):
        nvfp4_dequantize(codes, scales, amax, e4m3_scale_bound=128)
    with pytest.raises(ValueError, match="scales must have shape"):
        nvfp4_dequantize(codes, scales[:, :8], amax)
    with pytest.raises(ValueError, match="row vector"):
        nvfp4_dequantize(
            codes, scales, torch.ones(7, dtype=torch.float32, device="cuda")
        )


@_skip_no_sm100
@pytest.mark.parametrize("row_scaled_activation", [False, True])
@pytest.mark.parametrize("bias", [False, True])
def test_linear_forward_backward(row_scaled_activation, bias):
    torch.manual_seed(0)
    M, K, N = 256, 512, 384
    x = torch.randn(M, K, dtype=torch.bfloat16, device="cuda", requires_grad=True)
    w = (torch.randn(N, K, dtype=torch.bfloat16, device="cuda") * 0.1).requires_grad_(
        True
    )
    b = (
        torch.randn(N, dtype=torch.bfloat16, device="cuda", requires_grad=True)
        if bias
        else None
    )
    y = four_over_six_linear(x, w, b, "mae", 256, row_scaled_activation)
    assert y.shape == (M, N)
    dy = torch.randn_like(y)
    y.backward(dy)

    y_ref = x.detach().float() @ w.detach().float().t()
    if bias:
        y_ref = y_ref + b.detach().float()
    dx_ref = dy.float() @ w.detach().float()
    dw_ref = dy.float().t() @ x.detach().float()
    assert compute_error(y_ref, y.float()).item() > 14.0
    assert compute_error(dx_ref, x.grad.float()).item() > 14.0
    assert compute_error(dw_ref, w.grad.float()).item() > 14.0
    if bias:
        # grad_bias is reduced in bf16, matching nvfp4_linear.
        torch.testing.assert_close(b.grad, dy.sum(dim=0))


@_skip_no_sm100
def test_linear_module():
    torch.manual_seed(0)
    lin = NVFP4FourOverSixLinear(512, 384, device="cuda", dtype=torch.bfloat16)
    x = torch.randn(128, 512, dtype=torch.bfloat16, device="cuda", requires_grad=True)
    y = lin(x)
    y.sum().backward()
    assert y.shape == (128, 384)
    assert lin.weight.grad is not None


@_skip_no_cuda
def test_linear_rejects_unaligned_dims():
    x = torch.randn(100, 512, dtype=torch.bfloat16, device="cuda")
    w = torch.randn(384, 512, dtype=torch.bfloat16, device="cuda")
    with pytest.raises(ValueError, match="divisible by 128"):
        four_over_six_linear(x, w, None, "mae", 256, False)


@_skip_no_sm100
@pytest.mark.parametrize("row_scaled_activation", [False, True])
def test_backward_override_high_precision(row_scaled_activation):
    """dx/dw are the plain bf16 GEMMs on the original operands."""
    torch.manual_seed(0)
    M, K, N = 256, 512, 384
    x = torch.randn(M, K, dtype=torch.bfloat16, device="cuda", requires_grad=True)
    w = (torch.randn(N, K, dtype=torch.bfloat16, device="cuda") * 0.1).requires_grad_(
        True
    )
    y = four_over_six_linear(
        x, w, None, "mae", 256, row_scaled_activation, "high_precision"
    )
    dy = torch.randn_like(y)
    y.backward(dy)
    torch.testing.assert_close(x.grad, dy @ w.detach(), atol=0, rtol=0)
    torch.testing.assert_close(w.grad, dy.t() @ x.detach(), atol=0, rtol=0)


@_skip_no_sm100
@pytest.mark.parametrize("row_scaled_activation", [False, True])
@pytest.mark.parametrize("weight_block", ["16x16", "1x16"])
def test_backward_override_dequantized(row_scaled_activation, weight_block):
    """dx/dw are bf16 GEMMs on dequantizations of the rowwise fprop operands."""
    torch.manual_seed(0)
    M, K, N = 256, 512, 384
    x = torch.randn(M, K, dtype=torch.bfloat16, device="cuda", requires_grad=True)
    w = (torch.randn(N, K, dtype=torch.bfloat16, device="cuda") * 0.1).requires_grad_(
        True
    )
    y = four_over_six_linear(
        x, w, None, "mae", 256, row_scaled_activation, "dequantized", weight_block
    )
    dy = torch.randn_like(y)
    y.backward(dy)

    x_hp, w_hp = x.detach(), w.detach()
    x_amax = (
        x_hp.abs().amax(dim=1) if row_scaled_activation else x_hp.abs().amax()
    ).to(torch.float32)
    w_amax = w_hp.abs().amax().to(torch.float32)
    x_codes, x_scales = four_over_six_quantize(x_hp, x_amax)
    w_codes, w_scales = four_over_six_quantize(w_hp, w_amax, block=weight_block)
    x_dq = nvfp4_dequantize(x_codes, x_scales, x_amax)
    w_dq = nvfp4_dequantize(w_codes, w_scales, w_amax)
    torch.testing.assert_close(x.grad, dy @ w_dq, atol=0, rtol=0)
    torch.testing.assert_close(w.grad, dy.t() @ x_dq, atol=0, rtol=0)


@_skip_no_sm100
def test_row_scaled_default_backward_is_high_precision():
    """row_scaled + backward_override=None keeps the pre-override behavior."""
    torch.manual_seed(0)
    M, K, N = 256, 512, 384
    x_hp = torch.randn(M, K, dtype=torch.bfloat16, device="cuda")
    w_hp = torch.randn(N, K, dtype=torch.bfloat16, device="cuda") * 0.1
    dy = torch.randn(M, N, dtype=torch.bfloat16, device="cuda")

    def run(override):
        x = x_hp.clone().requires_grad_(True)
        w = w_hp.clone().requires_grad_(True)
        y = four_over_six_linear(x, w, None, "mae", 256, True, override)
        y.backward(dy)
        return y.detach(), x.grad, w.grad

    y0, dx0, dw0 = run(None)
    y1, dx1, dw1 = run("high_precision")
    torch.testing.assert_close(y0, y1, atol=0, rtol=0)
    torch.testing.assert_close(dx0, dx1, atol=0, rtol=0)
    torch.testing.assert_close(dw0, dw1, atol=0, rtol=0)


@_skip_no_sm100
def test_weight_block_1x16_forward():
    """weight_block='1x16' quantizes the fprop weight with 1x16 blocks."""
    from torchao.prototype.moe_training.nvfp4_training.four_over_six import (
        _global_decode_scale,
        _scaled_mm_nvfp4,
    )

    torch.manual_seed(0)
    M, K, N = 256, 512, 384
    x = torch.randn(M, K, dtype=torch.bfloat16, device="cuda")
    w = torch.randn(N, K, dtype=torch.bfloat16, device="cuda") * 0.1
    y = four_over_six_linear(x, w, None, "mae", 256, False, None, "1x16")

    x_amax = x.abs().amax().to(torch.float32)
    w_amax = w.abs().amax().to(torch.float32)
    x_codes, x_scales = four_over_six_quantize(x, x_amax)
    w_codes, w_scales = four_over_six_quantize(w, w_amax, block="1x16")
    y_ref = _scaled_mm_nvfp4(
        x_codes,
        x_scales,
        _global_decode_scale(x_amax, 256),
        w_codes.t(),
        w_scales,
        _global_decode_scale(w_amax, 256),
        torch.bfloat16,
    )
    torch.testing.assert_close(y, y_ref, atol=0, rtol=0)


@_skip_no_cuda
def test_backward_override_validation():
    x = torch.randn(128, 256, dtype=torch.bfloat16, device="cuda")
    w = torch.randn(128, 256, dtype=torch.bfloat16, device="cuda")
    with pytest.raises(ValueError, match="no quantized backward"):
        four_over_six_linear(x, w, None, "mae", 256, True, "quantized")
    with pytest.raises(ValueError, match="backward_override"):
        four_over_six_linear(x, w, None, "mae", 256, False, "bf16")


@_skip_no_sm100
def test_linear_module_backward_override():
    lin = NVFP4FourOverSixLinear(
        512,
        384,
        backward_override="dequantized",
        weight_block="1x16",
        device="cuda",
        dtype=torch.bfloat16,
    )
    x = torch.randn(128, 512, dtype=torch.bfloat16, device="cuda", requires_grad=True)
    y = lin(x)
    y.sum().backward()
    assert y.shape == (128, 384)
    assert lin.weight.grad is not None
    assert x.grad is not None


def test_training_config_four_over_six_recipe_swap():
    model = torch.nn.Sequential(
        torch.nn.Linear(512, 384, bias=True), torch.nn.Linear(384, 512, bias=False)
    )
    weight = model[0].weight
    quantize_(
        model,
        NVFP4TrainingConfig(
            recipe="four_over_six",
            err_mode="mse",
            e4m3_scale_bound=448,
            row_scaled_activation=True,
            backward_override="dequantized",
            weight_block="1x16",
        ),
    )
    for mod in model:
        assert type(mod) is NVFP4FourOverSixLinear
        assert mod.err_mode == "mse"
        assert mod.e4m3_scale_bound == 448
        assert mod.row_scaled_activation is True
        assert mod.backward_override == "dequantized"
        assert mod.weight_block == "1x16"
    assert model[0].weight is weight
    assert model[0].bias is not None
    assert model[1].bias is None
    # Re-quantizing leaves already-converted modules alone — under the same
    # recipe and under the other one (no silent cross-recipe rewrap).
    converted = model[0]
    quantize_(model, NVFP4TrainingConfig(recipe="four_over_six"))
    assert model[0] is converted
    quantize_(model, NVFP4TrainingConfig())
    assert model[0] is converted


def test_training_config_default_recipe_swap():
    model = torch.nn.Sequential(torch.nn.Linear(512, 384, bias=False))
    quantize_(model, NVFP4TrainingConfig())
    assert type(model[0]) is NVFP4Linear
    converted = model[0]
    quantize_(model, NVFP4TrainingConfig(recipe="four_over_six"))
    assert model[0] is converted


def test_training_config_recipe_validation():
    with pytest.raises(ValueError, match="recipe must be"):
        NVFP4TrainingConfig(recipe="4over6")
    with pytest.raises(ValueError, match="err_mode configures the 'four_over_six'"):
        NVFP4TrainingConfig(err_mode="mse")
    with pytest.raises(ValueError, match="stay at its default under recipe='default'"):
        NVFP4TrainingConfig(backward_override="dequantized")
    with pytest.raises(ValueError, match="err_mode must be"):
        NVFP4TrainingConfig(recipe="four_over_six", err_mode="rmse")
    with pytest.raises(ValueError, match="e4m3_scale_bound must be"):
        NVFP4TrainingConfig(recipe="four_over_six", e4m3_scale_bound=384)
    with pytest.raises(ValueError, match="backward_override must be"):
        NVFP4TrainingConfig(recipe="four_over_six", backward_override="bf16")
    with pytest.raises(ValueError, match="weight_block must be"):
        NVFP4TrainingConfig(recipe="four_over_six", weight_block="32x32")
    with pytest.raises(ValueError, match="world_size configures the 'default'"):
        NVFP4TrainingConfig(recipe="four_over_six", world_size=2)


@_skip_no_cutedsl
@pytest.mark.parametrize("err_mode", ["mae", "mse"])
@pytest.mark.parametrize("e4m3_scale_bound", [256, 448])
@pytest.mark.parametrize("block", ["1x16", "16x16"])
@pytest.mark.parametrize("row_scaled", [False, True])
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float32])
@pytest.mark.parametrize(
    "init_data", ["random", "boundary", "maxes", "denormal", "negzero"]
)
def test_cutedsl_bitwise_matches_reference(
    err_mode, e4m3_scale_bound, block, row_scaled, dtype, init_data
):
    """CuTe DSL fast path is bitwise identical to the pure-PyTorch body.

    Shapes cover R < the kernel's 128-row tile (TMA-clipped stores), R not a
    multiple of 16 (1x16 only), multi-tile rows/columns, and R > 128 (a
    grid.y > 1 launch). The data constructions target the encode edge
    cases: rounding-boundary straddles, dtype-max saturation, bf16
    subnormals, and negative zeros.
    """
    if row_scaled and block == "16x16":
        pytest.skip("row-scaled is 1x16 only")
    shapes = [(128, 256), (64, 1024), (384, 256)]
    if block == "1x16":
        shapes.append((100, 320))
    for shape in shapes:
        torch.manual_seed(0)
        x = _make_test_data(init_data, shape, dtype)
        amax = (x.abs().amax(dim=1) if row_scaled else x.abs().amax()).to(torch.float32)
        assert four_over_six_module._cutedsl_quantize_eligible(x)
        codes, scales = four_over_six_quantize(
            x, amax, block=block, err_mode=err_mode, e4m3_scale_bound=e4m3_scale_bound
        )
        ref_codes, ref_scales = _reference_quantize(
            x, amax, block=block, err_mode=err_mode, e4m3_scale_bound=e4m3_scale_bound
        )
        torch.testing.assert_close(codes, ref_codes, atol=0, rtol=0)
        torch.testing.assert_close(
            scales.view(torch.uint8), ref_scales.view(torch.uint8), atol=0, rtol=0
        )


@_skip_no_cutedsl
@pytest.mark.parametrize("block", ["1x16", "16x16"])
def test_cutedsl_special_values(block):
    """Zeros, Inf injections, and amax==0 rows stay bitwise vs the reference."""
    torch.manual_seed(0)
    # all zeros: S_enc falls back to 1.0, zero scales, zero codes
    x = torch.zeros(64, 256, dtype=torch.bfloat16, device="cuda")
    amax = x.abs().amax().to(torch.float32)
    codes, scales = four_over_six_quantize(x, amax, block=block)
    ref_codes, ref_scales = _reference_quantize(x, amax, block=block)
    torch.testing.assert_close(codes, ref_codes, atol=0, rtol=0)
    torch.testing.assert_close(
        scales.view(torch.uint8), ref_scales.view(torch.uint8), atol=0, rtol=0
    )
    # Inf injections: block scale caps at 448, Inf encodes as +/-6, both
    # candidate errors go Inf and the tie picks map-to-6
    x = torch.randn(64, 256, dtype=torch.bfloat16, device="cuda")
    x[7, 32] = float("inf")
    x[23, 100] = float("-inf")
    amax = x.abs().amax().to(torch.float32)
    codes, scales = four_over_six_quantize(x, amax, block=block)
    ref_codes, ref_scales = _reference_quantize(x, amax, block=block)
    torch.testing.assert_close(codes, ref_codes, atol=0, rtol=0)
    torch.testing.assert_close(
        scales.view(torch.uint8), ref_scales.view(torch.uint8), atol=0, rtol=0
    )
    if block == "1x16":
        # row-scaled with amax == 0 rows over nonzero data: identity S_enc
        x = torch.randn(64, 256, dtype=torch.bfloat16, device="cuda")
        row_amax = x.abs().amax(dim=1).to(torch.float32)
        row_amax[::3] = 0.0
        codes, scales = four_over_six_quantize(x, row_amax, block=block)
        ref_codes, ref_scales = _reference_quantize(x, row_amax, block=block)
        torch.testing.assert_close(codes, ref_codes, atol=0, rtol=0)
        torch.testing.assert_close(
            scales.view(torch.uint8), ref_scales.view(torch.uint8), atol=0, rtol=0
        )


@_skip_no_cutedsl
def test_cutedsl_nan_semantics():
    """NaN handling is the kernel's one documented divergence from the
    pure-PyTorch body (torch.amax propagates NaN into the block scales
    while the kernel's fmaxf drops it): an all-NaN group gets amax 0 -> scale byte
    0x00, and NaN elements encode to +6 (satfinite), i.e. code bytes 0x77."""
    torch.manual_seed(0)
    x = torch.randn(32, 256, dtype=torch.bfloat16, device="cuda")
    x[3, 32:48] = float("nan")  # group (3, 2)
    # a NaN-free global amax, as a NaN-dropping amax kernel produces
    amax = torch.nan_to_num(x.float(), nan=0.0).abs().amax().to(torch.float32)
    codes, scales = four_over_six_quantize(x, amax, block="1x16")
    assert scales.view(torch.uint8)[3, 2].item() == 0x00
    assert (codes[3, 16:24] == 0x77).all()
    # NaN-free groups are still bitwise vs the reference
    ref_codes, ref_scales = _reference_quantize(x, amax, block="1x16")
    keep = torch.ones_like(codes, dtype=torch.bool)
    keep[3, 16:24] = False
    torch.testing.assert_close(codes[keep], ref_codes[keep], atol=0, rtol=0)


@_skip_no_cutedsl
def test_cutedsl_ineligible_falls_back():
    """Ineligible shapes/layouts silently use the pure-PyTorch body."""
    x = torch.randn(64, 272, dtype=torch.bfloat16, device="cuda")  # C % 64 != 0
    assert not four_over_six_module._cutedsl_quantize_eligible(x)
    amax = x.abs().amax().to(torch.float32)
    codes, scales = four_over_six_quantize(x, amax)
    assert codes.shape == (64, 136) and scales.shape == (64, 17)
    x_t = torch.randn(64, 256, dtype=torch.bfloat16, device="cuda").t()
    assert not four_over_six_module._cutedsl_quantize_eligible(x_t)


@_skip_no_sm100
@pytest.mark.skipif(not _cutedsl_available, reason="requires the CuTe DSL runtime")
def test_cutedsl_linear_compile():
    """torch.compile traces through the dispatch (custom op + fake impl)."""
    torch.manual_seed(0)
    x = torch.randn(128, 256, dtype=torch.bfloat16, device="cuda")
    w = torch.randn(384, 256, dtype=torch.bfloat16, device="cuda") * 0.1

    def fn(x, w):
        return four_over_six_linear(x, w, None, "mae", 256, False)

    y_eager = fn(x, w)
    y_compiled = torch.compile(fn, fullgraph=True)(x, w)
    torch.testing.assert_close(y_compiled, y_eager, atol=0, rtol=0)


@_skip_no_sm100
@pytest.mark.skipif(not _cutedsl_available, reason="requires the CuTe DSL runtime")
@pytest.mark.parametrize("backward_override", ["high_precision", "dequantized"])
def test_linear_compile_backward_overrides(backward_override):
    """fullgraph compile of the override backwards, bitwise vs eager.

    The quantize stays an opaque custom op under compile; the override
    backwards are bf16 GEMMs (on original or dequantized operands), so
    compiled gradients must match eager exactly.
    """
    torch.manual_seed(0)
    x = torch.randn(128, 256, dtype=torch.bfloat16, device="cuda")
    w = torch.randn(384, 256, dtype=torch.bfloat16, device="cuda") * 0.1

    def fn(x, w):
        return four_over_six_linear(x, w, None, "mae", 256, False, backward_override)

    x_e = x.clone().requires_grad_(True)
    w_e = w.clone().requires_grad_(True)
    y_eager = fn(x_e, w_e)
    dy = torch.randn_like(y_eager)
    y_eager.backward(dy)

    x_c = x.clone().requires_grad_(True)
    w_c = w.clone().requires_grad_(True)
    y_compiled = torch.compile(fn, fullgraph=True)(x_c, w_c)
    y_compiled.backward(dy)

    torch.testing.assert_close(y_compiled, y_eager, atol=0, rtol=0)
    torch.testing.assert_close(x_c.grad, x_e.grad, atol=0, rtol=0)
    torch.testing.assert_close(w_c.grad, w_e.grad, atol=0, rtol=0)


@_skip_no_sm100
@pytest.mark.skipif(not _cutedsl_available, reason="requires the CuTe DSL runtime")
def test_linear_compile_weight_block_1x16():
    """fullgraph compile of the 1x16-weight forward, bitwise vs eager."""
    torch.manual_seed(0)
    x = torch.randn(128, 256, dtype=torch.bfloat16, device="cuda")
    w = torch.randn(384, 256, dtype=torch.bfloat16, device="cuda") * 0.1

    def fn(x, w):
        return four_over_six_linear(x, w, None, "mae", 256, False, None, "1x16")

    y_eager = fn(x, w)
    y_compiled = torch.compile(fn, fullgraph=True)(x, w)
    torch.testing.assert_close(y_compiled, y_eager, atol=0, rtol=0)
