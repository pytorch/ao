# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import copy
from enum import IntEnum

import pytest
import torch
import torch.nn as nn

from torchao.prototype.mx_formats.constants import SUPPORTED_ELEM_DTYPES
from torchao.prototype.mx_formats.mx_linear import (
    MXInferenceLinear,
    MXLinear,
    swap_linear_with_mx_inference_linear,
    swap_linear_with_mx_linear,
)
from torchao.prototype.mx_formats.mx_tensor import MXTensor
from torchao.quantization.utils import compute_error
from torchao.utils import (
    TORCH_VERSION_AT_LEAST_2_4,
    is_sm_at_least_89,
    is_sm_at_least_100,
)

torch.manual_seed(2)

if not TORCH_VERSION_AT_LEAST_2_4:
    pytest.skip("Unsupported PyTorch version", allow_module_level=True)


# not for land, https://www.internalfb.com/phabricator/paste/view/P1717686991
class DataType(IntEnum):
    DEFAULT = 0
    E8M0 = 1
    FP4 = 2
    UFP8 = 3


# source: https://stackoverflow.com/a/22638709
@pytest.fixture(autouse=True)
def run_around_tests():
    # 1. before test - set up (currently do nothing)
    # 2. run test
    yield
    # 3. after test - teardown
    torch._dynamo.reset()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
@pytest.mark.parametrize("elem_dtype", SUPPORTED_ELEM_DTYPES)
@pytest.mark.parametrize("bias", [True, False])
@pytest.mark.parametrize("input_shape", [(4, 8), (1, 4, 8), (1, 1, 4, 8)])
def test_linear_eager(elem_dtype, bias, input_shape):
    """
    Smoke test for training linear module with mx weight
    """
    grad_shape = list(input_shape)
    grad_shape[-1] = 6

    m = nn.Sequential(
        nn.Linear(8, 6, bias=bias, device="cuda"),
    )
    m_mx = copy.deepcopy(m)
    block_size = 2
    swap_linear_with_mx_linear(m_mx, elem_dtype, block_size)

    x_ref = torch.randn(*input_shape, device="cuda").requires_grad_()
    x = copy.deepcopy(x_ref)
    g = torch.randn(*grad_shape, device="cuda")
    with torch.autocast("cuda", dtype=torch.bfloat16):
        y_ref = m(x_ref)
        y_mx = m_mx(x)

    y_ref.backward(g)
    y_mx.backward(g)

    y_sqnr = compute_error(y_ref, y_mx)
    w_g_sqnr = compute_error(m[0].weight.grad, getattr(m_mx, "0").weight.grad)
    x_g_sqnr = compute_error(x_ref.grad, x.grad)

    if elem_dtype is torch.float8_e4m3fn:
        assert y_sqnr >= 18.0
        assert w_g_sqnr >= 18.0
        assert x_g_sqnr >= 12.0
    else:
        assert y_sqnr >= 8.0
        assert w_g_sqnr >= 10.0
        assert x_g_sqnr >= 8.0


# TODO(future): enable compile support
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_activation_checkpointing():
    input_shape = (2, 4)
    grad_shape = (2, 6)
    elem_dtype = torch.float8_e4m3fn

    m = nn.Sequential(
        nn.Linear(4, 6, bias=True, device="cuda"),
        nn.Linear(6, 6, bias=True, device="cuda"),
    )
    block_size = 2
    swap_linear_with_mx_linear(m, elem_dtype, block_size)

    x = torch.randn(*input_shape, device="cuda").requires_grad_()
    g = torch.randn(*grad_shape, device="cuda")
    y = torch.utils.checkpoint.checkpoint(m, x, use_reentrant=False)
    y.backward(g)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
@pytest.mark.skipif(
    is_sm_at_least_100(), reason="triton does not work yet on CUDA capability 10.0"
)
@pytest.mark.parametrize("elem_dtype", SUPPORTED_ELEM_DTYPES)
@pytest.mark.parametrize("bias", [False, True])
# TODO(future PR): figure out why torch.compile does not match eager when
# autocast is on
@pytest.mark.parametrize(
    "use_autocast",
    [
        False,
    ],
)
def test_linear_compile(elem_dtype, bias, use_autocast):
    """
    Verify that compile does not change numerics of MX linear fw + bw
    """
    if elem_dtype in (torch.float8_e4m3fn, torch.float8_e5m2):
        if not is_sm_at_least_89():
            pytest.skip("CUDA capability >= 8.9 required for float8 in triton")
    M, K, N = 4, 8, 6
    input_shape = (M, K)
    grad_shape = (M, N)
    m_mx = nn.Sequential(
        nn.Linear(K, N, bias=bias, device="cuda"),
    )
    block_size = 2
    swap_linear_with_mx_linear(m_mx, elem_dtype, block_size)
    m_mx_c = copy.deepcopy(m_mx)
    m_mx_c = torch.compile(m_mx_c, fullgraph=True, backend="inductor")

    x_ref = torch.randn(*input_shape, device="cuda").requires_grad_()
    x = copy.deepcopy(x_ref)
    g = torch.randn(*grad_shape, device="cuda")

    if use_autocast:
        with torch.autocast("cuda", dtype=torch.bfloat16):
            y_ref = m_mx(x_ref)
            y = m_mx_c(x)
    else:
        y_ref = m_mx(x_ref)
        y = m_mx_c(x)
    torch.testing.assert_close(y_ref, y, atol=0, rtol=0)

    y_ref.backward(g)
    y.backward(g)
    w_g_ref = m_mx[0].weight.grad
    w_g = getattr(m_mx_c, "0").weight.grad
    # TODO(future): investigate why we can't match with rtol=0 atol=0
    # after moving to torchao repo. Technically compile does not give
    # bit exactness guarantees, but there also might be a bug lurking
    # around.
    torch.testing.assert_close(w_g_ref, w_g, atol=0.02, rtol=0.02)

    x_g_ref = x_ref.grad
    x_g = x.grad
    # TODO(future): investigate why we can't match with rtol=0 atol=0
    # after moving to torchao repo. Technically compile does not give
    # bit exactness guarantees, but there also might be a bug lurking
    # around.
    torch.testing.assert_close(x_g_ref, x_g, atol=0.02, rtol=0.02)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
@pytest.mark.parametrize("elem_dtype", SUPPORTED_ELEM_DTYPES)
@pytest.mark.parametrize("bias", [True, False])
@pytest.mark.parametrize("input_shape", [(2, 4), (1, 2, 4), (1, 1, 2, 4)])
def test_inference_linear(elem_dtype, bias, input_shape):
    """
    Smoke test for inference linear module with mx weight
    """
    m = nn.Sequential(nn.Linear(4, 6, bias=bias, dtype=torch.bfloat16))
    m = m.cuda()
    m_mx = copy.deepcopy(m)
    block_size = 2
    swap_linear_with_mx_inference_linear(m_mx, elem_dtype, block_size)

    x = torch.randn(*input_shape, device="cuda", dtype=torch.bfloat16)
    y_ref = m(x)
    y_mx = m_mx(x)
    sqnr = compute_error(y_ref, y_mx)
    if elem_dtype is torch.float8_e4m3fn:
        assert sqnr >= 20.0
    else:
        assert sqnr >= 11.0


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
@pytest.mark.skipif(
    is_sm_at_least_100(), reason="triton does not work yet on CUDA capability 10.0"
)
@pytest.mark.parametrize("elem_dtype", SUPPORTED_ELEM_DTYPES)
def test_inference_compile_simple(elem_dtype):
    """
    Smoke test for inference compile
    """
    if elem_dtype in (torch.float8_e4m3fn, torch.float8_e5m2):
        if not is_sm_at_least_89():
            pytest.skip("CUDA capability >= 8.9 required for float8 in triton")
    m = nn.Sequential(nn.Linear(4, 6, bias=False, dtype=torch.bfloat16))
    m = m.cuda()
    m_mx = copy.deepcopy(m)
    block_size = 2
    swap_linear_with_mx_inference_linear(m_mx, elem_dtype, block_size)
    m_mx = torch.compile(m_mx, fullgraph="true")

    x = torch.randn(2, 4, device="cuda", dtype=torch.bfloat16)
    y_ref = m(x)
    y_mx = m_mx(x)
    sqnr = compute_error(y_ref, y_mx)
    if elem_dtype is torch.float8_e4m3fn:
        assert sqnr >= 20.0
    else:
        assert sqnr >= 13.5


def test_filter_fn():
    m1 = nn.Sequential(
        nn.Linear(32, 32),
        nn.Linear(32, 32),
    )
    m2 = copy.deepcopy(m1)
    filter_fn = lambda mod, fqn: fqn != "1"  # noqa: E731

    swap_linear_with_mx_linear(m1, torch.float8_e4m3fn, 32, filter_fn)
    assert type(m1[0]) == MXLinear
    assert type(m1[1]) == torch.nn.Linear

    swap_linear_with_mx_inference_linear(m2, torch.float8_e4m3fn, 32, filter_fn)  # noqa: E501
    assert type(m2[0]) == MXInferenceLinear
    assert type(m2[1]) == torch.nn.Linear

# copy-pasted from https://github.com/drisspg/transformer_nuggets/blob/12bf63d334900d57958f839f273f5bca78a8f4a1/transformer_nuggets/mx/to_blocked.py#L54C1-L62C76 
# and modified to return 128x4 instead of 32x16
def _to_blocked_single(scales: torch.Tensor) -> torch.Tensor:
    """Assume that we have a 128x4 block of scales in K Major order

    To see more information on the individual tile layout:
    https://docs.nvidia.com/cuda/cublas/index.html#d-block-scaling-factors-layout
    """
    assert scales.shape == (128, 4)
    scales_tiled = scales.view(4, 32, 4)  # view as 4 - (32, 4) tiles
    return scales_tiled.transpose(0, 1).reshape(128, 4).contiguous()  # Interleave tiles

def test_to_blocked():
    scales = torch.arange(128 * 4).reshape(128, 4) / 4
    print('orig')
    print(scales)
    print('blocked')
    print(_to_blocked_single(scales))
    # looks right!


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
@pytest.mark.skipif(
    not is_sm_at_least_100(),
    reason="blockwise torch._scaled_mm requires CUDA 10.0 or higher",
)
def test_scaled_mm_mxfp8_scales_one():
    # basic numerics with all scales 1.0
    # next: other scale values

    # M, K, N = 8192, 4096, 8192
    M, K, N = 128, 128, 128
    BLOCK_SIZE = 32
    a = torch.ones(M, K, device="cuda", dtype=torch.float32).to(torch.float8_e4m3fn)
    b = torch.ones(N, K, device="cuda", dtype=torch.float32).to(torch.float8_e4m3fn).t()

    # 127 is 1.0 in e8m0
    scale_val = 127

    a_scales = torch.full(
        (M, K // BLOCK_SIZE), scale_val, device="cuda", dtype=torch.uint8
    )
    b_scales = torch.full(
        (K // BLOCK_SIZE, N), scale_val, device="cuda", dtype=torch.uint8
    ).t()
    # b_scales[0][0] = 128
    out = torch._scaled_mm(
        a,
        b,
        a_scales,
        b_scales,
        None,
        None,
        torch.bfloat16,
        False,
        None,
        None,
        DataType.E8M0,
    )

    # [[1, 0, ...], ..., [0, ..., 1]] - correct
    torch.set_printoptions(profile="full", linewidth=280)
    print(out)
    print(torch.max(out))

@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
@pytest.mark.skipif(
    not is_sm_at_least_100(),
    reason="blockwise torch._scaled_mm requires CUDA 10.0 or higher",
)
def test_scaled_mm_mxfp8_mxtensor():
    # baseline 1: fp32
    # experiment 1: emulated MX from MXTensor
    # experiment 2: real MX gemm

    # results so far:
    # * experiment 1 is very close to experiment 2
    # * experiments 1 and 2 are far from baseline (lol!)

    # M, K, N = 8192, 4096, 8192
    M, K, N = 128, 128, 128
    BLOCK_SIZE = 32
    a_fp32 = torch.randn(M, K, device="cuda", dtype=torch.float32)
    b_fp32 = torch.randn(N, K, device="cuda", dtype=torch.float32).t().contiguous()

    a_mx = MXTensor.to_mx(a_fp32, torch.float8_e4m3fn, BLOCK_SIZE)
    b_mx = MXTensor.to_mx(b_fp32, torch.float8_e4m3fn, BLOCK_SIZE).t()
    a_s0 = a_mx._scale_e8m0.reshape(M, -1)
    a_s1 = _to_blocked_single(a_s0)
    b_s0 = b_mx._scale_e8m0.reshape(N, -1)
    b_s1 = _to_blocked_single(b_s0)

    # ones_scale = torch.full((M, K // BLOCK_SIZE), 127, dtype=torch.uint8, device="cuda")

    out_ref = a_fp32 @ b_fp32.t()
    print('baseline', out_ref)

    out_mx_emulated = a_mx @ b_mx
    print('mx_emulated', out_mx_emulated)

    out_mx_real = torch._scaled_mm(
        a_mx._data,
        b_mx._data,
        # a_scales is really b_scales, and vice versa. Probably switched in cuBLAS kernel?
        _to_blocked_single(b_mx._scale_e8m0.reshape(N, -1)),
        _to_blocked_single(a_mx._scale_e8m0.reshape(M, -1)),
        None,
        None,
        torch.float32,
        False,
        None,
        None,
        DataType.E8M0,
    )
    print('mx_real', out_mx_real)

    sqnr_baseline_to_emulated_mx = compute_error(out_ref, out_mx_emulated)
    sqnr_baseline_to_real_mx = compute_error(out_ref, out_mx_real)
    sqnr_emulated_mx_to_real_mx = compute_error(out_mx_emulated, out_mx_real)
    print('sqnr baseline -> emulated_mx', sqnr_baseline_to_emulated_mx)
    print('sqnr baseline -> real_mx', sqnr_baseline_to_real_mx)
    print('sqnr emulated_mx -> real_mx', sqnr_emulated_mx_to_real_mx)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
@pytest.mark.skipif(
    not is_sm_at_least_100(),
    reason="blockwise torch._scaled_mm requires CUDA 10.0 or higher",
)
def test_scaled_mm_mx_reconstruct_scale_a_layout():
    # brute force the expected layout format
    # basic numerics with all scales 1.0
    # next: other scale values

    # M, K, N = 8192, 4096, 8192
    M, K, N = 128, 128, 128
    BLOCK_SIZE = 32
    a = torch.ones(M, K, device="cuda", dtype=torch.float32).to(torch.float8_e4m3fn)

    # 127 is 1.0 in e8m0
    scale_val = 127

    print()

    # Probe torch._scaled_mm to deduce the actual layout used for the scale
    # arguments. Specifically, here is what the code below would do if we had
    # A and B as 4x4 matrices with MX block size 2. All matrices are shown in float32
    # format, not their actual storage format, to demonstrate the algorithm.
    #
    # A matrix - set to all-ones
    #
    # A =   1111
    #       1111
    #       1111
    #       1111
    #
    # B matrix variants - all-zeros, except a single one for each mx block in the first column
    #
    # B_0 = 1000   B_1 = 0000
    #       0000         0000
    #       0000         1000
    #       0000         0000
    #
    # A scale - starts as a matrix of all-ones
    #
    # A_s = 11
    #       11
    #       11
    #       11
    #
    # for each row in rows of A:
    #   for each ol in cols of A:
    #     initialize A to all-ones
    #     set A[row][col] = 2.0
    #     for each B in [Bs]:
    #       C = torch._scaled_mm(A, B, A_s, B_s, ...)
    #       if max(C) > 1.0:
    #         the scale incremented in A_s was corresponding to the current block

    for scale_row in range(M):
        for scale_col in range(K // BLOCK_SIZE):

            a_scales = torch.full(
                (M, K // BLOCK_SIZE), scale_val, device="cuda", dtype=torch.uint8
            )
            b_scales = torch.full(
                # (K // BLOCK_SIZE, N), scale_val, device="cuda", dtype=torch.uint8
                (N, K // BLOCK_SIZE), scale_val, device="cuda", dtype=torch.uint8
            )

            # TODO: it looks like blockwise scales are switched in cuBLAS? 
            # incrementing scale of b looks like it's actually affecting scaling of a
            b_scales[scale_row][scale_col] = scale_val + 1

            # We test every BLOCK_SIZE to deduce which of the blocks is
            # responsible for the scale value. Note that this isn't the most
            # efficient way to test, but I'm optimizing for dev time here.
            for block_idx in range(K // BLOCK_SIZE):

                b = torch.zeros(N, K, device="cuda", dtype=torch.float32)
                # set a single one inside the block
                b[0][block_idx * BLOCK_SIZE] = 1
                b = b.to(torch.float8_e4m3fn).t()

                out = torch._scaled_mm(
                    a,
                    b,
                    a_scales,
                    b_scales,
                    None,
                    None,
                    torch.bfloat16,
                    False,
                    None,
                    None,
                    DataType.E8M0,
                )

                # print(scale_row, scale_col, block_idx)
                # torch.set_printoptions(profile="full", linewidth=320)
                # print(out)
                # print(torch.max(out, keepdim=True))

                max_val = torch.max(out).item()
                if max_val > 1:
                    max_flat_index = torch.argmax(out).item()
                    max_row = max_flat_index // M
                    max_col = max_flat_index % M
                    assert max_col == 0
                    assert max_val == 2.0
                    print('scale_coords', scale_row, scale_col, 'block_idx', block_idx, 'max_coords', max_row, max_col, 'max_val', max_val)

            # break
        # break

@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
@pytest.mark.skipif(
    not is_sm_at_least_100(),
    reason="blockwise torch._scaled_mm requires CUDA 10.0 or higher",
)
def test_scaled_mm_mx_reconstruct_scale_b_layout():
    # brute force the expected layout format
    # basic numerics with all scales 1.0
    # next: other scale values

    # M, K, N = 8192, 4096, 8192
    M, K, N = 128, 128, 128
    BLOCK_SIZE = 32
    b = torch.ones(M, K, device="cuda", dtype=torch.float32).to(torch.float8_e4m3fn).t()

    # 127 is 1.0 in e8m0
    scale_val = 127

    print()

    # Probe torch._scaled_mm to deduce the actual layout used for the scale
    # arguments. Specifically, here is what the code below would do if we had
    # A and B as 4x4 matrices with MX block size 2. All matrices are shown in float32
    # format, not their actual storage format, to demonstrate the algorithm.
    #
    # A matrix variants - all-zeros, except a single one for each mx block in the first row
    #
    # A_0 = 1000   A_1 = 0010
    #       0000         0000
    #       0000         0000
    #       0000         0000
    #
    # B matrix - set to all-ones
    #
    # B =   1111
    #       1111
    #       1111
    #       1111
    #
    # B scale - starts as a matrix of all-ones
    #
    # B_s = 11
    #       11
    #       11
    #       11
    #
    # for each row in rows of B:
    #   for each col in cols of B:
    #     initialize B to all-ones
    #     set B[row][col] = 2.0
    #     for each A in [As]:
    #       C = torch._scaled_mm(A, B, A_s, B_s, ...)
    #       if max(C) > 1.0:
    #         the scale incremented in B_s was corresponding to the current block

    for scale_row in range(M):
        for scale_col in range(K // BLOCK_SIZE):

            a_scales = torch.full(
                (M, K // BLOCK_SIZE), scale_val, device="cuda", dtype=torch.uint8
            )
            b_scales = torch.full(
                # (K // BLOCK_SIZE, N), scale_val, device="cuda", dtype=torch.uint8
                (N, K // BLOCK_SIZE), scale_val, device="cuda", dtype=torch.uint8
            )

            # TODO: it looks like blockwise scales are switched in cuBLAS? 
            # incrementing scale of a looks like it's actually affecting scaling of b
            a_scales[scale_row][scale_col] = scale_val + 1

            # We test every BLOCK_SIZE to deduce which of the blocks is
            # responsible for the scale value. Note that this isn't the most
            # efficient way to test, but I'm optimizing for dev time here.
            for block_idx in range(K // BLOCK_SIZE):

                a = torch.zeros(M, K, device="cuda", dtype=torch.float32)
                # set a single one inside the block
                a[0][block_idx * BLOCK_SIZE] = 1
                a = a.to(torch.float8_e4m3fn)

                out = torch._scaled_mm(
                    a,
                    b,
                    a_scales,
                    b_scales,
                    None,
                    None,
                    torch.bfloat16,
                    False,
                    None,
                    None,
                    DataType.E8M0,
                )

                # print(scale_row, scale_col, block_idx)
                # torch.set_printoptions(profile="full", linewidth=320)
                # print(out)
                # print(torch.max(out, keepdim=True))

                max_val = torch.max(out).item()
                if max_val > 1:
                    max_flat_index = torch.argmax(out).item()
                    max_row = max_flat_index // M
                    max_col = max_flat_index % M
                    assert max_row == 0
                    assert max_val == 2.0
                    print('scale_coords', scale_row, scale_col, 'block_idx', block_idx, 'max_coords', max_row, max_col, 'max_val', max_val)

            # break
        # break

@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
@pytest.mark.skipif(
    not is_sm_at_least_100(),
    reason="blockwise torch._scaled_mm requires CUDA 10.0 or higher",
)
def test_scaled_mm_nvfp4():
    # hello world
    # next: basic numerics

    M, K, N = 8192, 4096, 8192
    BLOCK_SIZE = 16
    a = torch.randint(128, (M, K // 2), device="cuda", dtype=torch.uint8)
    b = torch.randint(128, (N, K // 2), device="cuda", dtype=torch.uint8).t()
    a_scales = torch.randint(
        128, (M, K // BLOCK_SIZE), device="cuda", dtype=torch.uint8
    )
    b_scales = torch.randint(
        128, (N, K // BLOCK_SIZE), device="cuda", dtype=torch.uint8
    ).t()
    out = torch._scaled_mm(
        a,
        b,
        a_scales,
        b_scales,
        None,
        None,
        torch.bfloat16,
        False,
        DataType.FP4,
        DataType.FP4,
        DataType.UFP8,
    )
    print(out)
