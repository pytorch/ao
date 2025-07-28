# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import copy

import pytest
import torch
import torch.nn as nn

from torchao.prototype.mx_formats.config import (
    MXGemmKernelChoice,
)
from torchao.prototype.mx_formats.inference_workflow import (
    MXFPInferenceConfig,
    NVFP4InferenceConfig,
    NVFP4MMConfig,
)
from torchao.quantization import quantize_
from torchao.quantization.utils import compute_error
from torchao.testing.utils import skip_if_rocm
from torchao.utils import (
    TORCH_VERSION_AT_LEAST_2_8,
    is_sm_at_least_89,
    is_sm_at_least_100,
)

torch.manual_seed(2)

if not TORCH_VERSION_AT_LEAST_2_8:
    pytest.skip("Unsupported PyTorch version", allow_module_level=True)


# source: https://stackoverflow.com/a/22638709
@pytest.fixture(autouse=True)
def run_around_tests():
    # 1. before test - set up (currently do nothing)
    # 2. run test
    yield
    # 3. after test - teardown
    torch._dynamo.reset()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
@pytest.mark.skipif(
    not TORCH_VERSION_AT_LEAST_2_8, reason="torch.compile requires PyTorch 2.8+"
)
@pytest.mark.parametrize("elem_dtype", [torch.float8_e4m3fn, torch.float4_e2m1fn_x2])
@pytest.mark.parametrize("bias", [True, False])
@pytest.mark.parametrize("compile", [True, False])
@torch.no_grad()
@skip_if_rocm(
    "ROCm float4 gemm require gfx950"
)  # TODO(future): deploy gfx950 in ROCM CI
@pytest.mark.skipif(not is_sm_at_least_100(), reason="CUDA capability >= 10.0 required")
def test_inference_workflow(elem_dtype, bias: bool, compile: bool):
    """
    Smoke test for inference compile
    """
    # TODO(future): figure out why these CUDA capability conditions are not properly
    # applied when inside `pytest.mark.skipif` for this test
    if elem_dtype in (torch.float8_e4m3fn, torch.float8_e5m2):
        if not is_sm_at_least_89():
            pytest.skip("CUDA capability >= 8.9 required for float8 in triton")
    elif elem_dtype == torch.float4_e2m1fn_x2:
        if not is_sm_at_least_100():
            pytest.skip("CUDA capability >= 10.0 required for float4 gemm")

    m = nn.Linear(32, 128, bias=bias, dtype=torch.bfloat16, device="cuda")
    m_mx = copy.deepcopy(m)
    kernel_choice = (
        MXGemmKernelChoice.CUTLASS
        if elem_dtype == torch.float4_e2m1fn_x2
        else MXGemmKernelChoice.CUBLAS
    )
    config = MXFPInferenceConfig(
        activation_dtype=elem_dtype,
        weight_dtype=elem_dtype,
        gemm_kernel_choice=kernel_choice,
    )
    quantize_(m_mx, config=config)
    if compile:
        m_mx = torch.compile(m_mx, fullgraph=True)

    x = torch.randn(128, 32, device="cuda", dtype=torch.bfloat16)
    y_ref = m(x)
    y_mx = m_mx(x)
    sqnr = compute_error(y_ref, y_mx)
    SQNR_THRESHOLD = 25.0 if elem_dtype == torch.float8_e4m3fn else 15.0
    assert sqnr >= SQNR_THRESHOLD, (
        f"Got a sqnr of {sqnr} for {elem_dtype} and bias={bias}"
    )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
@pytest.mark.skipif(
    not TORCH_VERSION_AT_LEAST_2_8, reason="torch.compile requires PyTorch 2.8+"
)
@pytest.mark.parametrize("bias", [True, False])
@pytest.mark.parametrize("compile", [True, False])
@pytest.mark.parametrize(
    "mm_config", [NVFP4MMConfig.DYNAMIC, NVFP4MMConfig.WEIGHT_ONLY]
)
@pytest.mark.parametrize("inpt_dtype", [torch.bfloat16, torch.float32])
@pytest.mark.parametrize("use_triton_kernel", [True, False])
@pytest.mark.parametrize(
    "shapes",
    [
        (128, 64, 256),
        (256, 128, 512),
        (145, 64, 256),
        (128, 96, 256),
        (128, 160, 256),
        (64, 64, 256),
        (200, 192, 256),
    ],
    ids=lambda s: f"{s[0]}x{s[1]}x{s[2]}",
)
@torch.no_grad()
@skip_if_rocm("ROCm float4 gemm require gfx950")
def test_inference_workflow_nvfp4(
    bias: bool,
    compile: bool,
    mm_config: NVFP4MMConfig,
    inpt_dtype: torch.dtype,
    use_triton_kernel: bool,
    shapes: tuple,
):
    """
    Test NVFP4 recipe with scale_dtype=float8_e4m3fn and block_size=16
    Tests both DYNAMIC and WEIGHT_ONLY mm_config modes
    """
    # DYNAMIC mode requires SM100+, but WEIGHT_ONLY works on older GPUs
    if mm_config == NVFP4MMConfig.DYNAMIC and not is_sm_at_least_100():
        pytest.skip("CUDA capability >= 10.0 required for DYNAMIC float4 gemm")

    if bias and inpt_dtype == torch.float32:
        pytest.xfail("Bias is not supported when module weight is in fp32")

    if mm_config == NVFP4MMConfig.WEIGHT_ONLY and compile:
        pytest.skip("TODO: NVFP4MMConfig.WEIGHT_ONLY currently errors w/ compile")
    batch_size, in_features, out_features = shapes

    m = nn.Linear(in_features, out_features, bias=bias, dtype=inpt_dtype, device="cuda")
    m_mx = copy.deepcopy(m)

    config = NVFP4InferenceConfig(
        mm_config=mm_config, use_triton_kernel=use_triton_kernel
    )
    quantize_(m_mx, config=config)

    if compile:
        m_mx = torch.compile(m_mx, fullgraph=True, backend="aot_eager")

    x = torch.randn(batch_size, in_features, device="cuda", dtype=inpt_dtype)
    y_ref = m(x)
    y_mx = m_mx(x)
    sqnr = compute_error(y_ref, y_mx)

    if mm_config == NVFP4MMConfig.WEIGHT_ONLY:
        SQNR_THRESHOLD = 18.0
    else:
        SQNR_THRESHOLD = 15.0

    assert y_mx.dtype == inpt_dtype, f"Got {y_mx.dtype} for inpt_dtype={inpt_dtype}"
    assert sqnr >= SQNR_THRESHOLD, (
        f"Got a sqnr of {sqnr} for NVFP4 recipe with bias={bias}, mm_config={mm_config}"
    )
