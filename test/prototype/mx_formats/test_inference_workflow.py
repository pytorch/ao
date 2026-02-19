# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import copy
from contextlib import contextmanager

import pytest
import torch
import torch.nn as nn
from torch.profiler import ProfilerActivity, profile

from torchao.prototype.mx_formats.config import QuantizeToNVFP4KernelChoice
from torchao.prototype.mx_formats.inference_workflow import (
    MXDynamicActivationMXWeightConfig,
    NVFP4DynamicActivationNVFP4WeightConfig,
    NVFP4WeightOnlyConfig,
)
from torchao.quantization import quantize_
from torchao.quantization.quantize_.common import KernelPreference
from torchao.quantization.utils import compute_error
from torchao.testing.utils import TorchAOIntegrationTestCase, skip_if_rocm
from torchao.utils import (
    is_sm_at_least_89,
    is_sm_at_least_100,
    torch_version_at_least,
)

torch.manual_seed(2)

if not torch_version_at_least("2.8.0"):
    pytest.skip("Unsupported PyTorch version", allow_module_level=True)


# source: https://stackoverflow.com/a/22638709
@pytest.fixture(autouse=True)
def run_around_tests():
    # 1. before test - set up (currently do nothing)
    # 2. run test
    yield
    # 3. after test - teardown
    torch._dynamo.reset()


@contextmanager
def cuda_kernel_profiler(kernel_pattern):
    """Context manager for profiling CUDA kernels."""
    result = {"found": False, "kernel_names": []}

    with profile(activities=[ProfilerActivity.CUDA]) as prof:
        yield result

    kernel_names = [
        evt.name
        for evt in prof.events()
        if evt.device_type == torch.autograd.DeviceType.CUDA and evt.name
    ]
    result["kernel_names"] = kernel_names
    result["found"] = any(kernel_pattern in name for name in kernel_names)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
@pytest.mark.skipif(
    not torch_version_at_least("2.8.0"), reason="torch.compile requires PyTorch 2.8+"
)
@pytest.mark.parametrize("elem_dtype", [torch.float8_e4m3fn, torch.float4_e2m1fn_x2])
@pytest.mark.parametrize("bias", [True, False])
@pytest.mark.parametrize("compile", [True, False])
@pytest.mark.parametrize("emulate", [True, False])
@pytest.mark.parametrize("use_inference_mode", [True, False])
@pytest.mark.parametrize("x_rank", [2, 3])
@torch.no_grad()
@skip_if_rocm(
    "ROCm float4 gemm require gfx950"
)  # TODO(future): deploy gfx950 in ROCM CI
def test_inference_workflow_mx(
    elem_dtype,
    bias: bool,
    compile: bool,
    emulate: bool,
    use_inference_mode: bool,
    x_rank: int,
):
    """
    Smoke test for inference compile
    """
    # TODO(future): figure out why these CUDA capability conditions are not properly
    # applied when inside `pytest.mark.skipif` for this test
    if elem_dtype in (torch.float8_e4m3fn, torch.float8_e5m2):
        if not is_sm_at_least_89():
            pytest.skip("CUDA capability >= 8.9 required for float8 in triton")
        elif not is_sm_at_least_100() and not emulate:
            pytest.skip("CUDA capability >= 10.0 required for mxfp8 gemm")
    elif elem_dtype == torch.float4_e2m1fn_x2:
        if not is_sm_at_least_100() and not emulate:
            pytest.skip("CUDA capability >= 10.0 required for mxfp4 gemm")
        elif compile:
            # TODO(future PR): investigate and fix this
            pytest.skip("mxfp4 + compile currently does not work, low SQNR")

    m = nn.Linear(32, 128, bias=bias, dtype=torch.bfloat16, device="cuda")
    m_mx = copy.deepcopy(m)

    if emulate:
        kernel_choice = KernelPreference.EMULATED
    else:
        kernel_choice = KernelPreference.AUTO
    config = MXDynamicActivationMXWeightConfig(
        activation_dtype=elem_dtype,
        weight_dtype=elem_dtype,
        kernel_preference=kernel_choice,
    )
    quantize_(m_mx, config=config)
    if compile:
        m_mx = torch.compile(m_mx, fullgraph=True)

    x = torch.randn(128, 32, device="cuda", dtype=torch.bfloat16)
    if x_rank == 3:
        x = x.unsqueeze(0)

    y_ref = m(x)
    if use_inference_mode:
        with torch.inference_mode():
            y_mx = m_mx(x)
    else:
        y_mx = m_mx(x)
    sqnr = compute_error(y_ref, y_mx)
    SQNR_THRESHOLD = 25.0 if elem_dtype == torch.float8_e4m3fn else 12.0
    assert sqnr >= SQNR_THRESHOLD, (
        f"Got a sqnr of {sqnr} for {elem_dtype} and bias={bias}"
    )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
@pytest.mark.skipif(
    not torch_version_at_least("2.8.0"), reason="torch.compile requires PyTorch 2.8+"
)
@pytest.mark.parametrize("bias", [True, False])
@pytest.mark.parametrize("compile", [True, False])
@pytest.mark.parametrize("quant_type", ["dynamic", "weight_only"])
@pytest.mark.parametrize("inpt_dtype", [torch.bfloat16, torch.float32])
@pytest.mark.parametrize(
    "quantize_to_nvfp4_kernel_choice",
    [QuantizeToNVFP4KernelChoice.TORCH, QuantizeToNVFP4KernelChoice.TRITON],
)
@pytest.mark.parametrize("use_dynamic_per_tensor_scale", [True, False])
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
@pytest.mark.parametrize("use_inference_mode", [False, True])
@pytest.mark.parametrize("x_rank", [2, 3])
@torch.no_grad()
@skip_if_rocm("ROCm float4 gemm require gfx950")
def test_inference_workflow_nvfp4(
    bias: bool,
    compile: bool,
    quant_type: str,
    inpt_dtype: torch.dtype,
    quantize_to_nvfp4_kernel_choice: QuantizeToNVFP4KernelChoice,
    use_dynamic_per_tensor_scale: bool,
    shapes: tuple,
    use_inference_mode: bool,
    x_rank: int,
):
    """
    Test NVFP4 recipe with scale_dtype=float8_e4m3fn and block_size=16
    Tests both DYNAMIC and WEIGHT_ONLY mm_config modes
    """
    # DYNAMIC mode requires SM100+, but WEIGHT_ONLY works on older GPUs
    if quant_type == "dynamic" and not is_sm_at_least_100():
        pytest.skip("CUDA capability >= 10.0 required for DYNAMIC float4 gemm")
    if quant_type == "weight_only" and compile:
        pytest.skip("TODO: weight_only quant currently errors w/ compile")
    if (
        quant_type == "weight_only"
        and quantize_to_nvfp4_kernel_choice == QuantizeToNVFP4KernelChoice.TRITON
    ):
        pytest.skip("unsupported configuration")

    if use_inference_mode and (
        shapes != (128, 64, 256)
        or inpt_dtype != torch.bfloat16
        or quantize_to_nvfp4_kernel_choice == QuantizeToNVFP4KernelChoice.TRITON
    ):
        pytest.skip("skipping unnecessary tests for inference mode")
    if x_rank == 3 and (
        shapes != (128, 64, 256)
        or inpt_dtype != torch.bfloat16
        or quantize_to_nvfp4_kernel_choice == QuantizeToNVFP4KernelChoice.TRITON
    ):
        pytest.skip("skipping unnecessary tests for x_rank 3")

    batch_size, in_features, out_features = shapes

    m = nn.Linear(in_features, out_features, bias=bias, dtype=inpt_dtype, device="cuda")
    m_mx = copy.deepcopy(m)

    if quant_type == "dynamic":
        config = NVFP4DynamicActivationNVFP4WeightConfig(
            quantize_to_nvfp4_kernel_choice=quantize_to_nvfp4_kernel_choice,
            use_dynamic_per_tensor_scale=use_dynamic_per_tensor_scale,
        )
    else:
        config = NVFP4WeightOnlyConfig(
            use_dynamic_per_tensor_scale=use_dynamic_per_tensor_scale,
        )
    quantize_(m_mx, config=config)

    if compile:
        m_mx = torch.compile(m_mx, fullgraph=True, backend="aot_eager")

    x = torch.randn(batch_size, in_features, device="cuda", dtype=inpt_dtype)
    if x_rank == 3:
        x = x.unsqueeze(0)

    y_ref = m(x)

    if (
        quantize_to_nvfp4_kernel_choice == QuantizeToNVFP4KernelChoice.TRITON
        and quant_type == "dynamic"
    ):
        with cuda_kernel_profiler("quantize_nvfp4_triton_kernel") as result:
            y_mx = m_mx(x)
        assert result["found"], "Expected quantize_nvfp4 kernel to be found"
    else:
        if use_inference_mode:
            with torch.inference_mode():
                y_mx = m_mx(x)
        else:
            y_mx = m_mx(x)

    sqnr = compute_error(y_ref, y_mx)

    if quant_type == "weight_only":
        SQNR_THRESHOLD = 18.0
    else:
        SQNR_THRESHOLD = 15.0

    assert y_mx.dtype == inpt_dtype, f"Got {y_mx.dtype} for inpt_dtype={inpt_dtype}"
    assert sqnr >= SQNR_THRESHOLD, (
        f"Got a sqnr of {sqnr} for NVFP4 recipe with bias={bias}, {quant_type=}"
    )


class VLLMIntegrationTestCase(TorchAOIntegrationTestCase):
    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    @pytest.mark.skipif(
        not torch_version_at_least("2.8.0"),
        reason="torch.compile requires PyTorch 2.8+",
    )
    def test_slice_and_copy_similar_to_vllm(self):
        config = MXDynamicActivationMXWeightConfig(
            activation_dtype=torch.float8_e4m3fn,
            weight_dtype=torch.float8_e4m3fn,
            kernel_preference=KernelPreference.EMULATED,
        )
        self._test_slice_and_copy_similar_to_vllm(config)

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    @pytest.mark.skipif(
        not torch_version_at_least("2.8.0"),
        reason="torch.compile requires PyTorch 2.8+",
    )
    def test_narrow_similar_to_vllm(self):
        config = MXDynamicActivationMXWeightConfig(
            activation_dtype=torch.float8_e4m3fn,
            weight_dtype=torch.float8_e4m3fn,
            kernel_preference=KernelPreference.EMULATED,
        )
        self._test_narrow_similar_to_vllm(config)

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    @pytest.mark.skipif(
        not torch_version_at_least("2.8.0"),
        reason="torch.compile requires PyTorch 2.8+",
    )
    def test_nvfp4_quantize_3d_param_similar_to_vllm(self):
        config = NVFP4WeightOnlyConfig(
            use_dynamic_per_tensor_scale=False,
        )
        self._test_quantize_3d_param_similar_to_vllm(config)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
@pytest.mark.skipif(
    not torch_version_at_least("2.8.0"), reason="torch.compile requires PyTorch 2.8+"
)
@pytest.mark.skipif(
    not is_sm_at_least_100(), reason="CUDA capability >= 10.0 required for NVFP4"
)
@pytest.mark.parametrize("bias", [True, False])
@torch.no_grad()
@skip_if_rocm("ROCm float4 gemm require gfx950")
def test_nvfp4_static_quantization_flow(
    bias: bool,
):
    """
    Test the prepare -> calibrate -> convert static quantization flow for NVFP4.

    This verifies that:
    1. prepare step inserts NVFP4ObservedLinear
    2. Calibration runs without error and tracks amax
    3. convert step produces quantized weights with static activation scale
    4. Inference produces reasonable results (SQNR check)
    """
    from torchao.prototype.mx_formats.inference_workflow import NVFP4ObservedLinear

    in_features, out_features = 64, 256
    batch_size = 128

    m = nn.Sequential(
        nn.Linear(
            in_features, out_features, bias=bias, dtype=torch.bfloat16, device="cuda"
        ),
    )
    m_ref = copy.deepcopy(m)

    # Step 1: Prepare - insert observers
    quantize_(
        m,
        NVFP4DynamicActivationNVFP4WeightConfig(step="prepare"),
    )

    # Verify observers were inserted
    assert isinstance(m[0], NVFP4ObservedLinear)

    # Step 2: Calibrate with representative data
    calibration_data = [
        torch.randn(batch_size, in_features, device="cuda", dtype=torch.bfloat16)
        for _ in range(5)
    ]
    for data in calibration_data:
        m(data)

    # Step 3: Convert - extract scale and quantize
    quantize_(
        m,
        NVFP4DynamicActivationNVFP4WeightConfig(step="convert"),
    )

    # Verify quantization was applied
    from torchao.prototype.mx_formats.nvfp4_tensor import NVFP4Tensor

    assert isinstance(m[0].weight, NVFP4Tensor), (
        f"Expected NVFP4Tensor weight, got {type(m[0].weight)}"
    )
    assert m[0].weight.act_per_tensor_scale is not None, (
        "Expected static act_per_tensor_scale to be set"
    )
    assert not m[0].weight.act_quant_kwargs.use_dynamic_per_tensor_scale, (
        "Expected use_dynamic_per_tensor_scale=False for static quant"
    )

    # Step 4: Verify inference produces reasonable results
    x = torch.randn(batch_size, in_features, device="cuda", dtype=torch.bfloat16)
    y_ref = m_ref(x)
    y_static = m(x)

    sqnr = compute_error(y_ref, y_static)
    SQNR_THRESHOLD = 15.0
    assert sqnr >= SQNR_THRESHOLD, (
        f"Got a sqnr of {sqnr} for NVFP4 static quant with bias={bias}"
    )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
@pytest.mark.skipif(
    not torch_version_at_least("2.8.0"), reason="torch.compile requires PyTorch 2.8+"
)
@pytest.mark.skipif(
    not is_sm_at_least_100(), reason="CUDA capability >= 10.0 required for NVFP4"
)
@torch.no_grad()
@skip_if_rocm("ROCm float4 gemm require gfx950")
def test_nvfp4_static_vs_dynamic_quantization():
    """
    Test that static quantization matches dynamic quantization when calibrated
    on the same input. Since NVFP4 static quant only determines the activation
    per_tensor_scale statically, calibrating on the exact test input should
    produce the same per_tensor_scale as the dynamic path, yielding identical results.
    """
    in_features, out_features = 64, 256
    batch_size = 128

    m_dynamic = nn.Sequential(
        nn.Linear(
            in_features, out_features, bias=True, dtype=torch.bfloat16, device="cuda"
        ),
    )
    m_static = copy.deepcopy(m_dynamic)

    # Dynamic quantization
    quantize_(
        m_dynamic,
        NVFP4DynamicActivationNVFP4WeightConfig(
            quantize_to_nvfp4_kernel_choice=QuantizeToNVFP4KernelChoice.TORCH,
            use_dynamic_per_tensor_scale=True,
        ),
    )

    # Create the test input upfront so we can use it for calibration too
    x = torch.randn(batch_size, in_features, device="cuda", dtype=torch.bfloat16)

    # Static quantization: prepare -> calibrate -> convert
    quantize_(
        m_static,
        NVFP4DynamicActivationNVFP4WeightConfig(
            step="prepare",
            quantize_to_nvfp4_kernel_choice=QuantizeToNVFP4KernelChoice.TORCH,
        ),
    )
    # Calibrate with the same input used for testing
    m_static(x)

    quantize_(
        m_static,
        NVFP4DynamicActivationNVFP4WeightConfig(
            step="convert",
            quantize_to_nvfp4_kernel_choice=QuantizeToNVFP4KernelChoice.TORCH,
        ),
    )

    # Test inference - when calibrating on the same input, static and dynamic
    # should produce the same result since the per_tensor_scale will match
    y_dynamic = m_dynamic(x)
    y_static = m_static(x)

    assert torch.equal(y_dynamic, y_static), (
        "Expect dynamic and static quant result to be equal"
    )
