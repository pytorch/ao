# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD 3-Clause license found in the
# LICENSE file in the root directory of this source tree.

import pytest
import torch

from torchao.quantization import (
    Int8DynamicActivationInt8WeightConfig,
    Int8StaticActivationInt8WeightConfig,
    Int8Tensor,
    PerTensor,
    quantize_,
)
from torchao.quantization.quant_primitives import MappingType


@pytest.fixture(
    params=[("cpu", False), ("cuda", False), ("cuda", True)],
    ids=["cpu-eager", "cuda-eager", "cuda-compiled"],
)
def execution(request):
    device, compiled = request.param
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA required")
    if compiled:
        torch.compiler.reset()
    return device, compiled


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32])
@pytest.mark.parametrize("scale_dtype", [None, torch.bfloat16, torch.float64])
@pytest.mark.parametrize("bias", [False, True])
def test_asymmetric_linear_single_rounding(dtype, bias, scale_dtype, execution):
    device, compiled = execution
    torch.manual_seed(7)
    model = torch.nn.Linear(256, 32, bias=bias).to(device=device, dtype=dtype)
    x = torch.randn(4 if device == "cpu" else 32, 256).abs().clamp_max(2).to(dtype)
    x[:, :6] = 2
    x = x.to(device)
    if scale_dtype is not None:
        config = Int8StaticActivationInt8WeightConfig(
            act_quant_scale=torch.tensor(
                [[0.0100000003]],
                dtype=scale_dtype,
                device=device,
            ),
            act_quant_zero_point=torch.tensor(
                [[-128]], dtype=torch.int8, device=device
            ),
            act_mapping_type=MappingType.ASYMMETRIC,
            granularity=PerTensor(),
        )
    else:
        config = Int8DynamicActivationInt8WeightConfig(
            act_mapping_type=MappingType.ASYMMETRIC
        )
    quantize_(model, config)
    weight = model.weight
    kwargs = weight.act_quant_kwargs

    def quantize_activation(tensor):
        activation = Int8Tensor.from_hp(
            tensor,
            kwargs.granularity,
            mapping_type=kwargs.mapping_type,
            reduce_range=kwargs.reduce_range,
            scale=weight.act_quant_scale,
            zero_point=weight.act_quant_zero_point,
        )
        return activation.qdata, activation.scale, activation.zero_point

    # Match inference quantization: compilation may elide low-precision casts.
    quantizer = (
        torch.compile(quantize_activation, fullgraph=True)
        if compiled
        else quantize_activation
    )
    qdata, activation_scale, zero_point = quantizer(x)
    xi = qdata.cpu().long()
    wi = weight.qdata.cpu().long()
    zp = zero_point.cpu().reshape(-1, 1).long()
    reference = ((xi - zp) @ wi.t()).double()
    reference *= activation_scale.cpu().reshape(-1, 1).double()
    reference *= weight.scale.cpu().reshape(1, -1).double()
    if bias:
        reference += model.bias.cpu().double()
    runner = torch.compile(model, fullgraph=True) if compiled else model
    actual = runner(x)
    actual = actual.cpu()
    assert torch.isfinite(actual).all()
    error = (actual.double() - reference).norm()
    floor = (reference.to(dtype).double() - reference).norm()
    assert error <= floor * 1.05 + 1e-6


def test_symmetric_linear_preserves_early_rounding(execution):
    from torchao.quantization.quantize_.workflows.int8.kernels import (
        _int_scaled_matmul,
    )

    device, compiled = execution
    torch.manual_seed(8)
    model = torch.nn.Linear(256, 32, bias=True).to(device=device, dtype=torch.bfloat16)
    x = torch.randn(4 if device == "cpu" else 32, 256, device=device).to(torch.bfloat16)
    quantize_(model, Int8DynamicActivationInt8WeightConfig())
    weight = model.weight
    kwargs = weight.act_quant_kwargs

    def original_order(tensor):
        quantized = Int8Tensor.from_hp(
            tensor,
            kwargs.granularity,
            mapping_type=kwargs.mapping_type,
            reduce_range=kwargs.reduce_range,
        )
        expected = _int_scaled_matmul(
            quantized.qdata,
            weight.qdata.contiguous().t(),
            quantized.scale.reshape(-1, 1),
        ).to(tensor.dtype)
        expected = expected * weight.scale.flatten()
        expected += model.bias
        return expected.to(tensor.dtype)

    reference_runner = (
        torch.compile(original_order, fullgraph=True) if compiled else original_order
    )
    expected = reference_runner(x)
    runner = torch.compile(model, fullgraph=True) if compiled else model
    actual = runner(x)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
