# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import pytest
import torch
from torch.utils._triton import has_triton

from torchao.prototype.moe_training.nvfp4_training.hadamard_utils import (
    prepare_for_cuda_graph,
)
from torchao.prototype.moe_training.nvfp4_training.nvfp4_training import NVFP4Linear
from torchao.quantization.quantize_.common.kernel_preference import KernelPreference
from torchao.quantization.utils import compute_error
from torchao.utils import is_sm_at_least_100, torch_version_at_least


def test_nvfp4_linear_rht_sign_vector_state_dict_roundtrip():
    torch.manual_seed(123)
    layer = NVFP4Linear(128, 128, bias=False, kernel_preference=KernelPreference.TRITON)
    expected_sign_vector = layer.rht_sign_vector
    state_dict = layer.state_dict()

    torch.manual_seed(456)
    loaded = NVFP4Linear(
        128, 128, bias=False, kernel_preference=KernelPreference.TRITON
    )
    loaded.load_state_dict(state_dict)

    assert loaded.rht_sign_vector == expected_sign_vector
    torch.testing.assert_close(
        loaded._rht_sign_vector.cpu(),
        layer._rht_sign_vector.cpu(),
        atol=0,
        rtol=0,
    )


def test_nvfp4_linear_from_linear_preserves_rht_sign_vector():
    sign_vector = tuple(1 if i % 2 == 0 else -1 for i in range(16))
    layer = NVFP4Linear(
        128,
        128,
        bias=False,
        kernel_preference=KernelPreference.TRITON,
        rht_sign_vector=sign_vector,
    )

    converted = NVFP4Linear.from_linear(layer)

    assert converted.rht_sign_vector == sign_vector
    torch.testing.assert_close(
        converted._rht_sign_vector.cpu(),
        layer._rht_sign_vector.cpu(),
        atol=0,
        rtol=0,
    )


@pytest.mark.skipif(not has_triton(), reason="unsupported without triton")
@pytest.mark.skipif(not is_sm_at_least_100(), reason="Requires SM100+")
@pytest.mark.skipif(
    not torch_version_at_least("2.10.0"), reason="torch.compile requires PyTorch 2.10+"
)
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_nvfp4_mm_triton_cuda_graph_compile():
    """nvfp4_linear TRITON path works under reduce-overhead CUDA graphs."""
    M, K, N = 128, 256, 128
    layer = (
        NVFP4Linear(K, N, bias=False, kernel_preference=KernelPreference.TRITON)
        .cuda()
        .to(torch.bfloat16)
    )
    prepare_for_cuda_graph(torch.device("cuda"), sign_vectors=(layer.rht_sign_vector,))
    x = torch.randn(M, K, dtype=torch.bfloat16, device="cuda", requires_grad=True)

    compiled_layer = torch.compile(layer, mode="reduce-overhead", fullgraph=True)
    compiled_bwd = torch.compile(fullgraph=True, mode="reduce-overhead")

    for _ in range(3):
        with torch._dynamo.compiled_autograd._enable(compiled_bwd):
            compiled_layer(x).sum().backward()

    r1 = compiled_layer(x)
    r2 = compiled_layer(x)
    torch.testing.assert_close(r1, r2)

    x_hp = torch.randn(M, K, dtype=torch.bfloat16, device="cuda")
    ref = torch.nn.functional.linear(x_hp, layer.weight)
    nvfp4_out = layer(x_hp)
    sqnr = compute_error(ref, nvfp4_out)
    assert sqnr >= 15.0, f"Forward SQNR {sqnr:.2f} dB < 15 dB"

    # Use a fixed non-constant upstream gradient. With .sum().backward(), grad_output
    # is all ones, whose RHT quantization lands on exact values and can be
    # deterministic even when stochastic rounding is enabled.
    grad_out = torch.randn_like(r1)

    def one_step():
        x.grad = None
        layer.weight.grad = None
        with torch._dynamo.compiled_autograd._enable(compiled_bwd):
            compiled_layer(x).backward(grad_out)
        return layer.weight.grad.detach().clone()

    for _ in range(3):
        one_step()

    g1 = one_step()
    g2 = one_step()

    assert not torch.equal(g1, g2), (
        "Backward SR grad_weight must differ across steps because default CUDA RNG "
        "advances each replay"
    )
