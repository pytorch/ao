# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD 3-Clause license found in the
# LICENSE file in the root directory of this source tree.

import copy

import pytest
import torch
from torch._dynamo.testing import CompileCounterWithBackend

from torchao.utils import is_sm_at_least_90

triton = pytest.importorskip("triton", reason="Triton required to run this test")
if not is_sm_at_least_90():
    pytest.skip("This test requires SM90 or higher", allow_module_level=True)


from torchao.float8.float8_utils import compute_error
from torchao.prototype.blockwise_fp8_training.linear import Float8BlockwiseLinear

torch.random.manual_seed(0)


def _run_blockwise_quant_linear_fwd_bwd(
    in_features,
    out_features,
    batch_size,
    block_size,
    *,
    compile_mode: bool = False,
):
    if in_features % block_size != 0 or out_features % block_size != 0:
        pytest.skip(f"Dimensions must be divisible by block_size={block_size}")

    layer_ref = torch.nn.Linear(
        in_features=in_features,
        out_features=out_features,
        bias=False,
    ).cuda()
    layer_test = Float8BlockwiseLinear.from_float(copy.deepcopy(layer_ref))
    compiled_frame_counter = None
    layer_under_test = layer_test
    compiled_step = None

    if compile_mode:
        with torch._dynamo.config.patch(trace_autograd_ops=True):
            torch._dynamo.reset()
            compiled_frame_counter = CompileCounterWithBackend("inductor")

            def step(x):
                y = layer_test(x)
                x_grad, weight_grad = torch.autograd.grad(
                    y.sum(),
                    (x, layer_test.weight),
                )
                return y.detach(), x_grad, weight_grad

            compiled_step = torch.compile(
                step,
                backend=compiled_frame_counter,
                fullgraph=True,
            )

    x_test = torch.randn(batch_size, 256, in_features).cuda().requires_grad_(True)
    x_ref = x_test.clone().detach().requires_grad_(True)

    if compile_mode:
        assert compiled_step is not None
        with torch._dynamo.config.patch(trace_autograd_ops=True):
            y_test, x_grad_test, weight_grad_test = compiled_step(x_test)
    else:
        y_test = layer_under_test(x_test)

    y_ref = layer_ref(x_ref)

    if compile_mode:
        assert compiled_frame_counter is not None
        assert compiled_frame_counter.frame_count == 1, (
            "Compiled blockwise linear should run in a single frame"
        )

    sqnr = compute_error(y_ref, y_test)
    assert not y_test.isnan().any(), "Output must not contain NaNs"
    assert sqnr >= 25.0, f"SQNR: {sqnr.item()} must be >= 25.0"
    assert not sqnr.isinf().any(), "SQNR must not be inf"

    if compile_mode:
        x_grad_ref, weight_grad_ref = torch.autograd.grad(
            y_ref.sum(),
            (x_ref, layer_ref.weight),
        )
    else:
        y_test.sum().backward()
        y_ref.sum().backward()
        x_grad_test = x_test.grad
        weight_grad_test = layer_test.weight.grad
        x_grad_ref = x_ref.grad
        weight_grad_ref = layer_ref.weight.grad

    sqnr = compute_error(x_grad_ref, x_grad_test)
    assert not x_grad_test.isnan().any(), "Input grad must not contain NaNs"
    assert sqnr >= 30.0, f"SQNR: {sqnr} must be >= 25.0"

    sqnr = compute_error(weight_grad_ref, weight_grad_test)
    assert not weight_grad_test.isnan().any(), "Weight grad must not contain NaNs"
    assert sqnr >= 30.0, f"SQNR: {sqnr} must be >= 25.0"


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
@pytest.mark.parametrize("in_features", [4096])
@pytest.mark.parametrize("out_features", [128256])
@pytest.mark.parametrize("batch_size", [1, 8])
@pytest.mark.parametrize("block_size", [128])
def test_blockwise_quant_linear_fwd_bwd(
    in_features,
    out_features,
    batch_size,
    block_size,
):
    _run_blockwise_quant_linear_fwd_bwd(
        in_features,
        out_features,
        batch_size,
        block_size,
    )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
@pytest.mark.parametrize("in_features", [4096])
@pytest.mark.parametrize("out_features", [4096])
@pytest.mark.parametrize("batch_size", [1])
@pytest.mark.parametrize("block_size", [128])
def test_blockwise_quant_linear_compile_fullgraph_fwd_bwd(
    in_features,
    out_features,
    batch_size,
    block_size,
):
    _run_blockwise_quant_linear_fwd_bwd(
        in_features,
        out_features,
        batch_size,
        block_size,
        compile_mode=True,
    )
