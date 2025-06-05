# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD 3-Clause license found in the
# LICENSE file in the root directory of this source tree.
import pytest
import torch

from torchao.float8.config import Float8LinearConfig
from torchao.float8.float8_linear import _matmul_with_hp_or_float8_args
from torchao.float8.float8_tensor import LinearMMConfig, ScaledMMConfig
from torchao.prototype.float8nocompile.float8nocompile_linear import (
    matmul_with_args_in_hp,
)
from torchao.prototype.float8nocompile.float8nocompile_scaling_utils import (
    KernelAlgorithm,
)


# unit test comparing the two implementations
@pytest.mark.parametrize(
    "input_shape",
    [(32, 16), (1, 32, 16), (2, 32, 16)],
)
def test_matmul_with_args_in_hp(input_shape: tuple[int, int]):
    assert torch.cuda.is_available()
    device = "cuda"

    # high precision inputs
    input_bf16 = torch.randn(
        input_shape, dtype=torch.bfloat16, device=device, requires_grad=True
    )
    prod_input_bf16 = input_bf16.clone().detach().to(device).requires_grad_(True)
    prototype_input_bf16 = input_bf16.clone().detach().to(device).requires_grad_(True)

    # high precision weights
    # nn.Linear stores weights in transposed form
    weight_bf16 = torch.randn(
        (32, input_bf16.shape[-1]),
        dtype=torch.bfloat16,
        device=device,
        requires_grad=True,
    )
    prod_weight_bf16 = weight_bf16.clone().detach().to(device).requires_grad_(True)
    prototype_weight_bf16 = weight_bf16.clone().detach().to(device).requires_grad_(True)

    # default configs
    config = Float8LinearConfig()
    emulate = False
    linear_mm_config = linear_mm_config = LinearMMConfig(
        # output
        ScaledMMConfig(
            emulate,
            config.gemm_config_output.use_fast_accum,
            False,
            config.pad_inner_dim,
        ),
        # grad_input
        ScaledMMConfig(
            emulate,
            config.gemm_config_grad_input.use_fast_accum,
            False,
            config.pad_inner_dim,
        ),
        # grad_weight
        ScaledMMConfig(
            emulate,
            config.gemm_config_grad_weight.use_fast_accum,
            False,
            config.pad_inner_dim,
        ),
    )

    # prod forward. expects transposed weight.
    out_prod = _matmul_with_hp_or_float8_args.apply(
        prod_input_bf16, prod_weight_bf16.t(), linear_mm_config, config
    )

    # prototype forward. expects non-transposed weight
    out_prototype = matmul_with_args_in_hp.apply(
        prototype_input_bf16,
        prototype_weight_bf16,
        config,
        linear_mm_config,
        KernelAlgorithm.ATOMIC_MAX,
    )

    # compare model outputs
    assert torch.allclose(out_prod, out_prototype, atol=0, rtol=0)

    out_prod.sum().backward()
    out_prototype.sum().backward()

    # compare input gradients
    assert torch.allclose(
        prod_input_bf16.grad, prototype_input_bf16.grad, atol=0, rtol=0
    )

    # compare weight gradients
    assert torch.allclose(
        prod_weight_bf16.grad, prototype_weight_bf16.grad, atol=0, rtol=0
    )
