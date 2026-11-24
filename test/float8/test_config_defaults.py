# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD 3-Clause license found in the
# LICENSE file in the root directory of this source tree.

import pytest
import torch

from torchao.float8.config import (
    CastConfig,
    Float8LinearConfig,
    e4m3_dtype,
    e5m2_dtype,
)


@pytest.mark.parametrize(
    "field,override,dtype",
    [
        ("cast_config_input", "cast_config_input_for_grad_weight", e4m3_dtype),
        ("cast_config_weight", "cast_config_weight_for_grad_input", e4m3_dtype),
        (
            "cast_config_grad_output",
            "cast_config_grad_output_for_grad_weight",
            e5m2_dtype,
        ),
    ],
)
def test_resolving_default_dtype_does_not_mutate_cast_config(field, override, dtype):
    cast_config = CastConfig()
    configs = {cast_config: "original"}
    config = Float8LinearConfig(**{field: cast_config})

    assert cast_config.target_dtype is None
    assert configs[cast_config] == "original"
    assert getattr(config, field).target_dtype == dtype
    assert getattr(config, override).target_dtype == dtype


def test_shared_cast_config_gets_defaults_per_operand():
    cast_config = CastConfig()
    config = Float8LinearConfig(
        cast_config_input=cast_config,
        cast_config_weight=cast_config,
        cast_config_grad_output=cast_config,
    )

    assert cast_config.target_dtype is None
    assert config.cast_config_input.target_dtype == e4m3_dtype
    assert config.cast_config_weight.target_dtype == e4m3_dtype
    assert config.cast_config_grad_output.target_dtype == e5m2_dtype
    assert config.cast_config_grad_output_for_grad_weight.target_dtype == e5m2_dtype


def test_reused_cast_config_is_independent_between_linear_configs():
    cast_config = CastConfig()
    input_config = Float8LinearConfig(cast_config_input=cast_config)
    grad_config = Float8LinearConfig(cast_config_grad_output=cast_config)

    assert input_config.cast_config_input.target_dtype == e4m3_dtype
    assert grad_config.cast_config_grad_output.target_dtype == e5m2_dtype
    assert cast_config.target_dtype is None


def test_explicit_dtype_and_incompatible_override():
    cast_config = CastConfig(target_dtype=torch.float8_e5m2)
    config = Float8LinearConfig(cast_config_input=cast_config)
    assert config.cast_config_input.target_dtype == torch.float8_e5m2
    assert config.cast_config_input_for_grad_weight.target_dtype == torch.float8_e5m2

    with pytest.raises(AssertionError, match="input must be cast to the same dtype"):
        Float8LinearConfig(
            cast_config_input=CastConfig(target_dtype=torch.float8_e4m3fn),
            cast_config_input_for_grad_weight=CastConfig(
                target_dtype=torch.float8_e5m2
            ),
        )


@pytest.mark.skipif(
    not torch.cuda.is_available()
    or torch.version.cuda is None
    or torch.cuda.get_device_capability() < (8, 9),
    reason="FP8 matmul requires a supported CUDA device",
)
def test_shared_cast_config_gpu_forward_and_backward():
    import copy

    from torchao.float8.float8_linear import Float8Linear

    torch.manual_seed(0)
    linear = torch.nn.Linear(64, 32, device="cuda", dtype=torch.bfloat16)
    shared = CastConfig()
    shared_config = Float8LinearConfig(
        cast_config_input=shared,
        cast_config_weight=shared,
        cast_config_grad_output=shared,
    )
    explicit_config = Float8LinearConfig(
        cast_config_input=CastConfig(target_dtype=e4m3_dtype),
        cast_config_weight=CastConfig(target_dtype=e4m3_dtype),
        cast_config_grad_output=CastConfig(target_dtype=e5m2_dtype),
    )
    shared_model = Float8Linear.from_float(copy.deepcopy(linear), shared_config)
    explicit_model = Float8Linear.from_float(copy.deepcopy(linear), explicit_config)
    x = torch.randn(32, 64, device="cuda", dtype=torch.bfloat16, requires_grad=True)
    x_ref = x.detach().clone().requires_grad_(True)
    grad = torch.randn(32, 32, device="cuda", dtype=torch.bfloat16)

    out = shared_model(x)
    out_ref = explicit_model(x_ref)
    out.backward(grad)
    out_ref.backward(grad)
    torch.testing.assert_close(out, out_ref, rtol=0, atol=0)
    torch.testing.assert_close(x.grad, x_ref.grad, rtol=0, atol=0)
    torch.testing.assert_close(
        shared_model.weight.grad, explicit_model.weight.grad, rtol=0, atol=0
    )
