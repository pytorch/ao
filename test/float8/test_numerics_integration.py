# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD 3-Clause license found in the
# LICENSE file in the root directory of this source tree.

# Tests LLaMa FeedForward numerics with float8

import copy
from typing import Optional

import pytest

from torchao.utils import (
    TORCH_VERSION_AT_LEAST_2_5,
    is_sm_at_least_89,
    is_sm_at_least_90,
)

if not TORCH_VERSION_AT_LEAST_2_5:
    pytest.skip("Unsupported PyTorch version", allow_module_level=True)

import torch
import torch.nn as nn
import torch.nn.functional as F

from torchao.float8.config import (
    Float8LinearConfig,
    Float8LinearRecipeName,
    ScalingType,
)
from torchao.float8.float8_linear_utils import (
    convert_to_float8_training,
)
from torchao.float8.float8_utils import IS_ROCM, compute_error
from torchao.testing.training.test_utils import get_test_float8_linear_config

torch.manual_seed(0)


# copied from https://github.com/pytorch/torchtitan/blob/main/torchtitan/models/llama/model.py
class FeedForward(nn.Module):
    """
    FeedForward module

    Args:
        dim (int): Input dimension.
        hidden_dim (int): Hidden dimension of the feedforward layer.
        multiple_of (int): Value to ensure hidden dimension is a multiple of this value.
        ffn_dim_multiplier (Optional[float]): Custom multiplier for hidden dimension. Defaults to None.

    Attributes:
        w1 (Linear): Linear transformation for the first layer.
        w2 (Linear): Linear transformation for the second layer.
        w3 (Linear): Linear transformation for the third layer.

    """

    def __init__(
        self,
        dim: int,
        hidden_dim: int,
        multiple_of: int,
        ffn_dim_multiplier: Optional[float],
    ):
        super().__init__()
        hidden_dim = int(2 * hidden_dim / 3)
        # custom dim factor multiplier
        if ffn_dim_multiplier is not None:
            hidden_dim = int(ffn_dim_multiplier * hidden_dim)
        hidden_dim = multiple_of * ((hidden_dim + multiple_of - 1) // multiple_of)

        self.w1 = nn.Linear(dim, hidden_dim, bias=False)
        self.w2 = nn.Linear(hidden_dim, dim, bias=False)
        self.w3 = nn.Linear(dim, hidden_dim, bias=False)

    def forward(self, x):
        return self.w2(F.silu(self.w1(x)) * self.w3(x))

    def init_weights(self, init_std: float):
        nn.init.trunc_normal_(self.w1.weight, mean=0.0, std=0.02)
        for linear in (self.w2, self.w3):
            nn.init.trunc_normal_(linear.weight, mean=0.0, std=init_std)


class TestFloat8NumericsIntegrationTest:
    def _test_impl(self, config: Float8LinearConfig) -> None:
        data_dtype = torch.bfloat16
        # LLaMa 3 70B shapes
        model_ref = (
            FeedForward(
                dim=4096,
                hidden_dim=16384,
                multiple_of=1024,
                ffn_dim_multiplier=1.3,
            )
            .cuda()
            .to(data_dtype)
        )

        # for now just test the encoder to simplify things
        model_fp8 = copy.deepcopy(model_ref)

        convert_to_float8_training(
            model_fp8,
            config=config,
        )

        lr = 0.01
        optim_ref = torch.optim.SGD(model_ref.parameters(), lr=lr)
        optim_fp8 = torch.optim.SGD(model_fp8.parameters(), lr=lr)

        # Note: you need two different inputs to properly test numerics
        # of delayed scaling, because the first time around the initialization
        # logic of delayed scaling behaves as dynamic scaling
        # TODO(future PR): delete ^, since we deleted delayed scaling
        shape = (1, 8192, 4096)
        data1 = torch.randn(*shape, device="cuda", dtype=data_dtype)
        data2 = torch.randn(*shape, device="cuda", dtype=data_dtype)

        model_ref(data1).sum().backward()
        # zero out grads without stepping, since we just want to compare grads
        # of the second datum
        optim_ref.zero_grad()
        model_ref_out = model_ref(data2)
        model_ref_out.sum().backward()

        model_fp8(data1).sum().backward()
        # zero out grads without stepping, since we just want to compare grads
        # of the second datum
        optim_fp8.zero_grad()
        model_fp8_out = model_fp8(data2)
        model_fp8_out.sum().backward()

        out_sqnr = compute_error(model_ref_out, model_fp8_out)
        assert out_sqnr > 20.0

        ref_name_to_grad = {
            name: param.grad for name, param in model_ref.named_parameters()
        }

        grad_sqnr_threshold = 20.0

        for name, param in model_fp8.named_parameters():
            ref_grad = ref_name_to_grad[name]
            cur_grad = param.grad
            sqnr = compute_error(ref_grad, cur_grad)
            assert sqnr > grad_sqnr_threshold

    @pytest.mark.parametrize(
        "scaling_type_input",
        [ScalingType.DYNAMIC],
    )
    @pytest.mark.parametrize(
        "scaling_type_weight",
        [ScalingType.DYNAMIC],
    )
    @pytest.mark.parametrize(
        "scaling_type_grad_output",
        [ScalingType.DYNAMIC],
    )
    @pytest.mark.skipif(
        not is_sm_at_least_89(), reason="requires SM89 compatible machine"
    )
    @pytest.mark.skipif(IS_ROCM, reason="test doesn't currently work on the ROCm stack")
    def test_encoder_fw_bw_from_config_params(
        self,
        scaling_type_input: ScalingType,
        scaling_type_weight: ScalingType,
        scaling_type_grad_output: ScalingType,
    ):
        config = get_test_float8_linear_config(
            scaling_type_input,
            scaling_type_weight,
            scaling_type_grad_output,
            emulate=False,
        )
        self._test_impl(config)

    @pytest.mark.parametrize(
        "recipe_name",
        [
            Float8LinearRecipeName.ROWWISE,
            Float8LinearRecipeName.ROWWISE_WITH_GW_HP,
        ],
    )
    @pytest.mark.skipif(
        not is_sm_at_least_90(), reason="requires SM90 compatible machine"
    )
    @pytest.mark.skipif(IS_ROCM, reason="test doesn't currently work on the ROCm stack")
    def test_encoder_fw_bw_from_recipe(
        self,
        recipe_name: str,
    ):
        config = Float8LinearConfig.from_recipe_name(recipe_name)
        self._test_impl(config)


if __name__ == "__main__":
    pytest.main([__file__])
