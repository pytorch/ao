# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD 3-Clause license found in the
# LICENSE file in the root directory of this source tree.

"""
Tests for unified MoE and dense layer training using FqnToConfig.
"""

from collections import OrderedDict

import torch
from torch import nn

from torchao.prototype.moe_training.conversion_utils import (
    FP8GroupedMMRecipe,
    GroupedMMConfig,
    MXFP8GroupedMMRecipe,
)
from torchao.prototype.moe_training.tensor import ScaledGroupedMMTensor
from torchao.prototype.mx_formats.config import MXLinearConfig
from torchao.prototype.mx_formats.mx_linear import MXLinear
from torchao.quantization import FqnToConfig
from torchao.quantization.quant_api import quantize_


class ExpertWeights(nn.Module):
    """Container for expert parameters."""

    def __init__(self, num_experts, hidden_dim, dim):
        super().__init__()
        self.w1 = nn.Parameter(
            torch.randn(num_experts, hidden_dim, dim, dtype=torch.bfloat16)
        )
        self.w2 = nn.Parameter(
            torch.randn(num_experts, dim, hidden_dim, dtype=torch.bfloat16)
        )


class MoEModel(nn.Module):
    """Model with both MoE-style expert layers and regular dense layers."""

    def __init__(self, num_experts=8, dim=128, hidden_dim=256):
        super().__init__()
        self.experts = ExpertWeights(num_experts, hidden_dim, dim)

        # Dense layers
        self.pre_moe = nn.Linear(dim, dim, bias=False, dtype=torch.bfloat16)
        self.post_moe = nn.Linear(dim, dim, bias=False, dtype=torch.bfloat16)

    def forward(self, x):
        # just testing conversion here, no need to implement real forward pass
        return x


def test_fqn_to_config_simple():
    """Test simple FQN-based configuration with single quantize_() call."""
    model = MoEModel()

    # Configure different quantization for experts vs dense layers using FqnToConfig
    config = FqnToConfig(
        fqn_to_config=OrderedDict(
            [
                # Apply GroupedMMConfig to expert parameters
                ("experts", GroupedMMConfig(recipe=FP8GroupedMMRecipe.ROWWISE)),
                # Apply MXLinearConfig to dense layers
                (
                    "pre_moe",
                    MXLinearConfig(elem_dtype=torch.float8_e4m3fn, block_size=32),
                ),
                (
                    "post_moe",
                    MXLinearConfig(elem_dtype=torch.float8_e4m3fn, block_size=32),
                ),
            ]
        )
    )

    # Single quantize_() call transforms both layer types!
    quantize_(model, config, filter_fn=None)

    # Verify transformations
    assert isinstance(model.experts.w1.data, ScaledGroupedMMTensor), (
        "w1 should be ScaledGroupedMMTensor"
    )
    assert isinstance(model.experts.w2.data, ScaledGroupedMMTensor), (
        "w2 should be ScaledGroupedMMTensor"
    )
    assert model.experts.w1.data.recipe == FP8GroupedMMRecipe.ROWWISE
    assert isinstance(model.pre_moe, MXLinear), "pre_moe should be MXLinear"
    assert isinstance(model.post_moe, MXLinear), "post_moe should be MXLinear"


def test_fqn_to_config_with_regex():
    """Test FQN-based configuration using regex patterns."""
    model = MoEModel()

    # Use regex patterns to match multiple modules
    config = FqnToConfig(
        fqn_to_config=OrderedDict(
            [
                ("re:.*experts.*", GroupedMMConfig(recipe=MXFP8GroupedMMRecipe.RCEIL)),
                (
                    "re:^(pre_moe|post_moe)$",
                    MXLinearConfig(elem_dtype=torch.float8_e4m3fn, block_size=32),
                ),
            ]
        )
    )

    quantize_(model, config, filter_fn=None)

    # Verify transformations
    assert isinstance(model.experts.w1.data, ScaledGroupedMMTensor), (
        "w1 should be ScaledGroupedMMTensor"
    )
    assert model.experts.w1.data.recipe == MXFP8GroupedMMRecipe.RCEIL
    assert isinstance(model.experts.w2.data, ScaledGroupedMMTensor), (
        "w2 should be ScaledGroupedMMTensor"
    )
    assert isinstance(model.pre_moe, MXLinear), "pre_moe should be MXLinear"
    assert isinstance(model.post_moe, MXLinear), "post_moe should be MXLinear"


def test_fqn_to_config_experts_only():
    """Test FQN-based configuration for experts only."""
    model = MoEModel()

    # Only quantize expert parameters using FqnToConfig
    config = FqnToConfig(
        fqn_to_config=OrderedDict(
            [
                ("re:.*experts.*", GroupedMMConfig(recipe=FP8GroupedMMRecipe.ROWWISE)),
            ]
        )
    )

    quantize_(model, config, filter_fn=None)

    # Verify transformations
    assert isinstance(model.experts.w1.data, ScaledGroupedMMTensor), (
        "w1 should be ScaledGroupedMMTensor"
    )
    assert isinstance(model.experts.w2.data, ScaledGroupedMMTensor), (
        "w2 should be ScaledGroupedMMTensor"
    )
    # Dense layers should remain unchanged
    assert isinstance(model.pre_moe, nn.Linear) and not isinstance(
        model.pre_moe, MXLinear
    ), "pre_moe should remain nn.Linear"
    assert isinstance(model.post_moe, nn.Linear) and not isinstance(
        model.post_moe, MXLinear
    ), "post_moe should remain nn.Linear"


def test_fqn_to_config_selective_layers():
    """Test selective layer quantization using FqnToConfig."""
    model = MoEModel()

    # Quantize experts and only pre_moe
    config = FqnToConfig(
        fqn_to_config=OrderedDict(
            [
                ("re:.*experts.*", GroupedMMConfig(recipe=FP8GroupedMMRecipe.ROWWISE)),
                (
                    "pre_moe",
                    MXLinearConfig(elem_dtype=torch.float8_e4m3fn, block_size=32),
                ),
            ]
        )
    )

    quantize_(model, config, filter_fn=None)

    # Verify transformations
    assert isinstance(model.experts.w1.data, ScaledGroupedMMTensor), (
        "w1 should be ScaledGroupedMMTensor"
    )
    assert isinstance(model.experts.w2.data, ScaledGroupedMMTensor), (
        "w2 should be ScaledGroupedMMTensor"
    )
    assert isinstance(model.pre_moe, MXLinear), "pre_moe should be MXLinear"
    # post_moe should remain unchanged
    assert isinstance(model.post_moe, nn.Linear) and not isinstance(
        model.post_moe, MXLinear
    ), "post_moe should remain nn.Linear"


def test_fqn_to_config_mxfp8_wgrad_with_hp():
    """Test FqnToConfig with MXFP8 high-precision weight gradients recipe."""
    model = MoEModel()

    config = FqnToConfig(
        fqn_to_config=OrderedDict(
            [
                (
                    "re:.*experts.*",
                    GroupedMMConfig(recipe=MXFP8GroupedMMRecipe.RCEIL_WGRAD_WITH_HP),
                ),
                (
                    "re:^(pre_moe|post_moe)$",
                    MXLinearConfig(elem_dtype=torch.float8_e4m3fn, block_size=32),
                ),
            ]
        )
    )

    quantize_(model, config, filter_fn=None)

    # Verify transformations
    assert isinstance(model.experts.w1.data, ScaledGroupedMMTensor), (
        "w1 should be ScaledGroupedMMTensor"
    )
    assert model.experts.w1.data.recipe == MXFP8GroupedMMRecipe.RCEIL_WGRAD_WITH_HP, (
        "w1 should use RCEIL_WGRAD_WITH_HP recipe"
    )
    assert isinstance(model.experts.w2.data, ScaledGroupedMMTensor), (
        "w2 should be ScaledGroupedMMTensor"
    )
    assert isinstance(model.pre_moe, MXLinear), "pre_moe should be MXLinear"
    assert isinstance(model.post_moe, MXLinear), "post_moe should be MXLinear"


def test_fqn_to_config_dense_only():
    """Test FqnToConfig for dense layers only, leaving experts unchanged."""
    model = MoEModel()

    # Only quantize dense layers
    config = FqnToConfig(
        fqn_to_config=OrderedDict(
            [
                (
                    "re:^(pre_moe|post_moe)$",
                    MXLinearConfig(elem_dtype=torch.float8_e4m3fn, block_size=32),
                ),
            ]
        )
    )

    quantize_(model, config, filter_fn=None)

    # Verify only Linear layers were transformed
    assert not isinstance(model.experts.w1.data, ScaledGroupedMMTensor), (
        "w1 should remain regular tensor"
    )
    assert not isinstance(model.experts.w2.data, ScaledGroupedMMTensor), (
        "w2 should remain regular tensor"
    )
    assert isinstance(model.pre_moe, MXLinear), "pre_moe should be MXLinear"
    assert isinstance(model.post_moe, MXLinear), "post_moe should be MXLinear"


def test_fqn_to_config_specific_expert_params():
    """Test FqnToConfig with different configs for different expert parameters."""
    model = MoEModel()

    # Apply different configs to w1 vs w2
    config = FqnToConfig(
        fqn_to_config=OrderedDict(
            [
                # we wouldn't normally mix fp8 and mxfp8, but just to test granular fqn selection is working
                ("experts.w1", GroupedMMConfig(recipe=FP8GroupedMMRecipe.ROWWISE)),
                ("experts.w2", GroupedMMConfig(recipe=MXFP8GroupedMMRecipe.RCEIL)),
                (
                    "re:^(pre_moe|post_moe)$",
                    MXLinearConfig(elem_dtype=torch.float8_e4m3fn, block_size=32),
                ),
            ]
        )
    )
    quantize_(model, config, filter_fn=None)

    # Verify different recipes were applied
    assert isinstance(model.experts.w1.data, ScaledGroupedMMTensor), (
        "w1 should be ScaledGroupedMMTensor"
    )
    assert model.experts.w1.data.recipe == FP8GroupedMMRecipe.ROWWISE, (
        "w1 should use FP8 ROWWISE"
    )
    assert isinstance(model.experts.w2.data, ScaledGroupedMMTensor), (
        "w2 should be ScaledGroupedMMTensor"
    )
    assert model.experts.w2.data.recipe == MXFP8GroupedMMRecipe.RCEIL, (
        "w2 should use MXFP8 RCEIL"
    )
    assert isinstance(model.pre_moe, MXLinear), "pre_moe should be MXLinear"
    assert isinstance(model.post_moe, MXLinear), "post_moe should be MXLinear"
