from functools import partial

import pytest
import torch.nn as nn

from torchao.float8 import _auto_filter_for_recipe
from torchao.float8.config import Float8LinearRecipeName
from torchao.float8.float8_linear_utils import (
    _auto_filter_for_rowwise,
    _auto_filter_for_tensorwise,
)


@pytest.fixture
def sample_filter_fqns():
    """Fixture providing sample filter FQNs."""
    return ["layer1", "layer2", "filtered_layer"]


def test_tensorwise_recipe_returns_partial_function(sample_filter_fqns):
    """Test that tensorwise recipe returns a partial function."""
    filter_func = _auto_filter_for_recipe(
        Float8LinearRecipeName.TENSORWISE, sample_filter_fqns
    )

    assert isinstance(filter_func, partial)
    assert filter_func.keywords == {"filter_fqns": sample_filter_fqns}


def test_rowwise_recipe_returns_partial_function(sample_filter_fqns):
    """Test that rowwise recipe returns a partial function."""
    filter_func = _auto_filter_for_recipe(
        Float8LinearRecipeName.ROWWISE, sample_filter_fqns
    )

    assert isinstance(filter_func, partial)
    assert filter_func.keywords == {"filter_fqns": sample_filter_fqns}


def test_rowwise_with_gw_hp_raises_not_implemented(sample_filter_fqns):
    """Test that ROWWISE_WITH_GW_HP recipe raises NotImplementedError."""
    with pytest.raises(NotImplementedError, match="Unsupported recipe"):
        _auto_filter_for_recipe(
            Float8LinearRecipeName.ROWWISE_WITH_GW_HP, sample_filter_fqns
        )


@pytest.mark.parametrize(
    "invalid_recipe",
    ["invalid_recipe_name", "tensorwise_typo", "rowwise_typo", "", None],
)
def test_invalid_recipe_raises_value_error(invalid_recipe, sample_filter_fqns):
    """Test that invalid recipes raise ValueError."""
    with pytest.raises(ValueError):
        _auto_filter_for_recipe(invalid_recipe, sample_filter_fqns)


@pytest.mark.parametrize(
    "recipe_type,module_dims,fqn,filter_fqns,expected",
    [
        # Tensorwise tests
        ("tensorwise", (8192, 2048), "valid.layer", [], True),
        # FQN matches filter
        ("tensorwise", (8192, 2048), "skip_layer.linear", ["skip_layer"], False),
        # Threshold fail
        ("tensorwise", (4096, 1024), "valid.layer", [], False),
        # Rowwise tests
        ("rowwise", (4096, 8192), "valid.layer", [], True),
        ("rowwise", (4096, 8192), "skip_layer.linear", ["skip_layer"], False),
        # Combined threshold fail
        (
            "rowwise",
            (2048, 4096),
            "valid.layer",
            [],
            False,
        ),
    ],
)
def test_end_to_end_filtering(recipe_type, module_dims, fqn, filter_fqns, expected):
    """Test complete filtering workflow for both recipe types."""
    in_features, out_features = module_dims

    # Get the filter function
    filter_func = _auto_filter_for_recipe(recipe_type, filter_fqns)

    # Create test module
    test_module = nn.Linear(in_features, out_features)

    # Test filtering
    result = filter_func(test_module, fqn)
    assert result is expected


def test_exact_boundary_dimensions_rowwise():
    """Test exact boundary dimensions for rowwise filtering."""
    # Test exact thresholds
    module_n_2048 = nn.Linear(4096, 2048)  # N exactly 2048
    assert _auto_filter_for_rowwise(module_n_2048, "layer", []) is False

    module_k_1024 = nn.Linear(1024, 4112)  # K exactly 1024
    assert _auto_filter_for_rowwise(module_k_1024, "layer", []) is False


def test_exact_boundary_dimensions_tensorwise():
    """Test exact boundary dimensions for tensorwise filtering."""
    # Test exact combined threshold
    module_boundary = nn.Linear(4096, 1024)  # K=4096, N=1024
    assert _auto_filter_for_tensorwise(module_boundary, "layer", []) is False


def test_partial_fqn_matching():
    """Test partial FQN matching behavior."""
    filter_fqns = ["embed", "norm"]
    large_module = nn.Linear(8192, 4096)

    # (fqn, expected result from filter func)
    test_cases = [
        ("model.embeddings.linear", False),  # Contains "embed"
        ("layer.norm.weight", False),  # Contains "norm"
        ("model.transformer.layer", True),  # Doesn't contain either
        ("embedding_layer", False),  # Contains "embed" as substring
    ]

    for fqn, expected_result in test_cases:
        result_tensorwise = _auto_filter_for_tensorwise(large_module, fqn, filter_fqns)
        result_rowwise = _auto_filter_for_rowwise(large_module, fqn, filter_fqns)
        assert result_tensorwise is expected_result, (
            f"Tensorwise result mismatch: fqn={fqn}, expected={expected_result}, actual={result_tensorwise}"
        )
        assert result_rowwise is expected_result, (
            f"Rowwise result mismatch: fqn={fqn}, expected={expected_result}, actual={result_rowwise}"
        )
