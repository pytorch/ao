import torch.nn as nn
from torch.testing._internal.common_utils import (
    TestCase,
    instantiate_parametrized_tests,
    parametrize,
    run_tests,
)

from torchao.float8 import _auto_filter_for_recipe
from torchao.float8.float8_linear_utils import (
    _auto_filter_for_rowwise,
    _auto_filter_for_tensorwise,
)


class TestAutoFilter(TestCase):
    @parametrize(
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
    def test_end_to_end_filtering(
        self, recipe_type, module_dims, fqn, filter_fqns, expected
    ):
        """Test complete filtering workflow for both recipe types."""
        in_features, out_features = module_dims

        # Get the filter function
        filter_func = _auto_filter_for_recipe(recipe_type, filter_fqns)

        # Create test module
        test_module = nn.Linear(in_features, out_features)

        # Test filtering
        result = filter_func(test_module, fqn)
        assert result is expected

    def test_exact_boundary_dimensions_rowwise(self):
        """Test exact boundary dimensions for rowwise filtering."""
        # Test exact thresholds
        module_n_2048 = nn.Linear(4096, 2048)  # N exactly 2048
        assert _auto_filter_for_rowwise(module_n_2048, "layer", []) is False

        module_k_1024 = nn.Linear(1024, 4112)  # K exactly 1024
        assert _auto_filter_for_rowwise(module_k_1024, "layer", []) is False

    def test_exact_boundary_dimensions_tensorwise(self):
        """Test exact boundary dimensions for tensorwise filtering."""
        # Test exact combined threshold
        module_boundary = nn.Linear(4096, 1024)  # K=4096, N=1024
        assert _auto_filter_for_tensorwise(module_boundary, "layer", []) is False

    def test_partial_fqn_matching(self):
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
            result_tensorwise = _auto_filter_for_tensorwise(
                large_module, fqn, filter_fqns
            )
            result_rowwise = _auto_filter_for_rowwise(large_module, fqn, filter_fqns)
            assert result_tensorwise is expected_result, (
                f"Tensorwise result mismatch: fqn={fqn}, expected={expected_result}, actual={result_tensorwise}"
            )
            assert result_rowwise is expected_result, (
                f"Rowwise result mismatch: fqn={fqn}, expected={expected_result}, actual={result_rowwise}"
            )


instantiate_parametrized_tests(TestAutoFilter)

if __name__ == "__main__":
    run_tests()
