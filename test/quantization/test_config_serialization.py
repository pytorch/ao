# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD 3-Clause license found in the
# LICENSE file in the root directory of this source tree.

import json
import os
import tempfile
from dataclasses import dataclass
from unittest import mock

import pytest
import torch

from torchao.core.config import (
    AOBaseConfig,
    VersionMismatchError,
    config_from_dict,
    config_to_dict,
)
from torchao.quantization.quant_api import (
    Float8DynamicActivationFloat8WeightConfig,
    Float8WeightOnlyConfig,
    FPXWeightOnlyConfig,
    GemliteUIntXWeightOnlyConfig,
    Int4DynamicActivationInt4WeightConfig,
    Int4WeightOnlyConfig,
    Int8DynamicActivationInt4WeightConfig,
    Int8DynamicActivationInt8WeightConfig,
    Int8WeightOnlyConfig,
    ModuleFqnToConfig,
    PerRow,
    UIntXWeightOnlyConfig,
)
from torchao.sparsity.sparse_api import BlockSparseWeightConfig, SemiSparseWeightConfig

# Define test configurations as fixtures
configs = [
    Float8DynamicActivationFloat8WeightConfig(),
    Float8DynamicActivationFloat8WeightConfig(granularity=PerRow()),
    Float8WeightOnlyConfig(
        weight_dtype=torch.float8_e4m3fn,
    ),
    UIntXWeightOnlyConfig(dtype=torch.uint1),
    Int4DynamicActivationInt4WeightConfig(),
    Int4WeightOnlyConfig(
        group_size=32,
    ),
    Int8DynamicActivationInt4WeightConfig(
        group_size=64,
    ),
    Int8DynamicActivationInt8WeightConfig(),
    # Int8DynamicActivationInt8WeightConfig(layout=SemiSparseLayout()),
    Int8WeightOnlyConfig(
        group_size=128,
    ),
    UIntXWeightOnlyConfig(
        dtype=torch.uint3,
        group_size=32,
        use_hqq=True,
    ),
    GemliteUIntXWeightOnlyConfig(
        group_size=128,  # Optional, has default of 64
        bit_width=8,  # Optional, has default of 4
    ),
    FPXWeightOnlyConfig(ebits=4, mbits=8),
    # Sparsity configs
    SemiSparseWeightConfig(),
    BlockSparseWeightConfig(blocksize=128),
    ModuleFqnToConfig({}),
    ModuleFqnToConfig({"_default": Int4WeightOnlyConfig(), "linear1": None}),
    ModuleFqnToConfig(
        {
            "linear1": Int4WeightOnlyConfig(),
            "linear2": Int8DynamicActivationInt4WeightConfig(),
        }
    ),
]


# Create ids for better test naming
def get_config_ids(configs):
    if not isinstance(configs, list):
        configs = [configs]
    return [config.__class__.__name__ for config in configs]


@pytest.mark.parametrize("config", configs, ids=get_config_ids)
def test_reconstructable_dict_file_round_trip(config):
    """Test saving and loading reconstructable dicts to/from JSON files."""
    # Get a reconstructable dict
    reconstructable = config_to_dict(config)

    # Create a temporary file to save the JSON
    with tempfile.NamedTemporaryFile(
        mode="w+", suffix=".json", delete=False
    ) as temp_file:
        # Write the reconstructable dict as JSON
        json.dump(reconstructable, temp_file)
        temp_file_path = temp_file.name

    try:
        # Read back the JSON file
        with open(temp_file_path, "r") as file:
            loaded_dict = json.load(file)

        # Reconstruct from the loaded dict
        reconstructed = config_from_dict(loaded_dict)

        # Check it's the right class
        assert isinstance(reconstructed, config.__class__)

        # Verify attributes match
        for attr_name in config.__dict__:
            if not attr_name.startswith("_"):  # Skip private attributes
                original_value = getattr(config, attr_name)
                reconstructed_value = getattr(reconstructed, attr_name)

                # Special handling for torch dtypes
                if (
                    hasattr(original_value, "__module__")
                    and original_value.__module__ == "torch"
                ):
                    assert str(original_value) == str(reconstructed_value), (
                        f"Attribute {attr_name} mismatch after file round trip for {config.__class__.__name__}"
                    )
                else:
                    assert original_value == reconstructed_value, (
                        f"Attribute {attr_name} mismatch after file round trip for {config.__class__.__name__}"
                    )

    finally:
        # Clean up the temporary file
        if os.path.exists(temp_file_path):
            os.unlink(temp_file_path)


# Define a dummy config in a non-allowed module
@dataclass
class DummyNonAllowedConfig(AOBaseConfig):
    VERSION = 2
    value: int = 42


def test_disallowed_modules():
    """Test that configs from non-allowed modules are rejected during reconstruction."""
    # Create a config from a non-allowed module
    dummy_config = DummyNonAllowedConfig()
    reconstructable = config_to_dict(dummy_config)

    with pytest.raises(
        ValueError,
        match="Failed to find class DummyNonAllowedConfig in any of the allowed modules",
    ):
        config_from_dict(reconstructable)

    # Use mock.patch as a context manager
    with mock.patch("torchao.core.config.ALLOWED_AO_MODULES", {__name__}):
        reconstructed = config_from_dict(reconstructable)
        assert isinstance(reconstructed, DummyNonAllowedConfig)
        assert reconstructed.value == 42
        assert reconstructed.VERSION == 2


def test_version_mismatch():
    """Test that version mismatch raises an error during reconstruction."""
    # Create a config
    dummy_config = DummyNonAllowedConfig()
    reconstructable = config_to_dict(dummy_config)

    # Modify the version in the dict to create a mismatch
    reconstructable["_version"] = 1

    # Patch to allow the module but should still fail due to version mismatch
    with mock.patch("torchao.core.config.ALLOWED_AO_MODULES", {__name__}):
        with pytest.raises(
            VersionMismatchError,
            match="Version mismatch for DummyNonAllowedConfig: stored version 1 != current version 2",
        ):
            config_from_dict(reconstructable)


if __name__ == "__main__":
    pytest.main([__file__])
