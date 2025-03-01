import json
import os
import tempfile

import pytest
import torch

from torchao.core.config import reconstruct_from_dict, to_reconstructable_dict
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
    PerRow,
    UIntXWeightOnlyConfig,
)

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
        packing_bitwidth=8,  # Optional, has default of 32
        contiguous=True,  # Optional, has default of None
    ),
    FPXWeightOnlyConfig(ebits=4, mbits=8),
]


# Create ids for better test naming
def get_config_ids(configs):
    return [config.__class__.__name__ for config in configs]


# Parametrized tests
@pytest.mark.parametrize("config", configs, ids=get_config_ids)
def test_to_dict_serialization(config):
    """Test that all configs can be serialized to a dictionary."""
    # Test to_dict method exists and returns a dict
    assert hasattr(
        config, "to_dict"
    ), f"{config.__class__.__name__} missing to_dict method"
    result = config.to_dict()
    assert isinstance(result, dict)

    # Check that all essential attributes are present in the dict
    for attr_name in config.__dict__:
        if not attr_name.startswith("_"):  # Skip private attributes
            assert attr_name in result, f"{attr_name} missing in serialized dict"


@pytest.mark.parametrize("config", configs, ids=get_config_ids)
def test_to_json_serialization(config):
    """Test that all configs can be serialized to JSON."""
    # Test to_json method exists and returns a string
    assert hasattr(
        config, "to_json"
    ), f"{config.__class__.__name__} missing to_json method"
    json_str = config.to_json()
    assert isinstance(json_str, str)

    # Verify it's valid JSON
    try:
        parsed = json.loads(json_str)
        assert isinstance(parsed, dict)
    except json.JSONDecodeError as e:
        pytest.fail(f"Invalid JSON for {config.__class__.__name__}: {e}")


@pytest.mark.parametrize("config", configs, ids=get_config_ids)
def test_from_dict_deserialization(config):
    """Test that all configs can be deserialized from a dictionary."""
    # Get the class of the instance
    cls = config.__class__

    # Serialize to dict
    data = config.to_dict()

    # Test from_dict class method exists
    assert hasattr(cls, "from_dict"), f"{cls.__name__} missing from_dict class method"

    # Deserialize back to instance
    deserialized = cls.from_dict(data)

    # Check it's the right class
    assert isinstance(deserialized, cls)

    # Compare key attributes
    for attr_name in config.__dict__:
        if not attr_name.startswith("_"):  # Skip private attributes
            original_value = getattr(config, attr_name)
            deserialized_value = getattr(deserialized, attr_name)

            # Special handling for torch dtypes
            if (
                hasattr(original_value, "__module__")
                and original_value.__module__ == "torch"
            ):
                assert str(original_value) == str(
                    deserialized_value
                ), f"Attribute {attr_name} mismatch for {cls.__name__}"
            else:
                assert (
                    original_value == deserialized_value
                ), f"Attribute {attr_name} mismatch for {cls.__name__}"


@pytest.mark.parametrize("config", configs, ids=get_config_ids)
def test_from_json_deserialization(config):
    """Test that all configs can be deserialized from JSON."""
    # Get the class of the instance
    cls = config.__class__

    # Serialize to JSON
    json_str = config.to_json()

    # Test from_json class method exists
    assert hasattr(cls, "from_json"), f"{cls.__name__} missing from_json class method"

    # Deserialize back to instance
    deserialized = cls.from_json(json_str)

    # Check it's the right class
    assert isinstance(deserialized, cls)

    # Verify the instance is equivalent to the original
    # This assumes __eq__ is properly implemented
    assert (
        config == deserialized
    ), f"Deserialized instance doesn't match original for {cls.__name__}"


@pytest.mark.parametrize("config", configs, ids=get_config_ids)
def test_round_trip_equivalence(config):
    """Test complete serialization and deserialization round trip."""
    # JSON round trip
    json_str = config.to_json()
    deserialized_from_json = config.__class__.from_json(json_str)
    assert (
        config == deserialized_from_json
    ), f"JSON round trip failed for {config.__class__.__name__}"

    # Dict round trip
    data_dict = config.to_dict()
    deserialized_from_dict = config.__class__.from_dict(data_dict)
    assert (
        config == deserialized_from_dict
    ), f"Dict round trip failed for {config.__class__.__name__}"


@pytest.mark.parametrize("config", configs, ids=get_config_ids)
def test_reconstructable_dict_file_round_trip(config):
    """Test saving and loading reconstructable dicts to/from JSON files."""
    # Get a reconstructable dict
    reconstructable = to_reconstructable_dict(config)
    breakpoint()

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
        reconstructed = reconstruct_from_dict(loaded_dict)

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
                    assert (
                        str(original_value) == str(reconstructed_value)
                    ), f"Attribute {attr_name} mismatch after file round trip for {config.__class__.__name__}"
                else:
                    assert (
                        original_value == reconstructed_value
                    ), f"Attribute {attr_name} mismatch after file round trip for {config.__class__.__name__}"

    finally:
        # Clean up the temporary file
        if os.path.exists(temp_file_path):
            os.unlink(temp_file_path)


if __name__ == "__main__":
    pytest.main([__file__])
