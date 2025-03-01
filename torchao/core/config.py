from typing import Any, Dict

import torch
from pydantic import BaseModel, ConfigDict, field_serializer, model_validator


class AOBaseConfig(BaseModel):
    """
    Base configuration class for TorchAO quantization workflows with native Pydantic handling for torch.dtype.

    When a workflow configuration inherits from AOBaseConfig, the `quantize_` function can automatically
    apply the appropriate transformation to a model based on the configuration type.

    Usage example:
        # 1. Define a configuration class for your workflow
        class WorkflowFooConfig(AOBaseConfig):
            # Configuration parameters for workflow 'Foo'
            bar: str = 'baz'

        # 2. Register a handler for this configuration (internal implementation)
        @register_quantize_module_handler(WorkflowFooConfig)
        def _transform(
            mod: torch.nn.Module,
            config: WorkflowFooConfig,
        ) -> torch.nn.Module:
            # Implementation of the transformation logic
            # Typically performs tensor subclass weight swapping or module replacement
            ...

        # 3. Apply the configuration to a model
        # The user simply calls `quantize_` with a model and config instance
        # The appropriate handler is automatically selected based on the config type
        model = ...
        quantized_model = quantize_(model, WorkflowFooConfig(bar='custom_value'))

    Note on serialization, if you add a new AOBaseConfig and want to support serialization,
    please add a test in test/quantization/test_config_serialization.py
    """

    model_config = ConfigDict(
        arbitrary_types_allowed=True,
        validate_assignment=True,
        extra="forbid",
        validate_default=True,
        populate_by_name=True,
    )

    @field_serializer("*")
    def serialize_torch_dtype(self, v, _info):
        if isinstance(v, torch.dtype):
            return str(v)
        return v

    @model_validator(mode="before")
    @classmethod
    def convert_dtypes(cls, data: Any) -> Any:
        """Simple converter for torch dtype strings"""
        if isinstance(data, str) and data.startswith("torch."):
            dtype_name = data.split("torch.")[1]
            if hasattr(torch, dtype_name):
                return getattr(torch, dtype_name)
        elif isinstance(data, dict):
            return {k: cls.convert_dtypes(v) for k, v in data.items()}
        elif isinstance(data, list):
            return [cls.convert_dtypes(item) for item in data]
        return data

    def to_dict(self) -> dict:
        """Convert the configuration to a dictionary that is compat w/ json.dump"""
        return self.model_dump(mode="json")

    def to_json(self) -> str:
        """Convert the configuration to a JSON string."""
        return self.model_dump_json()

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "AOBaseConfig":
        """Create a configuration from a dictionary."""
        return cls.model_validate(data)

    @classmethod
    def from_json(cls, json_str: str) -> "AOBaseConfig":
        """Create a configuration from a JSON string."""
        return cls.model_validate_json(json_str)


def to_reconstructable_dict(config: AOBaseConfig) -> dict:
    """
    Convert an AOBaseConfig instance to a dictionary format that can be reconstructed.
    The output is designed to be JSON-serializable for storage.

    Args:
        config: An instance of AOBaseConfig or its subclass

    Returns:
        A dictionary with 'class_name' and 'config' keys, where config contains
        a JSON string representation of the configuration

    Example:
        # Create a configuration
        config = SomeConfig(param1="value1", param2=42)

        # Convert to a reconstructable dict (JSON-serializable)
        serialized = to_reconstructable_dict(config)

        # Later, reconstruct
        reconstructed = reconstruct_from_dict(serialized)
    """
    # Use Pydantic's to_json to get a JSON string representation
    config_json = config.to_dict()

    return {"class_name": config.__class__.__name__, "config": config_json}


def reconstruct_from_dict(config_dict) -> AOBaseConfig:
    """
    Reconstruct a configuration class instance from a dictionary created by to_reconstructable_dict.

    Args:
        config_dict: Dictionary with 'class_name' and 'config' keys, where config contains
                     a JSON string representation of the configuration

    Returns:
        An instance of the specified configuration class
    """
    # Get the class name and configuration parameters as JSON string
    class_name = config_dict["class_name"]
    config_params_json = config_dict["config"]

    # Import the module where the class is defined
    import importlib

    module = importlib.import_module("torchao.quantization.quant_api")

    # Get the class from the module
    config_class = getattr(module, class_name)

    # Create an instance of the class from the JSON config
    return config_class.from_dict(config_params_json)
