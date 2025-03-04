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
        """Convert the configuration to a dictionary"""
        return self.model_dump()

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
