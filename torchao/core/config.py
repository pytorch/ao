from typing import Any, Dict

import torch
from pydantic import BaseModel, field_serializer, model_validator


class AOBaseConfig(BaseModel):
    """
    Base configuration class with native Pydantic handling for torch.dtype.
    """

    model_config = {
        "arbitrary_types_allowed": True,
        "validate_assignment": True,
        "extra": "forbid",
        "validate_default": True,
        "populate_by_name": True,
    }

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
