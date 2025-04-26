# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD 3-Clause license found in the
# LICENSE file in the root directory of this source tree.
import abc
import dataclasses
import enum
import importlib
import json
from typing import Any, ClassVar, Dict

import torch


class AOBaseConfig(abc.ABC):
    """
    If a workflow config inherits from this then `quantize_` knows
    how to a apply it to a model. For example::

        # user facing code
        class WorkflowFooConfig(AOBaseConfig): ...
            # configuration for workflow `Foo` is defined here
            bar = 'baz'

        # non user facing code
        @register_quantize_module_handler(WorkflowFooConfig)
        def _transform(
            mod: torch.nn.Module,
            config: WorkflowFooConfig,
        ) -> torch.nn.Module:
            # the transform is implemented here, usually a tensor sublass
            # weight swap or a module swap
            ...

        # then, the user calls `quantize_` with a config, and `_transform` is called
        # under the hood by `quantize_.

    """

    # Base Version of a config
    VERSION: ClassVar[int] = 1


class VersionMismatchError(Exception):
    """Raised when trying to deserialize a config with a different version"""

    def __init__(self, type_path, stored_version, current_version):
        self.type_path = type_path
        self.stored_version = stored_version
        self.current_version = current_version
        message = (
            f"Version mismatch for {type_path}: "
            f"stored version {stored_version} != current version {current_version}"
        )
        super().__init__(message)


class ConfigJSONEncoder(json.JSONEncoder):
    """Custom JSON encoder for AOBaseConfig objects"""

    def default(self, o):
        # Handle AOBaseConfig subclasses first (most specific case)
        if isinstance(o, AOBaseConfig):
            data_dict = {}
            # Process each attribute to handle nested objects
            for k, v in o.__dict__.items():
                if not k.startswith("_") and k != "VERSION":
                    # Recursively encode each value (important for nested objects)
                    data_dict[k] = self.encode_value(v)

            return {
                # Only store the class name, not the full module path
                "_type": o.__class__.__name__,
                "_version": getattr(o.__class__, "VERSION", 1),
                "_data": data_dict,
            }

        # Handle NamedTuple types
        if hasattr(o, "_fields") and hasattr(
            o, "_asdict"
        ):  # Check for NamedTuple characteristics
            asdict_data = o._asdict()
            # Process each field to handle nested objects
            processed_data = {k: self.encode_value(v) for k, v in asdict_data.items()}

            return {
                "_type": o.__class__.__name__,
                "_version": getattr(o.__class__, "VERSION", 1),
                "_data": processed_data,
            }

        # Handle dataclasses
        if dataclasses.is_dataclass(o) and not isinstance(o, type):
            data_dict = {}
            # Process each field to handle nested objects
            for f in dataclasses.fields(o):
                if f.name != "VERSION":
                    data_dict[f.name] = self.encode_value(getattr(o, f.name))

            return {
                # Only store the class name for dataclasses too
                "_type": o.__class__.__name__,
                "_version": getattr(o.__class__, "VERSION", 1),
                "_data": data_dict,
            }

        # Handle torch.dtype
        if hasattr(o, "__module__") and o.__module__ == "torch" and isinstance(o, type):
            return {"_type": "torch.dtype", "_data": str(o).split(".")[-1]}

        # Handle Layout objects
        if hasattr(o, "__class__") and "Layout" in o.__class__.__name__:
            return {
                "_type": o.__class__.__name__,
                "_data": {
                    k: self.encode_value(v)
                    for k, v in o.__dict__.items()
                    if not k.startswith("_")
                },
            }

        # Handle enum values
        if isinstance(o, enum.Enum):
            # Store the full path for enums to ensure uniqueness
            return {"_type": f"{o.__class__.__name__}", "_data": o.name}

        if isinstance(o, torch.dtype):
            return {"_type": "torch.dtype", "_data": str(o).split(".")[-1]}

        # For lists and dictionaries, recursively process their items
        if isinstance(o, list):
            return [self.encode_value(item) for item in o]

        if isinstance(o, dict):
            return {k: self.encode_value(v) for k, v in o.items()}

        # Default case - let the parent class handle it
        return super().default(o)

    def encode_value(self, value):
        """Helper method to recursively encode a value"""
        # Try to use default for custom type
        try:
            # This will handle all our special cases and raise TypeError
            # if it can't handle the type
            result = self.default(value)
            return result
        except TypeError:
            pass

        # Default case - return as is
        # (This will be processed by standard JSON encoder later)
        return value


def config_to_dict(config: AOBaseConfig) -> Dict[str, Any]:
    """
    Convert an AOBaseConfig instance to a dictionary suitable for serialization.

    Args:
        config: An instance of AOBaseConfig subclass

    Returns:
        Dict representation of the config
    """
    if not isinstance(config, AOBaseConfig):
        raise TypeError(f"Expected AOBaseConfig instance, got {type(config)}")

    # Use the existing JSON encoder but return the dict directly
    return json.loads(json.dumps(config, cls=ConfigJSONEncoder))


ALLOWED_AO_MODULES = {
    "torchao.quantization",
    "torchao.sparsity.sparse_api",
    "torchao.prototype.quantization",
    "torchao.prototype.mx_formats",
}


def config_from_dict(data: Dict[str, Any]) -> AOBaseConfig:
    """
    Create an AOBaseConfig subclass instance from a dictionary.

    Args:
        data: Dictionary containing serialized config data

    Returns:
        An instance of the appropriate AOBaseConfig subclass

    Raises:
        VersionMismatchError: If the stored version doesn't match the class version
        ValueError: If deserialization fails for other reasons
    """
    if not isinstance(data, dict):
        raise TypeError(f"Expected dictionary, got {type(data)}")

    if "_type" not in data or "_data" not in data:
        raise ValueError("Input dictionary missing required '_type' or '_data' fields")

    type_path = data["_type"]
    stored_version = data.get("_version", 1)
    obj_data = data["_data"]

    # Handle torch.dtype
    if type_path == "torch.dtype":
        import torch

        return getattr(torch, obj_data)
    # Try to find the class in any of the allowed modules
    cls = None
    for module_path in ALLOWED_AO_MODULES:
        try:
            module = importlib.import_module(module_path)
            cls = getattr(module, type_path)
            break  # Found the class, exit the loop
        except (ImportError, AttributeError):
            continue  # Try the next module

    # If we couldn't find the class in any allowed module, raise an error
    if cls is None:
        allowed_modules_str = ", ".join(ALLOWED_AO_MODULES)
        raise ValueError(
            f"Failed to find class {type_path} in any of the allowed modules: {allowed_modules_str}"
        )

    # Check version - require exact match
    current_version = getattr(cls, "VERSION", 1)
    if stored_version != current_version:
        raise VersionMismatchError(type_path, stored_version, current_version)

    # Handle the case where obj_data is not a dictionary
    if not isinstance(obj_data, dict):
        if issubclass(cls, enum.Enum):
            # For enums, convert string to enum value
            return getattr(cls, obj_data)
        else:
            # For other primitive types, create an instance with the value
            try:
                return cls(obj_data)
            except:
                return obj_data

    # Process nested structures for dictionary obj_data
    processed_data = {}
    for key, value in obj_data.items():
        if isinstance(value, dict) and "_type" in value and "_data" in value:
            # Recursively handle nested configs
            processed_data[key] = config_from_dict(value)
        elif isinstance(value, list):
            # Handle lists of possible configs
            processed_data[key] = [
                config_from_dict(item)
                if isinstance(item, dict) and "_type" in item and "_data" in item
                else item
                for item in value
            ]
        else:
            processed_data[key] = value

    # Create and return the instance
    try:
        return cls(**processed_data)
    except Exception as e:
        raise ValueError(f"Failed to create instance of {cls.__name__}: {e}")
