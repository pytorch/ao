import dataclasses
import enum
import json
from typing import Any, Dict

import torch

import torchao
from torchao.quantization import Float8Tensor
from torchao.quantization.quantize_.common import KernelPreference
from torchao.quantization.quantize_.workflows import QuantizeTensorToFloat8Kwargs

ALLOWED_CLASSES = {
    "Float8Tensor": Float8Tensor,
    "Float8MMConfig": torchao.float8.inference.Float8MMConfig,
    "QuantizeTensorToFloat8Kwargs": QuantizeTensorToFloat8Kwargs,
    "PerRow": torchao.quantization.PerRow,
    "PerTensor": torchao.quantization.PerTensor,
    "KernelPreference": KernelPreference,
}

ALLOWED_TENSORS = ["Float8Tensor", "Tensor"]

__all__ = [
    "Float8TensorAttributeJSONEncoder",
    "object_from_dict",
    "is_metadata_torchao",
]


class Float8TensorAttributeJSONEncoder(json.JSONEncoder):
    def default(self, o):
        if isinstance(o, Float8Tensor):
            tensor_attr_dict = {}
            all_tensor_attributes = (
                o.optional_tensor_attribute_names + o.tensor_attribute_names
            )

            for tensor_attribute_name in all_tensor_attributes:
                attribute = getattr(o, tensor_attribute_name)
                encoded_attribute = self.encode_value(attribute)
                tensor_attr_dict[tensor_attribute_name] = encoded_attribute

            return {"_type": o.__class__.__name__, "_data": tensor_attr_dict}

        if hasattr(o, "_fields") and hasattr(
            o, "_asdict"
        ):  # Check for NamedTuple characteristics
            asdict_data = o._asdict()
            # Process each field to handle nested objects
            processed_data = {k: self.encode_value(v) for k, v in asdict_data.items()}

            return {
                "_type": o.__class__.__name__,
                "_data": processed_data,
            }

        if dataclasses.is_dataclass(o) and not isinstance(o, type):
            data_dict = {}
            # Process each field to handle nested objects
            for f in dataclasses.fields(o):
                data_dict[f.name] = self.encode_value(getattr(o, f.name))

            return {
                "_type": o.__class__.__name__,
                "_data": data_dict,
            }

        if isinstance(o, torch.dtype):
            return {"_type": "torch.dtype", "_data": str(o).split(".")[-1]}

        if isinstance(o, enum.Enum):
            # Store the full class name for enums to ensure uniqueness
            return {"_type": f"{o.__class__.__name__}", "_data": o.name}

        if isinstance(o, list):
            return [self.encode_value(item) for item in o]

        if isinstance(o, dict):
            return {k: self.encode_value(v) for k, v in o.items()}

        # Default case
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


def object_from_dict(data: Dict[str, Any]):
    if not isinstance(data, dict):
        raise TypeError(f"Expected dictionary, got {type(data)}")

    if "_type" not in data or "_data" not in data:
        raise ValueError("Input dictionary missing required '_type' or '_data' fields")

    type_path = data["_type"]
    obj_data = data["_data"]

    if type_path == "torch.dtype":
        return getattr(torch, obj_data)

    cls = ALLOWED_CLASSES.get(type_path)

    # If we couldn't find the class in any allowed module, raise an error
    if cls is None:
        allowed_modules_str = ", ".join(ALLOWED_CLASSES)
        raise ValueError(
            f"Failed to find class {type_path} in any of the allowed modules: {allowed_modules_str}"
        )

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

    processed_data = {}

    for key, value in obj_data.items():
        if isinstance(value, dict) and "_type" in value and "_data" in value:
            # Recursively handle nested configs
            processed_data[key] = object_from_dict(value)
        elif isinstance(value, list):
            # Handle lists or tuples of possible configs
            processed_data[key] = [
                object_from_dict(item)
                if isinstance(item, dict) and "_type" in item and "_data" in item
                else item
                for item in value
            ]
        elif isinstance(value, tuple):
            raise NotImplementedError(
                "Tuples will be serialized as List in JSON, so we recommend to use "
                f"Lists instead to avoid surprises. got: {value}"
            )
        elif isinstance(value, dict):
            # Handle dicts of possible configs
            processed_data[key] = {
                k: object_from_dict(v)
                if isinstance(v, dict) and "_type" in v and "_data" in v
                else v
                for k, v in value.items()
            }
        else:
            processed_data[key] = value

    # Create and return the instance
    try:
        return cls(**processed_data)
    except Exception as e:
        raise ValueError(f"Failed to create instance of {cls.__name__}: {e}")


def is_metadata_torchao(metadata: Dict[str, Any]):
    if not metadata or "tensor_names" not in metadata:
        return False
    try:
        all_tensor_names = json.loads(metadata["tensor_names"])
    except (TypeError, json.JSONDecodeError, UnicodeDecodeError):
        return False

    if not all_tensor_names or not isinstance(all_tensor_names, list):
        return False

    for tensor_name in all_tensor_names:
        if tensor_name not in metadata or not isinstance(metadata[tensor_name], str):
            return False
        try:
            tensor_dict = json.loads(metadata[tensor_name])
        except (TypeError, json.JSONDecodeError, UnicodeDecodeError):
            return False

        # returns None if _type not in tensor_dict
        tensor_type = tensor_dict.get("_type")
        if tensor_type not in ALLOWED_TENSORS:
            return False

    return True
