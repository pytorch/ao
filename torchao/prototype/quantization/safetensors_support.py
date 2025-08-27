import json
from typing import Dict, Tuple

import torch
from safetensors.torch import load_file, save_file

from torchao.float8.inference import Float8MMConfig
from torchao.prototype.quantization.safetensors_JSONEncoders import (
    Float8TensorAttributeJSONEncoder,
    config_from_dict,
)
from torchao.quantization import Float8Tensor
from torchao.quantization.quantize_.workflows.float8.float8_tensor import (
    QuantizeTensorToFloat8Kwargs,
)


def load_tensor_subclass_dict(file_path: str, device: str):
    """
    Load a dictionary of tensor subclasses from a safetensors file.

    Args:
        file_path: Path to the safetensors file

    Returns:
        Dictionary of reconstructed tensor subclasses
    """
    loaded_tensors = load_file(file_path, device)

    with open(file_path, "rb") as f:
        import struct

        header_size = struct.unpack("<Q", f.read(8))[0]
        header_bytes = f.read(header_size)
        header = json.loads(header_bytes)
        metadata = header.get("__metadata__", {})

    if "tensor_names" not in metadata:
        raise ValueError("No tensors found")

    tensor_names = json.loads(metadata["tensor_names"])
    result = {}

    for tensor_name in tensor_names:
        tensor_tensors = {}
        for key, value in loaded_tensors.items():
            if key.startswith(f"{tensor_name}:"):
                # Remove the prefix
                tensor_tensors[key[len(tensor_name) + 1 :]] = value

        tensor_metadata = json.loads(metadata.get(tensor_name))
        tensor_type = tensor_metadata.get("_type")

        if tensor_type == "Float8Tensor":
            tensor_metadata["_data"].update(tensor_tensors)
            float8_tensor = config_from_dict(tensor_metadata)
            result[tensor_name] = float8_tensor

        elif tensor_type == "Tensor":
            data = tensor_tensors["tensor_data"]
            result[tensor_name] = data

    print(
        f"Loaded {len(tensor_names)} tensor subclasses from {file_path} with metadata"
    )
    return result


def create_metadata_for_tensor_subclass(
    tensor: torch.Tensor,
) -> Tuple[str, Dict[str, torch.Tensor]]:
    """
    Create metadata for tensor subclasses from torchao.

    Args:
        tensor: A tensor subclass (e.g., Float8Tensor)

    Returns:
        Tuple of (metadata, tensors_dict) where:
        - metadata: Dictionary with metadata needed to reconstruct the tensor
        - tensors_dict: Dictionary with tensors to save
    """
    metadata = ""
    tensors_dict = {}

    if isinstance(tensor, Float8Tensor):
        for tensor_data_name in tensor.tensor_data_names:
            tensors_dict[tensor_data_name] = getattr(tensor, tensor_data_name)

        metadata = json.dumps(tensor, cls=Float8TensorAttributeJSONEncoder)

    return metadata, tensors_dict


def save_tensor_subclass_dict(
    tensor_dict: Dict[str, Dict[str, torch.Tensor]],
    file_path: str,
):
    """
    Save a dictionary of tensor subclasses with appropriate metadata.

    Args:
        tensor_dict: Dictionary of tensor subclasses to save, with keys as tensor names
        file_path: Path where to save the tensors
    """

    combined_metadata = {}
    combined_tensors_dict = {}

    for tensor_name, tensor in tensor_dict.items():
        if isinstance(tensor, Float8Tensor):
            metadata, tensors_dict = create_metadata_for_tensor_subclass(tensor)
        elif isinstance(tensor, torch.Tensor):
            tensors_dict = {"tensor_data": tensor}
            metadata = json.dumps({"_type": "Tensor"})
        else:
            raise ValueError(f"Unsupported tensor type: {type(tensor)}")

        # Clone tensors to avoid memory sharing issues
        prefixed_tensors_dict = {
            f"{tensor_name}:{key}": (
                value.detach().clone() if isinstance(value, torch.Tensor) else value
            )
            for key, value in tensors_dict.items()
        }

        combined_metadata[tensor_name] = metadata
        combined_tensors_dict.update(prefixed_tensors_dict)

    combined_metadata["tensor_names"] = json.dumps(list(tensor_dict.keys()))

    save_file(combined_tensors_dict, file_path, metadata=combined_metadata)
    print(f"Saved {len(tensor_dict)} tensor subclasses to {file_path} with metadata")
