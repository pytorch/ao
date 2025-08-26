import json
from typing import Dict, Tuple

import torch
from safetensors.torch import load_file, save_file

from torchao.float8.inference import Float8MMConfig
from torchao.prototype.quantization.QuantizeTensorToFloat8KwargsJSON import (
    QuantizeTensorToFloat8KwargsJSONEncoder,
    config_from_dict,
    ALLOWED_GRANUALARITY,
    ALLOWED_QUANT_DTYPES,
)
from torchao.quantization import Float8Tensor, granularity
from torchao.quantization.granularity import PerRow
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
        tensor_metadata = {}
        for key, value in metadata.items():
            if key.startswith(f"{tensor_name}:"):
                # Remove the prefix
                tensor_metadata[key[len(tensor_name) + 1 :]] = value

        tensor_tensors = {}
        for key, value in loaded_tensors.items():
            if key.startswith(f"{tensor_name}:"):
                # Remove the prefix
                tensor_tensors[key[len(tensor_name) + 1 :]] = value

        tensor_type = tensor_metadata.get("tensor_type")

        if tensor_type == str(Float8Tensor):
            # not the same as mm_config in act_quant_kwargs
            mm_config = json.loads(tensor_metadata.get("mm_config"))
            if mm_config:
                mm_config = Float8MMConfig(*mm_config)

            act_quant_kwargs_dict = json.loads(tensor_metadata.get("act_quant_kwargs"))
            act_quant_kwargs = config_from_dict(act_quant_kwargs_dict)

            result[tensor_name] = Float8Tensor(
                qdata=tensor_tensors["qdata"],
                scale=tensor_tensors["scale"],
                block_size=json.loads(tensor_metadata.get("block_size")),
                mm_config=mm_config,
                hp_value_lb=act_quant_kwargs.hp_value_lb,
                hp_value_ub=act_quant_kwargs.hp_value_ub,
                act_quant_kwargs=act_quant_kwargs,
                kernel_preference=act_quant_kwargs.kernel_preference,
                dtype=act_quant_kwargs.float8_dtype,
            )
        elif tensor_type == str(torch.Tensor):
            data = tensor_tensors["data"]
            result[tensor_name] = data

    print(
        f"Loaded {len(tensor_names)} tensor subclasses from {file_path} with metadata"
    )
    return result


def create_metadata_for_tensor_subclass(
    tensor: torch.Tensor,
) -> Tuple[Dict[str, str], Dict[str, torch.Tensor]]:
    """
    Create metadata for tensor subclasses from torchao.

    Args:
        tensor: A tensor subclass (e.g., Float8Tensor)

    Returns:
        Tuple of (metadata, tensors_dict) where:
        - metadata: Dictionary with metadata needed to reconstruct the tensor
        - tensors_dict: Dictionary with tensors to save
    """
    metadata = {}
    tensors_dict = {}

    if isinstance(tensor, Float8Tensor):
        metadata["tensor_type"] = str(tensor.__class__)
        for item, value in tensor.__dict__.items():
            if isinstance(value, QuantizeTensorToFloat8Kwargs):
                metadata[item] = json.dumps(
                    obj=value, cls=QuantizeTensorToFloat8KwargsJSONEncoder
                )
            elif item == "qdata" or item == "scale":
                tensors_dict[item] = value
            elif value:
                metadata[item] = json.dumps(value)

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
        additional_metadata: Optional additional metadata to include
    """

    combined_metadata = {}
    combined_tensors_dict = {}

    for tensor_name, tensor in tensor_dict.items():
        if isinstance(tensor, Float8Tensor):
            metadata, tensors_dict = create_metadata_for_tensor_subclass(tensor)
        elif isinstance(tensor, torch.Tensor):
            tensors_dict = {"data": tensor}
            metadata = {"tensor_type": str(torch.Tensor)}
            metadata["device"] = str(tensor.device)
        else:
            raise ValueError(f"Unsupported tensor type: {type(tensor)}")

        # Clone tensors to avoid memory sharing issues
        prefixed_tensors_dict = {
            f"{tensor_name}:{key}": (
                value.detach().clone() if isinstance(value, torch.Tensor) else value
            )
            for key, value in tensors_dict.items()
        }

        for key, value in metadata.items():
            combined_metadata[f"{tensor_name}:{key}"] = value

        combined_tensors_dict.update(prefixed_tensors_dict)

    combined_metadata["tensor_names"] = json.dumps(list(tensor_dict.keys()))

    save_file(combined_tensors_dict, file_path, metadata=combined_metadata)
    print(f"Saved {len(tensor_dict)} tensor subclasses to {file_path} with metadata")
