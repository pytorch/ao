import json
from typing import Dict, Optional, Tuple

import torch
from safetensors.torch import load_file, save_file

from torchao import quantize_
from torchao.float8.inference import Float8MMConfig
from torchao.quantization import granularity
from torchao.quantization.granularity import PerRow
from torchao.quantization.quant_api import Float8DynamicActivationFloat8WeightConfig
from torchao.quantization.quantize_.workflows.float8.float8_tensor import (
    QuantizeTensorToFloat8Kwargs,
)

ALLOWED_QUANT_DTYPES = {
    "torch.float8_e4m3fn": torch.float8_e4m3fn,
    # add to me
}
ALLOWED_GRANUALARITY = {"PerRow": PerRow()}


def load_tensor_subclass_dict(file_path: str):
    """
    Load a dictionary of tensor subclasses from a safetensors file.

    Args:
        file_path: Path to the safetensors file

    Returns:
        Dictionary of reconstructed tensor subclasses
    """
    loaded_tensors = load_file(file_path)

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

        if tensor_type == "Float8Tensor":
            dtype = ALLOWED_QUANT_DTYPES.get(tensor_metadata.get("dtype"))
            granularity = ALLOWED_GRANUALARITY.get(tensor_metadata.get("granularity"))
            kernel_preference = json.loads(tensor_metadata.get("kernel_preference"))
            hp_value_lb = tensor_metadata.get("hp_value_lb")
            hp_value_ub = tensor_metadata.get("hp_value_ub")
            mm_config = Float8MMConfig(*json.loads(tensor_metadata.get("mm_config")))

            from torchao.quantization.quantize_.workflows import Float8Tensor

            result[tensor_name] = Float8Tensor(
                qdata=tensor_tensors["qdata"].to(tensor_metadata["qdata_device"]),
                scale=tensor_tensors["scale"].to(tensor_metadata["scale_device"]),
                block_size=json.loads(tensor_metadata.get("block_size")),
                mm_config=mm_config,
                hp_value_lb=hp_value_lb,
                hp_value_ub=hp_value_ub,
                act_quant_kwargs=QuantizeTensorToFloat8Kwargs(
                    float8_dtype=dtype,
                    granularity=granularity if granularity else PerRow(),
                    mm_config=mm_config,
                    hp_value_lb=hp_value_lb,
                    hp_value_ub=hp_value_ub,
                    kernel_preference=kernel_preference,
                ),
                kernel_preference=kernel_preference,
                dtype=dtype,
            )
        elif tensor_type == "Tensor":
            data = tensor_tensors["data"]
            data = data.to(tensor_metadata["device"])
            result[tensor_name] = data

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

    if tensor.__class__.__name__ == "Float8Tensor":
        metadata["tensor_type"] = "Float8Tensor"
        for item, value in tensor.__dict__.items():
            if isinstance(value, granularity.PerRow):
                metadata["granularity"] = "PerRow"
            elif isinstance(value, QuantizeTensorToFloat8Kwargs):
                metadata["dtype"] = "torch.float8_e4m3fn"
            elif item == "qdata":
                tensors_dict["qdata"] = value
                metadata["qdata_device"] = (
                    "cuda:0" if value.device == torch.device("cuda:0") else "cpu"
                )
            elif item == "scale":
                tensors_dict["scale"] = value
                metadata["scale_device"] = (
                    "cuda:0" if value.device == torch.device("cuda:0") else "cpu"
                )

            elif value:
                metadata[item] = json.dumps(value)

    return metadata, tensors_dict


def save_tensor_subclass_dict(
    tensor_dict: Dict[str, Dict[str, torch.Tensor]],
    file_path: str,
    additional_metadata: Optional[Dict[str, str]] = None,
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
        # TODO: handle case where tensor is a plain tensor
        if tensor.__class__.__name__ == "Tensor":
            tensors_dict = {"data": tensor}
            metadata = {"tensor_type": "Tensor"}
            metadata["device"] = (
                "cuda:0" if tensor.device == torch.device("cuda:0") else "cpu"
            )
        else:
            metadata, tensors_dict = create_metadata_for_tensor_subclass(tensor)

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

    if additional_metadata:
        combined_metadata.update(additional_metadata)

    save_file(combined_tensors_dict, file_path, metadata=combined_metadata)
    print(f"Saved {len(tensor_dict)} tensor subclasses to {file_path} with metadata")


if __name__ == "__main__":
    config = Float8DynamicActivationFloat8WeightConfig(granularity=PerRow())
    model = torch.nn.Sequential(
        torch.nn.Linear(32, 256, dtype=torch.bfloat16, device="cuda")
    )
    quantize_(model, config)
    example_inputs = (torch.randn(2, 32, dtype=torch.bfloat16, device="cuda"),)
    ref_output = model(*example_inputs)

    save_tensor_subclass_dict(model.state_dict(), "fp8_weights.safetensors")
    reconstructed_dict = load_tensor_subclass_dict("fp8_weights.safetensors")

    model = torch.nn.Sequential(
        torch.nn.Linear(32, 256, dtype=torch.bfloat16, device="cuda")
    )
    model.load_state_dict(reconstructed_dict, assign=True)
    output = model(*example_inputs)
    assert torch.equal(output, ref_output)
