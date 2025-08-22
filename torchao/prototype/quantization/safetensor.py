import inspect
import json
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import torch
import torchao
from safetensors.torch import load_file, save_file

from torchao.quantization.quant_api import Float8DynamicActivationFloat8WeightConfig

from torchao.quantization.granularity import (
    PerRow
)
from torchao import quantize_

def load_tensor_subclass_dict(file_path: str):
    pass

def create_metadata_for_tensor_subclass(tensor: torch.Tensor) -> Tuple[Dict[str, str], Dict[str, torch.Tensor]]:
    """
    Create metadata for tensor subclasses from torchao.

    Args:
        tensor: A tensor subclass (e.g., LinearActivationQuantizedTensor)

    Returns:
        Tuple of (metadata, tensors_dict) where:
        - metadata: Dictionary with metadata needed to reconstruct the tensor
        - tensors_dict: Dictionary with tensors to save
    """
    metadata = {}
    tensors_dict = {}

    if tensor.__class__.__name__ == "Float8Tensor":
        metadata["tensor_type"] = "Float8Tensor"
        metadata["block_size"] = tensor.block_size
        metadata["mm_config"] = tensor.mm_config
        metadata["hp_value_lb"] = tensor.hp_value_lb
        metadata["hp_value_ub"] = tensor.hp_value_ub
        metadata["act_quant_kwargs"] = tensor.act_quant_kwargs
        metadata["kernel_preference"] = tensor.kernel_preference
        metadata["dtype"] = tensor.dtype

        tensors_dict["qdata"] = tensor.qdata
        tensors_dict["scale"] = tensor.scale

    return metadata, tensors_dict

def save_tensor_subclass_dict(
    tensor_dict: Dict[str, Dict[str, torch.Tensor]],
    file_path: str,
    additional_metadata: Optional[Dict[str, str]] = None,
):
    combined_metadata = {}
    combined_tensors_dict = {}

    for tensor_name, tensor in tensor_dict.items():
        # TODO: handle case where tensor is a plain tensor
        if tensor.__class__.__name__ == "Tensor":
            metadata, tensors_dict["data"] = {}, tensor
        else:
            metadata, tensors_dict = create_metadata_for_tensor_subclass(tensor)

        prefixed_tensors_dict = {f"{tensor_name}:{key}": value for key, value in tensors_dict.items()}

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
    # reconstructed_dict = load_tensor_subclass_dict("fp8_weights.safetensors")

    # model = torch.nn.Sequential(
    #     torch.nn.Linear(32, 256, dtype=torch.bfloat16, device="cuda")
    # )
    # model.load_state_dict(reconstructed_dict, assign=True)
    # output = model(*example_inputs)
    # assert torch.equal(output, ref_output)
