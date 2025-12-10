import json
import logging
from typing import Any, Dict

import torch

from torchao.prototype.safetensors.safetensors_utils import (
    ALLOWED_TENSORS_SUBCLASSES,
    TensorSubclassAttributeJSONEncoder,
    object_from_dict,
)

logger: logging.Logger = logging.getLogger(__name__)


def unflatten_tensor_state_dict(
    tensors_data_dict: Dict[str, Any],
    metadata: Dict[str, Any],
):
    """
    Reconstructs tensor subclass state dict from provided torch.Tensor data and metadata dictionary
    The naming of metadata is so that it is consistent with safetensors naming to avoid confusion
    This function is used after loading in previously saved model state dict (using safetensors.save_file) to reconstruct tensor subclass structure

    For example, given a previously flattened tensors_data_dict and metadata:
    tensors_data_dict = {
        '0._weight_qdata': torch.Tensor(...),
        '0._weight_scale': torch.Tensor(...),
        '0.bias': torch.Tensor(...),
    }
    metadata = {
        '0.weight': {
            '_type': 'Float8Tensor',
            '_data': {
                'block_size': [1,32],
                ...
            },
            '_tensor_data_names': ['qdata', 'scale']
        }
        '0.bias': {
            '_type': 'torch.Tensor',
        }
        'tensor_names': ['0.weight', '0.bias']
    }

    We recover the structure of the original state dict:
    tensor_dict = {
        '0.weight': Float8Tensor(
            qdata=torch.Tensor(...),
            scale=torch.Tensor(...),
            block_size=[1,32],
            ...),
        '0.bias': torch.Tensor(...),
    }

    Args:
        tensors_data_dict: a dictionary from "{tensor_name}_{tensor_data_attribute_name}" to flattened torch.Tensor data for tensor subclass instance
        metadata: a dictionary from "tensor_name" to another dictionary that contains type and attributes for tensor subclass instance

    Returns:
        Dictionary of reconstructed tensor subclasses
    """
    combined_data = {**tensors_data_dict, **metadata}

    if "tensor_names" not in metadata:
        raise ValueError("No tensors found")

    tensor_names = json.loads(metadata["tensor_names"])
    result = {}
    leftover_state_dict = tensors_data_dict.copy()
    for tensor_name in tensor_names:
        processed_tensors = []

        module_fqn, weight_name = tensor_name.rsplit(".", 1)

        prefix = f"{module_fqn}._{weight_name}_"
        tensor_tensors = {}

        for key, value in combined_data.items():
            if key.startswith(prefix):
                # Remove the prefix
                tensor_tensors[key[len(prefix) :]] = value

        tensor_metadata = json.loads(metadata.get(tensor_name))
        tensor_type = tensor_metadata.get("_type")
        complete_tensor_data_names = tensor_metadata.get("_tensor_data_names")

        if tensor_type in ALLOWED_TENSORS_SUBCLASSES:
            # if not all tensor data is present (ie missing qdata) we wait for it
            # to be loaded in from a future call
            if not len(tensor_tensors) is len(complete_tensor_data_names):
                continue
            tensor_metadata["_data"].update(tensor_tensors)
            result[tensor_name] = object_from_dict(tensor_metadata)

            for suffix in complete_tensor_data_names:
                processed_tensors.append(prefix + suffix)
        elif tensor_type == torch.Tensor.__name__:
            # we allow the option of loading in state_dict info for a single tensor
            # if tensor state dict info is not loaded in yet, we wait for it to be provided
            # in a future call
            if tensor_name not in tensors_data_dict.keys():
                continue
            result[tensor_name] = tensors_data_dict[tensor_name]
            processed_tensors.append(
                tensor_name
            )  # add here because key for torch.Tensor has no prefix
        else:
            raise ValueError(f"Unsupported tensor type: {tensor_type}")

        for tensor_name in processed_tensors:
            del leftover_state_dict[tensor_name]

    return result, leftover_state_dict


def flatten_tensor_state_dict(
    tensors_dict: Dict[str, Dict[str, torch.Tensor]],
):
    """
    Flattens a dictionary of tensor subclasses so that it is compatible with safetensors.save_file
    We disconstruct tensor subclass structure into torch.Tensor data and metadata dictionary
    The naming of metadata is so that it is consistent with safetensors naming to avoid confusion

    For example, given something like:
    tensor_dict = {
        '0.weight': Float8Tensor(
            qdata=torch.Tensor(...),
            scale=torch.Tensor(...),
            block_size=[1,32],
            ...),
        '0.bias': torch.Tensor(...),
    }

    We flatten this to:
    tensors_data = {
        '0._weight_qdata': torch.Tensor(...),
        '0._weight_scale': torch.Tensor(...),
        '0.bias': torch.Tensor(...),
    }
    metadata = {
        '0.weight': {
            '_type': 'Float8Tensor',
            '_data': {
                'block_size': [1,32],
                ...
            },
            '_tensor_data_names': ['qdata', 'scale']
        }
        '0.bias': {
            '_type': 'torch.Tensor',
        }
        'tensor_names': ['0.weight', '0.bias']
    }

    Args:
        tensor_dict: Dictionary of tensor subclasses to save, with keys as tensor names

    Returns:
        A tuple of (tensors_data, metadata) where
            tensors_data: Dict[str, torch.Tensor] contains the tensor data
            metadata: Dict[str, str] contains accompanying metadata from tensor subclass
        This structure is compatible with safetensors.save_file
    """

    metadata = {}
    tensors_data_dict = {}

    for tensor_name, tensor in tensors_dict.items():
        if tensor.__class__.__name__ in ALLOWED_TENSORS_SUBCLASSES:
            tensor_dict = {}

            all_tensor_data = list(tensor.tensor_data_names)  # create a copy
            if hasattr(tensor, "optional_tensor_data_names"):
                all_tensor_data += tensor.optional_tensor_data_names

            for tensor_data_name in all_tensor_data:
                if getattr(tensor, tensor_data_name) is not None:
                    tensor_dict[tensor_data_name] = getattr(tensor, tensor_data_name)

            tensor_metadata = json.dumps(tensor, cls=TensorSubclassAttributeJSONEncoder)

            # Clone tensors to avoid memory sharing issues
            tensors_dict_to_save = {
                f"{tensor_name.rsplit('.', 1)[0]}._{tensor_name.rsplit('.', 1)[1]}_{key}": (
                    value.detach().clone() if isinstance(value, torch.Tensor) else value
                )
                for key, value in tensor_dict.items()
            }

        elif type(tensor) is torch.Tensor:
            tensor_metadata = json.dumps({"_type": torch.Tensor.__name__})
            tensors_dict_to_save = {tensor_name: tensor}
        else:
            raise ValueError(f"Unsupported tensor type: {type(tensor)}")

        metadata[tensor_name] = tensor_metadata
        tensors_data_dict.update(tensors_dict_to_save)

    metadata["tensor_names"] = json.dumps(list(tensors_dict.keys()))
    return tensors_data_dict, metadata
