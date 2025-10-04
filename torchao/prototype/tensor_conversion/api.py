# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD 3-Clause license found in the
# LICENSE file in the root directory of this source tree.

import torch
import torch.nn as nn

# TODO: move the function to torchao.utils
from torchao.dtypes.utils import is_device
from torchao.quantization import (
    Int4PreshuffledTensor,
    Int4Tensor,
    IntxUnpackedToInt8Tensor,
)
from torchao.utils import TorchAOBaseTensor, _is_fbgemm_gpu_genai_available


def _convert_linear_weight_to_int8_lut_tensor(module):
    from torchao.prototype.quantization.int8_lut_tensor import Int8LutTensor

    assert isinstance(module, nn.Linear)
    weight = module.weight
    new_weight = Int8LutTensor.from_intx_unpacked_to_int8_tensor(
        weight, bias=module.bias
    )
    module.weight = torch.nn.Parameter(new_weight, requires_grad=False)
    module.bias = None


def _convert_module_weight_to_intx_opaque_tensor(module, intx_packing_format):
    from torchao.quantization.quantize_.workflows.intx.intx_opaque_tensor import (
        IntxOpaqueTensor,
    )

    assert isinstance(module, nn.Linear) or isinstance(module, nn.Embedding)
    weight = module.weight
    new_weight = IntxOpaqueTensor.from_intx_unpacked_to_int8_tensor(
        weight,
        bias=module.bias if hasattr(module, "bias") else None,
        intx_packing_format=intx_packing_format,
    )
    module.weight = torch.nn.Parameter(new_weight, requires_grad=False)
    if hasattr(module, "bias"):
        module.bias = None


def _find_tied_module_names_for_embedding(embedding_weight, model):
    assert isinstance(embedding_weight, IntxUnpackedToInt8Tensor)
    tied_names = []
    for name, module in model.named_modules():
        is_linear = isinstance(module, nn.Linear)
        is_embedding = isinstance(module, nn.Embedding)
        if not (is_linear or is_embedding):
            continue

        weight = module.weight
        if not isinstance(weight, IntxUnpackedToInt8Tensor):
            continue

        # We only have tied kernels for dynamically quantized linears
        if is_linear and weight.activation_quantization != "int8_asym_per_token":
            continue

        # We only have tied kernels for linear layers with no bias
        if is_linear and module.bias is not None:
            continue

        are_tied = (
            (embedding_weight.shape == weight.shape)
            and (embedding_weight.block_size == weight.block_size)
            and (embedding_weight.dtype == weight.dtype)
            and (embedding_weight.qdata == weight.qdata).all()
            and (embedding_weight.scale == weight.scale).all()
            and (embedding_weight.zero_point == weight.zero_point).all()
        )

        if are_tied:
            tied_names.append(name)

    return tied_names


def _find_tied_params(model):
    from torchao.quantization.quantize_.workflows.intx.intx_opaque_tensor import (
        IntxOpaqueTensor,
    )

    module_name_to_tied_param = {}
    for name, module in model.named_modules():
        if not isinstance(module, nn.Embedding):
            continue

        weight = module.weight
        if not isinstance(weight, IntxUnpackedToInt8Tensor):
            continue

        tied_module_names = _find_tied_module_names_for_embedding(weight, model)
        if not tied_module_names:
            continue

        if name in module_name_to_tied_param:
            tied_param = module_name_to_tied_param[name]
        else:
            # Construct a new tied param
            # IntxOpaqueTensor requires activation_quantization = int8_asym_per_token
            prev = weight.activation_quantization
            weight.activation_quantization = "int8_asym_per_token"
            tied_param = IntxOpaqueTensor.from_intx_unpacked_to_int8_tensor(
                weight,
                bias=None,
                intx_packing_format="opaque_torchao_lowbit",
            )
            weight.activation_quantization = prev
            tied_param = nn.Parameter(tied_param, requires_grad=False)
            module_name_to_tied_param[name] = tied_param

        for t in tied_module_names:
            if t not in module_name_to_tied_param:
                module_name_to_tied_param[t] = tied_param

    return module_name_to_tied_param


def _convert_model_for_aarch64(
    model,
    *,
    tensor_type="auto",
    intx_packing_format="opaque_torchao_auto",
    convert_tied_embedding=True,
    convert_linear=True,
):
    module_name_to_tied_param = (
        _find_tied_params(model) if convert_tied_embedding else {}
    )

    # Iterate through modules in model and convert IntxUnpackedToInt8Tensor tensors to Int8LutTensor
    for name, module in model.named_modules():
        if name in module_name_to_tied_param:
            module.weight = module_name_to_tied_param[name]
            continue

        if isinstance(module, nn.Embedding):
            print("Skipping converting nn.Embedding {name} because it is not tied")
            continue

        if not (convert_linear and isinstance(module, nn.Linear)):
            continue

        weight = module.weight
        if not isinstance(weight, IntxUnpackedToInt8Tensor):
            print(
                f"Skipping converting {name} to IntxOpaqueTensor because its weight is not an IntxUnpackedToInt8Tensor"
            )
            continue

        if tensor_type == "int8_lut_tensor":
            _convert_linear_weight_to_int8_lut_tensor(module)
        elif tensor_type == "intx_opaque_tensor":
            _convert_module_weight_to_intx_opaque_tensor(module, intx_packing_format)
        elif tensor_type == "auto":
            if weight._has_float_zero_point() and isinstance(module, nn.Linear):
                _convert_linear_weight_to_int8_lut_tensor(module)
            else:
                _convert_module_weight_to_intx_opaque_tensor(
                    module, intx_packing_format
                )
        else:
            raise ValueError(f"Unexpected tensor_type={tensor_type}")

    return model


def convert_to_packed_tensor_based_on_current_hardware(tensor: TorchAOBaseTensor):
    """Convert a plain / unpacked torchao tensor to a packed one based on hardware

    Goal is to have an optimized performance on current hardware, while also allow
    us to
    (1). distribute a single unpacked / plain format that can be used in multiple hardwares
    (2). support the vLLM use case, where we need to slice the weights for distributed
    inference. Since slice is not always supported in packed weight, we would like to first
    load plain / unpacked weight, slice it and then convert to packed weight to get the best
    inference speed
    """
    if (
        isinstance(tensor, Int4Tensor)
        and is_device("cuda", tensor.device)
        and _is_fbgemm_gpu_genai_available()
    ):
        return Int4PreshuffledTensor.from_int4_tensor(tensor)
    return tensor
