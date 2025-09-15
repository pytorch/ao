# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD 3-Clause license found in the
# LICENSE file in the root directory of this source tree.

import torch
import torch.nn as nn


def _convert_linear_weight_to_int8_lut_tensor(module):
    from torchao.prototype.quantization.int8_lut_tensor import Int8LutTensor

    assert isinstance(module, nn.Linear)
    weight = module.weight
    new_weight = Int8LutTensor.from_intx_unpacked_to_int8_tensor(
        weight, bias=module.bias
    )
    module.weight = torch.nn.Parameter(new_weight, requires_grad=False)
    module.bias = None


def _convert_model_for_aarch64(
    model,
    *,
    tensor_type="int8_lut_tensor",
):
    from torchao.quantization.quantize_.workflows import IntxUnpackedToInt8Tensor

    # Iterate through modules in model and convert IntxUnpackedToInt8Tensor tensors to Int8LutTensor
    for name, module in model.named_modules():
        if not isinstance(module, nn.Linear):
            print(f"Skipping converting {name} because it is not a linear layer")
            continue

        weight = module.weight
        if not isinstance(weight, IntxUnpackedToInt8Tensor):
            print(
                f"Skipping converting {name} to IntxOpaqueTensor because its weight is not an IntxUnpackedToInt8Tensor"
            )
            continue

        if tensor_type == "int8_lut_tensor":
            _convert_linear_weight_to_int8_lut_tensor(module)
        else:
            raise ValueError(f"Unexpected tensor_type={tensor_type}")

    return model
