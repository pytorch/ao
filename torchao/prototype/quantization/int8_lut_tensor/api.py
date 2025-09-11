# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD 3-Clause license found in the
# LICENSE file in the root directory of this source tree.

import torch
import torch.nn as nn

from torchao.prototype.quantization.int8_lut_tensor.int8_lut_tensor import (
    Int8LutTensor,
)
from torchao.quantization.quantize_.workflows import IntxUnpackedToInt8Tensor


def convert_model(model):
    # Iterate through modules in model and convert IntxUnpackedToInt8Tensor tensors to Int8LutTensor
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear):
            weight = module.weight
            if not isinstance(weight, IntxUnpackedToInt8Tensor):
                print(
                    f"Skipping converting {name} to Int8LutTensor because its weight is not an IntxUnpackedToInt8Tensor"
                )
                continue
            try:
                new_weight = Int8LutTensor.from_intx_unpacked_to_int8_tensor(
                    weight, bias=module.bias
                )
                module.weight = torch.nn.Parameter(new_weight, requires_grad=False)
                module.bias = None
            except Exception as e:
                print(
                    f"Skipping converting {name} to Int8LutTensor because an error occurred: {e}"
                )
                continue
    return model
