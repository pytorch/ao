# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""
This file defines the top level torch ops that are extended by MXTensor
See: https://docs.pytorch.org/docs/stable/notes/extending.html#extending-torch-with-a-tensor-wrapper-type
for more details.
"""

from typing import Any, Dict

import torch

from torchao.prototype.mx_formats.mx_ops import _addmm_mx_dispatch
from torchao.prototype.mx_formats.mx_tensor import (  # noqa: E501
    MXTensor,
)

aten = torch.ops.aten

MX_FUNC_TABLE: Dict[Any, Any] = {}


def implements_func(torch_ops):
    """Register torch ops to the mx op table for torch function"""

    def decorator(func):
        for op in torch_ops:
            MX_FUNC_TABLE[op] = func
        return func

    return decorator


@implements_func([aten.linear.default])
def mx_linear(func, types, args, kwargs):
    a, b = args[0], args[1]
    assert isinstance(a, MXTensor) and isinstance(b, MXTensor)
    bias = args[2] if len(args) == 3 else None
    return _addmm_mx_dispatch(a, b.t(), func, bias=bias)
