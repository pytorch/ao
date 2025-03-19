# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD 3-Clause license found in the
# LICENSE file in the root directory of this source tree.
from typing import Any, Dict, Tuple

import torch

import torchao.ops
from torchao.swizzle.swizzle_tensor import SwizzleTensor

aten = torch.ops.aten
SWIZZLE_OPS_TABLE: Dict[Any, Any] = {}


def implements(aten_ops):
    """Register aten ops to the swizzle op table"""

    def decorator(func):
        for op in aten_ops:
            SWIZZLE_OPS_TABLE[op] = func
        return func

    return decorator


@implements([aten.mm.default])
def swizzle_mm(aten_op, args, kwargs=None):
    a = args[0]
    b = args[1]

    if torch.is_floating_point(a) and torch.is_floating_point(b):
        a_is_swizzled = False
        b_is_swizzled = False
        if isinstance(a, SwizzleTensor):
            a = a.as_tensor()
            a_is_swizzled = True
        if isinstance(b, SwizzleTensor):
            b = b.as_tensor()
            b_is_swizzled = True
        tensor_out = torchao.ops.swizzle_mm(a, b, a_is_swizzled, b_is_swizzled)
    else:
        a = a.unswizzle() if isinstance(a, SwizzleTensor) else a
        b = b.unswizzle() if isinstance(b, SwizzleTensor) else b
        tensor_out = aten_op(a, b, **kwargs)
    return tensor_out


@implements([aten.bmm.default])
def swizzle_mm(aten_op, args, kwargs=None):
    a = args[0]
    b = args[1]

    a = a.unswizzle() if isinstance(a, SwizzleTensor) else a
    b = b.unswizzle() if isinstance(b, SwizzleTensor) else b
    return aten_op(a, b, **kwargs)


@implements([aten.addmm.default])
def swizzle_addmm(aten_op, args, kwargs=None):
    bias = args[0]
    a = args[1]
    b = args[2]
    a = a.unswizzle() if isinstance(a, SwizzleTensor) else a
    b = b.unswizzle() if isinstance(b, SwizzleTensor) else b
    return aten_op(bias, a, b, args[3:], **kwargs)


@implements([aten._scaled_mm.default])
def swizzle_scaled_mm(aten_op, args, kwargs=None):
    a = args[0]
    b = args[1]
    scale_a = args[2]
    scale_b = args[3]
    bias = None if len(args) <= 4 else args[4]
    scale_result = None if len(args) <= 5 else args[5]
    out_dtype = None if len(args) <= 6 else args[6]

    a_is_swizzled = False
    b_is_swizzled = False
    if isinstance(a, SwizzleTensor):
        a = a.as_tensor()
        a_is_swizzled = True
    if isinstance(b, SwizzleTensor):
        b = b.as_tensor()
        b_is_swizzled = True
    return torchao.ops.swizzle_scaled_mm(a, b, a_is_swizzled, b_is_swizzled, scale_a, scale_b, bias, scale_result, out_dtype, **kwargs)


@implements([aten.permute.default])
def swizzle_permute(aten_op, args, kwargs=None):
    tensor = args[0]
    dims = args[1]
    if len(dims) == 2 and dims[0] == 1 and dims[1] == 0:
        return tensor.shallow_transpose()
    return aten_op(tensor.unswizzle(), dims)
