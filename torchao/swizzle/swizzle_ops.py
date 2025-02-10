# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD 3-Clause license found in the
# LICENSE file in the root directory of this source tree.
from typing import Any, Dict, Tuple

import torch

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
    tensor_out = aten_op(a, b, **kwargs)
    return tensor_out


@implements([aten.addmm.default])
def swizzle_addmm(aten_op, args, kwargs=None):
    bias = args[0]
    a = args[1]
    b = args[2]
    a = a.unswizzle() if isinstance(a, SwizzleTensor) else a
    b = b.unswizzle() if isinstance(b, SwizzleTensor) else b
    return aten_op(bias, a, b, args[3:], **kwargs)


