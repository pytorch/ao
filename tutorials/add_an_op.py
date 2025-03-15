# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD 3-Clause license found in the
# LICENSE file in the root directory of this source tree.
import torch

import torchao
from torchao.dtypes import to_nf4

# To create coverage for a new nf4 op we first attempt to run it

# Construct a small nf4 Tensor of desired shaped
a = torch.randn(64)
a[0] = 0

# Don't forget to pick block and scalar shapes that work for your shape
a_nf4 = to_nf4(a, 32, 2)

# Trust is good, print better
print(f"a: {a}")
print(f"a_nf4: {a_nf4}")


# If GELU is not supported you'll get the following error
# NotImplementedError: NF4Tensor dispatch: attempting to run aten.gelu.default, this is not supported
# torch.nn.functional.gelu(a_nf4)


# Next you can add this function using the implements decorator
@torchao.dtypes.nf4tensor.implements([torch.ops.aten.gelu.default])
def gelu(func, *args, **kwargs):
    # The torch dispatch convention is to pass all args and kwargs via the
    # args input.
    # args[0] here corresponds to the original *args
    # args[1] here corresponds to the original *kwargs
    # We're getting the first argument of the original args
    inp = args[0][0]
    # There's a way very inefficient way to implement it
    return to_nf4(
        torch.nn.functional.gelu(inp.to(torch.float32)),
        inp.block_size,
        inp.scaler_block_size,
    )


print(f"gelu(a): {torch.nn.functional.gelu(a)}")
print(f"gelu(a_nf4): {torch.nn.functional.gelu(a_nf4)}")

# We collect these implementations in torchao.dtypes.nf4tensor, but you can also
# just roll your own.
