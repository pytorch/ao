# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD 3-Clause license found in the
# LICENSE file in the root directory of this source tree.

from dataclasses import dataclass

import torch


@dataclass
class Test:
    x: torch.Tensor = torch.ones(256, 256).cuda()


@torch.compile(fullgraph=True)
def test_trace(inp, asdf):
    return inp.x.mm(asdf)


a = Test()
b = torch.ones(256, 512).cuda()

res = test_trace(a, b)
print(res.shape)
