# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD 3-Clause license found in the
# LICENSE file in the root directory of this source tree.

import torch.nn as nn
import torch.nn.functional as F


class FeedForward(nn.Module):
    """MLP based model"""

    def __init__(self):
        super(FeedForward, self).__init__()
        self.w1 = nn.Linear(16, 32, bias=False)
        self.w2 = nn.Linear(16, 32, bias=False)
        self.out_proj = nn.Linear(32, 16, bias=False)

    def forward(self, x):
        return self.out_proj(F.silu(self.w1(x)) * self.w2(x))


class ToyModel(nn.Module):
    def __init__(self):
        super(ToyModel, self).__init__()
        self.ffn = FeedForward()

    def forward(self, x):
        return self.ffn(x)
