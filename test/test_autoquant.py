# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

# mypy: ignore-errors
import copy
import unittest

import torch
import torch.nn as nn
from torchao.quantization.quant_api import (
    change_linears_to_autoquantizable,
    change_autoquantizable_to_quantized
)
from torchao.quantization.autoquant import do_autoquant
from torch._dynamo import config
torch.manual_seed(0)
config.cache_size_limit = 100


class AutoquantTests(unittest.TestCase):
    def test_autoquant_e2e(self):
        model = torch.nn.Sequential(torch.nn.Linear(32,32), torch.nn.ReLU(), torch.nn.Linear(32,32)).cuda().to(torch.bfloat16)
        print(model, model[0].weight)
        example_input = torch.randn((1,64,32), dtype=torch.bfloat16, device=torch.cuda)
        out=model(example_input)
        print(out.sum())
        do_autoquant(model)
        print(model, model[0].weight)
        print(model(example_input).sum())

if __name__ == "__main__":
    unittest.main()
