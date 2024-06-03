# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.
#
# This source code is licensed under the BSD license found in the
# LICENSE file in the root directory of this source tree.
# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.
#
# This source code is licensed under the BSD license found in the
# LICENSE file in the root directory of this source tree.
from typing import Tuple

import torch
import torch.nn.functional as F
from torch import nn
from xformers_benchmark_utils import DTYPE2STR, benchmark_main_helper2, product_dict

from torchao.sparsity.prototype.training import SemiSparseLinear
from torchao.sparsity.prototype.training.autograd import semi_sparse_sparsify

min_run_time = 0.5
device = torch.device("cuda")

CASES = list(
    product_dict(
        B_in_hidden_out_ft=[
            # DINO ViT-L: lg + sm crops (patch16)
            (64 * 2 * (14 * 14 + 1) + 64 * 8 * (6 * 6 + 1), 1024, 1024 * 4, 1024),
        ],
        dtype=[torch.half],
        bias=[False],
    )
)

class Mlp(nn.Module):
    LINEAR_CLS = nn.Linear

    def __init__(
        self, B_in_hidden_out_ft: Tuple[int, int, int, int], dtype, bias: bool, bw: bool
    ) -> None:
        B, in_ft, hid_ft, out_ft = B_in_hidden_out_ft
        super().__init__()
        self.label = "mlp"
        self.sub_label = (
            f"{DTYPE2STR[dtype]} ({B},{in_ft},{hid_ft},{out_ft}){' b' if bias else ''}"
        )
        self.fc1 = self.LINEAR_CLS(in_ft, hid_ft, bias=bias)
        self.act = nn.GELU()
        self.fc2 = self.LINEAR_CLS(hid_ft, out_ft, bias=bias)
        self.grad = torch.randn([B, out_ft], device="cuda", dtype=dtype)
        self.input = torch.randn(
            [B, in_ft], device="cuda", dtype=dtype, requires_grad=True
        )
        self.out = self.input
        self.to("cuda").to(dtype)

    def fw(self):
        x = self.input
        x = self.fc1(x)
        x = self.act(x)
        x = self.fc2(x)
        self.out = x

    def bw(self):
        self.out.backward(self.grad, retain_graph=True)


class MlpAct24(Mlp):
    def fw(self):
        x = self.input
        x = self.fc1(x)

        x = semi_sparse_sparsify(x)

        x = self.act(x)
        x = self.fc2(x)
        self.out = x



class MlpW24(Mlp):
    LINEAR_CLS = SemiSparseLinear


class MicrobenchmarkBase:
    def __init__(
        self, B_in_hidden_out_ft: Tuple[int, int, int, int], dtype, bias: bool, bw: bool
    ) -> None:
        B, in_ft, hid_ft, out_ft = B_in_hidden_out_ft
        super().__init__()
        self.label = "mlp"
        self.sub_label = (
            f"{DTYPE2STR[dtype]} ({B},{in_ft},{hid_ft},{out_ft}){' b' if bias else ''}"
        )
        self.input = torch.randn(
            [B, in_ft], device="cuda", dtype=dtype, requires_grad=True
        )
        self.input_colMajor = self.input.t().contiguous().t()
        self.input_sp = semi_sparse_sparsify(self.input)

    def bw(self) -> None:
        return None


class MicrobenchmarkSparsify24(MicrobenchmarkBase):
    def fw(self) -> torch.Tensor:
        semi_sparse_sparsify(self.input)
        return self.input


class MicrobenchmarkInputClone(MicrobenchmarkBase):
    def fw(self) -> torch.Tensor:
        self.input.clone()
        return self.input


functions = {
    "act24": MlpAct24,
    "dense": Mlp,
    "w24": MlpW24,
    "s24_inp_sparsify24": MicrobenchmarkSparsify24,
    "s24_inp_clone": MicrobenchmarkInputClone,
}
benchmark_main_helper2(
    "sp24_fwbw",
    fw=True,
    bw=True,
    cases=CASES,
    functions=functions,
    min_run_time=min_run_time,
)
