# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from typing import Optional

import torch
import torch.nn as nn

torch.ops.load_library(
    "/tmp/cmake-out/torch_ao/examples/torch_custom_op/libtorch_custom_op.dylib"
)


def quantize(vals: torch.Tensor, group_size: int, nbit: int, scale_only: bool):
    assert nbit >= 2 and nbit <= 8
    qmin = -(1 << (nbit - 1))
    qmax = (1 << (nbit - 1)) - 1

    n, k = vals.shape
    vals = vals.reshape(-1, group_size)
    vmins, _ = torch.min(vals, axis=1)
    vmaxs, _ = torch.max(vals, axis=1)
    group_scales = (vmaxs - vmins) / (qmax - qmin)

    if scale_only:
        group_qvals = torch.round(vals / group_scales.reshape(-1, 1))
    else:
        group_zeros = qmin - torch.round(vmins / group_scales)
        group_qvals = torch.round(
            group_zeros.reshape(-1, 1) + vals / group_scales.reshape(-1, 1)
        )

    group_qvals = torch.clip(group_qvals, qmin, qmax).reshape(n, k).to(torch.int8)

    if scale_only:
        return group_qvals, group_scales
    return group_qvals, group_scales, group_zeros


class Chn8ActGrp3WgtQuantizedLinear(nn.Module):
    nbit = 3

    def __init__(self, squeeze_unsqueeze_dim0=False):
        super().__init__()
        self.squeeze_unsqueeze_dim0 = squeeze_unsqueeze_dim0

    def initialize_from_unpacked_weights(self, weight_qvals, weight_scales, group_size):
        n, k = weight_qvals.shape

        # TODO(T200095131): convert self.n, self.k, self.group_size to
        # int when supported by AOTI
        self.n = torch.empty(n)
        self.k = torch.empty(k)
        self.group_size = torch.empty(group_size)
        self.packed_weights = torch.ops.torchao._pack_weights_3bit(
            weight_qvals, weight_scales, self.group_size
        )

    def initialize_from_packed_weights(self, packed_weights, n, k, group_size):
        # TODO(T200095131): convert self.n, self.k, self.group_size to
        # int when supported by AOTI
        self.n = torch.empty(n)
        self.k = torch.empty(k)
        self.group_size = torch.empty(group_size)
        self.packed_weights = packed_weights

    def forward(self, x):
        if self.squeeze_unsqueeze_dim0:
            x = x.squeeze(0)

        res = torch.ops.torchao._linear_3bit(
            self.packed_weights, self.n, self.k, self.group_size, x
        )

        if self.squeeze_unsqueeze_dim0:
            res = res.unsqueeze(0)
        return res


class Chn8ActGrp4WgtQuantizedLinear(nn.Module):
    nbit = 4

    def __init__(self, squeeze_unsqueeze_dim0=False):
        super().__init__()
        self.squeeze_unsqueeze_dim0 = squeeze_unsqueeze_dim0

    def initialize_from_unpacked_weights(self, weight_qvals, weight_scales, group_size):
        n, k = weight_qvals.shape

        # TODO(T200095131): convert self.n, self.k, self.group_size to
        # int when supported by AOTI
        self.n = torch.empty(n)
        self.k = torch.empty(k)
        self.group_size = torch.empty(group_size)
        self.packed_weights = torch.ops.torchao._pack_weights_4bit(
            weight_qvals, weight_scales, self.group_size
        )

    def initialize_from_packed_weights(self, packed_weights, n, k, group_size):
        # TODO(T200095131): convert self.n, self.k, self.group_size to
        # int when supported by AOTI
        self.n = torch.empty(n)
        self.k = torch.empty(k)
        self.group_size = torch.empty(group_size)
        self.packed_weights = packed_weights

    def forward(self, x):
        if self.squeeze_unsqueeze_dim0:
            x = x.squeeze(0)

        res = torch.ops.torchao._linear_4bit(
            self.packed_weights, self.n, self.k, self.group_size, x
        )

        if self.squeeze_unsqueeze_dim0:
            res = res.unsqueeze(0)
        return res


def replace_linear_with_quantized_linear(module: nn.Module, kwargs={}):
    group_size = kwargs["group_size"]
    nbit = kwargs["nbit"]
    squeeze_unsqueeze_dim0 = (
        kwargs["squeeze_unsqueeze_dim0"]
        if "squeeze_unsqueeze_dim0" in kwargs
        else False
    )

    for name, child in module.named_children():
        if isinstance(child, nn.Linear):
            assert child.bias is None
            n, k = child.weight.shape
            weight_qvals, weight_scales = quantize(
                child.weight, group_size=group_size, nbit=nbit, scale_only=True
            )

            if nbit == 3:
                setattr(
                    module, name, Chn8ActGrp3WgtQuantizedLinear(squeeze_unsqueeze_dim0)
                )
                getattr(module, name).initialize_from_unpacked_weights(
                    weight_qvals,
                    weight_scales,
                    group_size,
                )
            elif nbit == 4:
                setattr(
                    module, name, Chn8ActGrp4WgtQuantizedLinear(squeeze_unsqueeze_dim0)
                )
                getattr(module, name).initialize_from_unpacked_weights(
                    weight_qvals,
                    weight_scales,
                    group_size,
                )
            else:
                raise ValueError(f"Unsupported nbit: {nbit}")
        else:
            replace_linear_with_quantized_linear(child, kwargs)
