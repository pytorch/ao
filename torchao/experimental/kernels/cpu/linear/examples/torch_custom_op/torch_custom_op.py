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


def linear_a8sz_w_lowbit_reference_impl(
    weights, activations, group_size, nbit, has_weight_zeros
):
    n, k = weights.shape
    m, k = activations.shape
    assert m == 1
    assert k % group_size == 0

    if has_weight_zeros:
        weight_qvals, weight_scales, weight_zeros = quantize(
            weights, group_size, nbit, scale_only=False
        )
        weights_dequantized = (
            weight_scales.reshape(-1, 1)
            * (weight_qvals.reshape(-1, group_size) - weight_zeros.reshape(-1, 1))
        ).reshape(n, k)
    else:
        weight_qvals, weight_scales = quantize(
            weights, group_size, nbit, scale_only=True
        )
        weights_dequantized = (
            weight_scales.reshape(-1, 1) * (weight_qvals.reshape(-1, group_size))
        ).reshape(n, k)

    activation_qvals, activations_scales, activations_zeros = quantize(
        activations, k, 8, False
    )
    activations_dequantized = activations_scales * (
        activation_qvals - activations_zeros
    ).reshape(m, k)
    return torch.matmul(activations_dequantized, weights_dequantized.transpose(1, 0))


class _quantized_linear(nn.Module):
    def __init__(
        self,
        nbit,
        has_weight_zeros,
        pack_weight_op,
        linear_op,
        squeeze_unsqueeze_dim0=False,
    ):
        super().__init__()
        self.squeeze_unsqueeze_dim0 = squeeze_unsqueeze_dim0
        self.nbit = nbit

        self._has_weight_zeros = has_weight_zeros
        self._pack_weights_op = pack_weight_op
        self._linear_op = linear_op

    def pack_weights(self, weight_qvals, weight_scales_and_zeros, group_size):
        n, k = weight_qvals.shape

        # TODO(T200095131): convert self.n, self.k, self.group_size to
        # int when supported by AOTI
        self.n = torch.empty(n)
        self.k = torch.empty(k)
        self.group_size = torch.empty(group_size)

        if self._has_weight_zeros:
            weight_scales, weight_zeros = weight_scales_and_zeros
            self.packed_weights = self._pack_weights_op(
                weight_qvals, weight_scales, weight_zeros, self.group_size
            )
        else:
            weight_scales = weight_scales_and_zeros
            self.packed_weights = self._pack_weights_op(
                weight_qvals, weight_scales, self.group_size
            )

    def forward(self, x):
        if self.squeeze_unsqueeze_dim0:
            x = x.squeeze(0)

        res = self._linear_op(self.packed_weights, self.n, self.k, self.group_size, x)

        if self.squeeze_unsqueeze_dim0:
            res = res.unsqueeze(0)
        return res


def replace_linear_with_quantized_linear(module: nn.Module, kwargs={}):
    group_size = kwargs["group_size"]
    nbit = kwargs["nbit"]
    has_weight_zeros = kwargs["has_weight_zeros"]
    squeeze_unsqueeze_dim0 = (
        kwargs["squeeze_unsqueeze_dim0"]
        if "squeeze_unsqueeze_dim0" in kwargs
        else False
    )

    for name, child in module.named_children():
        if isinstance(child, nn.Linear):
            assert child.bias is None

            if not has_weight_zeros:
                weight_qvals, weight_scales = quantize(
                    child.weight, group_size=group_size, nbit=nbit, scale_only=True
                )
                weight_scales_and_zeros = weight_scales
            else:
                weight_qvals, weight_scales, weight_zeros = quantize(
                    child.weight, group_size=group_size, nbit=nbit, scale_only=False
                )
                weight_scales_and_zeros = (weight_scales, weight_zeros.to(torch.int8))

            qlinear = None
            if nbit == 2:
                if has_weight_zeros:
                    qlinear = _quantized_linear(
                        nbit=nbit,
                        has_weight_zeros=has_weight_zeros,
                        pack_weight_op=torch.ops.torchao._pack_weights_a8sz_w2sz,
                        linear_op=torch.ops.torchao._linear_a8sz_w2sz,
                        squeeze_unsqueeze_dim0=squeeze_unsqueeze_dim0,
                    )
                else:
                    qlinear = _quantized_linear(
                        nbit=nbit,
                        has_weight_zeros=has_weight_zeros,
                        pack_weight_op=torch.ops.torchao._pack_weights_a8sz_w2s,
                        linear_op=torch.ops.torchao._linear_a8sz_w2s,
                        squeeze_unsqueeze_dim0=squeeze_unsqueeze_dim0,
                    )
            elif nbit == 3:
                if has_weight_zeros:
                    qlinear = _quantized_linear(
                        nbit=nbit,
                        has_weight_zeros=has_weight_zeros,
                        pack_weight_op=torch.ops.torchao._pack_weights_a8sz_w3sz,
                        linear_op=torch.ops.torchao._linear_a8sz_w3sz,
                        squeeze_unsqueeze_dim0=squeeze_unsqueeze_dim0,
                    )
                else:
                    qlinear = _quantized_linear(
                        nbit=nbit,
                        has_weight_zeros=has_weight_zeros,
                        pack_weight_op=torch.ops.torchao._pack_weights_a8sz_w3s,
                        linear_op=torch.ops.torchao._linear_a8sz_w3s,
                        squeeze_unsqueeze_dim0=squeeze_unsqueeze_dim0,
                    )
            elif nbit == 4:
                if has_weight_zeros:
                    qlinear = _quantized_linear(
                        nbit=nbit,
                        has_weight_zeros=has_weight_zeros,
                        pack_weight_op=torch.ops.torchao._pack_weights_a8sz_w4sz,
                        linear_op=torch.ops.torchao._linear_a8sz_w4sz,
                        squeeze_unsqueeze_dim0=squeeze_unsqueeze_dim0,
                    )
                else:
                    qlinear = _quantized_linear(
                        nbit=nbit,
                        has_weight_zeros=has_weight_zeros,
                        pack_weight_op=torch.ops.torchao._pack_weights_a8sz_w4s,
                        linear_op=torch.ops.torchao._linear_a8sz_w4s,
                        squeeze_unsqueeze_dim0=squeeze_unsqueeze_dim0,
                    )
            elif nbit == 5:
                if has_weight_zeros:
                    qlinear = _quantized_linear(
                        nbit=nbit,
                        has_weight_zeros=has_weight_zeros,
                        pack_weight_op=torch.ops.torchao._pack_weights_a8sz_w5sz,
                        linear_op=torch.ops.torchao._linear_a8sz_w5sz,
                        squeeze_unsqueeze_dim0=squeeze_unsqueeze_dim0,
                    )
                else:
                    qlinear = _quantized_linear(
                        nbit=nbit,
                        has_weight_zeros=has_weight_zeros,
                        pack_weight_op=torch.ops.torchao._pack_weights_a8sz_w5s,
                        linear_op=torch.ops.torchao._linear_a8sz_w5s,
                        squeeze_unsqueeze_dim0=squeeze_unsqueeze_dim0,
                    )
            else:
                raise ValueError(
                    f"Unsupported nbit ({nbit}) and has_weight_zeros ({has_weight_zeros}) combination"
                )

            assert qlinear is not None
            setattr(module, name, qlinear)
            getattr(module, name).pack_weights(
                weight_qvals,
                weight_scales_and_zeros,
                group_size,
            )
        else:
            replace_linear_with_quantized_linear(child, kwargs)
