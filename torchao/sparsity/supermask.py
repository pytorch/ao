#  Copyright (c) Meta Platforms, Inc. and affiliates.

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

SCORES_MIN = None
SCORES_MAX = 9e9


def percentile(t, q):
    """Return the value that is larger than q% of t"""
    k = 1 + round(0.01 * float(q) * (t.numel() - 1))
    return t.view(-1).kthvalue(k).values


class GetSubnet(torch.autograd.Function):
    """Supermask STE function"""

    @staticmethod
    def forward(ctx, scores, zeros, ones, sparsity):
        clamped_scores = scores.clamp(min=SCORES_MIN, max=SCORES_MAX)
        k_val = percentile(clamped_scores, sparsity * 100)
        return torch.where(
            clamped_scores < k_val, zeros.to(scores.device), ones.to(scores.device)
        )

    @staticmethod
    def backward(ctx, g):
        return g, None, None, None


class ApplyMask(torch.autograd.Function):
    """Supermask STE function"""

    @staticmethod
    def forward(ctx, weight, scores):
        return weight * scores

    @staticmethod
    def backward(ctx, grad_output):
        grad_weight = grad_scores = None
        if ctx.needs_input_grad[0]:
            grad_weight = grad_output
        if ctx.needs_input_grad[1]:
            grad_scores = grad_output
        return grad_weight, grad_scores


class SupermaskLinear(nn.Linear):
    """Supermask class for Linear layer"""

    def __init__(
        self, sparsity_level, blocksize, fixed_mask, fixed_weight, *args, **kwargs
    ):
        super(SupermaskLinear, self).__init__(*args, **kwargs)
        # calculate the maximum sparsity given blocksize for the layer
        max_sparsity_level = 1 - (
            1 / math.prod([math.ceil(k / blocksize) for k in self.weight.size()])
        )
        self.sparsity_level = sparsity_level
        if self.sparsity_level > max_sparsity_level:
            print(
                f"reducing sparsity from {self.sparsity} to {max_sparsity_level}",
                f"(maximum sparsity for layer with shape {self.weight.size()} and tile size {blocksize})",
            )
            self.sparsity_level = max_sparsity_level
        self.blocksize = blocksize
        self.sparsify_weights = False
        self.scores = nn.Parameter(
            torch.empty(
                [max(1, int(math.ceil(wn / blocksize))) for wn in self.weight.size()]
            ),
            requires_grad=not fixed_mask,
        )
        nn.init.kaiming_uniform_(self.scores, a=math.sqrt(5))

        # NOTE: the previous implementation of Supermask supported quantizing the weights, this has been removed.

        self.weight.requires_grad = not fixed_weight

    def get_mask(self):
        subnet = GetSubnet.apply(
            self.scores,
            torch.zeros_like(self.scores),
            torch.ones_like(self.scores),
            self.sparsity_level,
        )

        if self.blocksize != 1:
            for i, k in enumerate(self.weight.shape):
                subnet = subnet.repeat_interleave(self.blocksize, dim=i)
                subnet = torch.narrow(subnet, i, 0, k)

        return subnet

    def forward(self, x):
        subnet = self.get_mask()
        w = ApplyMask.apply(self.weight, subnet)
        return F.linear(x, w, self.bias)

    @classmethod
    def from_linear(
        cls,
        linear,
        sparsity_level=0.0,
        blocksize=1,
    ):
        """
        Main entrypoint for creating a SupermaskLinear from a Linear layer.
        """
        assert isinstance(linear, torch.nn.Linear)

        supermask_linear = SupermaskLinear(
            sparsity_level,
            blocksize,
            False,
            False,
            linear.in_features,
            linear.out_features,
            bias=linear.bias is not None,
        ).to(device=linear.weight.device, dtype=linear.weight.dtype)
        supermask_linear.weight.data.copy_(linear.weight.data)
        if linear.bias is not None:
            supermask_linear.bias.data.copy_(linear.bias.data)
        return supermask_linear

    @classmethod
    def to_linear(cls, supermask_linear):
        """
        Convert a SupermaskLinear to a Linear layer.
        Replaces the old sparsify_offline() function.
        """
        self = supermask_linear

        linear = torch.nn.Linear(
            self.in_features,
            self.out_features,
            bias=self.bias is not None,
        ).to(device=self.weight.device, dtype=self.weight.dtype)

        mask = self.get_mask()
        linear.weight.data.copy_(self.weight * mask)
        if self.bias is not None:
            linear.bias.data.copy_(self.bias.data)
        return linear
