#  Copyright (c) Meta Platforms, Inc. and affiliates.

import torch.nn as nn
import math
import torch
from torch.autograd import Variable
import torch.nn.functional as F
import numpy as np

from torchao.quantization.quant_api import _replace_with_custom_fn_if_matches_filter

# original supermask
scores_min=None
scores_max=9e9
uniform_init_01 = False

# adjusted supermask, initialize scores with uniform distribution in [0,1], clamp scores in each step in [0,1]
# scores_min=0.
# scores_max=1.
# uniform_init_01 = True

def percentile(t, q):
    """Return the value that is larger than q% of t"""
    k = 1 + round(.01 * float(q) * (t.numel() - 1))
    return t.view(-1).kthvalue(k).values


class GetSubnet(torch.autograd.Function):
    """Supermask STE function"""
    @staticmethod
    def forward(ctx, scores, zeros, ones, sparsity):
        clamped_scores = scores.clamp(min=scores_min,max=scores_max)
        k_val = percentile(clamped_scores, sparsity*100)
        return torch.where(clamped_scores < k_val, zeros.to(scores.device), ones.to(scores.device))
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
    def __init__(self, sparsity, fixed_mask, fixed_weight, bitwidth, transform, fixed_transform, *args, **kwargs):
        tile_size = kwargs.pop("tile_size", 1)
        super(SupermaskLinear, self).__init__(*args, **kwargs)
        # initialize the scores
        max_sparsity = 1 - (1 / math.prod([math.ceil(k / tile_size) for k in self.weight.size()]))
        self.sparsity = sparsity
        if self.sparsity > max_sparsity:
            print(
                f"reducing sparsity from {self.sparsity} to {max_sparsity}",
                f"(maximum sparsity for layer with shape {self.weight.size()} and tile size {tile_size})"
            )
            self.sparsity = max_sparsity
        self.tile_size = tile_size
        self.sparsify_weights = False
        self.scores = nn.Parameter(
            torch.empty(
                [max(1, int(math.ceil(wn / tile_size))) for wn in self.weight.size()]
            ),
            requires_grad=not fixed_mask,
        )
        nn.init.uniform_(self.scores) if uniform_init_01 else nn.init.kaiming_uniform_(self.scores, a=math.sqrt(5))

        # the shift and the scale are transformation parameters 
        # the actually used weights = self.weight*self.scale+self.shift
        # the transformation is activated only for quantized weights
        self.shift=nn.Parameter(torch.Tensor(1).fill_(0.), requires_grad=False)
        self.scale=nn.Parameter(torch.Tensor(1).fill_(1.), requires_grad=False)
        
        with torch.no_grad():
            # if bitwidth is None, then use floating point values in self.weight
            # if bitwidth is not None, then quantize self.weight into k-bit (k=bitwidth)
            # quantized values are -2^(k-1), -2^(k-1)+1, ..., 0, 1, ..., 2^(k-1)-1 
            # these quantized values are uniformly distributed
            if bitwidth is not None:
                weights_max = torch.max(self.weight).item()
                weights_min = torch.min(self.weight).item()
                least_step = (weights_max-weights_min)/pow(2,bitwidth)
                left_bound = weights_min-1e-6
                right_bound = weights_min+least_step+1e-6
                # self.shift=nn.Parameter(torch.Tensor(1).fill_( (weights_min+(pow(2,bitwidth-1)+0.5)*least_step) if transform[0] is None else transform[0] ), requires_grad=not fixed_transform[0])
                # self.scale=nn.Parameter(torch.Tensor(1).fill_( least_step if transform[1] is None else transform[1] ), requires_grad=not fixed_transform[1])
                # for example, if using binary weights (k=1) with -a, +a, set transform = [a,2a]; if using binary weights (k=1) with a, 0, set transform = [0,-a];
                self.shift=nn.Parameter(torch.Tensor(1).fill_( 0. if transform[0] is None else transform[0] ), requires_grad=not fixed_transform[0])
                self.scale=nn.Parameter(torch.Tensor(1).fill_( 1. if transform[1] is None else transform[1] ), requires_grad=not fixed_transform[1])
                for i in range(-int(pow(2,bitwidth-1)),int(pow(2,bitwidth-1))):
                    self.weight[torch.logical_and(self.weight>left_bound, self.weight<=right_bound)] = i                 
                    left_bound = right_bound
                    right_bound += least_step

        self.weight.requires_grad = not fixed_weight

    def get_mask(self):
        subnet = GetSubnet.apply(self.scores,
                                 torch.zeros_like(self.scores),
                                 torch.ones_like(self.scores),
                                 self.sparsity)

        if self.tile_size != 1:
            for i, k in enumerate(self.weight.shape):
                subnet = subnet.repeat_interleave(self.tile_size, dim=i)
                subnet = torch.narrow(subnet, i, 0, k)

        return subnet
    
    def sparsify_offline(self):
        subnet = self.get_mask()
        self.weight.data = (self.weight*self.scale+self.shift) * subnet
        self.sparsify_weights = True

    def forward(self, x):
        if not self.sparsify_weights:
            subnet = self.get_mask()
            # w = (self.weight*self.scale+self.shift)
            w = ApplyMask.apply(self.weight, subnet)
            return F.linear(x, w, self.bias)
        return F.linear(x, self.weight, self.bias)

    @classmethod
    def from_linear(cls, linear, sparsity_level=0.0, blocksize=1, inference=True):
        module_new = None

        assert isinstance(linear, torch.nn.Linear)
        module_new = SupermaskLinear(
            sparsity_level, False, False, None, None, None,
            linear.in_features,
            linear.out_features,
            bias=linear.bias is not None,
            tile_size=blocksize,
        ).to(device=linear.weight.device, dtype=linear.weight.dtype)
        module_new.weight.data.copy_(linear.weight.data)
        if linear.bias is not None:
            module_new.bias.data.copy_(linear.bias.data)
        if inference:
            module_new.sparsify_offline()
        return module_new


def apply_supermask(
    model,
    linear_sparsity=0.0,
    linear_sp_tilesize=1,
    skip_last_layer_sparsity=False,
    skip_first_transformer_sparsity=False,
    device="cuda",
    verbose=False,
):
    sparsified_modules = {}

    for n, m in model.named_modules():
        # check conditions for skipping sparsity
        if skip_last_layer_sparsity and n == "heads.head":
            continue
        if skip_first_transformer_sparsity and "encoder.layers.encoder_layer_0" in n:
            continue

        if linear_sparsity != 0.0 and isinstance(m, torch.nn.Linear):
            new_m = SupermaskLinear(
                linear_sparsity,
                False,
                False,
                None,
                None,
                None,
                m.in_features,
                m.out_features,
                bias=m.bias is not None,
                device=device,
                tile_size=linear_sp_tilesize,
            )
            new_m.weight.data.copy_(m.weight.data)
            if m.bias is not None:
                new_m.bias.data.copy_(m.bias.data)
            sparsified_modules[n] = new_m
            continue

    # add modules to model
    for k, v in sparsified_modules.items():
        sm_name, ch_name = k.rsplit(".", 1)
        sm = model.get_submodule(sm_name)
        sm.add_module(ch_name, v)

        if verbose:
            print(
                f'sparsified module "{k}" with sparsity={v.sparsity}, tile size={v.tile_size}'
            )

    return model
