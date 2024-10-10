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
            w = (self.weight*self.scale+self.shift) * subnet
        else:
            w = self.weight
        return F.linear(x, w, self.bias)
    

class SupermaskConv2d(nn.Conv2d):
    """Supermask class for Conv2d layer"""
    def __init__(self, sparsity, fixed_mask, fixed_weight, bitwidth, transform, fixed_transform, *args, **kwargs):
        tile_size = kwargs.pop("tile_size", 1)
        super(SupermaskConv2d, self).__init__(*args, **kwargs)
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
                # self.scale=nn.Parameter(torch.Tensor(1).fill_( least_step if transform[1] is None else transform[1]), requires_grad=not fixed_transform[1])
                # for example, if using binary weights (k=1) with -a, +a, set transform = [a,2a]; if using binary weights (k=1) with a, 0, set transform = [0,-a];
                self.shift=nn.Parameter(torch.Tensor(1).fill_( 0. if transform[0] is None else transform[0] ), requires_grad=not fixed_transform[0])
                self.scale=nn.Parameter(torch.Tensor(1).fill_( 1. if transform[1] is None else transform[1] ), requires_grad=not fixed_transform[1])
                for i in range(-int(pow(2,bitwidth-1)),int(pow(2,bitwidth-1))):
                    self.weight[torch.logical_and(self.weight>left_bound, self.weight<=right_bound)] = i                 
                    left_bound = right_bound
                    right_bound += least_step

        self.weight.requires_grad = not fixed_weight

    def forward(self, x):
        subnet = GetSubnet.apply(self.scores,
                                 torch.zeros_like(self.scores),
                                 torch.ones_like(self.scores),
                                 self.sparsity)
    
        if self.tile_size != 1:
            for i, k in enumerate(self.weight.shape):
                # if k == 1: continue
                subnet = subnet.repeat_interleave(self.tile_size, dim=i)
                subnet = torch.narrow(subnet, i, 0, k)

        w = (self.weight*self.scale+self.shift) * subnet
        return F.conv2d(x, w, self.bias, self.stride, self.padding, self.dilation, self.groups)

def apply_supermask(
    model,
    linear_sparsity=0.0,
    linear_sp_tilesize=1,
    conv1x1_sparsity=0.0,
    conv1x1_sp_tilesize=1,
    conv_sparsity=0.0,
    conv_sp_tilesize=1,
    skip_last_layer_sparsity=False,
    skip_first_transformer_sparsity=False,
    device="cuda",
    verbose=False,
):
    # create filter function
    # TODO: it might be better to move the filtering function to the script calling this function
    is_last_layer = lambda module, name: name == "heads.head"
    is_first_transformer_layer = lambda module, name: name == "encoder.layers.encoder_layer_0"
    # TODO: create condition for ffn, k,v,q,o projections
    reject_fn = lambda module, name : (skip_last_layer_sparsity and is_last_layer(module, name)) or (skip_first_transformer_sparsity and is_first_transformer_layer(module, name))
    filter_fn = lambda module, name : not reject_fn(module, name) and isinstance(module, (torch.nn.Linear, torch.nn.Conv2d))

    _replace_with_custom_fn_if_matches_filter(
        model,
        SuperMaskReplacementClass(
            linear_sparsity=linear_sparsity,
            linear_sp_tilesize=linear_sp_tilesize,
            conv1x1_sparsity=conv1x1_sparsity,
            conv1x1_sp_tilesize=conv1x1_sp_tilesize,
            conv_sparsity=conv_sparsity,
            conv_sp_tilesize=conv_sp_tilesize,
            device=device,
            verbose=verbose,
        ),
        filter_fn,
    )

class SuperMaskReplacementClass:
    def __init__(
        self,
        linear_sparsity=0.0,
        linear_sp_tilesize=1,
        conv1x1_sparsity=0.0,
        conv1x1_sp_tilesize=1,
        conv_sparsity=0.0,
        conv_sp_tilesize=1,
        device="cuda",
        verbose=False,
    ):
        self.linear_sparsity = linear_sparsity
        self.linear_sp_tilesize = linear_sp_tilesize
        self.conv1x1_sparsity = conv1x1_sparsity
        self.conv1x1_sp_tilesize = conv1x1_sp_tilesize
        self.conv_sparsity = conv_sparsity
        self.conv_sp_tilesize = conv_sp_tilesize
        self.device = device
        self.verbose = verbose

    def __call__(self, module):
        module_new = None

        if self.conv1x1_sparsity != 0.0 and isinstance(module, torch.nn.Conv2d) and module.kernel_size == (1, 1):
            # convert 1x1 convolutions
            module_new = SupermaskConv2d(
                self.conv1x1_sparsity, False, False, None, None, None,
                module.in_channels,
                module.out_channels,
                module.kernel_size, 
                stride=module.stride,
                padding=module.padding,
                dilation=module.dilation,
                groups=module.groups,
                bias=module.bias is not None,
                padding_mode=module.padding_mode,
                tile_size=self.conv1x1_sp_tilesize,
            ).to(device=self.device, dtype=module.weight.dtype)
            module_new.weight.data.copy_(module.weight.data)
            if module.bias is not None:
                module_new.bias.data.copy_(module.bias.data)
        elif self.conv_sparsity != 0.0 and isinstance(module, torch.nn.Conv2d):
            # convert all other convolutions (not tested!)
            module_new = SupermaskConv2d(
                self.conv_sparsity, False, False, None, None, None,
                module.in_channels,
                module.out_channels,
                module.kernel_size, 
                stride=module.stride,
                padding=module.padding,
                dilation=module.dilation,
                groups=module.groups,
                bias=module.bias is not None,
                padding_mode=module.padding_mode,
                tile_size=self.conv_sp_tilesize,
            ).to(device=self.device, dtype=module.weight.dtype)
            module_new.weight.data.copy_(module.weight.data)
            if module.bias is not None:
                module_new.bias.data.copy_(module.bias.data)
        elif self.linear_sparsity != 0.0 and isinstance(module, torch.nn.Linear):
            module_new = SupermaskLinear(
                self.linear_sparsity, False, False, None, None, None,
                module.in_features,
                module.out_features,
                bias=module.bias is not None,
                tile_size=self.linear_sp_tilesize,
            ).to(device=self.device, dtype=module.weight.dtype)
            module_new.weight.data.copy_(module.weight.data)
            if module.bias is not None:
                module_new.bias.data.copy_(module.bias.data)
        else:
            return module

        if self.verbose:
            print(f'sparsified module "{module}" with sparsity={module_new.sparsity}, tile size={module_new.tile_size}')

        return module_new
