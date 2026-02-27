# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD 3-Clause license found in the
# LICENSE file in the root directory of this source tree.

from typing import Union

import torch
import torch._prims_common as utils
from torch import Tensor, nn
from torch._prims_common.wrappers import _maybe_convert_to_dtype


def _nz_normalize(a: Tensor, norm_dims: Union[int, list[int]], eps: float) -> Tensor:
    """Computes the normalized tensor, ignoring zeroed out values.
    See torch._refs._normalize for more reference.
    """
    computation_dtype = utils.get_computation_dtype(a.dtype)
    a = _maybe_convert_to_dtype(a, computation_dtype)

    norm_dims = utils.canonicalize_dims(a.ndim, norm_dims)
    nz_mask = a.ne(0).detach()
    count = torch.sum(nz_mask, dim=norm_dims, keepdim=True).clamp_(min=1)
    mean = torch.sum(a, dim=norm_dims, keepdim=True) / count

    a_center = torch.where(nz_mask, a - mean, torch.zeros_like(a))
    biased_var = torch.sum(a_center.pow(2), dim=norm_dims, keepdim=True) / count
    out = a_center * torch.rsqrt(biased_var + eps)
    return out


class MaskedLayerNorm(nn.LayerNorm):
    """Layer normalization that ignores zeroed out elements in the input tensor.
    See torch._refs.native_layer_norm for reference.
    """

    def forward(self, input: Tensor) -> Tensor:
        normalized_ndim = len(self.normalized_shape)
        axis = input.ndim - normalized_ndim
        reduction_dims = list(range(axis, input.ndim))
        out = _nz_normalize(input, reduction_dims, self.eps)

        if self.elementwise_affine:
            out = out * self.weight + self.bias
        out = _maybe_convert_to_dtype(out, input.dtype)
        return out
