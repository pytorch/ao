# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import torch
from torch import Tensor
from torch.library import impl

torchao_lib = torch.library.Library("torchao", "IMPL")
for nbit in range(1, 8):

    @impl(torchao_lib, f"_linear_fp_act_{nbit}bit_weight", "Meta")
    def _(
        activations: Tensor,
        packed_weights: Tensor,
        group_size: int,
        scales: int,
        zeros: int,
    ):
        assert activations.dtype in [torch.float32, torch.float16, torch.bfloat16]
        assert activations.is_contiguous()
        assert activations.dim() == 2

        assert packed_weights.dtype == torch.uint8
        assert packed_weights.is_contiguous()

        m = activations.size(0)
        k = activations.size(1)
        n = packed_weights.size(0)

        assert k % 8 == 0
        assert n % 4 == 0

        assert group_size in [32, 64, 128, 256]

        assert scales.is_contiguous()
        assert scales.dim() == 2
        assert scales.size(1) == n

        assert zeros.is_contiguous()
        assert zeros.dim() == 2
        assert zeros.size(1) == n

        return torch.empty(m, n, dtype=activations.dtype, device="meta")
