# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import torch
from torch import Tensor
from torch.library import impl

# Define meta ops.  To support dynamic shapes, some meta ops need to
# be defined in python instead of C++.
torchao_lib = torch.library.Library("torchao", "IMPL")


def _op_is_registered(name: str) -> bool:
    return hasattr(torch.ops.torchao, name)


for weight_nbit in range(1, 9):
    linear_op = f"_linear_8bit_act_{weight_nbit}bit_weight"
    if _op_is_registered(linear_op):

        @impl(torchao_lib, linear_op, "Meta")
        def _(
            activations: Tensor,
            packed_weights: Tensor,
            group_size: int,
            n: int,
            k: int,
        ):
            assert activations.dim() == 2
            m, k_ = activations.shape
            assert k_ == k
            return torch.empty(m, n, dtype=activations.dtype, device="meta")

    embedding_op = f"_embedding_{weight_nbit}bit"
    if _op_is_registered(embedding_op):

        @impl(torchao_lib, embedding_op, "Meta")
        def _(
            packed_weight_qvals: Tensor,
            num_embeddings: int,
            embedding_dim: int,
            weight_scales: Tensor,
            weight_zeros: Tensor,
            indices: Tensor,
        ):
            assert indices.dim() == 1
            num_out = indices.shape[0]
            return torch.empty(
                num_out, embedding_dim, dtype=torch.float32, device="meta"
            )

    shared_embedding_op = f"_shared_embedding_{weight_nbit}bit"
    if _op_is_registered(shared_embedding_op):

        @impl(torchao_lib, shared_embedding_op, "Meta")
        def _(
            packed_weights: Tensor,
            group_size: int,
            n: int,
            k: int,
            indices: Tensor,
        ):
            assert indices.dim() == 1
            num_out = indices.shape[0]
            return torch.empty(num_out, k, dtype=torch.float32, device="meta")


for weight_nbit in range(1, 5):
    lut_op = f"_linear_groupwise_{weight_nbit}bit_weight_with_lut"
    if not _op_is_registered(lut_op):
        continue

    @impl(torchao_lib, lut_op, "Meta")
    def _(
        activations: Tensor,
        packed_weights: Tensor,
        scale_group_size: int,
        lut_group_size: int,
        n: int,
        k: int,
    ):
        assert activations.dim() == 2
        m, k_ = activations.shape
        assert k_ == k
        return torch.empty(m, n, dtype=activations.dtype, device="meta")
