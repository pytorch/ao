# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD 3-Clause license found in the
# LICENSE file in the root directory of this source tree.


from enum import Enum

import torch

__all__ = [
    "Float8TensorPackingFormat",
]


# can switch to StrEnum (https://docs.python.org/3/library/enum.html#enum.StrEnum)
# after python 3.10 is end of life (https://devguide.python.org/versions/)
class Float8TensorPackingFormat(str, Enum):
    """
    Sparse packing formats for 2:4 sparsity + FP8 quantization

    SPARSE_CUTLASS will pack the quantized_data into two tensors, qdata and sparse_metadata, for the specified values and metadata respectively.
    This packing format will dispatch to `rowwise_scaled_linear_sparse_cutlass_f8f8`, which will fuse the per-row scaling into the sparse matmul.
    """

    SPARSE_CUTLASS = "sparse_cutlass"


torch.serialization.add_safe_globals([Float8TensorPackingFormat])
