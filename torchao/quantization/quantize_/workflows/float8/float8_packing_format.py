# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD 3-Clause license found in the
# LICENSE file in the root directory of this source tree.


from enum import Enum

import torch

__all__ = [
    "Float8PackingFormat",
]


# can switch to StrEnum (https://docs.python.org/3/library/enum.html#enum.StrEnum)
# after python 3.10 is end of life (https://devguide.python.org/versions/)
class Float8PackingFormat(str, Enum):
    """
    plain packing format for Float8Tensor will lay out elements in Tensor sequentially,
    for example:                                                                                                                                                                                          for a Tensor of shape (4, 6):
    a_0_0, a_0_1, ..., a_0_5,
    ...
    a_3_0, a_3_1, ..., a_3_5
    """

    PLAIN = "plain"
    """
    Sparse packing format for 2:4 sparsity + FP8 quantization

    SPARSE_2D_DATA_2D_METADATA will pack the quantized_data into two tensors, qdata and sparse_metadata, for the specified values and metadata respectively.
    This packing format will dispatch to `rowwise_scaled_linear_sparse_cutlass_f8f8`, which will fuse the per-row scaling into the sparse matmul.
    """
    SPARSE_2D_DATA_2D_METADATA = "sparse_2d_data_2d_metadata"
    """
    Sparse packing format for 2:4 sparsity + FP8 quantization using hipSPARSELt (ROCm/AMD only).

    SPARSE_1D_DATA_1D_METADATA will pack the quantized_data into a single tensor containing both the quantized data and metadata
    as a 1D tensor of r*c/2 + r*c/8 bytes with the following layout: [compressed_data | metadata]

    - compressed_data: r*c/2 bytes
        The 2 non-zero FP8 values per group of 4 elements, stored row-major:
        row0_group0_val0, row0_group0_val1, row0_group1_val0, row0_group1_val1, ..., row1_group0_val0, ...
    - metadata: r*c/8 bytes
        4 bits per group of 4 elements encoding the positions of the 2 kept values
        (2 bits per kept element index), groups packed contiguously row-major:
        row0_group0_meta, row0_group1_meta, ..., row1_group0_meta, ...

    This packing format will dispatch to torch._cslt_sparse_mm for matmul, with per-tensor scaling passed as alpha.
    """
    SPARSE_1D_DATA_1D_METADATA = "sparse_1d_data_1d_metadata"


Float8PackingFormat.SPARSE_CUTLASS = Float8PackingFormat.SPARSE_2D_DATA_2D_METADATA

torch.serialization.add_safe_globals([Float8PackingFormat])
