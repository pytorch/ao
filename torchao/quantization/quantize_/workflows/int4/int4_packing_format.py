# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD 3-Clause license found in the
# LICENSE file in the root directory of this source tree.

from enum import Enum


# can switch to StrEnum (https://docs.python.org/3/library/enum.html#enum.StrEnum)
# after python 3.10 is end of life (https://devguide.python.org/versions/)
class Int4PackingFormat(str, Enum):
    """Packing format for quantized data in Int4 Tensor subclasses in torchao, represents how
    the values in quantized data are packed and laid out in memory.
    """

    """
    plain means the format that quantized Tensor data lays out elements in Tensor sequentially,
    for example:                                                                                                                                                                                            for a Tensor of shape (4, 6):
    a_0_0, a_0_1, ..., a_0_5,
    ...
    a_3_0, a_3_1, ..., a_3_5

    For example for int4, we will
    pack two adjacent int4 elements into one uint8/int8 value for plain packing format
    """
    PLAIN = "plain"

    """
    preshuffled is referring to the preshuffled format used by fbgemm kernels
    """
    PRESHUFFLED = "preshuffled"

    """
    marlin_sparse is referring to the format used by marlin kernels, requires symmetric quantization
    """
    MARLIN_SPARSE = "marlin_sparse"

    """
    CHANGE THIS
    """
    MARLIN_QQQ = "marlin_qqq"

    """
    plain_int32 is a format that 2 adjacent int4 values are packed in a byte and 4 such packed bytes are stored in a int32 value.
    """
    PLAIN_INT32 = "plain_int32"

    """
    tile_packed_to_4d is referring to the format used by tinygemm kernels for int4 quantization
    for a Tensor of shape (n, k), the packed weight will have dimension:
    [n / 8][k / (inner_k_tiles * 16)][32][inner_k_tiles / 2], where inner_k_tiles is 8 currently
    for simplication of Int4TilePackedTo4dTensor API
    """
    TILE_PACKED_TO_4D = "tile_packed_to_4d"

    """
    Opaque packing format that's used for tensors that does not have a predefined packing format
    (that may be decided on hardware, tensor shape, library availability etc.) and it's not
    needed for the rest of the system to understand the specific format that's adopted.
    """
    OPAQUE = "opaque"
