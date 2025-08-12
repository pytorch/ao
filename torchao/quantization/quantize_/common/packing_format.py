# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD 3-Clause license found in the
# LICENSE file in the root directory of this source tree.

from enum import Enum


# can switch to StrEnum (https://docs.python.org/3/library/enum.html#enum.StrEnum)
# after python 3.10 is end of life (https://devguide.python.org/versions/)
class PackingFormat(str, Enum):
    """Packing format for quantized data in Tensor subclasses in torchao, represents how
    the values are packed and laid out in the quantized data.
    """

    """
    plain means the format that quantized Tensor data lays out elements in Tensor sequentially,
    for example:                                                                                                                                                                                                          for a Tensor of shape (4, 6):
    a_0_0, a_0_1, ..., a_0_5,
    ...
    a_3_0, a_3_1, ..., a_3_5

    Note that it's different for different dtypes, for example for int4, we will
    pack two adjacent int4 elements into one uint8/int8 value for plain packing format
    """
    PLAIN = "plain"

    """
    preshuffled is referring to the preshuffled format used by fbgemm kernels
    """
    PRESHUFFLED = "preshuffled"

    """
    marlin_sparse is referring to the format used by marlin kernels, only supports symmetric quantization
    """
    MARLIN_SPARSE = "marlin_sparse"

    """
    Unpacked means the subbyte quantized data is stored as int8
    """
    UNPACKED_TO_INT8 = "unpacked_to_int8"

    """
    Tiled means the tiling used by CPU kernels
    """
    TILED = "tiled"
