# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD 3-Clause license found in the
# LICENSE file in the root directory of this source tree.

from enum import Enum


# can switch to StrEnum (https://docs.python.org/3/library/enum.html#enum.StrEnum)
# after python 3.10 is end of life (https://devguide.python.org/versions/)
class Float8PackingFormat(str, Enum):
    """Packing format for quantized data in Float8 Tensor subclasses in torchao, represents how
    the values in quantized data are packed and laid out in memory.
    """

    """
    plain means the format that quantized Tensor data lays out elements in Tensor sequentially,
    for example, for a Tensor of shape (4, 6):
        a_0_0, a_0_1, ..., a_0_5,
        ...
        a_3_0, a_3_1, ..., a_3_5
    """
    PLAIN = "plain"

    """
    Opaque packing format that's used for tensors that does not have a predefined packing format
    (that may be decided on hardware, tensor shape, library availability etc.) and it's not
    needed for the rest of the system to understand the specific format that's adopted.
    """
    OPAQUE = "opaque"
