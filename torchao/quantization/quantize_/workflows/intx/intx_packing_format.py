# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD 3-Clause license found in the
# LICENSE file in the root directory of this source tree.

from enum import Enum


# can switch to StrEnum (https://docs.python.org/3/library/enum.html#enum.StrEnum)
# after python 3.10 is end of life (https://devguide.python.org/versions/)
class IntxPackingFormat(str, Enum):
    """Packing format for quantized data in Tensor subclasses in torchao, represents how
    the values are packed and laid out in the quantized data.
    """

    """
    Unpacked to int8 means the subbyte quantized data is stored as int8
    """
    UNPACKED_TO_INT8 = "unpacked_to_int8"

    """
    Opaque packing format that's used for tensors that does not have a predefined packing format
    (that may be decided on hardware, tensor shape, library availability etc.) and it's not
    needed for the rest of the system to understand the specific format that's adopted.
    """
    OPAQUE = "opaque"
