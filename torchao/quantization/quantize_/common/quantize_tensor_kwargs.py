# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD 3-Clause license found in the
# LICENSE file in the root directory of this source tree.

import abc
from typing import ClassVar

__all__ = [
    "QuantizeTensorKwargs",
]


class QuantizeTensorKwargs(abc.ABC):
    """This is used to store the kwargs for some tensors

    e.g.

    class Float8Tensor(...)
        @classmethod
        def to_float8(cls, tensor, quant_kwargs: QuantizeTensorKwargs)
            ...
    """

    # Base Version of a config
    VERSION: ClassVar[int] = 1
