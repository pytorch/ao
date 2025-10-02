# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD 3-Clause license found in the
# LICENSE file in the root directory of this source tree.

from enum import Enum


# can switch to StrEnum (https://docs.python.org/3/library/enum.html#enum.StrEnum)
# after python 3.10 is end of life (https://devguide.python.org/versions/)
class IntxChooseQParamsAlgorithm(str, Enum):
    """Variant of quantization algorithm to calculate scale and zero_point"""

    """
    Uses `torchao.quantization.quant_primitives.choose_qparams_affine`
    """
    AFFINE = "affine"

    """
    Uses `torchao.quantization.quant_primitives._choose_qparams_and_quantize_scale_only_hqq`
    """
    HQQ_SCALE_ONLY = "hqq_scale_only"
