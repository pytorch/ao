# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD 3-Clause license found in the
# LICENSE file in the root directory of this source tree.

from enum import Enum


# can switch to StrEnum (https://docs.python.org/3/library/enum.html#enum.StrEnum)
# after python 3.10 is end of life (https://devguide.python.org/versions/)
class Int4ChooseQParamsAlgorithm(str, Enum):
    """Variant of quantization algorithm to calculate scale and zero_point"""

    """
    The choose qparams algorithm native for tinygemm kernel:
    scale = (max_val - min_val) / float(quant_max - quant_min), where
        max_val and min_val are the max/min for the slice of input Tensor based on block_size
        quant_max and quant_min and max/min for the quantized value, e.g. 0, 15 for uint4
    zero_point = min_val + scale * mid_point, where
        mid_point = (quant_max + quant_min + 1) / 2

    implemented in `torchao.quantization.quant_primitives._choose_qparams_affine_tinygemm
    """
    TINYGEMM = "tinygemm"

    """
    The choose qparams based on half-quadratic quantization: https://mobiusml.github.io/hqq_blog/

    implemented in `torchao.quantization.quant_primitives._choose_qparams_and_quantize_affine_hqq`
    """
    HQQ = "hqq"
