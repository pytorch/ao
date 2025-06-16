# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD 3-Clause license found in the
# LICENSE file in the root directory of this source tree.

from typing import Optional, Union

import torch
from torch import Tensor

from torchao.quantization.quant_primitives import (
    _DTYPE_TO_BIT_WIDTH,
    _DTYPE_TO_QVALUE_BOUNDS,
    MappingType,
    ZeroPointDomain,
    _choose_qparams_affine_dont_preserve_zero,
    _choose_qparams_affine_tinygemm,
    _dequantize_affine_no_zero_point,
    _dequantize_affine_tinygemm,
    _quantize_affine_no_zero_point,
    _quantize_affine_tinygemm,
    choose_qparams_affine,
    dequantize_affine,
    quantize_affine,
)

from .quantizer import Quantizer

_BIT_WIDTH_TO_DTYPE = {v: k for k, v in _DTYPE_TO_BIT_WIDTH.items()}


class UnifTorchaoQuantizer(Quantizer):
    """Uniform quantizer that uses torchao's quantization primitives"""

    def __init__(
        self,
        mapping_type: MappingType = MappingType.SYMMETRIC,
        target_dtype: torch.dtype = torch.int8,
        quant_min: Optional[Union[int, float]] = None,
        quant_max: Optional[Union[int, float]] = None,
        eps: Optional[float] = None,
        preserve_zero: bool = True,
        zero_point_domain: ZeroPointDomain = ZeroPointDomain.INT,
    ) -> None:
        super().__init__(center=False)

        self.mapping_type = mapping_type
        self.target_dtype = target_dtype
        self.quant_min = quant_min
        self.quant_max = quant_max
        self.eps = eps

        # defaults: zero_point_domain=ZeroPointDomain.INT, preserve_zero=True
        self._choose_qparams = choose_qparams_affine
        self._quantize = quantize_affine
        self._dequantize = dequantize_affine

        if zero_point_domain == ZeroPointDomain.FLOAT and not preserve_zero:
            self._choose_qparams = _choose_qparams_affine_tinygemm
            self._quantize = _quantize_affine_tinygemm
            self._dequantize = _dequantize_affine_tinygemm
        elif zero_point_domain == ZeroPointDomain.INT and not preserve_zero:
            self._choose_qparams = _choose_qparams_affine_dont_preserve_zero
            self._quantize = quantize_affine
            self._dequantize = dequantize_affine
        elif zero_point_domain == ZeroPointDomain.NONE:
            self._quantize = _quantize_affine_no_zero_point
            self._dequantize = _dequantize_affine_no_zero_point

    def _init_quant_min_max(self, b: int) -> None:
        if self.quant_min is None or self.quant_max is None:
            assert b in _BIT_WIDTH_TO_DTYPE, f"Unsupported bitwidth {b}"
            self.quant_min, self.quant_max = _DTYPE_TO_QVALUE_BOUNDS[
                _BIT_WIDTH_TO_DTYPE[b]
            ]

    def get_quant_size(self, b: int) -> int:
        self._init_quant_min_max(b)
        return self.quant_max - self.quant_min + 1

    def quantize(
        self, p: Tensor, b: int, dim: Optional[int] = None
    ) -> tuple[Tensor, Tensor]:
        self._init_quant_min_max(b)
        if self.eps is None:
            self.eps = torch.finfo(p.dtype).eps

        # assume that p has already been grouped in QuantOptimizer.step
        block_size = (1, p.size(-1)) if dim is not None else p.size()

        s, zero_point = self._choose_qparams(
            p,
            self.mapping_type,
            block_size,
            self.target_dtype,
            eps=self.eps,
            quant_min=self.quant_min,
            quant_max=self.quant_max,
        )
        q_args = (block_size, s, zero_point, self.target_dtype)
        q = self._quantize(
            p,
            *q_args,
            quant_min=self.quant_min,
            quant_max=self.quant_max,
        )
        q = self._dequantize(
            q,
            *q_args,
            output_dtype=p.dtype,
            quant_min=self.quant_min,
            quant_max=self.quant_max,
        )

        Q = torch.arange(
            self.quant_min, self.quant_max + 1, dtype=self.target_dtype, device=p.device
        )
        if dim is not None:
            Q = Q.view(1, -1).expand(q.size(0), -1)
            block_size = (1, Q.size(-1))
        else:
            block_size = Q.shape

        Q = self._dequantize(
            Q,
            block_size,
            *q_args[1:],
            output_dtype=p.dtype,
            quant_min=self.quant_min,
            quant_max=self.quant_max,
        )
        return q, Q


class Int4UnifTorchaoQuantizer(UnifTorchaoQuantizer):
    """Based on torchao.quantization.quant_api._int4_weight_only_transform"""

    def __init__(self) -> None:
        super().__init__(
            mapping_type=MappingType.ASYMMETRIC,
            target_dtype=torch.int32,
            quant_min=0,
            quant_max=15,
            eps=1e-6,
            preserve_zero=False,
            zero_point_domain=ZeroPointDomain.FLOAT,
        )
