# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD 3-Clause license found in the
# LICENSE file in the root directory of this source tree.

from typing import Optional, Union

import torch
from torch import Tensor

from torchao.quantization.quant_primitives import (
    _DTYPE_TO_QVALUE_BOUNDS,
    MappingType,
    ZeroPointDomain,
    choose_qparams_affine,
    dequantize_affine,
    quantize_affine,
)

from .quantizer import Quantizer


class UnifTorchaoQuantizer(Quantizer):
    """Uniform quantizer that uses torchao's quantization primitives"""
    def __init__(
        self,
        symmetric: bool,
        target_dtype: torch.dtype,
        quant_min: Optional[Union[int, float]] = None,
        quant_max: Optional[Union[int, float]] = None,
        eps: Optional[float] = None,
        preserve_zero: bool = True,
        zero_point_domain: ZeroPointDomain = ZeroPointDomain.FLOAT,
    ) -> None:
        super().__init__(center=False)

        self.mapping_type = (
            MappingType.SYMMETRIC if symmetric else MappingType.ASYMMETRIC
        )
        self.target_dtype = target_dtype
        self.quant_min = quant_min
        self.quant_max = quant_max
        self.eps = eps
        self.preserve_zero = preserve_zero
        self.zero_point_domain = zero_point_domain

    @property
    def q_kwargs(self) -> dict[str, Union[int, float]]:
        return {
            "quant_min": self.quant_min,
            "quant_max": self.quant_max,
            "zero_point_domain": self.zero_point_domain,
        }

    def get_quant_size(self, b: int) -> int:
        return 2 ** (b - 1) + 1 if self.mapping_type == MappingType.SYMMETRIC else 2**b

    def quantize(
        self, p: Tensor, b: int, dim: Optional[int] = None
    ) -> tuple[Tensor, Tensor]:
        if self.quant_min is None or self.quant_max is None:
            self.quant_min, self.quant_max = _DTYPE_TO_QVALUE_BOUNDS[p.dtype]

        if self.eps is None:
            self.eps = torch.finfo(p.dtype).eps

        # assume that p has already been grouped in QuantOptimizer.step
        block_size = (1, p.size(-1)) if dim is not None else p.size()
        s, zero_point = choose_qparams_affine(
            p,
            self.mapping_type,
            block_size,
            self.target_dtype,
            quant_min=self.quant_min,
            quant_max=self.quant_max,
            eps=self.eps,
            preserve_zero=self.preserve_zero,
            zero_point_domain=self.zero_point_domain,
        )
        q_args = (block_size, s, zero_point, self.target_dtype)
        q = quantize_affine(p, *q_args, **self.q_kwargs)
        q = dequantize_affine(q, *q_args, **self.q_kwargs, output_dtype=p.dtype)

        Q = torch.arange(
            self.quant_min, self.quant_max + 1, dtype=self.target_dtype, device=p.device
        )
        if dim is not None:
            Q = Q.unsqueeze(0).mul(s.unsqueeze(dim))
        else:
            Q.mul_(s)
        return q, Q
