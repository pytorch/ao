# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from typing import Optional

import torch

from torchao.quantization.granularity import (
    PerAxis,
    PerGroup,
    PerToken,
)
from torchao.quantization.quant_primitives import (
    _DTYPE_TO_BIT_WIDTH,
    _DTYPE_TO_QVALUE_BOUNDS,
)
from torchao.quantization.utils import (
    get_group_qparams_symmetric,
    get_groupwise_affine_qparams,
)

from .api import (
    FakeQuantizeConfig,
)
from .utils import (
    _choose_qparams_per_token_asymmetric,
    _fake_quantize_per_channel_group,
    _fake_quantize_per_token,
)


class FakeQuantizer(torch.nn.Module):
    """
    Generic module for applying fake quantization to a tensor, as specified in the config.
    """

    def __init__(self, config: FakeQuantizeConfig):
        super().__init__()
        self.config = config
        self.enabled = True
        self.scale: Optional[torch.Tensor] = None
        self.zero_point: Optional[torch.Tensor] = None

        # TODO: support range learinng
        if self.config.range_learning:
            raise NotImplementedError("Range learning is not supported yet")

    def forward(self, x: torch.Tensor):
        """
        Apply fake quantization to the tensor based on the bit-width,
        granularity, symmetry, and other properties specified in the config.
        """
        if not self.enabled:
            return x

        if isinstance(self.config.granularity, PerToken):
            return self._per_token_forward(x)
        elif isinstance(self.config.granularity, (PerAxis, PerGroup)):
            return self._per_channel_or_group_forward(x)
        else:
            raise ValueError("Unknown granularity '%s'" % self.config.granularity)

    def _per_token_forward(self, x: torch.Tensor):
        """
        Perform per token fake quantization on the tensor.
        """
        if self.config.is_symmetric:
            raise NotImplementedError("Symmetric per token is not supported yet")
        if self._should_compute_qparams():
            (self.scale, self.zero_point) = _choose_qparams_per_token_asymmetric(
                x,
                self.config.scale_precision,
                self.config.zero_point_precision,
            )
        qmin, qmax = _DTYPE_TO_QVALUE_BOUNDS[self.config.dtype]
        return _fake_quantize_per_token(x, self.scale, self.zero_point, qmin, qmax)

    def _per_channel_or_group_forward(self, x: torch.Tensor):
        """
        Perform per channel or per group fake quantization on the tensor.
        We express per channel using per group where the group size is the size
        of the last dimension of the tensor.
        """
        granularity = self.config.granularity
        scale_precision = self.config.scale_precision
        zero_point_precision = self.config.zero_point_precision
        zero_point_domain = self.config.zero_point_domain
        is_symmetric = self.config.is_symmetric

        # get group size
        if isinstance(granularity, PerAxis):
            assert granularity.axis == 0
            group_size = x.size()[-1]
        elif isinstance(granularity, PerGroup):
            group_size = granularity.group_size
        else:
            raise ValueError("Unexpected granularity '%s'" % granularity)

        # get scales and zero points
        if self._should_compute_qparams():
            bit_width = _DTYPE_TO_BIT_WIDTH[self.config.dtype]
            if is_symmetric:
                (self.scale, self.zero_point) = get_group_qparams_symmetric(
                    x,
                    bit_width,
                    group_size,
                    scale_precision,
                )
            else:
                (self.scale, self.zero_point) = get_groupwise_affine_qparams(
                    x,
                    bit_width,
                    group_size,
                    scale_precision,
                )
            self.zero_point = self.zero_point.to(zero_point_precision)

        qmin, qmax = _DTYPE_TO_QVALUE_BOUNDS[self.config.dtype]
        return _fake_quantize_per_channel_group(
            x,
            self.scale,
            self.zero_point,
            qmin,
            qmax,
            group_size,
            zero_point_domain,
        )

    def _should_compute_qparams(self) -> bool:
        """
        Return whether we need to compute new scales and zero points.
        """
        return self.config.is_dynamic or self.scale is None or self.zero_point is None

    def __repr__(self) -> str:
        """
        Return a human readable representation of this `FakeQuantizer` with config details.
        """
        return "FakeQuantizer(%s)" % self.config
