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
    MappingType,
    _Round,
    choose_qparams_affine,
)
from torchao.quantization.utils import (
    _get_per_token_block_size,
    get_group_qparams_symmetric,
    get_groupwise_affine_qparams,
)

from .fake_quantize_config import (
    FakeQuantizeConfigBase,
    IntxFakeQuantizeConfig,
)
from .utils import (
    _fake_quantize_per_channel_group,
    _fake_quantize_per_token,
    _Float8RowwiseFakeQuantize,
    _log_deprecation_warning,
)


class FakeQuantizerBase(torch.nn.Module):
    """
    Generic module for applying fake quantization to a tensor, as specified in the config.
    """

    config: FakeQuantizeConfigBase

    def __repr__(self) -> str:
        """
        Return a human readable representation of this `FakeQuantizer` with config details.
        """
        return "FakeQuantizer(%s)" % self.config

    @staticmethod
    def from_config(config: FakeQuantizeConfigBase) -> "FakeQuantizerBase":
        if isinstance(config, IntxFakeQuantizeConfig):
            return IntxFakeQuantizer(config)
        else:
            raise ValueError(f"Unknown config type: {config}")


class IntxFakeQuantizer(FakeQuantizerBase):
    """
    Generic module for applying integer fake quantization to a tensor, as specified in the config.
    """

    def __init__(self, config: IntxFakeQuantizeConfig):
        super().__init__()
        self.config = config
        self.enabled = True
        self.scale: Optional[torch.Tensor] = None
        self.zero_point: Optional[torch.Tensor] = None

        # For range learning only
        # TODO: make this configurable?
        self._scale_eps = 1e-9
        self._initialized = False

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply fake quantization to the tensor based on the bit-width,
        granularity, symmetry, and other properties specified in the config.
        """
        if not self.enabled:
            return x

        if (
            self.config.range_learning
            and not self._initialized
            and (self.scale is None or self.zero_point is None)
        ):
            raise ValueError(
                "Scales and zero points must be initialized for range learning. "
                "Please call `torchao.quantization.qat.initialize_fake_quantizers` "
                "before initializing the optimizer and beginning training."
            )

        if isinstance(self.config.granularity, PerToken):
            return self._per_token_forward(x)
        elif isinstance(self.config.granularity, (PerAxis, PerGroup)):
            return self._per_channel_or_group_forward(x)
        else:
            raise ValueError("Unknown granularity '%s'" % self.config.granularity)

    def _per_token_forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Perform per token fake quantization on the tensor.
        """
        if self.config.is_symmetric:
            raise NotImplementedError("Symmetric per token is not supported yet")
        qmin, qmax = _DTYPE_TO_QVALUE_BOUNDS[self.config.dtype]
        if self._should_compute_qparams():
            self.scale, self.zero_point = choose_qparams_affine(
                x,
                mapping_type=MappingType.ASYMMETRIC,
                block_size=_get_per_token_block_size(x),
                target_dtype=self.config.dtype,
                quant_min=qmin,
                quant_max=qmax,
                eps=self.config.eps,
                scale_dtype=self.config.scale_precision,
                zero_point_dtype=self.config.zero_point_precision,
            )
            self._maybe_update_qparams_for_range_learning()
        return _fake_quantize_per_token(x, self.scale, self.zero_point, qmin, qmax)

    def _per_channel_or_group_forward(self, x: torch.Tensor) -> torch.Tensor:
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
        # TODO: refactor this to use `choose_qparams_affine`
        if self._should_compute_qparams():
            bit_width = _DTYPE_TO_BIT_WIDTH[self.config.dtype]
            if is_symmetric:
                (self.scale, self.zero_point) = get_group_qparams_symmetric(
                    x,
                    bit_width,
                    group_size,
                    scale_precision,
                    eps=self.config.eps,
                )
            else:
                (self.scale, self.zero_point) = get_groupwise_affine_qparams(
                    x,
                    bit_width,
                    group_size,
                    scale_precision,
                    eps=self.config.eps,
                )
            self.zero_point = self.zero_point.to(zero_point_precision)
            self._maybe_update_qparams_for_range_learning()

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

    def _maybe_update_qparams_for_range_learning(self) -> None:
        """
        If range learning is enabled, turn scales and zero points into trainable parameters.
        This function is idempotent and should only be called once.
        """
        if (
            not self.config.range_learning
            or isinstance(self.scale, torch.nn.Parameter)
            or isinstance(self.zero_point, torch.nn.Parameter)
        ):
            return
        scale, zero_point = self.scale, self.zero_point
        qmin, qmax = _DTYPE_TO_QVALUE_BOUNDS[self.config.dtype]
        # Stabilize range learning
        scale = torch.clamp(scale, min=self._scale_eps)
        zero_point = _Round.apply(zero_point)
        zero_point = torch.clamp(zero_point, qmin, qmax)
        self.scale = torch.nn.Parameter(scale, requires_grad=True)
        self.zero_point = torch.nn.Parameter(zero_point, requires_grad=True)


# For BC
class FakeQuantizer(IntxFakeQuantizer):
    """
    (Deprecated) Please use :class:`~torchao.quantization.qat.IntxFakeQuantizer` instead.
    """

    def __init__(self, config: FakeQuantizeConfigBase):
        super().__init__(config)
        _log_deprecation_warning(self)


# TODO: make this a FakeQuantizerBase
class _Float8RowwiseActivationFakeQuantizer(torch.nn.Module):
    """
    Simple fake quantizer for float8 rowwise fake quantization, intended for activations only.
    """

    def __init__(self):
        super().__init__()
        self.enabled = True

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.enabled:
            return _Float8RowwiseFakeQuantize.apply(
                x,
                torch.float8_e4m3fn,
                -1,
            )
        else:
            return x
