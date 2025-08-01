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
    NVFP4FakeQuantizeConfig,
)
from .utils import (
    _fake_quantize_per_channel_group,
    _fake_quantize_per_token,
    _Float8RowwiseFakeQuantize,
)


class FakeQuantizer(torch.nn.Module):
    """
    Generic module for applying fake quantization to a tensor, as specified in the config.
    """

    def __init__(self, config: FakeQuantizeConfigBase):
        super().__init__()
        self.config = config
        self.enabled = True

        if isinstance(self.config, IntxFakeQuantizeConfig):
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

        if isinstance(self.config, NVFP4FakeQuantizeConfig):
            return self._nvfp4_forward(x)
        elif isinstance(self.config, IntxFakeQuantizeConfig):
            return self._intx_forward(x)
        else:
            raise ValueError(f"Unexpected config type {self.config}")

    def _nvfp4_forward(self, x: torch.Tensor):
        """
        Apply NVFP4 fake quantization to the tensor following `NVFP4Tensor`.
        """
        from torchao.prototype.mx_formats.nvfp4_tensor import (
            _nvfp4_quantize,
            per_tensor_amax_to_scale,
        )

        block_size = 16
        if self.config.use_per_tensor_scale:
            tensor_amax = torch.max(torch.abs(x))
            per_tensor_scale = per_tensor_amax_to_scale(tensor_amax)
        else:
            per_tensor_scale = None
        scale, q = _nvfp4_quantize(
            x,
            block_size=block_size,
            per_tensor_scale=per_tensor_scale,
            skip_dtype_cast_and_packing=True,
        )
        assert q.dtype == x.dtype
        assert scale.dtype == torch.float32
        M, K = q.shape[0], q.shape[1]
        q = q.view(M, K // block_size, block_size)
        scale = scale.view(M, K // block_size, 1)
        dq = q * scale
        return dq.view(x.shape)

    def _intx_forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply intx fake quantization to the tensor.
        """
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
            return self._intx_per_token_forward(x)
        elif isinstance(self.config.granularity, (PerAxis, PerGroup)):
            return self._intx_per_channel_or_group_forward(x)
        else:
            raise ValueError("Unknown granularity '%s'" % self.config.granularity)

    def _intx_per_token_forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Perform intx per token fake quantization on the tensor.
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

    def _intx_per_channel_or_group_forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Perform intx per channel or per group fake quantization on the tensor.
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

    def __repr__(self) -> str:
        """
        Return a human readable representation of this `FakeQuantizer` with config details.
        """
        return "FakeQuantizer(%s)" % self.config


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
