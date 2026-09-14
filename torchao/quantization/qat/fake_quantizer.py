# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import warnings
from typing import Optional

import torch

from torchao.quantization.granularity import (
    PerAxis,
    PerGroup,
    PerRow,
    PerTensor,
    PerToken,
)
from torchao.quantization.quant_primitives import (
    _DTYPE_TO_BIT_WIDTH,
    _DTYPE_TO_QVALUE_BOUNDS,
    _get_reduction_params,
    MappingType,
    _choose_scale_float8,
    _dequantize_affine_float8,
    _fake_quantize_affine,
    _quantize_affine_float8,
    _Round,
    choose_qparams_affine,
    choose_qparams_affine_with_min_max,
)
from torchao.quantization.utils import (
    _get_per_token_block_size,
    get_block_size,
    get_group_qparams_symmetric,
    get_groupwise_affine_qparams,
)

from .fake_quantize_config import (
    FakeQuantizeConfigBase,
    Float8FakeQuantizeConfig,
    Int4WeightFakeQuantizeConfig,
    IntxFakeQuantizeConfig,
)
from .utils import (
    _fake_quantize_per_channel_group,
    _fake_quantize_per_token,
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
        elif isinstance(config, Int4WeightFakeQuantizeConfig):
            return Int4WeightFakeQuantizer(config)
        elif isinstance(config, Float8FakeQuantizeConfig):
            return Float8FakeQuantizer(config)
        else:
            raise ValueError(f"Unknown config type: {config}")


class Float8FakeQuantizer(FakeQuantizerBase):
    """
    Generic module for applying float8 fake quantization to a tensor, as specified in the config.
    """

    def __init__(self, config: Float8FakeQuantizeConfig):
        super().__init__()
        self.config = config
        self.enabled = True
        torch._C._log_api_usage_once("torchao.quantization.qat.Float8FakeQuantizer")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not self.enabled:
            return x
        original_dtype = x.dtype
        block_size = get_block_size(x.shape, self.config.granularity)
        scale = _choose_scale_float8(
            x,
            block_size,
            self.config.dtype,
            hp_value_lb=self.config.hp_value_lb,
            hp_value_ub=self.config.hp_value_ub,
        )
        q = _quantize_affine_float8(x, scale, self.config.dtype)
        dq = _dequantize_affine_float8(q, scale, original_dtype)
        return dq


class Int4WeightFakeQuantizer(FakeQuantizerBase):
    """
    Generic module for applying int4 fake quantization to a weight tensor,
    targeting the following MSLK kernels:
        torch.ops.mslk.f8i4bf16_shuffled
        torch.ops.mslk.bf16i4bf16_shuffled
        torch.ops.mslk.bf16i4bf16_rowwise
    """

    def __init__(self, config: Int4WeightFakeQuantizeConfig):
        super().__init__()
        self.config = config
        self.enabled = True
        torch._C._log_api_usage_once("torchao.quantization.qat.Int4WeightFakeQuantizer")

    def forward(self, w: torch.Tensor) -> torch.Tensor:
        if not self.enabled:
            return w
        if self.config.activation_dtype == torch.float8_e4m3fn:
            return self._fp8_activations_forward(w)
        elif self.config.activation_dtype == torch.bfloat16:
            return self._bf16_activations_forward(w)
        else:
            raise ValueError(f"Unknown activation dtype {self.config.activation_dtype}")

    def _fp8_activations_forward(self, w: torch.Tensor) -> torch.Tensor:
        """
        Apply int4 fake quantization to the weight tensor where the input activations
        are expected to be rowwise fp8, using the following as a reference:
        https://github.com/meta-pytorch/MSLK/blob/main/mslk/quantize/shuffle.py
        """
        assert w.dim() == 2
        assert self.config.activation_dtype == torch.float8_e4m3fn

        # First quantize weights to fp8 per row
        # This simulates the numerics of mslk.quantize.triton.fp8_quantize.quantize_fp8_row
        per_row_block_size = get_block_size(w.shape, PerRow())
        fp8_scale = _choose_scale_float8(
            w,
            per_row_block_size,
            torch.float8_e4m3fn,
            hp_value_lb=1e-12,
        )
        w_fp8 = _quantize_affine_float8(w, fp8_scale, torch.float8_e4m3fn)
        w_fp8 = _dequantize_affine_float8(w_fp8, fp8_scale, w.dtype)

        # Now quantize to int4 per group
        # This simulates the numerics of mslk.quantize.shuffle.int4_row_quantize
        eps = 1e-6
        mslk_scale_quant_max = 8
        w_fp8_grouped = w_fp8.view(w_fp8.shape[0], -1, self.config.group_size)
        max_abs = torch.amax(torch.abs(w_fp8_grouped), dim=-1, keepdim=False)
        scale = torch.clamp(max_abs / mslk_scale_quant_max, min=eps)
        zero_point = torch.zeros_like(scale)
        per_group_block_size = (1, self.config.group_size)
        fq = _fake_quantize_affine(
            w_fp8,
            per_group_block_size,
            scale,
            zero_point,
            quant_dtype=torch.int8,
            quant_min=-8,
            quant_max=7,
        )
        return fq.to(w.dtype)

    def _bf16_activations_forward(self, w: torch.Tensor) -> torch.Tensor:
        """
        Apply int4 fake quantization to the weight tensor where the input activations
        are expected to be bf16, using the following as a reference:
        https://github.com/meta-pytorch/MSLK/blob/main/mslk/quantize/shuffle.py
        """
        assert w.dim() == 2
        assert self.config.activation_dtype == torch.bfloat16

        eps = 1e-6
        qmin, qmax = 0, 15
        mslk_symmetric_qmax = 8
        w_grouped = w.to(torch.float32).view(w.shape[0], -1, self.config.group_size)
        max_val = torch.amax(w_grouped, dim=-1, keepdim=True)
        min_val = torch.amin(w_grouped, dim=-1, keepdim=True)
        scale = torch.clamp(max_val - min_val, min=eps) / qmax
        zero_point = min_val + scale * mslk_symmetric_qmax
        fq = _Round.apply((w_grouped - min_val) / scale).clamp(qmin, qmax)
        fq = fq - mslk_symmetric_qmax
        fq = fq * scale + zero_point
        return fq.view(w.shape).to(w.dtype)


class IntxFakeQuantizer(FakeQuantizerBase):
    """
    Generic module for applying integer fake quantization to a tensor, as specified in the config.
    """

    def __init__(self, config: IntxFakeQuantizeConfig):
        super().__init__()
        torch._C._log_api_usage_once("torchao.quantization.qat.IntxFakeQuantizer")
        self.config = config
        # `enabled` controls fake quantization, while `observer_enabled` controls
        # calibration range collection. The two controls are independent:
        #   True / True: observe, then fake quantize
        #   True / False: observe, then return the original input
        #   False / True: fake quantize without observing
        #   False / False: return the original input without observing
        # The order above is observer_enabled / enabled.
        self.enabled = True
        self.observer_enabled = False
        self._enabled_before_calibration: Optional[bool] = None
        self.scale: Optional[torch.Tensor]
        self.zero_point: Optional[torch.Tensor]
        if config.is_dynamic or config.range_learning:
            self.scale = None
            self.zero_point = None
        else:
            self.register_buffer("scale", torch.empty(0, dtype=config.scale_precision))
            self.register_buffer(
                "zero_point", torch.empty(0, dtype=config.zero_point_precision)
            )
        # Calibration ranges are temporary. Finalized scale and zero point are
        # the only calibration results stored in the state dictionary.
        self.min_val: Optional[torch.Tensor] = None
        self.max_val: Optional[torch.Tensor] = None

        # For range learning only
        # TODO: make this configurable?
        self._scale_eps = 1e-9
        self._initialized = False

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply fake quantization to the tensor based on the bit-width,
        granularity, symmetry, and other properties specified in the config.
        """
        if self.observer_enabled:
            # Observe before the early return so calibration can bypass QDQ.
            self._update_calibration_ranges(x)
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
        elif isinstance(self.config.granularity, PerTensor):
            return self._per_tensor_forward(x)
        else:
            raise ValueError("Unknown granularity '%s'" % self.config.granularity)

    def enable_calibration(self) -> None:
        """Reset ranges, enable observation, and disable fake quantization."""
        self._validate_calibration_config()
        if not self.observer_enabled or self.enabled:
            self._enabled_before_calibration = self.enabled
        self.min_val = None
        self.max_val = None
        self.observer_enabled = True
        self.enabled = False

    def finalize_calibration(self) -> None:
        """Compute qparams from observed ranges and exit calibration mode.

        This method disables observation and restores the fake-quantization state
        that was active before :meth:`enable_calibration`.
        """
        self._validate_calibration_config()
        if (
            not self.observer_enabled
            or self.enabled
            or self._enabled_before_calibration is None
        ):
            raise ValueError(
                "Calibration must be enabled before calling finalize_calibration"
            )
        if self.min_val is None or self.max_val is None:
            self._restore_fake_quantization_state()
            raise ValueError("No calibration data was collected")
        qmin, qmax = self.config.quant_min, self.config.quant_max
        try:
            scale, zero_point = choose_qparams_affine_with_min_max(
                self.min_val,
                self.max_val,
                self.config.mapping_type,
                (),
                self.config.dtype,
                qmin,
                qmax,
                self.config.eps,
                self.config.scale_precision,
                self.config.zero_point_precision,
            )
        except Exception:
            self._restore_fake_quantization_state()
            raise
        self.scale = scale
        self.zero_point = zero_point
        self._restore_fake_quantization_state()

    def get_running_min_max(self) -> tuple[torch.Tensor, torch.Tensor]:
        """Return copies of the collected calibration range."""
        self._validate_calibration_config()
        if self.min_val is None or self.max_val is None:
            raise ValueError("No calibration data was collected")
        return self.min_val.clone(), self.max_val.clone()

    def set_running_min_max(self, min_val: torch.Tensor, max_val: torch.Tensor) -> None:
        """Replace the collected calibration range."""
        self._validate_calibration_config()
        if self.min_val is None or self.max_val is None:
            raise ValueError("No calibration data was collected")
        if min_val.shape != self.min_val.shape or max_val.shape != self.max_val.shape:
            raise ValueError("Calibration range shapes must match collected ranges")
        requires_conversion = (
            min_val.device != self.min_val.device
            or max_val.device != self.max_val.device
            or min_val.dtype != self.min_val.dtype
            or max_val.dtype != self.max_val.dtype
        )
        min_val = min_val.detach().to(self.min_val)
        max_val = max_val.detach().to(self.max_val)
        if not torch.isfinite(min_val).all().item() or not torch.isfinite(
            max_val
        ).all().item():
            raise ValueError("Calibration ranges must be finite")
        if torch.any(min_val > max_val).item():
            raise ValueError("Calibration minimum must not exceed the maximum")
        if requires_conversion:
            warnings.warn(
                "Converting calibration ranges to match the collected ranges",
                stacklevel=2,
            )
        self.min_val = min_val
        self.max_val = max_val

    def _validate_calibration_config(self) -> None:
        if (
            self.config.is_dynamic
            or self.config.range_learning
            or isinstance(self.config.granularity, PerToken)
        ):
            raise ValueError(
                "Calibration is only supported for static per-tensor, per-axis, "
                "or per-group quantization"
            )

    def _restore_fake_quantization_state(self) -> None:
        assert self._enabled_before_calibration is not None
        self.observer_enabled = False
        self.enabled = self._enabled_before_calibration
        self._enabled_before_calibration = None

    def _update_calibration_ranges(self, x: torch.Tensor) -> None:
        if x.numel() == 0:
            return
        x = x.detach().to(dtype=self.config.scale_precision)
        if isinstance(self.config.granularity, PerTensor):
            min_val, max_val = torch.aminmax(x)
        else:
            block_size = get_block_size(x.shape, self.config.granularity)
            shape_for_reduction, reduction_dims = _get_reduction_params(
                block_size, x.size()
            )
            x = x.view(shape_for_reduction)
            min_val = torch.amin(x, dim=reduction_dims)
            max_val = torch.amax(x, dim=reduction_dims)
        if self.min_val is None or self.max_val is None:
            self.min_val = min_val
            self.max_val = max_val
        else:
            if self.min_val.shape != min_val.shape or self.max_val.shape != max_val.shape:
                raise ValueError(
                    "Calibration range shape changed between calibration inputs"
                )
            self.min_val = torch.minimum(self.min_val, min_val)
            self.max_val = torch.maximum(self.max_val, max_val)

    def _per_token_forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Perform per token fake quantization on the tensor.
        """
        if self.config.is_symmetric:
            raise NotImplementedError("Symmetric per token is not supported yet")
        qmin, qmax = self.config.quant_min, self.config.quant_max
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
                (self.scale, self.zero_point) = self._choose_group_qparams_symmetric(
                    x, bit_width, group_size
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

        qmin, qmax = self.config.quant_min, self.config.quant_max
        return _fake_quantize_per_channel_group(
            x,
            self.scale,
            self.zero_point,
            qmin,
            qmax,
            group_size,
            zero_point_domain,
        )

    def _choose_group_qparams_symmetric(
        self,
        x: torch.Tensor,
        bit_width: int,
        group_size: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if (
            self.config.is_dynamic
            and bit_width == 4
            and self.config.mapping_type == MappingType.SYMMETRIC_NO_CLIPPING_ERR
        ):
            x_grouped = x.reshape(x.shape[0], -1, group_size)
            min_val = torch.amin(x_grouped, dim=-1)
            max_val = torch.amax(x_grouped, dim=-1)
            qmin, qmax = self.config.quant_min, self.config.quant_max
            return choose_qparams_affine_with_min_max(
                min_val,
                max_val,
                self.config.mapping_type,
                (1, group_size),
                self.config.dtype,
                qmin,
                qmax,
                self.config.eps,
                self.config.scale_precision,
                self.config.zero_point_precision,
            )

        return get_group_qparams_symmetric(
            x,
            bit_width,
            group_size,
            self.config.scale_precision,
            mapping_type=self.config.mapping_type,
            eps=self.config.eps,
        )

    def _per_tensor_forward(self, x: torch.Tensor) -> torch.Tensor:
        """Perform per-tensor fake quantization."""
        qmin, qmax = self.config.quant_min, self.config.quant_max
        block_size = get_block_size(x.shape, self.config.granularity)
        if self._should_compute_qparams():
            self.scale, self.zero_point = choose_qparams_affine(
                x,
                mapping_type=self.config.mapping_type,
                block_size=block_size,
                target_dtype=self.config.dtype,
                quant_min=qmin,
                quant_max=qmax,
                eps=self.config.eps,
                scale_dtype=self.config.scale_precision,
                zero_point_dtype=self.config.zero_point_precision,
            )
            self._maybe_update_qparams_for_range_learning()
        return _fake_quantize_affine(
            x,
            block_size,
            self.scale,
            self.zero_point,
            self.config.dtype,
            qmin,
            qmax,
            self.config.zero_point_domain,
        )

    def _should_compute_qparams(self) -> bool:
        """
        Return whether we need to compute new scales and zero points.
        """
        return (
            self.config.is_dynamic
            or self.scale is None
            or self.zero_point is None
            or self.scale.numel() == 0
            or self.zero_point.numel() == 0
        )

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
        qmin, qmax = self.config.quant_min, self.config.quant_max
        # Stabilize range learning
        scale = torch.clamp(scale, min=self._scale_eps)
        self.scale = torch.nn.Parameter(scale, requires_grad=True)
        if self.config.is_symmetric:
            self.zero_point.zero_()
        else:
            zero_point = _Round.apply(zero_point)
            zero_point = torch.clamp(zero_point, qmin, qmax)
            self.zero_point = torch.nn.Parameter(zero_point, requires_grad=True)

    def _load_from_state_dict(
        self,
        state_dict,
        prefix,
        local_metadata,
        strict,
        missing_keys,
        unexpected_keys,
        error_msgs,
    ):
        for name in ("scale", "zero_point"):
            key = prefix + name
            if name in self._buffers and key not in state_dict:
                state_dict[key] = self._buffers[name]
            elif (
                key in state_dict
                and name in self._buffers
                and self._buffers[name].numel() == 0
            ):
                self._buffers[name].resize_(state_dict[key].shape)
        super()._load_from_state_dict(
            state_dict,
            prefix,
            local_metadata,
            strict,
            missing_keys,
            unexpected_keys,
            error_msgs,
        )


# For BC
class FakeQuantizer(IntxFakeQuantizer):
    """
    (Deprecated) Please use :class:`~torchao.quantization.qat.IntxFakeQuantizer` instead.
    """

    def __init__(self, config: FakeQuantizeConfigBase):
        super().__init__(config)
        _log_deprecation_warning(self)
