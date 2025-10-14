# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

from typing import Optional

import torch
from torch.nn.parameter import Parameter

from torchao.quantization.pt2e.fake_quantize import FakeQuantizeBase

__all__ = ["LearnableFakeQuantize"]


class LearnableFakeQuantize(FakeQuantizeBase):
    r"""Generalized extension of the FakeQuantize module.

    This is an extension of the FakeQuantize module, which
    supports more generalized lower-bit quantization and supports learning of the scale
    and zero point parameters through backpropagation.

    In addition to the attributes in the original FakeQuantize module, the LearnableFakeQuantize
    module also includes the following attributes to support quantization parameter learning.

    * :attr:`use_grad_scaling` defines the flag for whether the gradients for scale and zero point are
      normalized by the constant, which is proportional to the square root of the number of
      elements in the tensor. The related literature justifying the use of this particular constant
      can be found here: https://openreview.net/pdf?id=rkgO66VKDS.

    * :attr:`learning_enabled` defines the flag for enabling backpropagation for scale and zero point.
    """

    def __init__(
        self,
        observer,
        quant_min=0,
        quant_max=255,
        use_grad_scaling=False,
        **observer_kwargs,
    ):
        super().__init__()
        assert quant_min < quant_max, "quant_min must be strictly less than quant_max."
        self.quant_min = quant_min
        self.quant_max = quant_max
        # also pass quant_min and quant_max to observer
        observer_kwargs["quant_min"] = quant_min
        observer_kwargs["quant_max"] = quant_max
        self.use_grad_scaling = use_grad_scaling

        # Initialize scale and zero_point as None, will be initialized during first forward pass
        self.scale: Optional[torch.nn.Parameter] = None
        self.zero_point: Optional[torch.nn.Parameter] = None

        self.activation_post_process = observer(**observer_kwargs)
        assert torch.iinfo(self.activation_post_process.dtype).min <= quant_min, (
            "quant_min out of bound"
        )
        assert quant_max <= torch.iinfo(self.activation_post_process.dtype).max, (
            "quant_max out of bound"
        )
        self.dtype = self.activation_post_process.dtype
        self.qscheme = self.activation_post_process.qscheme
        self.ch_axis = (
            self.activation_post_process.ch_axis
            if hasattr(self.activation_post_process, "ch_axis")
            else -1
        )
        self.register_buffer("learning_enabled", torch.tensor([0], dtype=torch.uint8))
        self.register_buffer("eps", torch.tensor([torch.finfo(torch.float32).eps]))

        self._initialized = False

    @torch.jit.export
    def enable_range_learning(self) -> None:
        r"""Enable quantization parameter learning.

        Enables learning of quantization parameters and
        disables observer estimates.
        """
        self.learning_enabled[0] = 1
        self.disable_observer()
        if self.scale is not None:
            self.scale.requires_grad = True
        if self.zero_point is not None:
            self.zero_point.requires_grad = True

    @torch.jit.export
    def disable_range_learning(self) -> None:
        r"""Disable quantization parameter learning.

        Disables learning of quantization parameters
        """
        self.learning_enabled[0] = 0
        if self.scale is not None:
            self.scale.requires_grad = False
        if self.zero_point is not None:
            self.zero_point.requires_grad = False

    @torch.jit.export
    def enable_observer(self, enabled: bool = True) -> None:
        r"""Enable observer.

        Enables observer estimates and disables learning of
        quantization parameters.
        """
        self.observer_enabled[0] = 1 if enabled else 0
        if enabled:
            self.disable_range_learning()

    @torch.jit.export
    def disable_observer(self):
        self.enable_observer(False)

    @torch.jit.export
    def enable_fake_quant(self, enabled: bool = True) -> None:
        self.fake_quant_enabled[0] = 1 if enabled else 0

    @torch.jit.export
    def disable_fake_quant(self):
        self.enable_fake_quant(False)

    @torch.jit.export
    def observe_quant_params(self):
        print(f"LearnableFakeQuantize Scale: {self.scale.detach()}")
        print(f"LearnableFakeQuantize Zero Point: {self.zero_point.detach()}")

    @torch.jit.export
    def calculate_qparams(self):
        self.scale.data.clamp_(min=self.eps.item())
        scale = self.scale.detach()
        zero_point = (
            self.zero_point.detach()
            .round()
            .clamp(self.quant_min, self.quant_max)
            .long()
        )
        return scale, zero_point

    @torch.jit.export
    def extra_repr(self):
        return (
            f"fake_quant_enabled={self.fake_quant_enabled}, observer_enabled={self.observer_enabled}, "
            f"learning_enabled={self.learning_enabled}, quant_min={self.quant_min}, quant_max={self.quant_max}, "
            f"dtype={self.dtype}, qscheme={self.qscheme}, ch_axis={self.ch_axis}, "
            f"use_grad_scaling={self.use_grad_scaling}, scale={self.scale}, zero_point={self.zero_point}"
        )

    def _initialize_or_update_qparams(
        self, scale: torch.Tensor, zero_point: torch.Tensor
    ) -> None:
        """
        Initialize scale and zero_point parameters if they are not initialized yet.
        Update them if they are already initialized.
        """
        if not self._initialized:
            self.scale = Parameter(scale)
            # Convert zero_point to float for learnable parameters
            self.zero_point = Parameter(zero_point.float())
            # Set requires_grad based on current learning state
            if self.learning_enabled[0] == 1:
                self.scale.requires_grad = True
                self.zero_point.requires_grad = True
            else:
                self.scale.requires_grad = False
                self.zero_point.requires_grad = False
            self._initialized = True
        else:
            self.scale.data.copy_(scale)
            self.zero_point.data.copy_(zero_point.float())

    def forward(self, X):
        if self.observer_enabled[0] == 1 or not self._initialized:
            self.activation_post_process(X.detach())
            _scale, _zero_point = self.activation_post_process.calculate_qparams()
            self._initialize_or_update_qparams(_scale, _zero_point)

        if self.fake_quant_enabled[0] == 1:
            if self.qscheme in (
                torch.per_channel_symmetric,
                torch.per_tensor_symmetric,
            ):
                self.zero_point.data.zero_()

            if self.use_grad_scaling:
                grad_factor = 1.0 / (X.numel() * self.quant_max) ** 0.5
            else:
                grad_factor = 1.0
            if self.qscheme in (torch.per_channel_symmetric, torch.per_channel_affine):
                X = torch._fake_quantize_learnable_per_channel_affine(
                    X,
                    self.scale,
                    self.zero_point,
                    self.ch_axis,
                    self.quant_min,
                    self.quant_max,
                    grad_factor,
                )
            else:
                X = torch._fake_quantize_learnable_per_tensor_affine(
                    X,
                    self.scale,
                    self.zero_point,
                    self.quant_min,
                    self.quant_max,
                    grad_factor,
                )

        return X


def enable_range_learning(mod):
    """Enable quantization parameter learning.

    Enable fake quantization for this module, if applicable. Example usage::

      # model is any PyTorch model
      model.apply(torchao.quantization.pt2e.enable_range_learning)

    """
    if isinstance(mod, LearnableFakeQuantize):
        mod.enable_range_learning()


def disable_range_learning(mod):
    """Enable quantization parameter learning.

    Enable fake quantization for this module, if applicable. Example usage::

      # model is any PyTorch model
      model.apply(torchao.quantization.pt2e.disable_range_learning)

    """
    if isinstance(mod, LearnableFakeQuantize):
        mod.disable_range_learning()
