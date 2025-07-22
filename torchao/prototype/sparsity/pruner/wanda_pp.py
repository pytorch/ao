# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD 3-Clause license found in the
# LICENSE file in the root directory of this source tree.

import torch
import torch.nn as nn

from torchao.sparsity import WandaSparsifier

__all__ = ["WandaPlusPlusSparsifier"]


# TODO: Implement Regional Optimization (RO)
# TODO: Add `prepare` function for building quantization configs same as WandaSparsifier
class WandaPlusPlusSparsifier(WandaSparsifier):
    r"""Wanda++ sparsifier extending Wanda with regional gradients
    Wanda++ (Pruning by Weights and activations with Regional Gradients), proposed in
    https://arxiv.org/abs/2503.04992, extends the Wanda method by incorporating
    regional gradients for more accurate pruning criteria.
    The sparsifier removes weights based on the Regional Gradient Score (RGS):
    S_ij = (α * G_ij + ||X_j||_2) * |W_ij|
    where:
    - G_ij: Regional gradient computed from L^l_RGS(X^l_n) = ||f^l(X^l_n)||_2
    - f^l: l-th decoder block function
    - X^l_n: n-th input sample to the l-th decoder block
    - α: Scaling factor for regional gradients (default: 100 from paper)
    Args:
        alpha: Regional gradient scaling factor (default: 100 from paper)
        calibration_samples: Number of samples for gradient computation (default: 32 from paper)
        **kwargs: Arguments passed to WandaSparsifier
    """

    def __init__(self, alpha: float = 100.0, calibration_samples: int = 32, **kwargs):
        super().__init__(**kwargs)
        self.defaults.update(
            {"alpha": alpha, "calibration_samples": calibration_samples}
        )
        self._calibration_inputs = {}
        self._current_decoder_block = None
        self._current_block_name = None

    def store_calibration_input(
        self, block_name: str, input_tensor: torch.Tensor
    ) -> None:
        """Store calibration inputs for regional gradient computation"""
        if block_name not in self._calibration_inputs:
            self._calibration_inputs[block_name] = []

        if (
            len(self._calibration_inputs[block_name])
            < self.defaults["calibration_samples"]
        ):
            self._calibration_inputs[block_name].append(input_tensor.detach().clone())

    def set_context(self, decoder_block: nn.Module, block_name: str) -> None:
        """Set decoder block and block name for regional gradient computation"""
        self._current_decoder_block = decoder_block
        self._current_block_name = block_name

    def update_mask(
        self, module: nn.Module, tensor_name: str, sparsity_level: float, **kwargs
    ) -> None:
        r"""Update mask using Wanda++ criteria with regional gradients

        Unlike Wanda, Wanda++ directly computes regional gradients
        from calibration inputs and applies sparsity based on the metric:
        S_ij = (α * G_ij + ||X_j||_2)
        where:
            - G_ij: Regional gradient computed from calibration inputs
            - ||X_j||_2: L2-norm of the activation post-process norm
            - α: Scaling factor for regional gradients (default: 100)
        """

        # Step 1: get the tensor and the mask from the parametrizations
        mask = getattr(module.parametrizations, tensor_name)[0].mask
        tensor = getattr(module.parametrizations, tensor_name).original
        activation_norm = module.activation_post_process.norm

        # Step 2: Compute regional gradients (RGS)
        regional_gradients = self._compute_regional_gradients(module, tensor_name)

        # Step 3 : Build the metric for sparsity
        metric = (
            self.defaults["alpha"] * regional_gradients + activation_norm.unsqueeze(0)
        ) * tensor.abs()

        # Apply sparsity using existing Wanda logic
        self._apply_sparsity(mask, metric, sparsity_level, kwargs)

    def _compute_regional_gradients(
        self, module: nn.Module, tensor_name: str
    ) -> torch.Tensor:
        """Compute regional gradients from calibration inputs"""

        inputs = self._calibration_inputs.get(self._current_block_name)
        target_param = getattr(module.parametrizations, tensor_name).original
        accumulated_gradients = torch.zeros_like(target_param)

        self._current_decoder_block.eval()

        # Compute L2-norm regional gradients
        for input_tensor in inputs:
            self._current_decoder_block.zero_grad()
            with torch.enable_grad():
                output = self._current_decoder_block(input_tensor)
                torch.norm(output, p=2).backward()
                if target_param.grad is not None:
                    accumulated_gradients += target_param.grad.abs()

        return accumulated_gradients / len(inputs)

    def _apply_sparsity(
        self,
        mask: torch.Tensor,
        metric: torch.Tensor,
        sparsity_level: float,
        kwargs: dict,
    ) -> None:
        """Apply sparsity pattern based on metric"""
        if kwargs.get("semi_structured_block_size"):
            block_size = kwargs["semi_structured_block_size"]
            indices = metric.view(-1, block_size).argsort(dim=1)[:, : block_size // 2]
            mask.data.view(-1, block_size).scatter_(1, indices, 0)
        else:
            num_prune = int(metric.numel() * sparsity_level)
            indices = metric.view(-1).argsort()[:num_prune]
            mask.data.view(-1).scatter_(0, indices, 0)

    def get_calibration_info(self) -> dict[str, int]:
        """Return calibration data info for debugging"""
        return {name: len(inputs) for name, inputs in self._calibration_inputs.items()}
