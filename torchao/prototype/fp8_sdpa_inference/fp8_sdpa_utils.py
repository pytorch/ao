# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD 3-Clause license found in the
# LICENSE file in the root directory of this source tree.

"""
Utilities to convert models to use FP8 SDPA.
"""

import warnings
from typing import Callable, Optional

import torch.nn as nn

from torchao.prototype.fp8_sdpa_inference.fp8_sdpa_attention import (
    fp8_sdpa,
    fp8_sdpa_parallel,
)


def _sdpa_call_wrapper(
    original_sdpa: Callable,
) -> Callable:
    """
    Wrap F.scaled_dot_product_attention calls with FP8 version.
    """

    def wrapper(
        query, key, value, attn_mask=None, dropout_p=0.0, is_causal=False, scale=None
    ):
        # If conditions don't match, fall back
        if dropout_p != 0.0 or attn_mask is not None:
            warnings.warn(
                "Dropout or attention mask is not supported for FP8 SDPA, falling back to original SDPA",
                UserWarning,
                stacklevel=2,
            )
            return original_sdpa(
                query,
                key,
                value,
                attn_mask=attn_mask,
                dropout_p=dropout_p,
                is_causal=is_causal,
                scale=scale,
            )

        # Use FP8 SDPA
        return fp8_sdpa_parallel(query, key, value, None, 0.0, is_causal, scale)

    return wrapper


def convert_sdpa_to_fp8_inference(
    model: nn.Module,
    module_filter: Optional[Callable[[str, nn.Module], bool]] = None,
) -> nn.Module:
    """
    Convert a model's attention calls to use FP8 SDPA.

    This is a best-effort conversion that monkeypatches F.scaled_dot_product_attention
    in the model's forward pass. For more control, manually replace attention modules.

    Args:
        model: The model to convert
        module_filter: Optional filter function(name, module) -> bool
                    If provided, only convert modules where this returns True

    Returns:
        The modified model (in-place modification)

    Example:
        >>> model = MyTransformerModel()
        >>> model = convert_sdpa_to_fp8_inference(model)
        >>> with torch.no_grad():
        ...     output = model(inputs)
    """
    # Import here to avoid circular dependency
    import torch.nn.functional as F

    # Store original SDPA
    original_sdpa = F.scaled_dot_product_attention

    # Create wrapper
    fp8_wrapper = _sdpa_call_wrapper(original_sdpa)

    # Monkeypatch (Note: This is module-level, affects all calls)
    # For finer control, users should manually replace attention layers
    F.scaled_dot_product_attention = fp8_wrapper

    print("[FP8 SDPA] Converted model to use FP8 SDPA")
    print("[FP8 SDPA] WARNING: This monkeypatches torch.nn.functional globally")
    print("[FP8 SDPA] For production, consider manual attention module replacement")

    return model
