# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD 3-Clause license found in the
# LICENSE file in the root directory of this source tree.

"""
Utilities to convert models to use FP8 SDPA.
"""

import warnings
from contextlib import contextmanager

import torch.nn.functional as F
from torch.nn.attention import (
    activate_flash_attention_impl,
    restore_flash_attention_impl,
)

from torchao.prototype.fp8_sdpa_inference.fp8_sdpa_attention import (
    fp8_sdpa_parallel,
)

# Store the original SDPA at module load time
_original_sdpa = F.scaled_dot_product_attention


@contextmanager
def fp8_sdpa_context():
    """
    Context manager that enables FP8 SDPA only within the context.

    Use this to selectively enable FP8 SDPA for specific forward passes
    (e.g., transformer) while leaving other modules (e.g., VAE) unaffected.

    Example:
        >>> with torch.no_grad():
        ...     with fp8_sdpa_context():
        ...         # Transformer uses FP8 SDPA
        ...         noise_pred = transformer(latents, ...)
        ...     # VAE uses regular SDPA (outside context)
        ...     image = vae.decode(latents)
    """
    activate_flash_attention_impl("FA3")

    def fp8_wrapper(
        query,
        key,
        value,
        attn_mask=None,
        dropout_p=0.0,
        is_causal=False,
        scale=None,
        enable_gqa=False,
    ):
        if dropout_p != 0.0 or attn_mask is not None or enable_gqa:
            warnings.warn(
                "Dropout, attention mask, and GQA are not supported for FP8 SDPA. Using regular SDPA instead."
            )
            return _original_sdpa(
                query,
                key,
                value,
                attn_mask=attn_mask,
                dropout_p=dropout_p,
                is_causal=is_causal,
                scale=scale,
                enable_gqa=enable_gqa,
            )
        # Activate FA3 backend (required for FP8 SDPA)
        return fp8_sdpa_parallel(query, key, value, None, 0.0, is_causal, scale, False)

    F.scaled_dot_product_attention = fp8_wrapper
    try:
        yield
    finally:
        restore_flash_attention_impl()
        F.scaled_dot_product_attention = _original_sdpa


def wrap_module_with_fp8_sdpa(module):
    """
    Wrap a module so its forward pass uses FP8 SDPA.
    Only the wrapped module's forward uses FP8; other modules are unaffected.
    """
    original_forward = module.forward

    def wrapped_forward(*args, **kwargs):
        with fp8_sdpa_context():
            return original_forward(*args, **kwargs)

    module.forward = wrapped_forward
    return module
