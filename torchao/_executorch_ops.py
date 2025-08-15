# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD 3-Clause license found in the
# LICENSE file in the root directory of this source tree.
import torch

# TODO: delete these ops


def _quantized_decomposed_quantize_per_channel_group_wrapper(*args, **kwargs):
    """
    Wrapper around torch.ops.quantized_decomposed.quantize_per_channel_group to mitigate
    availability issue until it can be supplanted by new quantize_affine function.
    """
    return torch.ops.quantized_decomposed.quantize_per_channel_group(*args, **kwargs)


def _quantized_decomposed_choose_qparams_per_token_asymmetric_wrapper(*args, **kwargs):
    """
    Wrapper around torch.ops.quantized_decomposed.choose_qparams_per_token_asymmetric to mitigate
    availability issue until it can be supplanted by new choose_qparams_affine function.
    """
    return torch.ops.quantized_decomposed.choose_qparams_per_token_asymmetric(
        *args, **kwargs
    )


def _quantized_decomposed_dequantize_per_channel_group_wrapper(*args, **kwargs):
    """
    Wrapper around torch.ops.quantized_decomposed.dequantize_per_channel_group to mitigate
    availability issue until it can be supplanted by new choose_qparams_affine function.
    """
    return torch.ops.quantized_decomposed.dequantize_per_channel_group(*args, **kwargs)


def _quantized_decomposed_quantize_per_token_wrapper(*args, **kwargs):
    """
    Wrapper around torch.ops.quantized_decomposed.quantize_per_token to mitigate
    availability issue until it can be supplanted by new choose_qparams_affine function.
    """
    return torch.ops.quantized_decomposed.quantize_per_token(*args, **kwargs)


def _quantized_decomposed_dequantize_per_token_wrapper(*args, **kwargs):
    """
    Wrapper around torch.ops.quantized_decomposed.dequantize_per_token to mitigate
    availability issue until it can be supplanted by new choose_qparams_affine function.
    """
    return torch.ops.quantized_decomposed.dequantize_per_token(*args, **kwargs)
