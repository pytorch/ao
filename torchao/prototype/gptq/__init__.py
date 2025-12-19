# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD 3-Clause license found in the
# LICENSE file in the root directory of this source tree.

import math
import types
from dataclasses import dataclass
from functools import partial

import torch
import torch.nn as nn
from fbgemm_gpu.experimental.gen_ai.quantize import int4_row_quantize_zp

from torchao.core.config import AOBaseConfig
from torchao.quantization import Int4Tensor, Int4WeightOnlyConfig
from torchao.quantization.quant_api import _module_extra_repr
from torchao.quantization.transform_module import register_quantize_module_handler

from .observer import ObserverTensor


@dataclass
class GPTQConfig(AOBaseConfig):
    """Unified config for GPTQ quantization with automatic phase detection.

    On first application: wraps weights as ObserverTensor for observation.
    On second application: applies GPTQ quantization to observed tensors.
    """

    acceleration_config = Int4WeightOnlyConfig()
    percdamp: int = 0.01
    gptq_quantize_block_size = 128


@register_quantize_module_handler(GPTQConfig)
def _gptq_config_transform(
    module: torch.nn.Module, config: GPTQConfig, *, parameter_name="weight"
) -> torch.nn.Module:
    """Unified transform handler that auto-detects observation vs quantization phase."""
    tensor = getattr(module, parameter_name)

    if isinstance(tensor, ObserverTensor):
        # Quantization phase: tensor is already an ObserverTensor
        hessian = _calculate_hessian(tensor.observed_data, device=tensor.hp_data.device)
        new_tensor = gptq_quantize(hessian, tensor.hp_data, config)
        new_quantized_tensor = nn.Parameter(new_tensor, requires_grad=False)
        setattr(module, parameter_name, new_quantized_tensor)
        return module
    else:
        # Observation phase: wrap as ObserverTensor
        new_tensor = ObserverTensor.from_hp(tensor)
        setattr(module, parameter_name, nn.Parameter(new_tensor, requires_grad=False))
        module.extra_repr = types.MethodType(
            partial(
                _module_extra_repr,
                original_extra_repr=module.extra_repr,
                parameter_name=parameter_name,
            ),
            module,
        )
        return module


def _int4_row_quantize_zp_precomputed_qparams(
    x: torch.Tensor,
    scales: torch.Tensor,
    zeros: torch.Tensor,
    group_size: int = 128,
) -> torch.Tensor:
    """Quantize tensor using precomputed scales and zero points."""
    n_bit = 4
    to_quant = torch.split(x.to(torch.float), group_size, dim=-1)

    scales_row = scales.t().contiguous()
    zeros_row = zeros.t().contiguous()
    scales_list = torch.split(scales_row, 1, dim=-1)
    zeros_list = torch.split(zeros_row, 1, dim=-1)

    min_val = [
        zero_chunk - scale_chunk * (2 ** (n_bit - 1))
        for zero_chunk, scale_chunk in zip(zeros_list, scales_list)
    ]
    max_int = 2**n_bit - 1
    min_int = 0

    out = [
        chunk.sub(min_chunk).div(scale_chunk).round().clamp_(min_int, max_int)
        for chunk, min_chunk, scale_chunk in zip(to_quant, min_val, scales_list)
    ]
    out = [(chunk - 2 ** (n_bit - 1)).to(dtype=torch.int8) for chunk in out]
    out = torch.cat(out, dim=-1)
    return out


def _pack_int4(x: torch.Tensor) -> torch.Tensor:
    """Pack int8 tensor containing int4 values into packed int4 format."""
    # Recenter from [-8, 7] to [0, 15]
    x = x + 8
    # Pack two 4-bit values into one byte
    assert x.shape[-1] % 2 == 0
    x = x.reshape(-1, x.shape[-1] // 2, 2)
    packed = x[:, :, 0] | (x[:, :, 1] << 4)
    return packed.reshape(x.shape[0], -1)


def _int4_row_dequantize_zp(
    x: torch.Tensor,
    scales: torch.Tensor,
    zeros: torch.Tensor,
    group_size: int = 128,
) -> torch.Tensor:
    """Dequantize int4 row-quantized tensor with zero point."""
    n_bit = 4

    scales = scales.t().contiguous()
    zeros = zeros.t().contiguous()

    x_chunks = torch.split(x, group_size, dim=-1)
    scales_list = torch.split(scales, 1, dim=-1)
    zeros_list = torch.split(zeros, 1, dim=-1)

    dequant_chunks = []
    for chunk, scale_chunk, zero_chunk in zip(x_chunks, scales_list, zeros_list):
        chunk_float = chunk.to(torch.float32) + 2 ** (n_bit - 1)
        min_val = zero_chunk - scale_chunk * (2 ** (n_bit - 1))
        dequant = chunk_float * scale_chunk + min_val
        dequant_chunks.append(dequant)

    return torch.cat(dequant_chunks, dim=-1)


def gptq_quantize(H, W, config):
    print("gptq quantizing weight of shape: ", W.shape)
    block_size = [1, config.acceleration_config.group_size]
    gptq_quantize_block_size = config.gptq_quantize_block_size
    percdamp = config.percdamp
    group_size = config.acceleration_config.group_size

    assert W.dim() == 2
    assert group_size > 0

    W = W.view(-1, W.shape[-1]).detach()
    columns = W.shape[1]
    device = W.device

    gptq_quantize_block_size = (
        math.ceil(gptq_quantize_block_size / group_size) * group_size
    )

    dead = torch.diag(H) == 0
    H[dead, dead] = 1
    W[:, dead] = 0

    damp = percdamp * torch.mean(torch.diag(H))
    diag = torch.arange(columns, device=device)
    H[diag, diag] += damp
    H = torch.linalg.cholesky(H)
    H = torch.cholesky_inverse(H)
    H = torch.linalg.cholesky(H, upper=True)
    Hinv = H

    all_qparams = []

    for W_quantize_block, block_start in zip(
        torch.split(W, gptq_quantize_block_size, dim=1),
        range(0, columns, gptq_quantize_block_size),
    ):
        block_end = min(block_start + gptq_quantize_block_size, columns)

        Err1 = torch.zeros_like(W_quantize_block, dtype=H.dtype)
        Hinv_quantize_block = Hinv[block_start:block_end, block_start:block_end]

        for W_group, group_start in zip(
            torch.split(W_quantize_block, group_size, dim=1),
            range(block_start, block_end, group_size),
        ):
            group_end = min(group_start + group_size, columns)

            if group_start % group_size == 0:
                # calculate qparams once per group
                _, scale, zero = int4_row_quantize_zp(W_group, group_size)
                all_qparams.append((scale, zero))

            # within each group
            for i in range(group_start - block_start, group_end - block_start):
                w = W_quantize_block[:, i].unsqueeze(1)

                q = _int4_row_quantize_zp_precomputed_qparams(
                    w, scale, zero, group_size
                )
                # Dequantize for error calculation
                dq = _int4_row_dequantize_zp(q, scale, zero, group_size)

                err1 = (w - dq) / Hinv_quantize_block[i, i]
                W_quantize_block[:, i:] -= err1.matmul(
                    Hinv_quantize_block[i, i:].unsqueeze(0)
                )
                Err1[:, i] = err1.flatten()

        W[:, block_end:] -= Err1.matmul(Hinv[block_start:block_end, block_end:])

    if "cuda" in device.type:
        torch.cuda.synchronize()

    # Create final Int4Tensor using standard from_hp method
    final_qparams = [torch.cat(x, dim=0) for x in zip(*all_qparams)]

    # Quantize using precomputed qparams
    wq = _int4_row_quantize_zp_precomputed_qparams(
        W, final_qparams[0], final_qparams[1], group_size
    )
    wq_packed = _pack_int4(wq)

    res = Int4Tensor(
        qdata=wq_packed,
        scale=final_qparams[0].to(W.dtype),
        zero_point=final_qparams[1].to(W.dtype),
        block_size=block_size,
        shape=W.shape,
        act_pre_scale=None,
    )
    return res


def _calculate_hessian(inputs, device=None):
    """Calculate Hessian matrix from input activations for GPTQ."""
    H = 0
    total_batches = 0

    for inp in inputs:
        # Setup x (activation tensor)
        x = inp.float()
        if device:
            x = x.to(device)
        shape = x.shape
        n = 1 if len(shape) == 2 else shape[0]
        x = x.reshape(-1, shape[-1])

        # Update Hessian with running average
        H *= total_batches / (total_batches + n)
        total_batches += n

        x = ((2 / total_batches) ** (1 / 2)) * x.t()
        H += x.matmul(x.t())

    return H


# Re-export public API
from .seq_quant_tracing import sequential_quantize_

# Backward compatibility alias
ObserverConfig = GPTQConfig

__all__ = [
    "ObserverConfig",  # Alias for backward compatibility
    "ObserverTensor",
    "GPTQConfig",
    "gptq_quantize",
    "sequential_quantize_",
]
