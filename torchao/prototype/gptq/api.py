# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD 3-Clause license found in the
# LICENSE file in the root directory of this source tree.

import types
from dataclasses import dataclass
from functools import partial
from typing import Union

import torch
import torch.nn as nn

try:
    from mslk.quantize import int4_row_quantize_zp, pack_int4
except:
    int4_row_quantize_zp = None
    pack_int4 = None

from torchao.core.config import AOBaseConfig
from torchao.prototype.mx_formats.inference_workflow import (
    MXDynamicActivationMXWeightConfig,
)
from torchao.prototype.mx_formats.mx_tensor import (
    MXTensor,
)
from torchao.quantization import Int4Tensor, Int8Tensor
from torchao.quantization.granularity import PerRow
from torchao.quantization.quant_api import (
    Int4WeightOnlyConfig,
    Int8WeightOnlyConfig,
    _module_extra_repr,
)
from torchao.quantization.quantize_.common.kernel_preference import KernelPreference
from torchao.quantization.transform_module import register_quantize_module_handler
from torchao.quantization.utils import get_block_size

from .observer import GPTQObserverTensor

CONFIG_TO_TORCHAO_BASE_TENSOR = {
    Int4WeightOnlyConfig: Int4Tensor,
    Int8WeightOnlyConfig: Int8Tensor,
}


@dataclass
class GPTQConfig(AOBaseConfig):
    """Config for GPTQ quantization

    GPTQ uses a two-step process:
    - step="observe": Wraps weights as ObserverTensor to collect input activations
    - step="convert": Applies GPTQ quantization using the collected observations

    Note: By default, the "observe" step uses unquantized weights during forward passes.
    For sequential quantization (where each layer observes quantized inputs from the
    previous layer), quantize the model one block at a time. See gptq_example.py for
    an example with HuggingFace models.

    To add support for sequential quantization (observing the quantized inputs) in torchao, we would likely need to trace the model, which introduces quite a bit of complexity to the code.
    A prototype implementation of this exists here: https://gist.github.com/jcaip/2750b5c0711500df48763bdb01d28a31, we plan to revisit adding support for this based on user feedback.

    Args:
        step: Either "observe" or "convert"
        base_config: Base quantization configuration that determines the target dtype.
            Use Int4WeightOnlyConfig() for int4, Int8WeightOnlyConfig() for int8,
            or MXDynamicActivationMXWeightConfig() for MX formats (mxfp8/mxfp4).
        percdamp: Damping factor for Hessian diagonal (default: 0.01)
        gptq_quantize_block_size: Block size for GPTQ algorithm (default: 256)
    """

    step: str = "observe"  # "observe" or "convert"
    base_config: Union[
        Int4WeightOnlyConfig, Int8WeightOnlyConfig, MXDynamicActivationMXWeightConfig
    ] = None
    percdamp: float = 0.01
    gptq_quantize_block_size: int = 256

    def __post_init__(self):
        if self.base_config is None:
            raise ValueError("base_config is required for GPTQ quantization.")
        if isinstance(self.base_config, Int4WeightOnlyConfig):
            if int4_row_quantize_zp is None:
                raise ValueError(
                    "fbgemm_gpu is not installed. Please install fbgemm_gpu to use int4 quantization."
                )


@register_quantize_module_handler(GPTQConfig)
def _gptq_config_transform(
    module: torch.nn.Module, config: GPTQConfig, *, parameter_name="weight"
) -> torch.nn.Module:
    """Unified transform handler that uses explicit step control."""
    tensor = getattr(module, parameter_name)

    if config.step == "observe":
        # Observation phase: wrap as GPTQObserverTensor which incrementally
        # computes the Hessian during forward passes.
        # For MX dynamic activation configs, pass a quantize_fn so that
        # activation quantization noise is captured during observation.
        quantize_fn = None
        if isinstance(config.base_config, MXDynamicActivationMXWeightConfig):
            base = config.base_config

            def quantize_fn(x):
                mx = MXTensor.to_mx(
                    x.to(torch.bfloat16),
                    base.activation_dtype,
                    block_size=base.block_size,
                    kernel_preference=KernelPreference.EMULATED,
                )
                return mx.dequantize(torch.float)

        new_tensor = GPTQObserverTensor.from_hp(tensor, quantize_fn=quantize_fn)
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
    elif config.step == "convert":
        # Quantization phase: tensor should be a GPTQObserverTensor
        if not isinstance(tensor, GPTQObserverTensor):
            raise ValueError(
                f"Expected {parameter_name} to be GPTQObserverTensor in 'convert' step, "
                f"but got {type(tensor)}. Did you run the 'observe' step first?"
            )

        # Validate that observations were recorded
        if tensor.hessian is None or tensor.total_batches == 0:
            raise ValueError(
                f"No observations recorded for {parameter_name}. "
                f"Hessian is empty. Did you run forward passes during the observe step?"
            )

        hessian = tensor.hessian
        new_tensor = gptq_quantize(hessian, tensor.hp_data, config)
        new_quantized_tensor = nn.Parameter(new_tensor, requires_grad=False)
        setattr(module, parameter_name, new_quantized_tensor)
        return module
    else:
        raise ValueError(
            f"Invalid step '{config.step}'. Must be 'observe' or 'convert'."
        )


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


def gptq_quantize(H: torch.Tensor, W: torch.Tensor, config: GPTQConfig):
    """
    This function implements the GPTQ algorithm described in this paper: https://arxiv.org/abs/2210.17323 (Algorithm 1)

    GPTQ quantizes weights column-by-column while propagating quantization errors to subsequent columns.
    This minimizes the overall error: argmin W' ||X@W - X@W'||_2^2

    For example we can see how GPTQ improves accuracy on a toy example (2x2 matrix, 2-bit symmetric quantization):

    Example:
        Given: W = [[1.2, 0.5],    X = [[1.0, 1.0]]    Original output: X @ W = [[2.0, 2.0]]
                    [0.8, 1.5]]

        We can find the naive RTN quantization error as follows:

        1. Compute damped inverse Hessian: H_inv = [[0.505, -0.495], [-0.495, 0.505]]

        2. Quantize column 0: [1.2, 0.8]
           Scale: max(abs(w)) / quant_max = 1.2 / 2 = 0.6
           Quant: round(w / scale).clamp(0, 2) → [2, 1]
           Dequant: quant * scale → [1.2, 0.6]
           Error: err = ([1.2, 0.8] - [1.2, 0.6]) / 0.505 = [0.0, 0.396]
           Update column 1: W[:, 1] = [0.5, 1.5] - [0.0, 0.396] @ [-0.495] = [0.5, 1.304]

        3. Quantize column 1 (updated): [0.5, 1.304]
           Scale: 1.304 / 2 = 0.652
           Quant: [1, 2]
           Dequant: [0.652, 1.304]

        GPTQ result:  quantized [[2, 1], [1, 2]]  →  dequantized [[1.2,   0.652],
                                                                  [0.6,   1.304]]
                      X @ W_gptq = [[1.801, 1.956]]    Error: ||X@W - X@W_gptq||_2^2 = 0.09

        Compare this to naive RTN quantization:
        Naive (RTN): Column 0 scale: 1.2 / 2 = 0.6,  Column 1 scale: 1.5 / 2 = 0.75
                     quantized [[2, 1], [1, 2]]  →  dequantized [[1.2,  0.75],
                                                                 [0.6,  1.5 ]]
                     X @ W_naive = [[1.8, 2.25]]   Error: ||X@W - X@W_naive||_2^2 = 0.11

    Args:
        H: Hessian matrix approximation (from input activations)
        W: Weight matrix to quantize
        config: GPTQ configuration

    Returns:
        Quantized weight matrix (Int4Tensor, Int8Tensor, or dequantized MXTensor)
    """
    assert W.dim() == 2
    gptq_quantize_block_size = config.gptq_quantize_block_size
    percdamp = config.percdamp
    base_config = config.base_config

    if isinstance(base_config, Int4WeightOnlyConfig):
        group_size = config.base_config.group_size
        block_size = [1, group_size]
    elif isinstance(base_config, Int8WeightOnlyConfig):
        assert isinstance(base_config.granularity, PerRow), (
            "GPTQ only supports per-row quantization"
        )
        block_size = get_block_size(W.shape, base_config.granularity)
        block_size = list(block_size)
        group_size = block_size[-1]

    assert group_size > 0

    W = W.view(-1, W.shape[-1]).detach()
    columns = W.shape[1]
    device = W.device

    assert device.type == "cuda", "GPTQ only supports CUDA currently"

    dead = torch.diag(H) == 0
    H[dead, dead] = 1
    W[:, dead] = 0

    # Apply damping and compute inverse Hessian for numerical stability
    damp = percdamp * torch.mean(torch.diag(H))
    diag = torch.arange(columns, device=device)
    H[diag, diag] += damp
    H = torch.linalg.cholesky(H)
    H = torch.cholesky_inverse(H)
    H = torch.linalg.cholesky(H, upper=True)
    Hinv = H

    group_qparams = []
    for W_quantize_block, block_start in zip(
        torch.split(W, gptq_quantize_block_size, dim=1),
        range(0, columns, gptq_quantize_block_size),
    ):
        block_end = min(block_start + gptq_quantize_block_size, columns)
        Err1 = torch.zeros_like(W_quantize_block, dtype=H.dtype)
        Hinv_quantize_block = Hinv[block_start:block_end, block_start:block_end]

        # If we are doing per-row quantization, the group_size is equal to the number of columns and this will only run once.
        # Otherwise, if we do per-group quantization, we need to iterate through the block one group at a time.
        for group_start in range(block_start, block_end, group_size):
            group_end = min(group_start + group_size, block_end)

            # We only need to calculate initial qparams for the group once
            if group_start % group_size == 0:
                if isinstance(base_config, Int4WeightOnlyConfig):
                    _, scale, zero_point = int4_row_quantize_zp(
                        W_quantize_block[
                            :, group_start - block_start : group_end - block_start
                        ],
                        group_size,
                    )
                    group_qparams.append((scale, zero_point))
                elif isinstance(base_config, Int8WeightOnlyConfig):
                    quantized_tensor = Int8Tensor.from_hp(
                        W_quantize_block[
                            :, group_start - block_start : group_end - block_start
                        ],
                        base_config.granularity,
                    )

            # Quantize each column and propagate errors to subsequent columns
            for i in range(group_start - block_start, group_end - block_start):
                w = W_quantize_block[:, i].unsqueeze(1)
                if isinstance(base_config, Int4WeightOnlyConfig):
                    q = _int4_row_quantize_zp_precomputed_qparams(
                        w, scale, zero_point, group_size
                    )
                    dq = _int4_row_dequantize_zp(q, scale, zero_point, group_size)
                elif isinstance(base_config, Int8WeightOnlyConfig):
                    q = Int8Tensor.from_hp(
                        w,
                        granularity=base_config.granularity,
                        scale=quantized_tensor.scale,
                    )
                    dq = q.dequantize(output_dtype=torch.float)

                err1 = (w - dq) / Hinv_quantize_block[i, i]
                W_quantize_block[:, i:] -= err1.matmul(
                    Hinv_quantize_block[i, i:].unsqueeze(0)
                )
                Err1[:, i] = err1.flatten()

        # Lazy Batch-Updates: We process B columns at a time with local updates above.
        # Once a block is fully processed, perform global updates to H^-1 and W using batched versions of the error propagation equations.
        W[:, block_end:] -= Err1.matmul(Hinv[block_start:block_end, block_end:])

    torch.cuda.synchronize()

    # Create the final quantized tensor, which has the same qparams (scale, zero_point), but different qdata
    if isinstance(base_config, Int4WeightOnlyConfig):
        scale, zero_point = [torch.cat(x, dim=0) for x in zip(*group_qparams)]
        wq = _int4_row_quantize_zp_precomputed_qparams(W, scale, zero_point, group_size)
        result = Int4Tensor(
            qdata=pack_int4(wq),
            scale=scale.to(W.dtype),
            zero_point=zero_point.to(W.dtype),
            block_size=block_size,
            shape=W.shape,
            act_pre_scale=None,
        )
    elif isinstance(base_config, Int8WeightOnlyConfig):
        result = Int8Tensor.from_hp(
            W, granularity=base_config.granularity, scale=quantized_tensor.scale
        )

    return result


__all__ = [
    "GPTQConfig",
    "gptq_quantize",
]
