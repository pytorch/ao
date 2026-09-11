# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD 3-Clause license found in the
# LICENSE file in the root directory of this source tree.

import os
import types
import warnings
from dataclasses import dataclass
from functools import partial
from typing import Literal, Union

import torch
import torch.nn as nn

from torchao.utils import torch_version_at_least

try:
    from mslk.quantize.shuffle import int4_row_quantize_zp, pack_int4
except:
    int4_row_quantize_zp = None
    pack_int4 = None

from torchao.core.config import AOBaseConfig
from torchao.prototype.mx_formats.constants import F4_E2M1_MAX
from torchao.prototype.mx_formats.inference_workflow import (
    NVFP4DynamicActivationNVFP4WeightConfig,
)
from torchao.prototype.mx_formats.kernels import (
    f4_unpacked_to_f32,
    f32_to_f4_unpacked,
    pack_uint4,
)
from torchao.prototype.mx_formats.nvfp4_tensor import (
    NVFP4Tensor,
    QuantizeTensorToNVFP4Kwargs,
    nvfp4_quantize,
    per_tensor_amax_to_scale,
)
from torchao.prototype.mx_formats.utils import (
    hp_data_dims_to_swizzled_scale_dims_nvfp4,
    to_blocked,
)
from torchao.quantization import Int4Tensor, Int8Tensor
from torchao.quantization.granularity import PerRow
from torchao.quantization.quant_api import (
    Int4WeightOnlyConfig,
    Int8WeightOnlyConfig,
)
from torchao.quantization.transform_module import register_quantize_module_handler
from torchao.quantization.utils import _module_extra_repr, get_block_size

from .observer import GPTQObserverTensor

CONFIG_TO_TORCHAO_BASE_TENSOR = {
    Int4WeightOnlyConfig: Int4Tensor,
    Int8WeightOnlyConfig: Int8Tensor,
    NVFP4DynamicActivationNVFP4WeightConfig: NVFP4Tensor,
}


@dataclass
class GPTQConfig(AOBaseConfig):
    """Config for GPTQ quantization

    GPTQ uses a two-step process:
    - step="prepare": Wraps weights as GPTQObserverTensor to collect Hessian information
    - step="convert": Applies GPTQ quantization using the collected observations

    Note: By default, the PREPARE step uses unquantized weights during forward passes.
    For sequential quantization (where each layer observes quantized inputs from the
    previous layer), quantize the model one block at a time. See gptq_example.py for
    an example with HuggingFace models.

    To add support for sequential quantization (observing the quantized inputs) in torchao, we would likely need to trace the model, which introduces quite a bit of complexity to the code.
    A prototype implementation of this exists here: https://gist.github.com/jcaip/2750b5c0711500df48763bdb01d28a31, we plan to revisit adding support for this based on user feedback.

    Args:
        step: The step for GPTQ process. Can be "prepare" or "convert".
            "prepare": insert GPTQ observers to collect Hessian information
            "convert": convert the observed linear modules to GPTQ quantized modules
        base_config: Base quantization configuration that determines the target dtype.
            Use Int4WeightOnlyConfig() for int4 or Int8WeightOnlyConfig() for int8.
        percdamp: Damping factor for Hessian diagonal (default: 0.01)
        gptq_quantize_block_size: Block size for GPTQ algorithm (default: 256)
    """

    step: str = "observe"  # "observe" or "convert"
    base_config: Union[
        Int4WeightOnlyConfig,
        Int8WeightOnlyConfig,
        NVFP4DynamicActivationNVFP4WeightConfig,
    ] = None
    percdamp: float = 0.01
    gptq_quantize_block_size: int = 256

    def __post_init__(self):
        if self.base_config is None:
            raise ValueError("base_config is required for GPTQ quantization.")
        if isinstance(self.base_config, Int4WeightOnlyConfig):
            if int4_row_quantize_zp is None:
                raise ValueError(
                    "mslk is not installed. Please install mslk to use int4 quantization."
                )


# simple progress counter for GPTQ convert
# TODO(future): make this cleaner, will require a refactor
gptq_convert_layer_counter = 0


@register_quantize_module_handler(GPTQConfig)
def _gptq_config_transform(
    module: torch.nn.Module, config: GPTQConfig, *, parameter_name="weight"
) -> torch.nn.Module:
    """Unified transform handler that uses explicit step control."""
    tensor = getattr(module, parameter_name)
    step = config.step

    if step == "prepare":
        # Observation phase: wrap as GPTQObserverTensor
        new_tensor = GPTQObserverTensor.from_hp(tensor)
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
        global gptq_convert_layer_counter
        print(f"gptq convert {gptq_convert_layer_counter}")
        gptq_convert_layer_counter += 1

        # Quantization phase: tensor should be an GPTQObserverTensor
        if not isinstance(tensor, GPTQObserverTensor):
            raise ValueError(
                f"Expected {parameter_name} to be GPTQObserverTensor in 'convert' step, "
                f"but got {type(tensor)}. Did you run the 'prepare' step first?"
            )

        # Validate that observations were recorded
        if (tensor.total_batches == 0).any():
            raise ValueError(
                f"No observations recorded for {parameter_name}. "
                f"total_batches is 0. Did you run forward passes during the observe step?"
            )

        # Use pre-computed Hessian directly
        hessian = tensor.hessian
        if len(tensor.shape) == 2:
            new_tensor = gptq_quantize(hessian, tensor.hp_data, config)
        else:
            assert len(tensor.shape) == 3, "unsupported"
            new_tensor = gptq_quantize_3d(hessian, tensor.hp_data, config)

        new_quantized_tensor = nn.Parameter(new_tensor, requires_grad=False)
        setattr(module, parameter_name, new_quantized_tensor)
        return module
    else:
        raise ValueError(
            f"Invalid step '{config.step}'. Expected 'prepare' or 'convert'."
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


def _nvfp4_with_precalculated_scales_qdq(
    data_hp: torch.Tensor,
    per_tensor_scale: torch.Tensor,
    block_scale: torch.Tensor,
) -> torch.Tensor:
    """
    Same as torchao.prototype.mx_formats.nvfp4_tensor.nvfp4_quantize, but with
    per_tensor_scale and block_scale precalculated and the end result dequantized.
    """
    assert per_tensor_scale.dtype is torch.float32
    assert block_scale.dtype is torch.float8_e4m3fn
    # this function only works for data_hp.shape == (N, k_slice)
    # and block_scale.shape == (N,)
    assert len(block_scale.shape) == 1

    scaled_block_scales_fp32 = block_scale.to(torch.float32)
    reciprocal_scale = (1.0 / per_tensor_scale) / scaled_block_scales_fp32
    data_scaled = data_hp * reciprocal_scale.unsqueeze(-1)
    data_scaled = torch.clamp(data_scaled, -F4_E2M1_MAX, F4_E2M1_MAX)
    data_lp = f32_to_f4_unpacked(data_scaled)
    data_lp_hp = f4_unpacked_to_f32(data_lp)
    data_lp_hp_unscaled = data_lp_hp / reciprocal_scale.unsqueeze(-1)
    return data_lp_hp_unscaled


def _nvfp4_with_precalculated_scales_q(
    data_hp: torch.Tensor,
    per_tensor_scale: torch.Tensor,
    block_scale: torch.Tensor,
) -> torch.Tensor:
    """
    Same as torchao.prototype.mx_formats.nvfp4_tensor.nvfp4_quantize, but with
    per_tensor_scale and block_scale precalculated.
    """
    assert per_tensor_scale.dtype is torch.float32
    assert block_scale.dtype is torch.float8_e4m3fn

    # TODO(future): figure out what to reuse vs leave copy-pasted vs
    # nvfp4_tensor.py
    scaled_block_scales_fp32 = block_scale.to(torch.float32)
    reciprocal_scale = (1.0 / per_tensor_scale) / scaled_block_scales_fp32
    N, K = data_hp.shape
    # reshape to 3d to properly broadcast for scaling
    data_hp = data_hp.view(N, K // 16, 16)
    data_scaled = data_hp * reciprocal_scale.unsqueeze(-1)
    data_scaled = torch.clamp(data_scaled, -F4_E2M1_MAX, F4_E2M1_MAX)
    data_lp = f32_to_f4_unpacked(data_scaled)
    data_lp_packed = pack_uint4(data_lp)
    # reshape back to 2d
    data_lp_packed = data_lp_packed.view(N, K // 2)
    return data_lp_packed


# Set to True to torch.compile the NVFP4 quantize/dequantize functions
# inside gptq_quantize. Gives ~3x speedup.
_use_torch_compile = True

if _use_torch_compile:
    _nvfp4_qdq_fn = torch.compile(_nvfp4_with_precalculated_scales_qdq)
    _nvfp4_q_fn = torch.compile(_nvfp4_with_precalculated_scales_q)

    if torch_version_at_least("2.11.0"):
        # Triton's default f32 division uses approximate reciprocal which
        # introduces ~1 ULP error per division. In GPTQ's error propagation
        # loop this compounds across columns. IEEE-compliant division rounding
        # eliminates the drift.
        import torch._inductor.config as _inductor_config

        if os.environ.get("TORCHINDUCTOR_EMULATE_DIVISION_ROUNDING") == "0":
            warnings.warn(
                "TORCHINDUCTOR_EMULATE_DIVISION_ROUNDING=0 may cause numerical "
                "drift in GPTQ with torch.compile. "
                "Consider unsetting it or setting it to 1."
            )
        else:
            _inductor_config.eager_numerics.division_rounding = True
    else:
        warnings.warn(
            "PyTorch < 2.11.0 detected. Upgrade to PyTorch 2.11.0+ for "
            "better GPTQ numerics with torch.compile (IEEE-compliant "
            "division rounding)."
        )
else:
    _nvfp4_qdq_fn = _nvfp4_with_precalculated_scales_qdq
    _nvfp4_q_fn = _nvfp4_with_precalculated_scales_q


def gptq_quantize(H: torch.Tensor, W_t: torch.Tensor, config: GPTQConfig):
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
        H: Hessian matrix approximation (from input activations), with shape (K, K)
        W_t: Weight matrix to quantize, with shape (N, K)
        config: GPTQ configuration

    Returns:
        Int4Tensor or Int8Tensor: Quantized weight matrix
    """
    assert W_t.dim() == 2
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
        block_size = get_block_size(W_t.shape, base_config.granularity)
        block_size = list(block_size)
        group_size = block_size[-1]
    elif isinstance(base_config, NVFP4DynamicActivationNVFP4WeightConfig):
        assert base_config.use_dynamic_per_tensor_scale, "unsupported"
        group_size = 16
        block_size = [1, group_size]
        # for per-tensor nvfp4, we need to calculate the global scale over the
        # entire tensor before we enter the GPTQ loop
        tensor_amax = torch.max(torch.abs(W_t))
        nvfp4_global_scale = per_tensor_amax_to_scale(tensor_amax)
    else:
        raise AssertionError("unsupported")

    assert group_size > 0

    W_t = W_t.view(-1, W_t.shape[-1]).detach()
    columns = W_t.shape[1]
    device = W_t.device

    assert device.type == "cuda", "GPTQ only supports CUDA currently"

    dead = torch.diag(H) == 0
    H[dead, dead] = 1
    W_t[:, dead] = 0

    # Apply damping and compute inverse Hessian for numerical stability
    damp = percdamp * torch.mean(torch.diag(H))
    diag = torch.arange(columns, device=device)
    H[diag, diag] += damp
    H = torch.linalg.cholesky(H)
    H = torch.cholesky_inverse(H)
    H = torch.linalg.cholesky(H, upper=True)
    Hinv = H

    # GPTQ update loop:
    #
    #                      W_t (below)
    #
    #    |------------------ K --------------------|
    #    |---B1----|---B2----| ...
    #    |-G1-|-G2-|-G1-|-G2-| ...
    #    |-----------------------------------------|
    #  N | 0 1 2 3 | 4 5 6 7 | ...
    #    | ...
    #    |-----------------------------------------|
    #
    # 1. start with W_t, with shape [N, K]
    #    * B1, ..., BN are chunks of size [N, B], where B is a hyperparameter of GPTQ
    #    * G1, ..., GN are chunks of size [N, G], where G is group_size of the quantization recipe
    #
    # 2. triple for loop, with every loop chunking along the K dimension:
    #
    #    for B_cur in (B1, ..., BN):
    #        # B_cur is of shape (N, B)
    #        # Hinv_cur corresponding to B_cur is of shape (B, B)
    #
    #        for G_cur in (G1, ..., GN):
    #            # G_cur is of shape (N, group_size)
    #            # Initialize qparams for all of G_cur, this freezes the quantization
    #            # grid for G_cur. The rest of this for loop will iteratively optimize
    #            # the quantized weight values.
    #
    #            for k in range(G_k_start - B_cur_k_start, G_k_end - B_cur_k_start):
    #                # k is relative to the start of B_cur
    #                w_t = B_cur[:, k]
    #                w_t_qdq = quant_dequant(w_t, base_config, qparams)
    #                err1 = (w_t - w_t_qdq) / Hinv_cur[k, k]
    #                # propagate errors to remaining columns in B_cur
    #                B_cur[:, k:] -= err1.matmul(Hinv_cur[k, k:])
    #                B_cur_Err1[:, k] = err1.flatten()
    #
    #        # batch propagate errors for all remaining blocks in W_t
    #        W_t[:, B_cur_k_end:] -= B_cur_Err1.matmul(Hinv[B_cur_k_start:B_cur_k_end, B_cur_k_end:])
    #

    group_qparams = []
    for B_cur, B_cur_k_start in zip(
        torch.split(W_t, gptq_quantize_block_size, dim=1),
        range(0, columns, gptq_quantize_block_size),
    ):
        B_cur_k_end = min(B_cur_k_start + gptq_quantize_block_size, columns)
        B_cur_Err1 = torch.zeros_like(B_cur, dtype=H.dtype)
        Hinv_cur = Hinv[B_cur_k_start:B_cur_k_end, B_cur_k_start:B_cur_k_end]

        # If we are doing per-row quantization, the group_size is equal to the number of columns and this will only run once.
        # Otherwise, if we do per-group quantization, we need to iterate through the block one group at a time.
        for G_k_start in range(B_cur_k_start, B_cur_k_end, group_size):
            G_k_end = min(G_k_start + group_size, B_cur_k_end)

            # We only need to calculate initial qparams for the group once
            if G_k_start % group_size == 0:
                if isinstance(base_config, Int4WeightOnlyConfig):
                    _, scale, zero_point = int4_row_quantize_zp(
                        B_cur[
                            :,
                            G_k_start - B_cur_k_start : G_k_end - B_cur_k_start,
                        ],
                        group_size,
                    )
                    group_qparams.append((scale, zero_point))
                elif isinstance(base_config, Int8WeightOnlyConfig):
                    quantized_tensor = Int8Tensor.from_hp(
                        B_cur[
                            :,
                            G_k_start - B_cur_k_start : G_k_end - B_cur_k_start,
                        ],
                        base_config.granularity,
                    )
                elif isinstance(base_config, NVFP4DynamicActivationNVFP4WeightConfig):
                    tensor_slice = B_cur[
                        :,
                        G_k_start - B_cur_k_start : G_k_end - B_cur_k_start,
                    ].contiguous()
                    # quantize this slice using pre-calculated global scale, to
                    # get the blockwise dynamic scales, they will be frozen
                    # after this point
                    scale, _data_lp = nvfp4_quantize(
                        tensor_slice,
                        per_tensor_scale=nvfp4_global_scale,
                    )
                    group_qparams.append(scale)
                    # TODO(future PR): simpler version of `nvfp4_quantize` which
                    # just calculates the scale, since we are throwing away the
                    # quantized packed data here. For now, just call the full
                    # one.
                    del _data_lp

                else:
                    raise AssertionError("unsupported")

            # Quantize each column and propagate errors to subsequent columns
            for k in range(G_k_start - B_cur_k_start, G_k_end - B_cur_k_start):
                # k is relative to the start of B_cur
                w_t = B_cur[:, k].unsqueeze(1)
                if isinstance(base_config, Int4WeightOnlyConfig):
                    q = _int4_row_quantize_zp_precomputed_qparams(
                        w_t, scale, zero_point, group_size
                    )
                    dq = _int4_row_dequantize_zp(q, scale, zero_point, group_size)
                elif isinstance(base_config, Int8WeightOnlyConfig):
                    q = Int8Tensor.from_hp(
                        w_t,
                        granularity=base_config.granularity,
                        scale=quantized_tensor.scale,
                    )
                    dq = q.dequantize(output_dtype=torch.float)
                elif isinstance(base_config, NVFP4DynamicActivationNVFP4WeightConfig):
                    dq = _nvfp4_qdq_fn(
                        w_t,
                        nvfp4_global_scale,
                        scale.squeeze(-1),
                    )

                err1 = (w_t - dq) / Hinv_cur[k, k]
                B_cur[:, k:] -= err1.matmul(Hinv_cur[k, k:].unsqueeze(0))
                B_cur_Err1[:, k] = err1.flatten()

        # Lazy Batch-Updates: We process B columns at a time with local updates above.
        # Once a block is fully processed, perform global updates to H^-1 and W using batched versions of the error propagation equations.
        W_t[:, B_cur_k_end:] -= B_cur_Err1.matmul(
            Hinv[B_cur_k_start:B_cur_k_end, B_cur_k_end:]
        )

    torch.cuda.synchronize()

    # Create the final quantized tensor, which has the same qparams (scale, zero_point), but different qdata
    if isinstance(base_config, Int4WeightOnlyConfig):
        scale, zero_point = [torch.cat(x, dim=0) for x in zip(*group_qparams)]
        wq_t = _int4_row_quantize_zp_precomputed_qparams(
            W_t, scale, zero_point, group_size
        )
        result = Int4Tensor(
            qdata=pack_int4(wq_t),
            scale=scale.to(W_t.dtype),
            zero_point=zero_point.to(W_t.dtype),
            block_size=block_size,
            shape=W_t.shape,
            act_pre_scale=None,
        )
    elif isinstance(base_config, Int8WeightOnlyConfig):
        result = Int8Tensor.from_hp(
            W_t, granularity=base_config.granularity, scale=quantized_tensor.scale
        )
    else:
        N, K = W_t.shape
        # TODO(future PR): clean up the line below. Context: the current nvfp4
        # code follows the int4 code - we save the blockwise scales to an array
        # and concat it. This leads to the necessity of t().contiguous().t() to
        # get the scales back into the right layout for W_t. This is not
        # intuitive, likely better to initialize the scales holder ahead of time
        # and write the scales directly to their final place.
        combined_scale = (
            torch.cat(group_qparams, dim=0).reshape(K // group_size, N).t().contiguous()
        )
        qdata = _nvfp4_q_fn(
            W_t,
            nvfp4_global_scale,
            combined_scale,
        )

        act_quant_kwargs = QuantizeTensorToNVFP4Kwargs(
            use_dynamic_per_tensor_scale=base_config.use_dynamic_per_tensor_scale,
            use_triton_kernel=base_config.use_triton_kernel,
            is_swizzled_scales=True,
        )

        # swizzle the block scales
        combined_scale_swizzled = to_blocked(combined_scale).flatten()
        scale_N, scale_K = hp_data_dims_to_swizzled_scale_dims_nvfp4(N, K)
        combined_scale_swizzled = combined_scale_swizzled.view(scale_N, scale_K)

        result = NVFP4Tensor(
            qdata,
            combined_scale_swizzled,
            block_size=group_size,
            orig_dtype=W_t.dtype,
            per_tensor_scale=nvfp4_global_scale,
            # TODO(future): get act_per_tensor_scale from calibration data?
            # for now, set it to None here to calculate it dynamically at
            # runtime
            act_per_tensor_scale=None,
            is_swizzled_scales=True,
            use_triton_kernel=base_config.use_triton_kernel,
            act_quant_kwargs=act_quant_kwargs,
        )

    return result


def gptq_quantize_3d(H: torch.Tensor, W_t: torch.Tensor, config: GPTQConfig):
    """3D variant of gptq_quantize for MoE expert weights.

    Args:
        H: per-expert Hessian of shape (E, K, K)
        W_t: stacked expert weights of shape (E, N, K)
        config: GPTQ configuration (NVFP4 only)

    Returns:
        NVFP4Tensor of shape (E, N, K) assembled from per-expert 2D results.
    """
    assert H.dim() == 3 and W_t.dim() == 3
    assert H.shape[0] == W_t.shape[0]
    base_config = config.base_config
    assert isinstance(base_config, NVFP4DynamicActivationNVFP4WeightConfig), (
        "gptq_quantize_3d only supports NVFP4"
    )

    E = W_t.shape[0]
    pieces = [gptq_quantize(H[e], W_t[e], config) for e in range(E)]

    # Stack inner NVFP4Tensor fields along a new expert dim 0. These are plain
    # tensors (uint8 / float8_e4m3fn / float32), so torch.stack goes through
    # normal aten dispatch, not NVFP4Tensor.
    qdata_3d = torch.stack([p.qdata for p in pieces], dim=0)
    scale_3d = torch.stack([p.scale for p in pieces], dim=0)
    per_tensor_scale_3d = torch.stack(
        [p.per_tensor_scale.view(1, 1) for p in pieces], dim=0
    )

    return NVFP4Tensor(
        qdata_3d,
        scale_3d,
        block_size=pieces[0].block_size,
        orig_dtype=pieces[0].orig_dtype,
        per_tensor_scale=per_tensor_scale_3d,
        act_per_tensor_scale=None,
        is_swizzled_scales=True,
        use_triton_kernel=pieces[0].use_triton_kernel,
        act_quant_kwargs=pieces[0].act_quant_kwargs,
    )


__all__ = [
    "GPTQConfig",
    "gptq_quantize",
    "gptq_quantize_3d",
]
