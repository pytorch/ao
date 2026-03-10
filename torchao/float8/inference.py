# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD 3-Clause license found in the
# LICENSE file in the root directory of this source tree.
"""
Defines an nn module designed to be used during inference
"""

import warnings
from typing import List, NamedTuple, Optional, Tuple, Union

import torch

from torchao.float8.float8_utils import is_row_major, pad_tensor_for_matmul
from torchao.float8.hifloat8_utils import (
    get_hifloat4_npu_dtype,
    hifloat4_raw_int8,
    hifloat8_raw_int8,
    hifloatx_scales_to_npu,
    is_hifloat4_tensor,
    is_hifloat8_tensor,
    is_hifloatx_tensor,
)
from torchao.float8.types import FP8Granularity
from torchao.quantization.granularity import (
    PerRow,
    PerTensor,
)
from torchao.utils import (
    is_MI300,
    is_sm_at_least_89,
)

Tensor = torch.Tensor
_STANDARD_FP8_DTYPES = tuple(
    d
    for d in (
        getattr(torch, "float8_e4m3fn", None),
        getattr(torch, "float8_e5m2", None),
        getattr(torch, "float8_e4m3fnuz", None),
        getattr(torch, "float8_e5m2fnuz", None),
    )
    if d is not None
)


def _is_standard_fp8_tensor(t: Tensor) -> bool:
    if not isinstance(t, torch.Tensor):
        return False
    if t.dtype in _STANDARD_FP8_DTYPES:
        return True
    try:
        dtype_str = str(t.dtype).lower()
    except Exception:
        return False
    return ("float8_e4m3fn" in dtype_str) or ("float8_e5m2" in dtype_str)


_WARNED_STANDARD_FP8_NPU_FALLBACK = False


def _warn_standard_fp8_npu_fallback_once() -> None:
    global _WARNED_STANDARD_FP8_NPU_FALLBACK
    if _WARNED_STANDARD_FP8_NPU_FALLBACK:
        return
    warnings.warn(
        "Standard float8 on NPU currently falls back to explicit dequantize + "
        "torch.mm (npu_quant_matmul is not used) to avoid _scaled_mm/triton path.",
        RuntimeWarning,
        stacklevel=2,
    )
    _WARNED_STANDARD_FP8_NPU_FALLBACK = True


def _should_use_npu_fp8_matmul(a_data: Tensor, b_data: Tensor) -> bool:
    if a_data.device.type != "npu" or b_data.device.type != "npu":
        return False
    if is_hifloatx_tensor(a_data) or is_hifloatx_tensor(b_data):
        return True
    return _is_standard_fp8_tensor(a_data) and _is_standard_fp8_tensor(b_data)


def _fp8_npu_matmul(
    a_data: Tensor,
    a_scale: Tensor,
    b_data: Tensor,
    b_scale: Tensor,
    output_dtype: torch.dtype,
    bias: Optional[Tensor] = None,
) -> Tensor:
    if torch.compiler.is_compiling():
        # Keep this op out of NPU inductor triton codegen to avoid
        # known scheduler/codegen failures in compile mode.
        return _fp8_npu_matmul_eager(
            a_data, a_scale, b_data, b_scale, output_dtype, bias=bias
        )
    return _fp8_npu_matmul_impl(
        a_data, a_scale, b_data, b_scale, output_dtype, bias=bias
    )


@torch._dynamo.disable
def _fp8_npu_matmul_eager(
    a_data: Tensor,
    a_scale: Tensor,
    b_data: Tensor,
    b_scale: Tensor,
    output_dtype: torch.dtype,
    bias: Optional[Tensor] = None,
) -> Tensor:
    return _fp8_npu_matmul_impl(
        a_data, a_scale, b_data, b_scale, output_dtype, bias=bias
    )


def _fp8_npu_matmul_impl(
    a_data: Tensor,
    a_scale: Tensor,
    b_data: Tensor,
    b_scale: Tensor,
    output_dtype: torch.dtype,
    bias: Optional[Tensor] = None,
) -> Tensor:
    try:
        import torch_npu  # type: ignore
    except Exception:
        out = torch.mm(a_data.float() / a_scale, b_data.float() / b_scale).to(
            output_dtype
        )
        return out + bias if bias is not None else out

    a_is_hif8 = is_hifloat8_tensor(a_data)
    b_is_hif8 = is_hifloat8_tensor(b_data)
    a_is_hif4 = is_hifloat4_tensor(a_data)
    b_is_hif4 = is_hifloat4_tensor(b_data)
    a_is_std_fp8 = _is_standard_fp8_tensor(a_data)
    b_is_std_fp8 = _is_standard_fp8_tensor(b_data)
    std_fp8_path = False

    if a_is_hif8 and b_is_hif8:
        x1 = hifloat8_raw_int8(a_data)
        x2 = hifloat8_raw_int8(b_data)
        x1_dtype = torch_npu.hifloat8
        x2_dtype = torch_npu.hifloat8
    elif a_is_hif4 and b_is_hif4:
        x1 = hifloat4_raw_int8(a_data)
        x2 = hifloat4_raw_int8(b_data)
        x1_dtype = get_hifloat4_npu_dtype()
        x2_dtype = x1_dtype
    elif a_is_std_fp8 and b_is_std_fp8:
        std_fp8_path = True
        # For standard float8_e4m3fn/e5m2, op-plugin meta check requires
        # x1_dtype/x2_dtype to be None. Pass fp8 tensors directly.
        x1 = a_data
        x2 = b_data
        x1_dtype = None
        x2_dtype = None
    else:
        # Mixed fp8 types are unsupported by npu_quant_matmul today.
        out = torch.mm(a_data.float() / a_scale, b_data.float() / b_scale).to(
            output_dtype
        )
        return out + bias if bias is not None else out

    scale, pertoken_scale = hifloatx_scales_to_npu(
        a_scale, b_scale, out_features=b_data.shape[-1]
    )
    device = x1.device
    if scale.device != device:
        scale = scale.to(device)
    if pertoken_scale is not None and pertoken_scale.device != device:
        pertoken_scale = pertoken_scale.to(device)
    if bias is not None and bias.device != device:
        bias = bias.to(device)

    try:
        quant_matmul_kwargs = {
            "pertoken_scale": pertoken_scale,
            "bias": bias,
            "output_dtype": output_dtype,
        }
        if x1_dtype is not None:
            quant_matmul_kwargs["x1_dtype"] = x1_dtype
        if x2_dtype is not None:
            quant_matmul_kwargs["x2_dtype"] = x2_dtype
        return torch_npu.npu_quant_matmul(x1, x2, scale, **quant_matmul_kwargs)
    except Exception:
        if std_fp8_path:
            _warn_standard_fp8_npu_fallback_once()
        # Keep an FP32 fallback on NPU to avoid _scaled_mm/triton path.
        out = torch.mm(a_data.float() / a_scale, b_data.float() / b_scale).to(
            output_dtype
        )
        return out + bias if bias is not None else out


class Float8MMConfig(NamedTuple):
    """
    Configuration for the scaled_mm in the forward and backward pass.

    Attributes:
        emulate (bool): Whether to emulate the matmuls in fp32.
        use_fast_accum (bool): Whether to use the fast-accumulation option for scaled_mm.
        pad_inner_dim (bool): Whether to pad the inner dimension of a and b with 0s.
                              This is needed for matmuls not aligned to 16.
    """

    emulate: bool = False
    use_fast_accum: bool = False
    pad_inner_dim: bool = False


def preprocess_data(
    a_data: Tensor,
    b_data: Tensor,
    scaled_mm_config: Float8MMConfig,
) -> Tuple[Tensor, Tensor]:
    """Preprocess the inner fp8 data tensors for admmm
    Args:
        a_data: Input tensor A.
        b_data: Input tensor B.
        scaled_mm_config: Configuration for _scaled_mm.
    Returns:
        Preprocessed tensors A and B in the format for _scaled_mm.
    """
    if scaled_mm_config.pad_inner_dim:
        assert a_data.size(1) == b_data.size(0), (
            f"Inner dims must match for mm, got {a_data.size(1)} and {b_data.size(0)}"
        )
        a_data = pad_tensor_for_matmul(a_data, dims=1)
        b_data = pad_tensor_for_matmul(b_data, dims=0)
    if not is_row_major(a_data.stride()):
        a_data = a_data.contiguous()
    if is_row_major(b_data.stride()):
        b_data = b_data.t().contiguous().t()
    return a_data, b_data


def preprocess_scale(input_scale: torch.Tensor, input_shape: Tuple[int, ...]):
    """Ensures input tensor is correctly formatted for _scaled_mm"""

    # For PerTensor quantization, scale should be a scalar or have shape [1]
    if input_scale.numel() == 1:
        # Already a scalar, ensure it has the right shape for _scaled_mm
        return input_scale.reshape(1, 1)

    # For per-row/block quantization, we need to handle the reshaping
    input_scale = input_scale.unsqueeze(-1)

    # Match: #input_data.reshape(-1, input_data.shape[-1])
    if input_scale.dim() > 2:
        input_scale = input_scale.reshape(-1, input_scale.shape[-1])

    return input_scale


def addmm_float8_unwrapped_inference(
    a_data: Tensor,
    a_scale: Tensor,
    b_data: Tensor,
    b_scale: Tensor,
    output_dtype: torch.dtype,
    output_scale: Optional[Tensor] = None,
    bias: Optional[Tensor] = None,
    use_fast_accum: bool = False,
) -> Tensor:
    """
    This is the unwrapped version of addmm_float8, which does not take in Float8TrainingTensors
    as inputs. This is used to standardize the logic between subclassed and non subclassed
    versions of the linear module.
    """

    # On NPU, route fp8 matmul through npu_quant_matmul to avoid
    # torch._scaled_mm and potential triton paths.
    if _should_use_npu_fp8_matmul(a_data, b_data):
        return _fp8_npu_matmul(
            a_data, a_scale, b_data, b_scale, output_dtype, bias=bias
        )

    if output_dtype == torch.float32 and bias is not None:
        # Bias is not supported by _scaled_mm when output is fp32
        output = torch._scaled_mm(
            a_data,
            b_data,
            scale_a=a_scale,
            scale_b=b_scale,
            scale_result=output_scale,
            out_dtype=output_dtype,
            use_fast_accum=use_fast_accum,
        )
        return output + bias
    return torch._scaled_mm(
        a_data,
        b_data,
        scale_a=a_scale,
        scale_b=b_scale,
        bias=bias,
        scale_result=output_scale,
        out_dtype=output_dtype,
        use_fast_accum=use_fast_accum,
    )


def _slice_scale_for_dimension(
    scale: torch.Tensor,
    data_shape: List[int],
    dim: int,
    start: int,
    end: int,
    step: int,
) -> torch.Tensor:
    """
    Slice the scale tensor appropriately based on the data tensor slicing.
    This function calculates how the scale should be sliced when the data tensor
    is sliced along a given dimension, taking into account the block structure.
    """
    aten = torch.ops.aten

    # Unsupported case for now, this would be 1 scale per data element
    if scale.shape == data_shape:
        return aten.slice.Tensor(scale, dim, start, end, step)

    # Reconstruct block sizes based on data shape and scale shape
    block_sizes = tuple(data_shape[i] // scale.shape[i] for i in range(len(data_shape)))

    if dim >= len(block_sizes):
        # Slicing beyond the dimensions we care about
        return scale

    block_size_for_dim = block_sizes[dim]

    if block_size_for_dim == 1:
        # Scale is per-element along this dimension
        # Slice away as normal
        return aten.slice.Tensor(scale, dim, start, end, step)
    else:
        # There is blocking in this dimension
        # Calculate which scale elements correspond to the sliced data
        scale_start = start // block_size_for_dim if start is not None else None
        scale_end = (
            (end + block_size_for_dim - 1) // block_size_for_dim
            if end is not None
            else None
        )

        # Error on Step > 1
        if step > 1:
            raise NotImplementedError(
                "Slicing with step > 1 is not implemented for scale tensors."
            )

        return aten.slice.Tensor(scale, dim, scale_start, scale_end, 1)


def _is_rowwise_scaled(x: torch.Tensor) -> bool:
    """Checks if a quantized tensor is rowwise scaled
    Args:
        x: quantized tensor (should have `block_size` attribute)
    """
    assert hasattr(x, "block_size"), "Expecting input to have `block_size` attribute"
    return tuple(x.block_size) == (1,) * (x.dim() - 1) + (x.shape[-1],)


def _is_tensorwise_scaled(x: torch.Tensor) -> bool:
    """Checks if a quantized tensor is rowwise scaled
    Args:
        x: quantized tensor (should have `block_size` attribute)
    """
    assert hasattr(x, "block_size"), "Expecting input to have `block_size` attribute"
    return all(
        x.block_size[i] == -1 or x.block_size[i] == x.shape[i] for i in range(x.ndim)
    )


def _normalize_granularity(
    granularity: Optional[
        Union[
            FP8Granularity,
            Tuple[FP8Granularity, FP8Granularity],
            list[FP8Granularity],
        ]
    ],
) -> Tuple[FP8Granularity, FP8Granularity]:
    processed_granularity = None
    if granularity is None:
        processed_granularity = (PerTensor(), PerTensor())
    elif isinstance(granularity, (PerTensor, PerRow)):
        processed_granularity = (granularity, granularity)
    elif isinstance(granularity, (tuple, list)) and len(granularity) == 2:
        if not (
            isinstance(granularity[0], (PerTensor, PerRow))
            and isinstance(granularity[1], (PerTensor, PerRow))
        ):
            raise ValueError(
                f"Invalid granularity types: {granularity}, only PerTensor or PerRow are supported."
            )
        if not isinstance(granularity[0], type(granularity[1])):
            raise ValueError(
                f"Different granularities for activation and weight are not supported: {granularity}, only PerTensor or PerRow are supported."
            )
        processed_granularity = tuple(granularity)
    else:
        raise ValueError(
            f"Invalid granularity specification: {granularity}, only PerTensor or PerRow are supported."
        )
    return processed_granularity


def _check_hardware_support(
    granularities: Tuple[FP8Granularity, FP8Granularity],
) -> None:
    """
    Validate that the hardware supports the requested granularities.

    Args:
        granularities: Tuple of (activation_granularity, weight_granularity)

    Raises:
        AssertionError: If hardware doesn't support the requested granularity
        ValueError: If invalid granularity type is provided
    """
    from torchao.quantization.granularity import (
        PerRow,
        PerTensor,
    )

    is_per_tensor = isinstance(granularities[0], PerTensor) and isinstance(
        granularities[1], PerTensor
    )
    is_per_row = isinstance(granularities[0], PerRow) and isinstance(
        granularities[1], PerRow
    )
    is_a_1_128_w_128_128 = _granularity_is_a_1_128_w_128_128(granularities)

    if is_per_tensor or is_per_row:
        npu_available = False
        try:
            import torch_npu  # type: ignore

            npu_available = torch_npu.npu.is_available()
        except Exception:
            npu_available = False
        assert (
            npu_available
            or torch.xpu.is_available()
            or (torch.cuda.is_available() and is_sm_at_least_89())
            or is_MI300()
        ), (
            "Float8 dynamic quantization requires CUDA compute capability ≥8.9 or MI300+ or XPU or NPU."
        )
