"""
FP4 QAT for MoE (Mixture of Experts) models using a tensor subclass.

This module provides a **model-agnostic** approach to NVFP4 fake quantization
for MoE expert layers.  It intercepts ``torch._grouped_mm`` via a tensor
subclass (following the same pattern as
:class:`torchao.prototype.moe_training.tensor.ScaledGroupedMMTensor`) and
injects block-scale FP4 (E2M1) quantize-dequantize roundtrips on both
activation and weight operands before each grouped GEMM.

This works with any HuggingFace MoE model that uses the ``grouped_mm``
expert backend (``experts_implementation="grouped_mm"``).

Usage::

    from torchao.prototype.qat.nvfp4_moe import apply_nvfp4_moe_qat, remove_nvfp4_moe_qat
    model = apply_nvfp4_moe_qat(model)
    # ... train ...
    model = remove_nvfp4_moe_qat(model)
"""

import logging

import torch
import torch.nn as nn
import torch.utils._pytree as pytree
from torch._prims_common import suggest_memory_format

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# FP4 E2M1 quantization (pure PyTorch, all on GPU)
# ---------------------------------------------------------------------------

# Representable positive E2M1 values: 0, 0.5, 1, 1.5, 2, 3, 4, 6
_E2M1_VALUES = [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0]
# Midpoint boundaries between consecutive values (for round-to-nearest)
_E2M1_BOUNDARIES = [0.25, 0.75, 1.25, 1.75, 2.5, 3.5, 5.0]
_SF_BLOCK_SIZE = 16


def _calculate_fp4_global_scale_factor(tensor: torch.Tensor) -> torch.Tensor:
    """Compute FP4 global scale: ``(448 * 6) / amax``.

    448 is the max representable FP8-E4M3 value and 6 is the max FP4-E2M1
    value; their product bounds the NvFP4 dynamic range.
    """
    return (448 * 6) / tensor.float().abs().nan_to_num().max()


def _quant_dequant_fp4(
    a: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """FP4 E2M1 block-scale quantize-dequantize roundtrip, all on GPU.

    Implements the same numerics as flashinfer's ``fp4_quantize`` followed by
    ``e2m1_and_ufp8sf_scale_to_float``, but without any CPU↔GPU transfers.

    Returns ``(dequantized, global_scale_factor)``.
    """
    device = a.device
    gsf = _calculate_fp4_global_scale_factor(a)

    # Scale by global scale factor.
    x = a.float() * gsf
    orig_shape = x.shape
    x_flat = x.reshape(-1, _SF_BLOCK_SIZE)

    # Per-block FP8-E4M3 scale factor: block_amax / fp4_max, quantized to FP8.
    block_amax = x_flat.abs().amax(dim=-1, keepdim=True)
    block_sf = (block_amax / 6.0).to(torch.float8_e4m3fn).float()

    # Normalize by block scale and round to nearest E2M1 value.
    x_norm = x_flat / block_sf.clamp(min=torch.finfo(torch.float8_e4m3fn).tiny)
    sign = x_norm.sign()
    x_abs = x_norm.abs()
    boundaries = torch.tensor(_E2M1_BOUNDARIES, device=device)
    values = torch.tensor(_E2M1_VALUES, device=device)
    indices = torch.bucketize(x_abs, boundaries)
    x_quant = sign * values[indices]

    # Dequantize: reverse the block scale and global scale.
    x_dq = (x_quant * block_sf).reshape(orig_shape) / gsf

    return x_dq, gsf


def _fp4_fake_quantize(x: torch.Tensor) -> torch.Tensor:
    """FP4 fake quantize with STE, using block-scale FP4 (E2M1).

    Matches the quantisation the ``trtllm_fp4_block_scale_moe`` kernel
    applies to activations and weights.  The forward pass uses the
    dequantized value; the backward treats quantisation as identity (STE).
    """
    dq_val, _ = _quant_dequant_fp4(x.to(torch.bfloat16).detach())
    return x + (dq_val.to(x.dtype) - x).detach()


# ---------------------------------------------------------------------------
# Ops that should preserve the tensor subclass wrapper
# ---------------------------------------------------------------------------

_ops_to_preserve_subclass = {
    torch.ops.aten.empty_like.default,
    torch.ops.aten.new_zeros.default,
    torch.ops.aten.slice.Tensor,
    torch.ops.aten.copy_.default,
    torch.ops.aten.view.default,
    torch.ops.aten.as_strided.default,
    torch.ops.aten._to_copy.default,
    torch.ops.aten._pin_memory.default,
    torch.ops.aten.split.Tensor,
    torch.ops.aten.clone.default,
    torch.ops.aten.transpose.int,
}


# ---------------------------------------------------------------------------
# Autograd function for fake-quantized grouped MM with STE
# ---------------------------------------------------------------------------


def _fp4_fake_quantize_forward(x: torch.Tensor) -> torch.Tensor:
    """Forward-only FP4 fake quantize (no STE, for use inside autograd.Function).

    Quantizes the entire tensor with a single global scale factor.
    """
    dq_val, _ = _quant_dequant_fp4(x.to(torch.bfloat16))
    return dq_val.to(x.dtype)


def _fp4_fake_quantize_per_expert(w: torch.Tensor) -> torch.Tensor:
    """Forward-only FP4 fake quantize with an independent global scale per expert.

    This matches the flashinfer ``trtllm_fp4_block_scale_moe`` kernel numerics
    where each expert's weight tensor is quantized independently (its own
    ``amax`` and global scale factor).

    Args:
        w: 3D weight tensor of shape ``[E, ...]`` where ``E`` is the expert
           dimension.
    """
    out = torch.empty_like(w)
    for i in range(w.shape[0]):
        dq_val, _ = _quant_dequant_fp4(w[i].to(torch.bfloat16))
        out[i] = dq_val.to(w.dtype)
    return out


class _NVFP4FakeQuantGroupedMM(torch.autograd.Function):
    """Autograd function that applies NVFP4 fake quantization in the forward
    pass of ``torch._grouped_mm`` and uses STE (straight-through estimator)
    for the backward pass.

    Both activation and weight are fake-quantized for every grouped GEMM.
    The activation uses a single global scale across all tokens; weights use
    an independent global scale per expert.

    **Weight quantization layout**: When ``is_transposed=False``, HF
    transposes the weight before calling ``_grouped_mm``, so the subclass
    sees the GEMM-ready layout rather than the storage layout.  To match
    the kernel's FP4 block boundaries (which are along the last dimension
    of the *storage* layout), we transpose back to storage layout before
    quantizing, then transpose the result back for the GEMM.

    The backward computes gradients as if no quantization happened (STE).
    """

    @staticmethod
    def forward(ctx, A, B_wrapper, offs):
        # B_wrapper is the NVFP4... subclass; unwrap to get the raw tensor.
        B_data = (
            B_wrapper._data
            if isinstance(B_wrapper, NVFP4FakeQuantizedScaledGroupedMMTensor)
            else B_wrapper
        )
        is_transposed = (
            B_wrapper._is_transposed
            if isinstance(B_wrapper, NVFP4FakeQuantizedScaledGroupedMMTensor)
            else False
        )
        ctx.save_for_backward(A, B_data, offs)

        # Weight: independent global scale per expert.
        # When is_transposed=False, HF has already transposed the weight
        # from storage layout [E, out, in] to GEMM layout [E, in, out].
        # We transpose back so FP4 block boundaries match the kernel's
        # storage-layout quantization, then transpose the result back.
        if not is_transposed and B_data.ndim == 3:
            B_fq = _fp4_fake_quantize_per_expert(
                B_data.transpose(-1, -2)
            ).transpose(-1, -2)
        else:
            B_fq = _fp4_fake_quantize_per_expert(B_data)

        # Activation: single global scale across all tokens.
        A_fq = _fp4_fake_quantize_forward(A)

        return torch._grouped_mm(A_fq, B_fq, offs=offs)

    @staticmethod
    def backward(ctx, grad_output):
        A, B_data, offs = ctx.saved_tensors
        # STE backward: compute gradients of grouped_mm(A, B) per group.
        # For group i with rows [start_i, end_i):
        #   C_i = A_i @ B_i          (forward)
        #   dA_i = dC_i @ B_i^T      (grad w.r.t. activation)
        #   dB_i = A_i^T @ dC_i      (grad w.r.t. weight)
        grad_A = torch.zeros_like(A)
        grad_B = torch.zeros_like(B_data)
        prev = 0
        for i, end in enumerate(offs.tolist()):
            if end > prev:
                grad_A[prev:end] = grad_output[prev:end] @ B_data[i].transpose(-1, -2)
                grad_B[i] = A[prev:end].transpose(-1, -2) @ grad_output[prev:end]
            prev = end
        return grad_A, grad_B, None


# ---------------------------------------------------------------------------
# Tensor subclass
# ---------------------------------------------------------------------------


class NVFP4FakeQuantizedScaledGroupedMMTensor(torch.Tensor):
    """Tensor subclass that intercepts ``torch._grouped_mm`` and injects
    NVFP4 fake quantization on both weight and activation operands, matching
    the numerics of the flashinfer ``trtllm_fp4_block_scale_moe`` fused kernel.

    Follows the same pattern as
    :class:`torchao.prototype.moe_training.tensor.ScaledGroupedMMTensor`.

    Args:
        tensor: The underlying weight tensor (3D, shape ``[E, ...]``).
        is_transposed: Whether the weight is stored in transposed
            (GEMM-ready) layout — i.e. ``[E, input_dim, output_dim]``.
            Mirrors the ``is_transposed`` flag from the HuggingFace
            ``@use_experts_implementation`` decorator.

            * ``False`` (default, e.g. Qwen3): weights are stored as
              ``[E, output_dim, input_dim]`` and HF transposes them before
              ``_grouped_mm``.  The subclass will transpose back to storage
              layout before quantizing, then transpose the result, so that
              FP4 block boundaries match the kernel.
            * ``True``: weights are already in GEMM-ready layout and are
              passed to ``_grouped_mm`` without transposing.  No layout
              fixup is needed.
    """

    grouped_mm_func_name = "_grouped_mm"
    offs_arg_name = "offs"

    @staticmethod
    def __new__(
        cls,
        tensor: torch.Tensor,
        is_transposed: bool = False,
    ):
        return torch.Tensor._make_wrapper_subclass(
            cls,
            tensor.size(),
            strides=tensor.stride(),
            storage_offset=tensor.storage_offset(),
            memory_format=suggest_memory_format(tensor),
            dtype=tensor.dtype,
            layout=tensor.layout,
            device=tensor.device,
            pin_memory=tensor.is_pinned(),
            requires_grad=tensor.requires_grad,
        )

    def __init__(
        self,
        tensor: torch.Tensor,
        is_transposed: bool = False,
    ):
        self._data = tensor
        self._is_transposed = is_transposed

    @classmethod
    def __torch_function__(cls, func, types, args, kwargs={}):
        if func.__name__ == cls.grouped_mm_func_name:
            A, B = args[0], args[1]
            A_is_2d = A.ndim == 2
            B_is_2d_or_3d = B.ndim == 2 or B.ndim == 3
            has_offs = kwargs.get(cls.offs_arg_name) is not None

            if (
                isinstance(B, NVFP4FakeQuantizedScaledGroupedMMTensor)
                and A_is_2d
                and B_is_2d_or_3d
                and has_offs
            ):
                return _NVFP4FakeQuantGroupedMM.apply(
                    A, B, kwargs[cls.offs_arg_name]
                )

        # Fall through for all other ops.
        with torch._C.DisableTorchFunctionSubclass():
            return func(*args, **kwargs)

    @classmethod
    def __torch_dispatch__(cls, func, types, args, kwargs={}):
        # Unwrap all subclass instances and extract metadata.
        is_transposed = False

        def unwrap(t):
            nonlocal is_transposed
            is_transposed = t._is_transposed
            return t._data

        args_unwrapped, kwargs_unwrapped = pytree.tree_map_only(
            NVFP4FakeQuantizedScaledGroupedMMTensor, unwrap, (args, kwargs or {})
        )

        # detach is a special case — always rewrap.
        if func == torch.ops.aten.detach.default:
            return NVFP4FakeQuantizedScaledGroupedMMTensor(
                args_unwrapped[0], is_transposed
            )

        out = func(*args_unwrapped, **kwargs_unwrapped)

        if func not in _ops_to_preserve_subclass:
            return out

        return pytree.tree_map_only(
            torch.Tensor,
            lambda x: NVFP4FakeQuantizedScaledGroupedMMTensor(
                x, is_transposed
            ),
            out,
        )

    def __repr__(self):
        return (
            f"NVFP4FakeQuantizedScaledGroupedMMTensor("
            f"data={self._data}, "
            f"is_transposed={self._is_transposed})"
        )

    def __tensor_flatten__(self):
        return ["_data"], {
            "is_transposed": self._is_transposed,
        }

    @staticmethod
    def __tensor_unflatten__(inner_tensors, flatten_spec, outer_size, outer_stride):
        return NVFP4FakeQuantizedScaledGroupedMMTensor(
            inner_tensors["_data"],
            flatten_spec["is_transposed"],
        )


# ---------------------------------------------------------------------------
# Model transforms
# ---------------------------------------------------------------------------


def apply_nvfp4_moe_qat(model: nn.Module) -> nn.Module:
    """Wrap MoE expert weight parameters with NVFP4 fake-quantized tensor subclass.

    This enables QAT (quantization-aware training) that matches the numerics of
    the flashinfer ``trtllm_fp4_block_scale_moe`` fused inference kernel.

    Requires a HuggingFace MoE model whose expert classes are decorated with
    ``@use_experts_implementation`` and loaded with
    ``experts_implementation="grouped_mm"``.  This covers most popular MoE
    architectures in HF transformers, including Qwen3-MoE, Qwen2-MoE,
    DeepSeek-V2/V3, Mixtral, OLMoE, Jamba, PhiMoE, and GLM4-MoE.

    Notable exception: Llama4 does **not** use ``@use_experts_implementation``
    (it dispatches via ``torch.bmm`` instead of ``torch._grouped_mm``) and
    is not supported yet.

    Args:
        model: A HuggingFace MoE model loaded with
            ``experts_implementation="grouped_mm"``.

    Returns:
        The same model, modified in-place.
    """
    for module in model.modules():
        if not hasattr(module, "num_experts"):
            continue
        # All HF @use_experts_implementation classes expose is_transposed.
        # If it's missing, this module likely doesn't use the grouped_mm
        # backend (e.g. Llama4 uses torch.bmm), so skip it.
        if not hasattr(module, "is_transposed"):
            logger.warning(
                "Skipping module %s: has num_experts but no is_transposed "
                "attribute (not decorated with @use_experts_implementation?)",
                type(module).__name__,
            )
            continue
        is_transposed = module.is_transposed
        for param_name, param in module.named_parameters(recurse=False):
            if param.ndim == 3 and not isinstance(
                param.data, NVFP4FakeQuantizedScaledGroupedMMTensor
            ):
                new_data = NVFP4FakeQuantizedScaledGroupedMMTensor(
                    param.data, is_transposed
                )
                new_param = nn.Parameter(new_data, requires_grad=param.requires_grad)
                setattr(module, param_name, new_param)
    return model


def remove_nvfp4_moe_qat(model: nn.Module) -> nn.Module:
    """Unwrap NVFP4 fake-quantized tensor subclass back to plain tensors.

    Args:
        model: A model previously modified by :func:`apply_nvfp4_moe_qat`.

    Returns:
        The same model, modified in-place.
    """
    for module in model.modules():
        for param_name, param in list(module.named_parameters(recurse=False)):
            if isinstance(param.data, NVFP4FakeQuantizedScaledGroupedMMTensor):
                new_param = nn.Parameter(
                    param.data._data, requires_grad=param.requires_grad
                )
                setattr(module, param_name, new_param)
    return model
