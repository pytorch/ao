"""
Simple FP4 QAT for MoE (Mixture of Experts) models using a tensor subclass.

This is a simplified version of :mod:`nvfp4_moe` that applies a straightforward
FP4 E2M1 fake quantization (per-tensor scale + round to nearest E2M1 value)
before each ``torch._grouped_mm`` call, **without** trying to match the exact
numerics of the flashinfer ``trtllm_fp4_block_scale_moe`` kernel.

Differences from :mod:`nvfp4_moe`:

- **Per-tensor scaling** (``amax / FP4_MAX``) instead of the two-level
  global + FP8-block-scale scheme that mirrors the NvFP4 kernel format.
- **No per-expert quantization** — the entire activation or weight tensor
  is quantized with a single scale factor, rather than computing an
  independent global scale for each expert slice.
- **No ``is_transposed`` handling** — no layout fixups to align FP4 block
  boundaries with the kernel's storage-layout quantization.

For the version that closely matches the flashinfer kernel numerics, see
:mod:`nvfp4_moe`.

Usage::

    from torchao.prototype.qat.nvfp4_moe_simple import (
        apply_simple_fp4_moe_qat,
        remove_simple_fp4_moe_qat,
    )
    model = apply_simple_fp4_moe_qat(model)
    # ... train ...
    model = remove_simple_fp4_moe_qat(model)
"""

import logging

import torch
import torch.nn as nn
import torch.utils._pytree as pytree
from torch._prims_common import suggest_memory_format

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# FP4 E2M1 quantization (block-scale, matching NvFP4 kernel numerics)
# ---------------------------------------------------------------------------

# Representable positive E2M1 values: 0, 0.5, 1, 1.5, 2, 3, 4, 6
_E2M1_VALUES = [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0]
# Midpoint boundaries between consecutive values (for round-to-nearest)
_E2M1_BOUNDARIES = [0.25, 0.75, 1.25, 1.75, 2.5, 3.5, 5.0]
_SF_BLOCK_SIZE = 16


def _fp4_quant_dequant(x: torch.Tensor) -> torch.Tensor:
    """FP4 E2M1 block-scale quantize-dequantize roundtrip.

    Uses two levels of scaling to map values into the FP4 E2M1 representable
    range, matching the NvFP4 kernel numerics:

    **Level 1 — Global scale factor (per-tensor):**
    Computed as ``(448 * 6) / amax(x)``, where 448 is the max FP8-E4M3 value
    and 6 is the max FP4-E2M1 value. This maps the tensor's dynamic range
    into the combined FP8 * FP4 representable range.

    **Level 2 — Block scale factor (per block of 16 elements):**
    The globally-scaled tensor is reshaped into blocks of 16. Each block gets
    its own FP8-E4M3 scale factor: ``block_amax / 6.0``, quantized to FP8.
    This adapts precision to local magnitude variation within each block.

    After both scales are applied, each element is rounded to the nearest
    E2M1 value (0, 0.5, 1, 1.5, 2, 3, 4, 6), then dequantized by reversing
    both scale factors.

    For 3D tensors (expert weights), quantizes each expert slice independently
    to avoid OOM from materializing the full tensor in float32.

    Returns the dequantized tensor in the same dtype as *x*.
    """
    if x.ndim == 3:
        out = torch.empty_like(x)
        for i in range(x.shape[0]):
            out[i] = _fp4_quant_dequant(x[i])
        return out

    device = x.device
    orig_dtype = x.dtype

    # Level 1: global scale factor maps tensor range into FP8*FP4 range
    gsf = (448 * 6) / x.float().abs().nan_to_num().max()
    x = x.float() * gsf

    # Level 2: per-block FP8 scale factor adapts to local magnitudes
    orig_shape = x.shape
    x_flat = x.reshape(-1, _SF_BLOCK_SIZE)
    block_amax = x_flat.abs().amax(dim=-1, keepdim=True)
    block_sf = (block_amax / 6.0).to(torch.float8_e4m3fn).float()

    # Round to nearest E2M1 value
    x_norm = x_flat / block_sf.clamp(min=torch.finfo(torch.float8_e4m3fn).tiny)
    sign = x_norm.sign()
    x_abs = x_norm.abs()
    boundaries = torch.tensor(_E2M1_BOUNDARIES, device=device)
    values = torch.tensor(_E2M1_VALUES, device=device)
    indices = torch.bucketize(x_abs, boundaries)
    x_quant = sign * values[indices]

    # Dequantize: reverse block scale, reshape, reverse global scale
    x_dq = (x_quant * block_sf).reshape(orig_shape) / gsf
    return x_dq.to(orig_dtype)


def _fp4_fake_quantize(x: torch.Tensor) -> torch.Tensor:
    """FP4 fake quantize with STE (straight-through estimator).

    Forward: returns the dequantized value.
    Backward: treats quantization as identity (STE).
    """
    dq = _fp4_quant_dequant(x.detach())
    return x + (dq - x).detach()


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


class _SimpleFP4FakeQuantGroupedMM(torch.autograd.Function):
    """Autograd function that applies simple per-tensor FP4 fake quantization
    in the forward pass of ``torch._grouped_mm`` and uses STE for the backward.

    Both activation and weight are fake-quantized with a single per-tensor
    scale before each grouped GEMM.
    """

    @staticmethod
    def forward(ctx, A, B_wrapper, offs):
        B_data = (
            B_wrapper._data
            if isinstance(B_wrapper, SimpleFP4FakeQuantizedGroupedMMTensor)
            else B_wrapper
        )
        ctx.save_for_backward(A, B_data, offs)

        A_fq = _fp4_quant_dequant(A)
        B_fq = _fp4_quant_dequant(B_data)

        return torch._grouped_mm(A_fq, B_fq, offs=offs)

    @staticmethod
    def backward(ctx, grad_output):
        A, B_data, offs = ctx.saved_tensors
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


class SimpleFP4FakeQuantizedGroupedMMTensor(torch.Tensor):
    """Tensor subclass that intercepts ``torch._grouped_mm`` and injects
    simple per-tensor FP4 E2M1 fake quantization on both weight and
    activation operands.

    Uses a single per-tensor scale (``amax / 6.0``) instead of the two-level
    global + FP8-block-scale scheme in
    :class:`~torchao.prototype.qat.nvfp4_moe.NVFP4FakeQuantizedScaledGroupedMMTensor`.
    Does not attempt to match any specific inference kernel's numerics.

    Follows the same interception pattern as
    :class:`torchao.prototype.moe_training.tensor.ScaledGroupedMMTensor`.
    """

    grouped_mm_func_name = "_grouped_mm"
    offs_arg_name = "offs"

    @staticmethod
    def __new__(cls, tensor: torch.Tensor):
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

    def __init__(self, tensor: torch.Tensor):
        self._data = tensor

    @classmethod
    def __torch_function__(cls, func, types, args, kwargs={}):
        if func.__name__ == cls.grouped_mm_func_name:
            A, B = args[0], args[1]
            A_is_2d = A.ndim == 2
            B_is_2d_or_3d = B.ndim == 2 or B.ndim == 3
            has_offs = kwargs.get(cls.offs_arg_name) is not None

            if (
                isinstance(B, SimpleFP4FakeQuantizedGroupedMMTensor)
                and A_is_2d
                and B_is_2d_or_3d
                and has_offs
            ):
                return _SimpleFP4FakeQuantGroupedMM.apply(
                    A, B, kwargs[cls.offs_arg_name]
                )

        with torch._C.DisableTorchFunctionSubclass():
            return func(*args, **kwargs)

    @classmethod
    def __torch_dispatch__(cls, func, types, args, kwargs={}):
        def unwrap(t):
            return t._data

        args_unwrapped, kwargs_unwrapped = pytree.tree_map_only(
            SimpleFP4FakeQuantizedGroupedMMTensor, unwrap, (args, kwargs or {})
        )

        if func == torch.ops.aten.detach.default:
            return SimpleFP4FakeQuantizedGroupedMMTensor(args_unwrapped[0])

        out = func(*args_unwrapped, **kwargs_unwrapped)

        if func not in _ops_to_preserve_subclass:
            return out

        return pytree.tree_map_only(
            torch.Tensor,
            lambda x: SimpleFP4FakeQuantizedGroupedMMTensor(x),
            out,
        )

    def __repr__(self):
        return (
            f"SimpleFP4FakeQuantizedGroupedMMTensor("
            f"data={self._data})"
        )

    def __tensor_flatten__(self):
        return ["_data"], {}

    @staticmethod
    def __tensor_unflatten__(inner_tensors, flatten_spec, outer_size, outer_stride):
        return SimpleFP4FakeQuantizedGroupedMMTensor(inner_tensors["_data"])


# ---------------------------------------------------------------------------
# Model transforms
# ---------------------------------------------------------------------------


def apply_simple_fp4_moe_qat(model: nn.Module) -> nn.Module:
    """Wrap MoE expert weight parameters with simple FP4 fake-quantized tensor subclass.

    Walks all modules and wraps 3D parameters on modules that have a
    ``num_experts`` attribute.  Unlike :func:`~torchao.prototype.qat.nvfp4_moe.apply_nvfp4_moe_qat`,
    this does not require or use ``is_transposed``.

    Args:
        model: A model whose MoE expert modules have a ``num_experts``
            attribute and 3D weight parameters.

    Returns:
        The same model, modified in-place.
    """
    for module in model.modules():
        if not hasattr(module, "num_experts"):
            continue
        for param_name, param in module.named_parameters(recurse=False):
            if param.ndim == 3 and not isinstance(
                param.data, SimpleFP4FakeQuantizedGroupedMMTensor
            ):
                logger.info(
                    "Replacing param %s (%s) with SimpleFP4FakeQuantizedGroupedMMTensor",
                    param_name,
                    param.shape,
                )
                new_data = SimpleFP4FakeQuantizedGroupedMMTensor(param.data)
                new_param = nn.Parameter(new_data, requires_grad=param.requires_grad)
                setattr(module, param_name, new_param)
    return model


def remove_simple_fp4_moe_qat(model: nn.Module) -> nn.Module:
    """Unwrap simple FP4 fake-quantized tensor subclass back to plain tensors.

    Args:
        model: A model previously modified by :func:`apply_simple_fp4_moe_qat`.

    Returns:
        The same model, modified in-place.
    """
    for module in model.modules():
        for param_name, param in list(module.named_parameters(recurse=False)):
            if isinstance(param.data, SimpleFP4FakeQuantizedGroupedMMTensor):
                new_param = nn.Parameter(
                    param.data._data, requires_grad=param.requires_grad
                )
                setattr(module, param_name, new_param)
    return model


# ---------------------------------------------------------------------------
# Torchtitan integration (sets fake_quant_fn on GroupedExperts modules)
# ---------------------------------------------------------------------------


def _fp4_fake_quantize_ste(x: torch.Tensor) -> torch.Tensor:
    """FP4 block-scale fake quantize with STE for use in torchtitan's
    GroupedExperts.fake_quant_fn. Applies quant-dequant to the detached
    tensor and uses straight-through estimator for the backward pass."""
    dq = _fp4_quant_dequant(x.detach())
    return x + (dq - x).detach()


def apply_simple_fp4_moe_qat_torchtitan(model: nn.Module) -> nn.Module:
    """Enable FP4 fake quantization on torchtitan GroupedExperts modules.

    Sets ``fake_quant_fn`` on each module that has both a ``num_experts``
    attribute and a ``fake_quant_fn`` attribute (torchtitan's GroupedExperts).
    This injects FP4 block-scale fake quantization on both weights and
    activations inside ``_experts_forward``, compatible with FSDP2/DTensor.

    Args:
        model: A torchtitan model with GroupedExperts modules.

    Returns:
        The same model, modified in-place.
    """
    count = 0
    for module in model.modules():
        if hasattr(module, "num_experts") and hasattr(module, "fake_quant_fn"):
            module.fake_quant_fn = _fp4_fake_quantize_ste
            count += 1
            logger.info(
                "Enabled FP4 fake quantization on %s (num_experts=%d)",
                type(module).__name__,
                module.num_experts,
            )
    logger.info("Applied FP4 QAT to %d GroupedExperts modules", count)
    return model


def apply_fp4_fake_quant_to_vllm_moe(model: nn.Module) -> None:
    """Monkey-patch FusedMoE layers in a vLLM model to apply NVFP4 fake
    quantization on activations during forward.

    Wraps each ``FusedMoE`` layer's ``quant_method.apply()`` so that input
    hidden states are quantize-dequantized (FP4 E2M1 block-scale) before
    entering the fused MoE kernel. Weight fake-quantization is handled
    separately by :func:`fake_quantize_vllm_moe_weights_inplace`, which
    should be called once after each weight sync.

    This is the vLLM-rollout counterpart of
    :func:`apply_simple_fp4_moe_qat_torchtitan` (which patches torchtitan's
    ``GroupedExperts``).
    """
    from vllm.model_executor.layers.fused_moe.layer import FusedMoE

    def _make_patched_apply(orig_apply):
        def patched_apply(layer, x, topk_weights, topk_ids,
                          shared_experts_input):
            x = _fp4_quant_dequant(x)
            return orig_apply(
                layer=layer, x=x, topk_weights=topk_weights,
                topk_ids=topk_ids,
                shared_experts_input=shared_experts_input,
            )
        return patched_apply

    count = 0
    for module in model.modules():
        if isinstance(module, FusedMoE):
            module.quant_method.apply = _make_patched_apply(
                module.quant_method.apply
            )
            count += 1
    logger.info(
        "Patched %d FusedMoE layers with activation FP4 fake quant for rollout",
        count,
    )


def fake_quantize_vllm_moe_weights_inplace(model: nn.Module) -> None:
    """Replace MoE expert weights with their FP4 fake-quantized versions.

    Should be called once after each weight sync (in
    ``update_weights_from_ipc``), before rollout generation begins. Since
    weights don't change during rollout, this is equivalent to per-forward
    fake-quantization but avoids the per-call clone+copy overhead.

    The next weight sync will overwrite these with fresh actor weights,
    so no cleanup is needed.
    """
    from vllm.model_executor.layers.fused_moe.layer import FusedMoE

    count = 0
    for module in model.modules():
        if isinstance(module, FusedMoE):
            module.w13_weight.data.copy_(
                _fp4_quant_dequant(module.w13_weight.data)
            )
            module.w2_weight.data.copy_(
                _fp4_quant_dequant(module.w2_weight.data)
            )
            count += 1
    logger.info(
        "Fake-quantized weights in %d FusedMoE layers in-place", count
    )


def remove_simple_fp4_moe_qat_torchtitan(model: nn.Module) -> nn.Module:
    """Remove FP4 fake quantization from torchtitan GroupedExperts modules.

    Args:
        model: A model previously modified by
            :func:`apply_simple_fp4_moe_qat_torchtitan`.

    Returns:
        The same model, modified in-place.
    """
    for module in model.modules():
        if hasattr(module, "fake_quant_fn") and module.fake_quant_fn is not None:
            module.fake_quant_fn = None
    return model
