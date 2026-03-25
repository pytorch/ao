# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD 3-Clause license found in the
# LICENSE file in the root directory of this source tree.

"""
Shared FX graph pattern detection and fusion utilities for low-precision attention.
"""

import logging
import operator
from dataclasses import dataclass
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.fx import Graph, Node

logger = logging.getLogger(__name__)


@dataclass
class RoPEMatch:
    """Result of detecting a RoPE pattern on a tensor."""

    pre_rope_input: Node  # tensor before RoPE: "x" in "x * cos + rotate_half(x) * sin"
    cos_node: Node  # cosine frequencies, traced back to [S, D] source
    sin_node: Node  # sine frequencies, traced back to [S, D] source
    rope_interleaved: bool  # True = FLUX interleaved, False = NeoX half-split


# FX Node Utilities


def _is_op(node: Node, *targets) -> bool:
    """Check if an FX node matches one of the given targets."""
    if node.op in ("call_function", "call_method"):
        return node.target in targets
    return False


def _get_fake_tensor(node: Node) -> Optional[torch.Tensor]:
    """Get the FakeTensor metadata from a node (pre-grad or post-grad)."""
    for key in ("val", "example_value"):
        if key in node.meta:
            val = node.meta[key]
            if isinstance(val, torch.Tensor):
                return val
    return None


def _get_node_shape(node: Node) -> Optional[Tuple[int, ...]]:
    fake = _get_fake_tensor(node)
    if fake is not None:
        return tuple(fake.shape)
    return None


def _reshape_cos_sin_to_2d(
    graph: Graph,
    cos_node: Node,
    sin_node: Node,
    insert_before: Node,
) -> Optional[Tuple[Node, Node]]:
    """Reshape cos/sin nodes to 2D [S, D] if they have leading size-1 dims.

    HuggingFace models produce cos/sin with shape [B, S, D] or [1, 1, S, D].
    """
    cos_shape = _get_node_shape(cos_node)
    sin_shape = _get_node_shape(sin_node)

    if cos_shape is None or sin_shape is None:
        return cos_node, sin_node

    if len(cos_shape) == 2 and len(sin_shape) == 2:
        return cos_node, sin_node

    for name, shape in [("cos", cos_shape), ("sin", sin_shape)]:
        if len(shape) < 2:
            logger.debug("RoPE %s has fewer than 2 dims: shape=%s", name, shape)
            return None
        non_unit = [d for d in shape if d != 1]
        if len(non_unit) != 2:
            logger.debug(
                "RoPE %s cannot be squeezed to 2D: shape=%s",
                name,
                shape,
            )
            return None

    cos_non_unit = [d for d in cos_shape if d != 1]
    s, d = cos_non_unit[0], cos_non_unit[1]
    with graph.inserting_before(insert_before):
        cos_2d = graph.call_function(
            torch.ops.aten.view.default,
            args=(cos_node, [s, d]),
        )
        sin_2d = graph.call_function(
            torch.ops.aten.view.default,
            args=(sin_node, [s, d]),
        )
    return cos_2d, sin_2d


def _trace_through_views(node: Node) -> Node:
    """Trace backward through view-like ops (unsqueeze, expand, clone, to.dtype, etc.)."""
    current = node
    while isinstance(current, Node):
        if current.op == "call_function" and current.target in (
            torch.ops.aten.unsqueeze.default,
            torch.ops.aten.clone.default,
            torch.ops.aten.contiguous.default,
            torch.ops.aten.expand.default,
            torch.ops.aten.to.dtype,
            torch.ops.aten._to_copy.default,
        ):
            current = current.args[0]
        elif current.op == "call_method" and current.target in (
            "unsqueeze",
            "clone",
            "contiguous",
            "expand",
            "to",
            "type_as",
            "float",
            "half",
            "bfloat16",
        ):
            current = current.args[0]
        elif (
            current.op == "call_function"
            and current.target is operator.getitem
            and len(current.args) >= 2
            and isinstance(current.args[1], tuple)
            and all(i is None or i == slice(None) for i in current.args[1])
        ):
            current = current.args[0]
        else:
            break
    return current


# Transpose Detection


def _unwrap_transpose(node: Node) -> Optional[Node]:
    """If node is transpose(1,2) or permute([0,2,1,3]), return its input.

    Also looks through contiguous()/clone() wrappers.
    """
    if not isinstance(node, Node):
        return None

    current = node
    while isinstance(current, Node) and _is_op(
        current,
        torch.ops.aten.contiguous.default,
        torch.ops.aten.clone.default,
        "contiguous",
        "clone",
    ):
        current = current.args[0]

    if not isinstance(current, Node):
        return None

    # aten.transpose.int(tensor, 1, 2)
    if _is_op(current, torch.ops.aten.transpose.int):
        if len(current.args) >= 3:
            dim0, dim1 = current.args[1], current.args[2]
            if (dim0 == 1 and dim1 == 2) or (dim0 == 2 and dim1 == 1):
                return current.args[0]

    # aten.permute.default(tensor, [0, 2, 1, 3])
    if _is_op(current, torch.ops.aten.permute.default):
        if len(current.args) >= 2:
            perm = current.args[1]
            if list(perm) == [0, 2, 1, 3]:
                return current.args[0]

    # call_method transpose(1, 2)
    if _is_op(current, "transpose"):
        if len(current.args) >= 3:
            dim0, dim1 = current.args[1], current.args[2]
            if (dim0 == 1 and dim1 == 2) or (dim0 == 2 and dim1 == 1):
                return current.args[0]

    # call_method permute(0, 2, 1, 3)
    if _is_op(current, "permute"):
        if len(current.args) >= 5:
            perm = list(current.args[1:5])
            if perm == [0, 2, 1, 3]:
                return current.args[0]
        elif len(current.args) >= 2 and isinstance(current.args[1], (list, tuple)):
            perm = current.args[1]
            if list(perm) == [0, 2, 1, 3]:
                return current.args[0]

    return None


def _unwrap_repeat_kv(node: Node) -> Optional[Node]:
    """Unwrap HuggingFace's repeat_kv (GQA head repetition) pattern.

    Pattern: reshape <- expand <- getitem(unsqueeze).
    Returns the pre-repetition tensor, or None.
    """
    if not isinstance(node, Node):
        return None

    if not _is_op(
        node,
        torch.ops.aten.reshape.default,
        torch.ops.aten.view.default,
        "reshape",
        "view",
    ):
        return None

    inner = node.args[0] if node.args else None
    if not isinstance(inner, Node):
        return None

    if not _is_op(inner, torch.ops.aten.expand.default, "expand"):
        return None

    inner2 = inner.args[0] if inner.args else None
    if not isinstance(inner2, Node):
        return None

    if _is_op(inner2, torch.ops.aten.unsqueeze.default, "unsqueeze"):
        return inner2.args[0] if inner2.args else None

    if inner2.op == "call_function" and inner2.target is operator.getitem:
        if len(inner2.args) >= 2 and isinstance(inner2.args[1], tuple):
            if any(i is None for i in inner2.args[1]):
                return inner2.args[0] if inner2.args else None

    return None


# SDPA Detection and Parameter Extraction


def _is_fp8_sdpa_node(node: Node, fp8_sdpa_op) -> bool:
    """Check if an FX node is our FP8 SDPA custom op."""
    return node.op == "call_function" and node.target is fp8_sdpa_op


def _is_sdpa_node(node: Node) -> bool:
    """Check if an FX node is ``aten.scaled_dot_product_attention``."""
    return _is_op(
        node,
        torch.ops.aten.scaled_dot_product_attention.default,
        torch._C._nn.scaled_dot_product_attention,
    )


def _sdpa_is_fusible(node: Node, strip_causal_mask: bool = False) -> Tuple[bool, bool]:
    """Check if an aten SDPA node is compatible with our FP8 fused kernel.

    Returns (is_fusible, needs_mask_strip).
    """
    args = node.args
    kwargs = node.kwargs

    attn_mask = args[3] if len(args) > 3 else kwargs.get("attn_mask", None)
    is_causal = args[5] if len(args) > 5 else kwargs.get("is_causal", False)

    needs_mask_strip = False
    if attn_mask is not None:
        if not is_causal and strip_causal_mask and isinstance(attn_mask, Node):
            needs_mask_strip = True
        else:
            return False, False

    dropout_p = args[4] if len(args) > 4 else kwargs.get("dropout_p", 0.0)
    if dropout_p != 0.0:
        return False, False

    return True, needs_mask_strip


def _strip_causal_mask(node: Node) -> None:
    """Strip a materialized causal mask from an aten SDPA node."""
    args = list(node.args)
    kwargs = dict(node.kwargs)

    if len(args) > 3:
        args[3] = None
    elif "attn_mask" in kwargs:
        kwargs["attn_mask"] = None

    if len(args) > 5:
        args[5] = True
    elif "is_causal" in kwargs:
        kwargs["is_causal"] = True
    else:
        kwargs["is_causal"] = True

    node.args = tuple(args)
    node.kwargs = kwargs

    logger.info("Stripped causal mask from SDPA node: %s", node.name)


def _is_lower_triangular_bool_mask(mask: torch.Tensor) -> bool:
    """Check if a tensor is a bool, square lower-triangular (causal) mask."""
    if mask.dtype != torch.bool or mask.ndim < 2:
        return False
    q_len, kv_len = mask.shape[-2], mask.shape[-1]
    if q_len != kv_len:
        return False
    ref = torch.tril(torch.ones(q_len, kv_len, dtype=torch.bool, device=mask.device))
    return torch.equal(mask.broadcast_to(mask.shape), ref.expand_as(mask))


def detect_causal_mask(
    model: nn.Module,
    sample_input_ids=None,
    flash_impl_name: str | None = None,
) -> bool:
    """Run one forward pass to detect whether the model uses causal masks.

    Returns True when every SDPA call used causal attention (either via
    a materialized lower-triangular bool mask, or via is_causal=True).
    """
    from torch.nn.attention import (
        activate_flash_attention_impl,
        restore_flash_attention_impl,
    )

    try:
        device = next(model.parameters()).device
    except StopIteration:
        return False

    if sample_input_ids is None:
        vocab_size = getattr(getattr(model, "config", None), "vocab_size", None)
        if vocab_size is None:
            return False
        sample_input_ids = torch.randint(0, vocab_size, (1, 16), device=device)

    all_causal: list[bool] = []
    saw_any_sdpa = False

    original_sdpa = F.scaled_dot_product_attention

    def _hook(*args, **kwargs):
        nonlocal saw_any_sdpa
        saw_any_sdpa = True
        attn_mask = args[3] if len(args) > 3 else kwargs.get("attn_mask", None)
        is_causal = kwargs.get("is_causal", False) if len(args) <= 5 else args[5]

        if attn_mask is not None and not is_causal:
            all_causal.append(_is_lower_triangular_bool_mask(attn_mask))
        elif attn_mask is None and is_causal:
            all_causal.append(True)

        return original_sdpa(*args, **kwargs)

    F.scaled_dot_product_attention = _hook
    if flash_impl_name is not None:
        activate_flash_attention_impl(flash_impl_name)
    try:
        with torch.no_grad():
            model(sample_input_ids)
    except Exception:
        logger.debug("detect_causal_mask: forward pass failed", exc_info=True)
        return False
    finally:
        F.scaled_dot_product_attention = original_sdpa
        if flash_impl_name is not None:
            restore_flash_attention_impl()

    if not saw_any_sdpa:
        return False

    return all(all_causal)


def _get_fp8_sdpa_params(node: Node) -> Tuple[bool, float, bool]:
    """Extract is_causal, scale, and enable_gqa from an FP8 SDPA custom op node.

    Custom op signature: (q, k, v, is_causal=False, scale=0.0, enable_gqa=False)
    Scale uses 0.0 as sentinel for "default" (1/sqrt(D)).
    """
    args = node.args
    kwargs = node.kwargs

    is_causal = args[3] if len(args) > 3 else kwargs.get("is_causal", False)
    scale = args[4] if len(args) > 4 else kwargs.get("scale", 0.0)
    enable_gqa = args[5] if len(args) > 5 else kwargs.get("enable_gqa", False)

    return is_causal, scale, enable_gqa


def _get_fp8_sdpa_qkv(node: Node) -> Optional[Tuple[Node, Node, Node]]:
    """Extract Q, K, V input nodes from an FP8 SDPA custom op node."""
    args = node.args
    kwargs = node.kwargs

    q = args[0] if len(args) > 0 else kwargs.get("q", None)
    k = args[1] if len(args) > 1 else kwargs.get("k", None)
    v = args[2] if len(args) > 2 else kwargs.get("v", None)

    if not all(isinstance(n, Node) for n in (q, k, v)):
        return None

    return q, k, v


# NeoX/LLaMA RoPE Pattern Detection
#
# NeoX/LLaMA RoPE:
#   rotate_half(x) = cat(-x[..., D//2:], x[..., :D//2], dim=-1)
#   apply_rope(x, cos, sin) = x * cos + rotate_half(x) * sin


def _detect_rotate_half(cat_node: Node) -> Optional[Node]:
    """Detect rotate_half(x) = cat(-x[..., D//2:], x[..., :D//2], dim=-1).

    Returns the source tensor x, or None.
    """
    if not _is_op(cat_node, torch.ops.aten.cat.default, torch.cat):
        return None

    if len(cat_node.args) < 1:
        return None

    tensors_list = cat_node.args[0]

    if len(cat_node.args) >= 2:
        cat_dim = cat_node.args[1]
    else:
        cat_dim = cat_node.kwargs.get("dim", 0)

    if cat_dim not in (-1, 3):
        return None

    if not isinstance(tensors_list, (list, tuple)) or len(tensors_list) != 2:
        return None

    neg_part = tensors_list[0]
    pos_part = tensors_list[1]

    if not isinstance(neg_part, Node) or not isinstance(pos_part, Node):
        return None

    if not _is_op(neg_part, torch.ops.aten.neg.default, operator.neg, torch.neg):
        return None

    neg_input = neg_part.args[0]
    if not isinstance(neg_input, Node):
        return None

    return _match_rotate_half_slices(neg_input, pos_part)


def _match_rotate_half_slices(neg_input: Node, pos_part: Node) -> Optional[Node]:
    """Match the slice patterns in rotate_half. Returns the source tensor x, or None."""
    # ATen slice pattern
    if _is_op(neg_input, torch.ops.aten.slice.Tensor) and _is_op(
        pos_part, torch.ops.aten.slice.Tensor
    ):
        slice_neg_source = neg_input.args[0]
        slice_pos_source = pos_part.args[0]

        if slice_neg_source is not slice_pos_source:
            return None

        slice_neg_dim = neg_input.args[1] if len(neg_input.args) > 1 else 0
        slice_pos_dim = pos_part.args[1] if len(pos_part.args) > 1 else 0

        if slice_neg_dim not in (-1, 3) or slice_pos_dim not in (-1, 3):
            return None

        pos_start = pos_part.args[2] if len(pos_part.args) > 2 else None
        pos_end = pos_part.args[3] if len(pos_part.args) > 3 else None
        neg_start = neg_input.args[2] if len(neg_input.args) > 2 else None

        if pos_start != 0:
            return None
        if neg_start is None or pos_end is None:
            return None
        if neg_start != pos_end:
            return None

        return slice_neg_source

    # Dynamo getitem pattern
    if _is_op(neg_input, operator.getitem) and _is_op(pos_part, operator.getitem):
        slice_neg_source = neg_input.args[0]
        slice_pos_source = pos_part.args[0]

        if slice_neg_source is not slice_pos_source:
            return None

        neg_idx = neg_input.args[1]
        pos_idx = pos_part.args[1]

        neg_slice = _extract_last_dim_slice(neg_idx)
        pos_slice = _extract_last_dim_slice(pos_idx)

        if neg_slice is None or pos_slice is None:
            return None

        if pos_slice.start not in (0, None):
            return None
        if pos_slice.stop is None:
            return None
        if neg_slice.start is None:
            return None
        if neg_slice.start != pos_slice.stop:
            return None

        return slice_neg_source

    return None


def _extract_last_dim_slice(idx) -> Optional[slice]:
    """Extract the slice on the last dimension from a getitem index."""
    if isinstance(idx, tuple):
        if len(idx) >= 2 and idx[0] is Ellipsis and isinstance(idx[1], slice):
            return idx[1]
        if len(idx) >= 1 and isinstance(idx[-1], slice):
            for i in range(len(idx) - 1):
                if idx[i] is not Ellipsis and idx[i] != slice(None):
                    return None
            return idx[-1]
    elif isinstance(idx, slice):
        return idx
    return None


def _detect_interleaved_rotation(node: Node) -> Optional[Node]:
    """Detect the FLUX-style interleaved rotation pattern.

    Pattern: x.reshape(..., -1, 2).unbind(-1) -> stack([-x_imag, x_real], dim=-1).flatten(3)
    Returns the source tensor x, or None.
    """
    if not _is_op(node, torch.ops.aten.flatten.using_ints, "flatten"):
        return None
    if len(node.args) < 1:
        return None

    stack_node = node.args[0]
    if not isinstance(stack_node, Node):
        return None

    if not _is_op(stack_node, torch.ops.aten.stack.default, torch.stack):
        return None
    if len(stack_node.args) < 1:
        return None

    tensors_list = stack_node.args[0]
    if len(stack_node.args) >= 2:
        stack_dim = stack_node.args[1]
    else:
        stack_dim = stack_node.kwargs.get("dim", 0)
    if stack_dim != -1:
        return None

    if not isinstance(tensors_list, (list, tuple)) or len(tensors_list) != 2:
        return None

    neg_part = tensors_list[0]
    pos_part = tensors_list[1]
    if not isinstance(neg_part, Node) or not isinstance(pos_part, Node):
        return None

    if not _is_op(neg_part, torch.ops.aten.neg.default, operator.neg, torch.neg):
        return None

    x_imag = neg_part.args[0]
    if not isinstance(x_imag, Node):
        return None

    if not _is_op(pos_part, operator.getitem) or not _is_op(x_imag, operator.getitem):
        return None

    real_source = pos_part.args[0]
    imag_source = x_imag.args[0]
    if real_source is not imag_source:
        return None
    if pos_part.args[1] != 0 or x_imag.args[1] != 1:
        return None

    unbind_node = real_source
    if not isinstance(unbind_node, Node):
        return None
    if not _is_op(unbind_node, torch.ops.aten.unbind.int, "unbind"):
        return None

    if len(unbind_node.args) >= 2:
        unbind_dim = unbind_node.args[1]
    else:
        unbind_dim = unbind_node.kwargs.get("dim", 0)
    if unbind_dim != -1:
        return None

    reshape_node = unbind_node.args[0]
    if not isinstance(reshape_node, Node):
        return None
    if not _is_op(
        reshape_node,
        torch.ops.aten.reshape.default,
        torch.ops.aten.view.default,
        "reshape",
        "view",
    ):
        return None

    source_x = reshape_node.args[0]
    return source_x if isinstance(source_x, Node) else None


def _detect_rotation(node: Node) -> Optional[Tuple[Node, bool]]:
    """Detect any supported rotation pattern.

    Returns (source_tensor, is_interleaved) or None.
    """
    result = _detect_rotate_half(node)
    if result is not None:
        return result, False

    result = _detect_interleaved_rotation(node)
    if result is not None:
        return result, True

    return None


def _detect_neox_rope(node: Node) -> Optional[RoPEMatch]:
    """Detect the NeoX/LLaMA RoPE pattern: add(mul(x, cos), mul(rotate_half(x), sin))."""
    if not _is_op(node, torch.ops.aten.add.Tensor, operator.add):
        return None
    if len(node.args) < 2:
        return None

    left = node.args[0]
    right = node.args[1]
    if not isinstance(left, Node) or not isinstance(right, Node):
        return None

    if not _is_op(left, torch.ops.aten.mul.Tensor, operator.mul):
        return None
    if not _is_op(right, torch.ops.aten.mul.Tensor, operator.mul):
        return None

    def _try_match(x_mul: Node, rot_mul: Node) -> Optional[RoPEMatch]:
        rot_a, rot_b = rot_mul.args[0], rot_mul.args[1]

        for rot_candidate, sin_candidate in [(rot_a, rot_b), (rot_b, rot_a)]:
            if not isinstance(rot_candidate, Node):
                continue

            rot_unwrapped = _trace_through_views(rot_candidate)
            rotation_result = _detect_rotation(rot_unwrapped)
            if rotation_result is None:
                continue
            x_from_rotate, is_interleaved = rotation_result

            x_a, x_b = x_mul.args[0], x_mul.args[1]
            for x_candidate, cos_candidate in [(x_a, x_b), (x_b, x_a)]:
                if not isinstance(x_candidate, Node):
                    continue

                x_traced = _trace_through_views(x_candidate)
                if x_traced is not x_from_rotate:
                    continue

                cos_source = _trace_through_views(cos_candidate)
                sin_source = _trace_through_views(sin_candidate)

                return RoPEMatch(
                    pre_rope_input=x_from_rotate,
                    cos_node=cos_source,
                    sin_node=sin_source,
                    rope_interleaved=is_interleaved,
                )

        return None

    # Try both orderings of the add (commutative).
    result = _try_match(left, right)
    if result is not None:
        return result
    return _try_match(right, left)


def _get_setitem_stride2_slice(node: Node) -> Optional[Tuple[int, Node]]:
    """Check if node is ``operator.setitem(tensor, (..., slice(start, None, 2)), value)``.

    Returns ``(start, value_node)`` or None.
    """
    if not _is_op(node, operator.setitem):
        return None
    if len(node.args) < 3:
        return None
    idx = node.args[1]
    value = node.args[2]
    if not isinstance(value, Node):
        return None
    # idx should be (Ellipsis, slice(start, None, 2))
    if not isinstance(idx, tuple) or len(idx) < 1:
        return None
    last = idx[-1]
    if not isinstance(last, slice) or last.step != 2:
        return None
    # Check preceding entries are Ellipsis or slice(None)
    for entry in idx[:-1]:
        if entry is not Ellipsis and entry != slice(None):
            return None
    start = last.start if last.start is not None else 0
    return start, value


def _get_getitem_stride2_slice(node: Node) -> Optional[Tuple[int, Node]]:
    """Check if node is ``operator.getitem(tensor, (..., slice(start, None, 2)))``.

    Returns ``(start, source_tensor_node)`` or None.
    """
    if not _is_op(node, operator.getitem):
        return None
    if len(node.args) < 2:
        return None
    source = node.args[0]
    idx = node.args[1]
    if not isinstance(source, Node):
        return None
    if not isinstance(idx, tuple) or len(idx) < 1:
        return None
    last = idx[-1]
    if not isinstance(last, slice) or last.step != 2:
        return None
    for entry in idx[:-1]:
        if entry is not Ellipsis and entry != slice(None):
            return None
    start = last.start if last.start is not None else 0
    return start, source


def _find_freq_slice_arg(mul_node: Node) -> Optional[Tuple[Node, Node]]:
    """Find which arg of a mul is a stride-2 getitem slice (freq), return (getitem_node, other_arg).

    Handles commutativity: checks both orderings.
    """
    if not _is_op(mul_node, torch.ops.aten.mul.Tensor, operator.mul):
        return None
    a, b = mul_node.args[0], mul_node.args[1]
    for freq_candidate, other in [(a, b), (b, a)]:
        if not isinstance(freq_candidate, Node):
            continue
        traced = _trace_through_views(freq_candidate)
        gi = _get_getitem_stride2_slice(traced)
        if gi is not None:
            return traced, other
    return None


def _get_freq_slice_start(node: Node) -> Optional[int]:
    """Get the start value from a stride-2 frequency slice node."""
    gi = _get_getitem_stride2_slice(node)
    if gi is not None:
        return gi[0]
    return None


def _get_freq_slice_source(node: Node) -> Node:
    """Get the source tensor from a stride-2 frequency slice node."""
    gi = _get_getitem_stride2_slice(node)
    if gi is not None:
        return _trace_through_views(gi[1])
    return node


def _detect_wan_rope(node: Node) -> Optional[RoPEMatch]:
    """Detect Wan-style indexed-write RoPE pattern (Pattern C).

    In the pre-grad FX graph (where the fusion pass runs), the pattern is:

        out = torch.empty_like(hidden_states)
        ...
        setitem(out, (..., slice(0, None, 2)), sub)   # even positions
        ...
        setitem(out, (..., slice(1, None, 2)), add)    # odd positions
        type_as(out, hidden_states)
        transpose(out, 1, 2)

    The detection starts from the ``empty_like`` node (reached after
    ``_unwrap_transpose`` + ``_trace_through_views`` strips the transpose and
    type_as).  We then scan the users of ``empty_like`` for the two stride-2
    setitem writes.

    This is mathematically identical to interleaved RoPE (pairs (2i, 2i+1)),
    so we return rope_interleaved=True.
    """
    # The node should be empty_like(hidden_states)
    if not _is_op(
        node,
        torch.ops.aten.empty_like.default,
        torch.empty_like,
    ):
        return None

    if not node.args or not isinstance(node.args[0], Node):
        return None
    pre_rope_input = node.args[0]

    # Find the two stride-2 setitem users: one with start=0 (even), one with start=1 (odd)
    even_val = None
    odd_val = None
    for user in node.users:
        si = _get_setitem_stride2_slice(user)
        if si is None:
            continue
        start, value = si
        if start == 0 and even_val is None:
            even_val = value
        elif start == 1 and odd_val is None:
            odd_val = value

    if even_val is None or odd_val is None:
        return None

    # even_val = sub(mul(x1, cos), mul(x2, sin))
    if not _is_op(even_val, torch.ops.aten.sub.Tensor, operator.sub):
        return None
    if len(even_val.args) < 2:
        return None
    even_left, even_right = even_val.args[0], even_val.args[1]
    if not isinstance(even_left, Node) or not isinstance(even_right, Node):
        return None
    if not _is_op(even_left, torch.ops.aten.mul.Tensor, operator.mul):
        return None
    if not _is_op(even_right, torch.ops.aten.mul.Tensor, operator.mul):
        return None

    # odd_val = add(mul(x1, sin), mul(x2, cos))
    if not _is_op(odd_val, torch.ops.aten.add.Tensor, operator.add):
        return None
    if len(odd_val.args) < 2:
        return None
    odd_left, odd_right = odd_val.args[0], odd_val.args[1]
    if not isinstance(odd_left, Node) or not isinstance(odd_right, Node):
        return None
    if not _is_op(odd_left, torch.ops.aten.mul.Tensor, operator.mul):
        return None
    if not _is_op(odd_right, torch.ops.aten.mul.Tensor, operator.mul):
        return None

    # From even_val's mul nodes, identify cos/sin (stride-2 slices) vs x1/x2
    even_left_match = _find_freq_slice_arg(even_left)
    even_right_match = _find_freq_slice_arg(even_right)
    if even_left_match is None or even_right_match is None:
        return None

    cos_slice_node, _x1 = even_left_match
    sin_slice_node, _x2 = even_right_match

    # Verify cos slice has start=0 (even positions) and sin slice has start=1 (odd positions)
    cos_start = _get_freq_slice_start(cos_slice_node)
    sin_start = _get_freq_slice_start(sin_slice_node)
    if cos_start != 0 or sin_start != 1:
        # Try swapping
        if sin_start == 0 and cos_start == 1:
            cos_slice_node, sin_slice_node = sin_slice_node, cos_slice_node
        else:
            return None

    # Unwrap stride-2 slices to get original cos/sin tensors
    cos_original = _get_freq_slice_source(cos_slice_node)
    sin_original = _get_freq_slice_source(sin_slice_node)

    return RoPEMatch(
        pre_rope_input=pre_rope_input,
        cos_node=cos_original,
        sin_node=sin_original,
        rope_interleaved=True,
    )


def _detect_rope(node: Node) -> Optional[RoPEMatch]:
    """Detect any supported RoPE variant at a given node."""
    result = _detect_neox_rope(node)
    if result is not None:
        return result
    return _detect_wan_rope(node)


# Graph Surgery


def _replace_with_fused_op(
    graph: Graph,
    sdpa_node: Node,
    pre_rope_q: Node,
    pre_rope_k: Node,
    v_input: Node,
    cos_node: Node,
    sin_node: Node,
    is_causal: bool,
    scale: float,
    enable_gqa: bool,
    rope_interleaved: bool,
    rope_sdpa_op,
) -> None:
    """Replace an SDPA node with a fused RoPE+SDPA custom op."""
    with graph.inserting_before(sdpa_node):
        fused_node = graph.call_function(
            rope_sdpa_op,
            args=(pre_rope_q, pre_rope_k, v_input, cos_node, sin_node),
            kwargs={
                "is_causal": is_causal,
                "scale": scale,
                "enable_gqa": enable_gqa,
                "rope_interleaved": rope_interleaved,
            },
        )

    fused_node.meta = sdpa_node.meta.copy()
    sdpa_node.replace_all_uses_with(fused_node)

    logger.info(
        "Fused RoPE + SDPA: replaced %s with %s",
        sdpa_node.name,
        fused_node.name,
    )


# Main Fusion Pass


def rope_sdpa_fusion_pass(
    graph: Graph,
    rope_sdpa_op,
    fp8_sdpa_op,
    backend_name: str = "FP8",
) -> None:
    """Detect RoPE patterns preceding FP8 SDPA nodes and fuse them.

    Scans the FX graph for FP8 SDPA custom op nodes (placed by the
    monkey-patch).  For each one where RoPE is detected on Q and K,
    replaces it with the fused RoPE+SDPA custom op.  Nodes without
    RoPE are left as-is — they are already low-precision from the
    monkey-patch.

    Supported patterns:
      - Pattern A (RoPE -> transpose -> FP8 SDPA): FLUX-style
      - Pattern B (transpose -> RoPE -> FP8 SDPA): HuggingFace-style
      - Pattern C (indexed-write RoPE -> transpose -> FP8 SDPA): Wan-style

    Note: KV caching must be disabled before compilation.
    DynamicCache.update() inserts torch.cat nodes that break pattern matching.
    """
    fp8_sdpa_nodes = [n for n in graph.nodes if _is_fp8_sdpa_node(n, fp8_sdpa_op)]

    if not fp8_sdpa_nodes:
        print(
            f"[low_precision_attention] RoPE fusion pass ({backend_name}): "
            f"found 0 FP8 SDPA nodes in graph"
        )
        return

    fused_count = 0

    for sdpa_node in fp8_sdpa_nodes:
        is_causal, scale, enable_gqa = _get_fp8_sdpa_params(sdpa_node)

        qkv = _get_fp8_sdpa_qkv(sdpa_node)
        if qkv is None:
            continue
        q_node, k_node, v_node = qkv

        v_pre_transpose = _unwrap_transpose(v_node)

        # Pattern A: RoPE -> transpose -> FP8 SDPA (FLUX-style)
        q_pre_transpose = _unwrap_transpose(q_node)
        k_pre_transpose = _unwrap_transpose(k_node)

        if q_pre_transpose is not None and k_pre_transpose is not None:
            q_pre_cast = _trace_through_views(q_pre_transpose)
            k_pre_cast = _trace_through_views(k_pre_transpose)

            q_rope = _detect_rope(q_pre_cast)
            k_rope = _detect_rope(k_pre_cast)

            if q_rope is not None and k_rope is not None:
                pre_rope_q = _trace_through_views(q_rope.pre_rope_input)
                pre_rope_k = _trace_through_views(k_rope.pre_rope_input)

                if v_pre_transpose is None:
                    logger.debug(
                        "Pattern A: V has no transpose, skipping: %s",
                        sdpa_node.name,
                    )
                    continue

                cos_sin = _reshape_cos_sin_to_2d(
                    graph,
                    q_rope.cos_node,
                    q_rope.sin_node,
                    sdpa_node,
                )
                if cos_sin is None:
                    logger.debug(
                        "Pattern A: cos/sin shape incompatible, skipping: %s",
                        sdpa_node.name,
                    )
                    continue
                cos_2d, sin_2d = cos_sin

                _replace_with_fused_op(
                    graph=graph,
                    sdpa_node=sdpa_node,
                    pre_rope_q=pre_rope_q,
                    pre_rope_k=pre_rope_k,
                    v_input=v_pre_transpose,
                    cos_node=cos_2d,
                    sin_node=sin_2d,
                    is_causal=is_causal,
                    scale=scale,
                    enable_gqa=enable_gqa,
                    rope_interleaved=q_rope.rope_interleaved,
                    rope_sdpa_op=rope_sdpa_op,
                )
                fused_count += 1
                continue

        # Pattern B: transpose -> RoPE -> FP8 SDPA (HuggingFace-style)
        # For GQA, K may go through repeat_kv after RoPE.
        q_rope = _detect_rope(_trace_through_views(q_node))

        k_rope = _detect_rope(_trace_through_views(k_node))
        gqa_unwrapped = False
        if k_rope is None:
            k_pre_repeat = _unwrap_repeat_kv(k_node)
            if k_pre_repeat is not None:
                k_rope = _detect_rope(_trace_through_views(k_pre_repeat))
                if k_rope is not None:
                    gqa_unwrapped = True

        if q_rope is not None and k_rope is not None:
            q_bshd = _unwrap_transpose(_trace_through_views(q_rope.pre_rope_input))
            k_bshd = _unwrap_transpose(_trace_through_views(k_rope.pre_rope_input))

            if q_bshd is not None and k_bshd is not None:
                v_for_fusion = v_node
                if gqa_unwrapped:
                    v_pre_repeat = _unwrap_repeat_kv(v_node)
                    if v_pre_repeat is not None:
                        v_for_fusion = v_pre_repeat

                v_bshd = _unwrap_transpose(v_for_fusion)
                if v_bshd is None:
                    logger.debug(
                        "Pattern B: V has no transpose, skipping: %s",
                        sdpa_node.name,
                    )
                    continue

                cos_sin = _reshape_cos_sin_to_2d(
                    graph,
                    q_rope.cos_node,
                    q_rope.sin_node,
                    sdpa_node,
                )
                if cos_sin is None:
                    logger.debug(
                        "Pattern B: cos/sin shape incompatible, skipping: %s",
                        sdpa_node.name,
                    )
                    continue
                cos_2d, sin_2d = cos_sin

                fused_enable_gqa = True if gqa_unwrapped else enable_gqa

                _replace_with_fused_op(
                    graph=graph,
                    sdpa_node=sdpa_node,
                    pre_rope_q=q_bshd,
                    pre_rope_k=k_bshd,
                    v_input=v_bshd,
                    cos_node=cos_2d,
                    sin_node=sin_2d,
                    is_causal=is_causal,
                    scale=scale,
                    enable_gqa=fused_enable_gqa,
                    rope_interleaved=q_rope.rope_interleaved,
                    rope_sdpa_op=rope_sdpa_op,
                )
                fused_count += 1
                continue

    print(
        f"[low_precision_attention] RoPE fusion pass ({backend_name}): "
        f"found {len(fp8_sdpa_nodes)} FP8 SDPA node(s), "
        f"{fused_count} fused with RoPE"
    )

    if fused_count > 0:
        graph.eliminate_dead_code()
