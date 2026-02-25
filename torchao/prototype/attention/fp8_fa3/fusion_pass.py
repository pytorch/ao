# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD 3-Clause license found in the
# LICENSE file in the root directory of this source tree.

"""
FX graph transformation pass: detects RoPE + SDPA patterns and replaces them
with a fused fp8_fa3_rope_sdpa operation.

This pass is registered as a pre-grad custom pass via
torch._inductor.config.pre_grad_custom_pass. It runs during torch.compile,
after Dynamo graph capture but before AOTAutograd / ATen decomposition.

At the pre-grad level, FX nodes use Python-level ops:
  - SDPA appears as torch._C._nn.scaled_dot_product_attention (builtin)
  - Arithmetic uses operator.add, operator.mul, operator.neg, etc.
  - Method calls appear as call_method nodes (e.g., "permute", "to", "float")

The pass walks the graph looking for SDPA nodes, traces their Q and K inputs
backward to detect RoPE patterns, and replaces the entire
(RoPE + transpose + SDPA) subgraph with a single fused custom op.

Supported RoPE rotation variants:
  - NeoX/LLaMA half-split: cat(-x[..., D//2:], x[..., :D//2], dim=-1)
  - FLUX interleaved: stack([-x_imag, x_real], dim=-1).flatten(3)
  Both applied as: x * cos + rotation(x) * sin

Supported layout patterns:
  - Pattern A (FLUX-style): RoPE on [B,S,H,D] -> transpose -> SDPA on [B,H,S,D]
  - Pattern B (HuggingFace-style): transpose -> RoPE on [B,H,S,D] -> SDPA on [B,H,S,D]
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


# ============================================================================
# Section 1: Custom Op Registration
# ============================================================================
#
# The fp8_fa3_rope_sdpa function in attention.py calls Triton kernels for
# the fused RoPE + FP8 quantization step (_fp8_rope_sdpa_quantize). These
# Triton kernels are not traceable by torch.compile -- they aren't built
# from standard PyTorch ops that the compiler can decompose.
#
# To make fp8_fa3_rope_sdpa usable inside a compiled graph, we register it
# as a torch.library.custom_op. This tells the compiler:
#   1. "This is an opaque operation -- don't try to trace into it."
#   2. "Here's how to compute output shapes/dtypes" (via register_fake).
#   3. "Here's the real implementation to call at runtime."
#
# The custom op is what we insert into the FX graph during fusion.
# After registration, it's accessible as torch.ops.torchao.fp8_fa3_rope_sdpa.

_CUSTOM_OP_LIB = "torchao"
_CUSTOM_OP_NAME = f"{_CUSTOM_OP_LIB}::fp8_fa3_rope_sdpa"


@torch.library.custom_op(_CUSTOM_OP_NAME, mutates_args=())
def _fp8_fa3_rope_sdpa_custom_op(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    is_causal: bool = False,
    scale: float = 0.0,
    enable_gqa: bool = False,
    rope_interleaved: bool = False,
) -> torch.Tensor:
    """Custom op wrapper around fp8_fa3_rope_sdpa.

    Args:
        q: Query tensor [B, S, H, D] in bf16/fp16.
        k: Key tensor [B, S, H, D] in bf16/fp16.
        v: Value tensor [B, S, H, D] in bf16/fp16.
        cos: RoPE cosine frequencies [S, D].
        sin: RoPE sine frequencies [S, D].
        is_causal: Whether to use causal masking.
        scale: Attention scale factor. We use 0.0 as a sentinel value meaning
               "use default (1/sqrt(D))" because custom_op doesn't support
               Optional[float], and scale=0.0 would never be a valid real scale.
        enable_gqa: Whether to enable grouped query attention.
        rope_interleaved: Whether to use interleaved (FLUX) or NeoX (LLaMA) RoPE.

    Returns:
        Attention output [B, H, S, D] in the input dtype.
    """
    from torchao.prototype.attention.fp8_fa3.attention import fp8_fa3_rope_sdpa

    # Convert sentinel scale=0.0 back to None (meaning "use default 1/sqrt(D)").
    actual_scale = scale if scale != 0.0 else None

    return fp8_fa3_rope_sdpa(
        q,
        k,
        v,
        cos,
        sin,
        is_causal=is_causal,
        scale=actual_scale,
        enable_gqa=enable_gqa,
        rope_interleaved=rope_interleaved,
    )


@_fp8_fa3_rope_sdpa_custom_op.register_fake
def _fp8_fa3_rope_sdpa_fake(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    is_causal: bool = False,
    scale: float = 0.0,
    enable_gqa: bool = False,
    rope_interleaved: bool = False,
) -> torch.Tensor:
    """FakeTensor implementation: tells the compiler the output shape and dtype.

    The fused kernel takes [B, S, H, D] input and produces [B, H, S, D] output
    (the transpose is baked into the kernel).
    """
    B, S, H, D = q.shape
    return torch.empty(B, H, S, D, dtype=q.dtype, device=q.device)


# Also register the non-rope fp8_fa3_sdpa as a custom op. This is used for
# SDPA nodes that don't have RoPE on their Q/K inputs. We need this because
# the fusion pass replaces ALL SDPA nodes (not just RoPE ones), and we can't
# use the monkey-patch approach (it would eat the SDPA nodes before the fusion
# pass sees them).

_NON_ROPE_CUSTOM_OP_NAME = f"{_CUSTOM_OP_LIB}::fp8_fa3_sdpa"


@torch.library.custom_op(_NON_ROPE_CUSTOM_OP_NAME, mutates_args=())
def _fp8_fa3_sdpa_custom_op(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    is_causal: bool = False,
    scale: float = 0.0,
    enable_gqa: bool = False,
) -> torch.Tensor:
    """Custom op wrapper around fp8_fa3_sdpa (non-rope version).

    Args:
        q: Query tensor [B, H, S, D] in bf16/fp16.
        k: Key tensor [B, H, S, D] in bf16/fp16.
        v: Value tensor [B, H, S, D] in bf16/fp16.
        is_causal: Whether to use causal masking.
        scale: Attention scale factor. 0.0 = use default (1/sqrt(D)).
        enable_gqa: Whether to enable grouped query attention.

    Returns:
        Attention output [B, H, S, D] in the input dtype.
    """
    from torchao.prototype.attention.fp8_fa3.attention import fp8_fa3_sdpa

    actual_scale = scale if scale != 0.0 else None

    return fp8_fa3_sdpa(
        q,
        k,
        v,
        is_causal=is_causal,
        scale=actual_scale,
        enable_gqa=enable_gqa,
    )


@_fp8_fa3_sdpa_custom_op.register_fake
def _fp8_fa3_sdpa_fake(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    is_causal: bool = False,
    scale: float = 0.0,
    enable_gqa: bool = False,
) -> torch.Tensor:
    """FakeTensor implementation for non-rope FP8 SDPA.

    Input and output are both [B, H, S, D] (standard SDPA layout).
    The output is always contiguous even if the input is a transposed view.
    """
    return torch.empty(q.shape, dtype=q.dtype, device=q.device)


# ============================================================================
# Section 2: Data Types
# ============================================================================


@dataclass
class RoPEMatch:
    """Result of successfully detecting a RoPE pattern on a tensor.

    When we determine that a graph node is the output of a RoPE operation,
    this dataclass captures the key information needed for fusion:

    Fields:
        pre_rope_input: The tensor node BEFORE RoPE was applied. This is the
            "x" in "x * cos + rotate_half(x) * sin". This is what we'll pass
            to the fused kernel as the query or key input.
        cos_node: The graph node for the cosine frequencies, traced back
            through unsqueeze/expand to the [S, D] source.
        sin_node: The graph node for the sine frequencies, traced back
            through unsqueeze/expand to the [S, D] source.
    """

    pre_rope_input: Node
    cos_node: Node
    sin_node: Node
    rope_interleaved: bool  # True = FLUX interleaved, False = NeoX half-split


# ============================================================================
# Section 3: FX Node Utilities
# ============================================================================


def _is_op(node: Node, *targets) -> bool:
    """Check if an FX node matches one of the given targets.

    Handles both levels of the FX graph:
      - call_function: ATen ops (e.g., torch.ops.aten.add.Tensor) or
        Python builtins (e.g., operator.add)
      - call_method: Python method calls (e.g., "permute", "transpose")

    Args:
        node: The FX graph node to check.
        *targets: One or more op targets to match against.
                  Can be function objects (for call_function) or
                  strings (for call_method).

    Returns:
        True if node matches any of the given targets.
    """
    if node.op in ("call_function", "call_method"):
        return node.target in targets
    return False


def _get_fake_tensor(node: Node) -> Optional[torch.Tensor]:
    """Get the FakeTensor metadata from a node.

    During torch.compile, FX nodes carry FakeTensor metadata that tells us
    the shape and dtype of the tensor each node produces without running real
    computation.  The metadata key varies by pass level:
      - Pre-grad (Dynamo) level: ``node.meta["example_value"]``
      - Post-grad (ATen/Inductor) level: ``node.meta["val"]``

    Returns:
        The FakeTensor, or None if metadata is unavailable.
    """
    for key in ("val", "example_value"):
        if key in node.meta:
            val = node.meta[key]
            if isinstance(val, torch.Tensor):
                return val
    return None


def _get_node_shape(node: Node) -> Optional[Tuple[int, ...]]:
    """Get the output tensor shape from a node's FakeTensor metadata.

    Returns:
        The shape as a tuple of ints, or None if metadata is unavailable.
    """
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

    HuggingFace models produce cos/sin with shape [B, S, D] (or even
    [1, 1, S, D]) from their rotary embeddings, but our fused kernel
    expects [S, D].  This function checks compatibility and inserts
    view nodes into the graph if reshaping is needed.

    Args:
        graph: The FX graph to insert reshape nodes into.
        cos_node: The cos frequencies node.
        sin_node: The sin frequencies node.
        insert_before: Insert any new nodes before this node.

    Returns:
        (cos_2d, sin_2d) nodes with shape [S, D], or None if the shapes
        are incompatible (e.g., a leading dim > 1).
    """
    cos_shape = _get_node_shape(cos_node)
    sin_shape = _get_node_shape(sin_node)

    # If metadata is unavailable, skip validation and pass through as-is.
    if cos_shape is None or sin_shape is None:
        return cos_node, sin_node

    # Already 2D — no reshaping needed.
    if len(cos_shape) == 2 and len(sin_shape) == 2:
        return cos_node, sin_node

    # Check that all leading dims are 1 (broadcastable to any batch size).
    for name, shape in [("cos", cos_shape), ("sin", sin_shape)]:
        if len(shape) < 2:
            logger.debug("RoPE %s has fewer than 2 dims: shape=%s", name, shape)
            return None
        for dim in shape[:-2]:
            if dim != 1:
                logger.debug(
                    "RoPE %s has non-unit leading dim: shape=%s",
                    name,
                    shape,
                )
                return None

    # Insert view nodes to reshape to [S, D].
    s, d = cos_shape[-2], cos_shape[-1]
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
    """Trace backward through view-like ops to find the underlying source tensor.

    Many ops in the graph change shape/dtype without computing anything new:
      - unsqueeze: adds a size-1 dimension  (e.g., [S, D] -> [1, S, D])
      - expand:    broadcasts              (e.g., [1, S, 1, D] -> [B, S, H, D])
      - clone:     copies the tensor
      - contiguous: changes memory layout
      - to.dtype:  casts dtype

    When we detect that a mul node uses "cos_expanded" (the broadcast version
    of cos), we want to trace back to the original [S, D] cos tensor before
    all these view ops were applied.

    Args:
        node: The starting node to trace backward from.

    Returns:
        The source node after peeling off all view-like operations.
    """
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
            "float",
            "half",
            "bfloat16",
        ):
            # Python method calls: tensor.to(dtype), tensor.float(), etc.
            current = current.args[0]
        elif (
            current.op == "call_function"
            and current.target is operator.getitem
            and len(current.args) >= 2
            and isinstance(current.args[1], tuple)
            and all(i is None or i == slice(None) for i in current.args[1])
        ):
            # Trace through indexing that only adds dimensions (unsqueeze-like).
            # e.g., cos[None, :, None, :] = (None, slice(None), None, slice(None))
            current = current.args[0]
        else:
            break
    return current


# ============================================================================
# Section 4: Transpose Detection
# ============================================================================
#
# In transformer models, Q/K/V are computed with shape [B, S, H, D] (the
# "natural" layout from the linear projection) and then transposed to
# [B, H, S, D] before being passed to SDPA (the layout SDPA expects).
#
# This transpose is done with either:
#   - transpose(1, 2):       swaps dims 1 and 2
#   - permute([0, 2, 1, 3]): equivalent reordering
#
# When we detect a transpose on an SDPA input, we "unwrap" it to get the
# pre-transpose tensor in [B, S, H, D] layout. This is important because
# our fused kernel expects [B, S, H, D] input (it does the transpose
# internally as part of the fused operation).
#
# Some models also insert a .contiguous() or .clone() after the transpose.
# We strip those before checking for the transpose pattern.


def _unwrap_transpose(node: Node) -> Optional[Node]:
    """If node is a transpose(1,2) or permute([0,2,1,3]), return its input.

    Also looks through contiguous()/clone() that may sit between the
    transpose and the consumer (SDPA or RoPE).

    Args:
        node: A graph node that might be a transpose operation (possibly
              wrapped in contiguous/clone).

    Returns:
        The input node in [B, S, H, D] layout (pre-transpose), or None if
        the node is not a matching transpose pattern.
    """
    if not isinstance(node, Node):
        return None

    # Strip contiguous/clone that may wrap the transpose output.
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

    # Pattern 1: aten.transpose.int(tensor, dim0=1, dim1=2)
    if _is_op(current, torch.ops.aten.transpose.int):
        if len(current.args) >= 3:
            dim0, dim1 = current.args[1], current.args[2]
            if (dim0 == 1 and dim1 == 2) or (dim0 == 2 and dim1 == 1):
                return current.args[0]

    # Pattern 2: aten.permute.default(tensor, [0, 2, 1, 3])
    if _is_op(current, torch.ops.aten.permute.default):
        if len(current.args) >= 2:
            perm = current.args[1]
            if list(perm) == [0, 2, 1, 3]:
                return current.args[0]

    # Pattern 3: call_method transpose(1, 2)
    if _is_op(current, "transpose"):
        if len(current.args) >= 3:
            dim0, dim1 = current.args[1], current.args[2]
            if (dim0 == 1 and dim1 == 2) or (dim0 == 2 and dim1 == 1):
                return current.args[0]

    # Pattern 4: call_method permute(0, 2, 1, 3)
    if _is_op(current, "permute"):
        if len(current.args) >= 5:
            # permute as separate args: tensor.permute(0, 2, 1, 3)
            perm = list(current.args[1:5])
            if perm == [0, 2, 1, 3]:
                return current.args[0]
        elif len(current.args) >= 2 and isinstance(current.args[1], (list, tuple)):
            # permute with list arg: tensor.permute([0, 2, 1, 3])
            perm = current.args[1]
            if list(perm) == [0, 2, 1, 3]:
                return current.args[0]

    return None


def _unwrap_repeat_kv(node: Node) -> Optional[Node]:
    """Unwrap a repeat_kv (GQA head repetition) pattern to get the original tensor.

    HuggingFace's repeat_kv expands KV heads for GQA compatibility:
        x[:, :, None, :, :].expand(B, nkv, n_rep, S, D).reshape(B, nkv*n_rep, S, D)

    In the FX graph this appears as: reshape ← expand ← getitem(unsqueeze).
    This function detects that pattern and returns the pre-repetition tensor.

    Args:
        node: A graph node that might be the output of repeat_kv.

    Returns:
        The pre-repetition tensor node, or None if not a repeat_kv pattern.
    """
    if not isinstance(node, Node):
        return None

    # Step 1: Must be reshape or view — merges (nkv, n_rep) → (nkv*n_rep).
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

    # Step 2: Input must be expand — broadcasts the rep dim.
    if not _is_op(
        inner,
        torch.ops.aten.expand.default,
        "expand",
    ):
        return None

    inner2 = inner.args[0] if inner.args else None
    if not isinstance(inner2, Node):
        return None

    # Step 3: Input to expand must add a size-1 dim (unsqueeze-like).
    # This can be getitem with None indexing or aten.unsqueeze.
    if _is_op(inner2, torch.ops.aten.unsqueeze.default, "unsqueeze"):
        return inner2.args[0] if inner2.args else None

    if inner2.op == "call_function" and inner2.target is operator.getitem:
        # e.g., x[:, :, None, :, :] — getitem with None in the index tuple.
        if len(inner2.args) >= 2 and isinstance(inner2.args[1], tuple):
            if any(i is None for i in inner2.args[1]):
                return inner2.args[0] if inner2.args else None

    return None


# ============================================================================
# Section 5: SDPA Detection and Parameter Extraction
# ============================================================================


def _is_sdpa_node(node: Node) -> bool:
    """Check if a node is scaled_dot_product_attention.

    At the pre-grad level, SDPA may appear as either:
      - torch.ops.aten.scaled_dot_product_attention.default (ATen level,
        after AOTAutograd functionalization)
      - torch._C._nn.scaled_dot_product_attention (Python builtin level,
        which is what F.scaled_dot_product_attention resolves to)

    We match both because the FA3 flash attention activation can prevent
    the normal ATen lowering during tracing.
    """
    return _is_op(
        node,
        torch.ops.aten.scaled_dot_product_attention.default,
        torch._C._nn.scaled_dot_product_attention,
    )


def _is_lower_triangular_bool_mask(mask: torch.Tensor) -> bool:
    """Check if a real tensor is a bool, square lower-triangular (causal) mask.

    Compares the mask against ``torch.tril(torch.ones(...))``.  Returns
    ``True`` only when the mask is boolean, the last two dimensions are
    square, and every element matches the canonical causal pattern.
    """
    if mask.dtype != torch.bool:
        return False

    if mask.ndim < 2:
        return False

    q_len, kv_len = mask.shape[-2], mask.shape[-1]
    if q_len != kv_len:
        return False

    # Build reference causal mask on the same device / shape and compare.
    ref = torch.tril(torch.ones(q_len, kv_len, dtype=torch.bool, device=mask.device))
    # Broadcast: mask may be [B, 1, Q, KV] or [B, H, Q, KV]; ref is [Q, KV].
    return torch.equal(mask.broadcast_to(mask.shape), ref.expand_as(mask))


def detect_causal_mask(model: nn.Module, sample_input_ids=None) -> bool:
    """Run one forward pass to detect whether the model uses causal masks.

    This pre-flight check monkey-patches ``F.scaled_dot_product_attention``
    to inspect every ``attn_mask`` argument.  It returns ``True`` only when
    **all** masks observed during the forward pass are lower-triangular
    boolean causal masks (i.e. safe to strip and replace with
    ``is_causal=True``).

    Args:
        model: The model to probe.  Must expose ``model.config.vocab_size``
            for automatic dummy-input generation.
        sample_input_ids: Optional ``[B, S]`` int tensor of input IDs.
            If ``None``, a ``[1, 16]`` tensor of random token IDs is
            created automatically.

    Returns:
        ``True`` if every SDPA call used a causal mask (safe to strip),
        ``False`` otherwise (conservatively disables mask stripping).
    """
    # Resolve device from model parameters.
    try:
        device = next(model.parameters()).device
    except StopIteration:
        return False

    # Build dummy input_ids if not provided.
    if sample_input_ids is None:
        vocab_size = getattr(getattr(model, "config", None), "vocab_size", None)
        if vocab_size is None:
            return False
        sample_input_ids = torch.randint(0, vocab_size, (1, 16), device=device)

    all_causal: list[bool] = []
    saw_any_mask = False

    original_sdpa = F.scaled_dot_product_attention

    def _hook(*args, **kwargs):
        nonlocal saw_any_mask
        # Extract attn_mask (positional arg 3 or kwarg).
        attn_mask = args[3] if len(args) > 3 else kwargs.get("attn_mask", None)
        is_causal = kwargs.get("is_causal", False) if len(args) <= 5 else args[5]

        if attn_mask is not None and not is_causal:
            saw_any_mask = True
            all_causal.append(_is_lower_triangular_bool_mask(attn_mask))

        return original_sdpa(*args, **kwargs)

    F.scaled_dot_product_attention = _hook
    try:
        with torch.no_grad():
            model(sample_input_ids)
    except Exception:
        logger.debug("detect_causal_mask: forward pass failed", exc_info=True)
        return False
    finally:
        F.scaled_dot_product_attention = original_sdpa

    if not saw_any_mask:
        # No masks observed — nothing to strip.
        return False

    return all(all_causal)


def _sdpa_is_fusible(node: Node, strip_causal_mask: bool = False) -> Tuple[bool, bool]:
    """Check if an SDPA node is compatible with our FP8 fused kernel.

    Args:
        node: The SDPA FX graph node to check.
        strip_causal_mask: If ``True``, the pre-flight ``detect_causal_mask``
            confirmed that every mask is a causal mask, so nodes with a
            mask and ``is_causal=False`` can be stripped.  If ``False``
            (default), any node that carries a mask is considered
            non-fusible.

    Returns:
        (is_fusible, needs_mask_strip) where needs_mask_strip indicates
        the attn_mask is a materialized causal mask that should be
        stripped (set to None) with is_causal set to True.
    """
    args = node.args
    kwargs = node.kwargs

    # attn_mask (arg index 3): must be None or a validated causal mask.
    attn_mask = args[3] if len(args) > 3 else kwargs.get("attn_mask", None)

    # is_causal (arg index 5, default False).
    is_causal = args[5] if len(args) > 5 else kwargs.get("is_causal", False)

    needs_mask_strip = False
    if attn_mask is not None:
        if not is_causal and strip_causal_mask and isinstance(attn_mask, Node):
            needs_mask_strip = True
        else:
            return False, False

    # dropout_p (arg index 4): must be 0.0.
    dropout_p = args[4] if len(args) > 4 else kwargs.get("dropout_p", 0.0)
    if dropout_p != 0.0:
        return False, False

    return True, needs_mask_strip


def _strip_causal_mask(node: Node) -> None:
    """Strip a materialized causal mask from an SDPA node.

    Sets attn_mask=None and is_causal=True in the node's args/kwargs.
    """
    args = list(node.args)
    kwargs = dict(node.kwargs)

    # Set attn_mask (index 3) to None
    if len(args) > 3:
        args[3] = None
    elif "attn_mask" in kwargs:
        kwargs["attn_mask"] = None

    # Set is_causal (index 5) to True
    if len(args) > 5:
        args[5] = True
    elif "is_causal" in kwargs:
        kwargs["is_causal"] = True
    else:
        # is_causal wasn't in args or kwargs; add to kwargs
        kwargs["is_causal"] = True

    node.args = tuple(args)
    node.kwargs = kwargs

    logger.info("Stripped causal mask from SDPA node: %s", node.name)


def _get_sdpa_params(node: Node) -> Tuple[bool, float, bool]:
    """Extract is_causal, scale, and enable_gqa from an SDPA node.

    These parameters are passed through to the fused kernel.

    Returns:
        (is_causal, scale, enable_gqa) where scale uses 0.0 as a sentinel
        for "default" (i.e., 1/sqrt(D)), matching the custom op convention.
    """
    args = node.args
    kwargs = node.kwargs

    # is_causal (arg index 5, default False).
    is_causal = args[5] if len(args) > 5 else kwargs.get("is_causal", False)

    # scale (arg index 6, default None meaning 1/sqrt(D)).
    scale = args[6] if len(args) > 6 else kwargs.get("scale", None)

    # enable_gqa (arg index 7, default False).
    enable_gqa = args[7] if len(args) > 7 else kwargs.get("enable_gqa", False)

    # Convert None to our sentinel value 0.0.
    if scale is None:
        scale = 0.0

    return is_causal, scale, enable_gqa


def _get_sdpa_qkv(
    node: Node,
) -> Optional[Tuple[Node, Node, Node]]:
    """Extract Q, K, V input nodes from an SDPA node.

    Handles both positional args and kwargs since the Python-level builtin
    (torch._C._nn.scaled_dot_product_attention) may use kwargs while the
    ATen op (torch.ops.aten.scaled_dot_product_attention.default) uses
    positional args.

    Returns:
        (q_node, k_node, v_node) or None if extraction fails.
    """
    args = node.args
    kwargs = node.kwargs

    q = args[0] if len(args) > 0 else kwargs.get("query", None)
    k = args[1] if len(args) > 1 else kwargs.get("key", None)
    v = args[2] if len(args) > 2 else kwargs.get("value", None)

    if not all(isinstance(n, Node) for n in (q, k, v)):
        return None

    return q, k, v


# ============================================================================
# Section 6: NeoX/LLaMA RoPE Pattern Detection
# ============================================================================
#
# The NeoX/LLaMA RoPE variant is the most common implementation, used in
# ~90% of modern transformer models (LLaMA, Mistral, FLUX, Gemma, etc.).
#
# The math:
#
#   rotate_half(x):
#     x1 = x[..., :D//2]          # first half of head dimension
#     x2 = x[..., D//2:]          # second half of head dimension
#     return cat(-x2, x1, dim=-1) # swap halves and negate the first
#
#   apply_rope(x, cos, sin):
#     return x * cos + rotate_half(x) * sin
#
# In the pre-grad FX graph, this becomes these ATen ops:
#
#   # rotate_half(x):
#   %slice_x1 = aten.slice.Tensor(x, 3, 0, D//2)          # x[..., :D//2]
#   %slice_x2 = aten.slice.Tensor(x, 3, D//2, maxint)     # x[..., D//2:]
#   %neg_x2   = aten.neg.default(%slice_x2)                # -x2
#   %rot_half = aten.cat.default([%neg_x2, %slice_x1], 3)  # cat(-x2, x1)
#
#   # apply_rope(x, cos, sin):
#   %mul_x_cos   = aten.mul.Tensor(x, %cos_expanded)       # x * cos
#   %mul_rot_sin = aten.mul.Tensor(%rot_half, %sin_expanded)# rotate_half(x)*sin
#   %rope_out    = aten.add.Tensor(%mul_x_cos, %mul_rot_sin)# final sum
#
# Note: In the FX graph, dim=-1 is normalized to dim=3 for 4D tensors.
# We check for both -1 and 3 to be safe.
#
# Our detection works bottom-up: starting from a candidate node, we check
# if it matches the "add(mul(x, cos), mul(rotate_half(x), sin))" shape.


def _detect_rotate_half(cat_node: Node) -> Optional[Node]:
    """Detect if a node is the output of rotate_half(x).

    rotate_half(x) = cat(-x[..., D//2:], x[..., :D//2], dim=-1)

    Handles both ATen-level and Dynamo-level (Python operator) graphs.

    Args:
        cat_node: A node to check.

    Returns:
        The source tensor x (the input to rotate_half), or None if the
        pattern doesn't match.
    """
    # Step 1: Must be a cat op with dim=-1 or dim=3.
    # ATen: aten.cat.default([tensors], dim)
    # Dynamo: torch.cat([tensors], dim=...) or torch.cat([tensors])
    if not _is_op(cat_node, torch.ops.aten.cat.default, torch.cat):
        return None

    if len(cat_node.args) < 1:
        return None

    tensors_list = cat_node.args[0]

    # Get cat dim from positional arg or kwarg.
    if len(cat_node.args) >= 2:
        cat_dim = cat_node.args[1]
    else:
        cat_dim = cat_node.kwargs.get("dim", 0)

    if cat_dim not in (-1, 3):
        return None

    # Step 2: Must concatenate exactly 2 tensors: [neg_part, pos_part].
    if not isinstance(tensors_list, (list, tuple)) or len(tensors_list) != 2:
        return None

    neg_part = tensors_list[0]  # Expected: neg(slice(x, ..., D//2, ...))
    pos_part = tensors_list[1]  # Expected: slice(x, ..., 0, D//2)

    if not isinstance(neg_part, Node) or not isinstance(pos_part, Node):
        return None

    # Step 3: neg_part should be neg(slice_of_second_half).
    if not _is_op(neg_part, torch.ops.aten.neg.default, operator.neg, torch.neg):
        return None

    neg_input = neg_part.args[0]  # Should be slice(x, ..., D//2, ...)
    if not isinstance(neg_input, Node):
        return None

    # Step 4: Both neg_input and pos_part should be slice ops.
    # ATen: aten.slice.Tensor(input, dim, start, end)
    # Dynamo: operator.getitem(input, (..., slice(start, end)))
    #   or aten.slice.Tensor (Dynamo sometimes lowers slicing to ATen)
    return _match_rotate_half_slices(neg_input, pos_part)


def _match_rotate_half_slices(neg_input: Node, pos_part: Node) -> Optional[Node]:
    """Match the slice patterns in rotate_half.

    Handles:
      - ATen: aten.slice.Tensor(x, dim, start, end)
      - Dynamo: operator.getitem(x, (slice(None), ..., slice(start, end)))

    Returns the source tensor x, or None.
    """
    # Try ATen slice pattern.
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

    # Try Dynamo getitem pattern: operator.getitem(x, (..., slice(s, e)))
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

        # pos_part: slice(0, D//2) or slice(None, D//2)
        if pos_slice.start not in (0, None):
            return None
        if pos_slice.stop is None:
            return None

        # neg_input: slice(D//2, None)
        if neg_slice.start is None:
            return None
        if neg_slice.start != pos_slice.stop:
            return None

        return slice_neg_source

    return None


def _extract_last_dim_slice(idx) -> Optional[slice]:
    """Extract the slice on the last dimension from a getitem index.

    Supports:
      - (Ellipsis, slice(...))
      - (slice(None), slice(None), slice(None), slice(...))
      - Just a slice on its own (for 1D)

    Returns the last-dimension slice, or None if not recognized.
    """
    if isinstance(idx, tuple):
        if len(idx) >= 2 and idx[0] is Ellipsis and isinstance(idx[1], slice):
            return idx[1]
        # (slice(None), ..., slice(start, end)) — last element is the slice
        if len(idx) >= 1 and isinstance(idx[-1], slice):
            # Check all preceding are slice(None) or Ellipsis
            for i in range(len(idx) - 1):
                if idx[i] is not Ellipsis and idx[i] != slice(None):
                    return None
            return idx[-1]
    elif isinstance(idx, slice):
        return idx
    return None


def _detect_interleaved_rotation(node: Node) -> Optional[Node]:
    """Detect the FLUX-style interleaved rotation pattern.

    FLUX uses an interleaved real/imaginary rotation instead of the
    NeoX half-split:

        x_real, x_imag = x.reshape(*x.shape[:-1], -1, 2).unbind(-1)
        x_rotated = torch.stack([-x_imag, x_real], dim=-1).flatten(3)

    This function detects that pattern and returns the source tensor x
    (before reshape/unbind).

    Args:
        node: A node to check (should be the flatten node after stack).

    Returns:
        The source tensor x (before reshape/unbind), or None if the
        pattern doesn't match.
    """
    # Step 1: Must be flatten(stack_result, start_dim).
    # At Dynamo level: call_method "flatten" or aten.flatten.using_ints.
    if not _is_op(
        node,
        torch.ops.aten.flatten.using_ints,
        "flatten",
    ):
        return None

    if len(node.args) < 1:
        return None

    stack_node = node.args[0]
    if not isinstance(stack_node, Node):
        return None

    # Step 2: Must be stack([neg_x_imag, x_real], dim=-1).
    if not _is_op(stack_node, torch.ops.aten.stack.default, torch.stack):
        return None

    if len(stack_node.args) < 1:
        return None

    tensors_list = stack_node.args[0]

    # Get stack dim from positional arg or kwarg.
    if len(stack_node.args) >= 2:
        stack_dim = stack_node.args[1]
    else:
        stack_dim = stack_node.kwargs.get("dim", 0)

    if stack_dim != -1:
        return None

    if not isinstance(tensors_list, (list, tuple)) or len(tensors_list) != 2:
        return None

    neg_part = tensors_list[0]  # Expected: neg(x_imag)
    pos_part = tensors_list[1]  # Expected: x_real

    if not isinstance(neg_part, Node) or not isinstance(pos_part, Node):
        return None

    # Step 3: neg_part should be neg(x_imag).
    if not _is_op(neg_part, torch.ops.aten.neg.default, operator.neg, torch.neg):
        return None

    x_imag = neg_part.args[0]
    if not isinstance(x_imag, Node):
        return None

    # Step 4: x_real (pos_part) and x_imag should come from getitem on the
    # same unbind result: getitem(unbind_result, 0) and getitem(unbind_result, 1).
    if not _is_op(pos_part, operator.getitem):
        return None
    if not _is_op(x_imag, operator.getitem):
        return None

    real_source = pos_part.args[0]
    imag_source = x_imag.args[0]

    if real_source is not imag_source:
        return None

    real_idx = pos_part.args[1]
    imag_idx = x_imag.args[1]

    if real_idx != 0 or imag_idx != 1:
        return None

    # Step 5: The common source should be unbind(-1) on a reshape result.
    unbind_node = real_source
    if not isinstance(unbind_node, Node):
        return None

    if not _is_op(unbind_node, torch.ops.aten.unbind.int, "unbind"):
        return None

    # Get unbind dim.
    if len(unbind_node.args) >= 2:
        unbind_dim = unbind_node.args[1]
    else:
        unbind_dim = unbind_node.kwargs.get("dim", 0)

    if unbind_dim != -1:
        return None

    reshape_node = unbind_node.args[0]
    if not isinstance(reshape_node, Node):
        return None

    # Step 6: reshape_node should be x.reshape(..., -1, 2) or x.view(..., -1, 2).
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
    """Detect any supported rotation pattern used in RoPE.

    Tries:
      1. NeoX half-split: cat(-x[..., D//2:], x[..., :D//2], dim=-1)
      2. FLUX interleaved: stack([-x_imag, x_real], dim=-1).flatten(3)

    Args:
        node: A node to check.

    Returns:
        (source_tensor, is_interleaved) or None.
        is_interleaved=False for NeoX half-split, True for FLUX interleaved.
    """
    result = _detect_rotate_half(node)
    if result is not None:
        return result, False  # NeoX half-split

    result = _detect_interleaved_rotation(node)
    if result is not None:
        return result, True  # Interleaved

    return None


def _detect_neox_rope(node: Node) -> Optional[RoPEMatch]:
    """Detect the NeoX/LLaMA RoPE pattern at a given node.

    The pattern is:
        add(mul(x, cos), mul(rotate_half(x), sin))

    Since both add and mul are commutative, we try both orderings:
      - add(mul(x, cos), mul(rotate_half(x), sin))
      - add(mul(rotate_half(x), sin), mul(x, cos))
    And within each mul, either operand could be the tensor vs. the scale.

    Args:
        node: A graph node to check.

    Returns:
        A RoPEMatch if the pattern is detected, None otherwise.
    """
    # Step 1: Must be an add node.
    if not _is_op(node, torch.ops.aten.add.Tensor, operator.add):
        return None

    if len(node.args) < 2:
        return None

    left = node.args[0]
    right = node.args[1]

    if not isinstance(left, Node) or not isinstance(right, Node):
        return None

    # Step 2: Both add inputs must be mul nodes.
    if not _is_op(left, torch.ops.aten.mul.Tensor, operator.mul):
        return None
    if not _is_op(right, torch.ops.aten.mul.Tensor, operator.mul):
        return None

    # Step 3: Figure out which mul is "x * cos" and which is
    # "rotation(x) * sin".
    #
    # We try both orderings of the add (left/right), and within each mul
    # we try both orderings of the operands (since mul is commutative).

    def _try_match(x_mul: Node, rot_mul: Node) -> Optional[RoPEMatch]:
        """Try matching x_mul as (x * cos) and rot_mul as (rotation(x) * sin)."""
        rot_a, rot_b = rot_mul.args[0], rot_mul.args[1]

        # Try both orderings within rot_mul to find the rotation output.
        for rot_candidate, sin_candidate in [(rot_a, rot_b), (rot_b, rot_a)]:
            if not isinstance(rot_candidate, Node):
                continue

            # Trace through dtype casts (e.g., .float()) to find the
            # actual rotation output. In FLUX, rotation is computed
            # in bf16 and then cast to float32 for the RoPE multiplication.
            rot_unwrapped = _trace_through_views(rot_candidate)

            # Check if rot_unwrapped is the output of a rotation(x).
            rotation_result = _detect_rotation(rot_unwrapped)
            if rotation_result is None:
                continue
            x_from_rotate, is_interleaved = rotation_result

            # Now match x_mul as (x * cos) where x == x_from_rotate.
            x_a, x_b = x_mul.args[0], x_mul.args[1]

            # Try both orderings within x_mul.
            for x_candidate, cos_candidate in [(x_a, x_b), (x_b, x_a)]:
                if not isinstance(x_candidate, Node):
                    continue

                # The x in "x * cos" must trace back to the SAME source
                # tensor as the x in "rotate_half(x)". In FLUX, both x and
                # rotate_half(x) are independently cast to float32, so we
                # trace through views on both sides before comparing.
                x_traced = _trace_through_views(x_candidate)
                if x_traced is not x_from_rotate:
                    continue

                # Success! Trace cos and sin back through view-like ops
                # (unsqueeze, expand, etc.) to find the [S, D] source tensors.
                cos_source = _trace_through_views(cos_candidate)
                sin_source = _trace_through_views(sin_candidate)

                return RoPEMatch(
                    pre_rope_input=x_from_rotate,
                    cos_node=cos_source,
                    sin_node=sin_source,
                    rope_interleaved=is_interleaved,
                )

        return None

    # Try both orderings of the add.
    # Case 1: left = x*cos, right = rotation(x)*sin
    result = _try_match(left, right)
    if result is not None:
        return result

    # Case 2: left = rotation(x)*sin, right = x*cos
    result = _try_match(right, left)
    if result is not None:
        return result

    return None


def _detect_rope(node: Node) -> Optional[RoPEMatch]:
    """Detect any supported RoPE variant at a given node.

    Currently supports:
      - NeoX/LLaMA variant (most common, ~90% of models)

    Future variants to add:
      - GPT-J interleaved (uses stride-2 indexing instead of half-split)
      - Complex multiplication (uses view_as_complex / view_as_real)

    Args:
        node: A graph node to check.

    Returns:
        A RoPEMatch if a RoPE pattern is detected, None otherwise.
    """
    # Try NeoX/LLaMA variant (most common).
    match = _detect_neox_rope(node)
    if match is not None:
        return match

    # Future: add GPT-J and complex variants here.
    # match = _detect_gptj_rope(node)
    # match = _detect_complex_rope(node)

    return None


# ============================================================================
# Section 7: Graph Surgery
# ============================================================================


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
) -> None:
    """Replace an SDPA node (and its RoPE + transpose predecessors) with the fused op.

    This inserts a new node calling our torchao::fp8_fa3_rope_sdpa custom op
    and redirects all users of the original SDPA output to use the new node
    instead.

    After this function returns, the old SDPA node and its predecessor ops
    (RoPE slices, neg, cat, mul, add, transpose) become dead code and will
    be removed by graph.eliminate_dead_code() in the main pass.

    Args:
        graph: The FX graph being modified.
        sdpa_node: The original SDPA node to replace.
        pre_rope_q: Pre-RoPE query tensor node, shape [B, S, H, D].
        pre_rope_k: Pre-RoPE key tensor node, shape [B, S, H, D].
        v_input: Value tensor node, shape [B, S, H, D].
        cos_node: Cosine frequencies node, shape [S, D].
        sin_node: Sine frequencies node, shape [S, D].
        is_causal: Whether causal masking is enabled.
        scale: Attention scale factor (0.0 = use default 1/sqrt(D)).
        enable_gqa: Whether grouped query attention is enabled.
        rope_interleaved: Whether to use interleaved (FLUX) or NeoX (LLaMA) RoPE.
    """
    # Insert the fused op node right before the old SDPA node in the graph.
    # This ensures all input nodes are defined before our new node.
    with graph.inserting_before(sdpa_node):
        fused_node = graph.call_function(
            torch.ops.torchao.fp8_fa3_rope_sdpa.default,
            args=(pre_rope_q, pre_rope_k, v_input, cos_node, sin_node),
            kwargs={
                "is_causal": is_causal,
                "scale": scale,
                "enable_gqa": enable_gqa,
                "rope_interleaved": rope_interleaved,
            },
        )

    # Copy metadata from the SDPA node to preserve shape/dtype info
    # for downstream compiler passes.
    fused_node.meta = sdpa_node.meta.copy()

    # Redirect all consumers of the SDPA output to use the fused op instead.
    sdpa_node.replace_all_uses_with(fused_node)

    logger.info(
        "Fused RoPE + SDPA: replaced %s with %s",
        sdpa_node.name,
        fused_node.name,
    )


def _replace_sdpa_with_fp8(
    graph: Graph,
    sdpa_node: Node,
    q_node: Node,
    k_node: Node,
    v_node: Node,
    is_causal: bool,
    scale: float,
    enable_gqa: bool,
) -> None:
    """Replace a plain SDPA node with fp8_fa3_sdpa (no RoPE fusion).

    This is a direct drop-in replacement: Q, K, V stay in [B, H, S, D] layout
    (standard SDPA layout). No transpose unwrapping is needed.

    Args:
        graph: The FX graph being modified.
        sdpa_node: The original SDPA node to replace.
        q_node: Query input node.
        k_node: Key input node.
        v_node: Value input node.
        is_causal: Whether causal masking is enabled.
        scale: Attention scale factor (0.0 = use default 1/sqrt(D)).
        enable_gqa: Whether grouped query attention is enabled.
    """

    with graph.inserting_before(sdpa_node):
        fp8_node = graph.call_function(
            torch.ops.torchao.fp8_fa3_sdpa.default,
            args=(q_node, k_node, v_node),
            kwargs={
                "is_causal": is_causal,
                "scale": scale,
                "enable_gqa": enable_gqa,
            },
        )

    fp8_node.meta = sdpa_node.meta.copy()
    sdpa_node.replace_all_uses_with(fp8_node)

    logger.info(
        "Replaced SDPA with FP8: %s -> %s",
        sdpa_node.name,
        fp8_node.name,
    )


# ============================================================================
# Section 8: Main Fusion Pass
# ============================================================================


def rope_sdpa_fusion_pass(
    graph: Graph, *, fuse_rope: bool = True, strip_causal_mask: bool = False
) -> None:
    """Main entry point: detect and replace SDPA patterns in the FX graph.

    This function is called by torch.compile as a pre-grad custom pass
    (registered via torch._inductor.config.pre_grad_custom_pass).

    It handles ALL fusible SDPA nodes in the graph:
      - SDPA with RoPE on Q/K → replaced with fused fp8_fa3_rope_sdpa custom op
      - SDPA without RoPE → replaced with fp8_fa3_sdpa custom op

    Algorithm:
      1. Collect all SDPA nodes in the graph.
      2. For each SDPA node, check fusibility (no mask, no dropout).
      3. Try Pattern A: unwrap transpose on Q/K inputs, then detect RoPE.
      4. Try Pattern B: detect RoPE directly on Q/K inputs, then unwrap
         transpose on the RoPE's inner input.
      5. If RoPE found on both Q and K, replace with fused rope custom op.
      6. Otherwise, replace with non-rope FP8 SDPA custom op.
      7. Run dead code elimination to clean up replaced nodes.

    Args:
        graph: The FX graph to transform (modified in-place).
        fuse_rope: If True (default), attempt to detect and fuse RoPE
            patterns (Patterns A and B).  If False, skip RoPE detection
            and replace all fusible SDPA nodes with the non-rope FP8
            SDPA kernel only.
        strip_causal_mask: If True, the pre-flight ``detect_causal_mask``
            confirmed that every attention mask is a materialized causal
            mask, so SDPA nodes carrying a mask can have it stripped and
            replaced with ``is_causal=True``.
    """
    # Collect SDPA nodes upfront. We snapshot the list before modifying
    # the graph to avoid iterator invalidation issues.
    sdpa_nodes = [n for n in graph.nodes if _is_sdpa_node(n)]

    if not sdpa_nodes:
        logger.debug("RoPE + SDPA fusion: found 0 SDPA nodes in graph")
        return

    fused_count = 0
    fp8_count = 0

    for sdpa_node in sdpa_nodes:
        # Check if this SDPA call is compatible with our fused kernel.
        is_fusible, needs_mask_strip = _sdpa_is_fusible(
            sdpa_node, strip_causal_mask=strip_causal_mask
        )
        if not is_fusible:
            logger.debug("Skipping non-fusible SDPA: %s", sdpa_node.name)
            continue

        if needs_mask_strip:
            _strip_causal_mask(sdpa_node)

        # Extract is_causal, scale, and enable_gqa to pass through to the fused op.
        is_causal, scale, enable_gqa = _get_sdpa_params(sdpa_node)

        # Get Q, K, V input nodes from the SDPA call.
        qkv = _get_sdpa_qkv(sdpa_node)
        if qkv is None:
            continue
        q_node, k_node, v_node = qkv

        # ----------------------------------------------------------------
        # Try RoPE fusion (Pattern A and Pattern B)
        # ----------------------------------------------------------------
        # When fuse_rope is False, skip RoPE detection entirely and fall
        # through to the non-rope FP8 SDPA replacement below.

        if fuse_rope:
            # V always needs its transpose unwrapped for the fused kernel.
            # The fused kernel expects V in [B, S, H, D] layout.
            v_pre_transpose = _unwrap_transpose(v_node)

            # ------------------------------------------------------------
            # Try Pattern A: RoPE -> transpose -> SDPA
            # ------------------------------------------------------------
            #
            # This is the FLUX-style layout where RoPE operates on the
            # "natural" [B, S, H, D] layout, then the result is transposed
            # to [B, H, S, D] for SDPA.
            #
            # Data flow:
            #   Q: pre_rope_q [B,S,H,D] -> RoPE -> q_roped [B,S,H,D]
            #                            -> transpose -> q_t [B,H,S,D] -> SDPA
            #   K: (same as Q)
            #   V: v [B,S,H,D] -> transpose -> v_t [B,H,S,D] -> SDPA
            #
            # Detection: unwrap transpose on Q/K, then detect RoPE.

            q_pre_transpose = _unwrap_transpose(q_node)
            k_pre_transpose = _unwrap_transpose(k_node)

            if q_pre_transpose is not None and k_pre_transpose is not None:
                # Q and K both go through transpose before SDPA.
                # Trace through dtype casts (e.g., .to(bf16)) between transpose
                # and RoPE. FLUX computes RoPE in float32, then casts back.
                q_pre_cast = _trace_through_views(q_pre_transpose)
                k_pre_cast = _trace_through_views(k_pre_transpose)

                q_rope = _detect_rope(q_pre_cast)
                k_rope = _detect_rope(k_pre_cast)

                if q_rope is not None and k_rope is not None:
                    # Both Q and K have RoPE applied before the transpose.

                    # Trace pre_rope_input through dtype casts (e.g., .float())
                    # to get the original bf16 tensor. FLUX computes RoPE in
                    # float32 (x.float() * cos + rotate_half(x.float()) * sin),
                    # but the fused kernel expects the bf16 input.
                    pre_rope_q = _trace_through_views(q_rope.pre_rope_input)
                    pre_rope_k = _trace_through_views(k_rope.pre_rope_input)

                    # V must also have a transpose we can unwrap.
                    if v_pre_transpose is None:
                        logger.debug(
                            "Pattern A: V has no transpose, skipping: %s",
                            sdpa_node.name,
                        )
                        continue

                    # Validate and reshape cos/sin to [S, D] for the fused kernel.
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
                    )
                    fused_count += 1
                    continue  # Move to next SDPA node.

            # ------------------------------------------------------------
            # Try Pattern B: transpose -> RoPE -> SDPA
            # ------------------------------------------------------------
            #
            # This is the HuggingFace-style layout where the tensor is first
            # transposed to [B, H, S, D], then RoPE is applied in that layout.
            #
            # Data flow (non-GQA):
            #   Q: q [B,S,H,D] -> transpose -> q_t [B,H,S,D]
            #                   -> RoPE -> q_roped [B,H,S,D] -> SDPA
            #   K: (same as Q)
            #   V: v [B,S,H,D] -> transpose -> v_t [B,H,S,D] -> SDPA
            #
            # Data flow (GQA with repeat_kv):
            #   Q: (same as non-GQA, Q has all heads)
            #   K: k [B,S,Hkv,D] -> transpose -> RoPE -> repeat_kv -> SDPA
            #   V: v [B,S,Hkv,D] -> transpose -> repeat_kv -> SDPA
            #
            # Detection: detect RoPE directly on SDPA's Q/K inputs (no
            # transpose between RoPE and SDPA), then trace the RoPE's inner
            # input backward through a transpose to find the [B,S,H,D] source.
            # For GQA, unwrap repeat_kv on K/V first, then detect RoPE.

            # Try to detect RoPE on Q (always direct, no repeat_kv on Q).
            q_rope = _detect_rope(_trace_through_views(q_node))

            # Try to detect RoPE on K. For GQA models, K may go through
            # repeat_kv (reshape←expand←unsqueeze) after RoPE. Try direct
            # detection first, then try unwrapping repeat_kv.
            k_rope = _detect_rope(_trace_through_views(k_node))
            gqa_unwrapped = False
            if k_rope is None:
                k_pre_repeat = _unwrap_repeat_kv(k_node)
                if k_pre_repeat is not None:
                    k_rope = _detect_rope(_trace_through_views(k_pre_repeat))
                    if k_rope is not None:
                        gqa_unwrapped = True

            if q_rope is not None and k_rope is not None:
                # RoPE detected directly on SDPA inputs (no intervening transpose).
                # The RoPE's pre_rope_input is in [B, H, S, D] (post-transpose).
                # We need to trace it back through dtype casts (e.g., .float())
                # and the transpose to get the original [B, S, H, D] bf16 tensor
                # for our fused kernel.
                q_bshd = _unwrap_transpose(_trace_through_views(q_rope.pre_rope_input))
                k_bshd = _unwrap_transpose(_trace_through_views(k_rope.pre_rope_input))

                if q_bshd is not None and k_bshd is not None:
                    # Found the [B, S, H, D] sources for Q and K.

                    # For V: if GQA was unwrapped on K, V also has repeat_kv.
                    # Unwrap repeat_kv on V first, then unwrap transpose.
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

                    # Validate and reshape cos/sin to [S, D] for the fused kernel.
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

                    # When we unwrapped repeat_kv, the fused kernel must handle
                    # GQA natively since K/V now have fewer heads than Q.
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
                    )
                    fused_count += 1
                    continue

        # ----------------------------------------------------------------
        # No RoPE detected (or fuse_rope=False) — replace with non-rope FP8 SDPA
        # ----------------------------------------------------------------
        #
        # This SDPA node doesn't have RoPE on its Q/K inputs (or the RoPE
        # pattern wasn't recognized). Replace it with fp8_fa3_sdpa which
        # does FP8 quantization + attention without RoPE fusion.
        # Q, K, V stay in [B, H, S, D] layout (standard SDPA layout).

        # Check head dimension compatibility. FA3 only supports head_dim <= 256.
        q_shape = _get_node_shape(q_node)
        if q_shape is not None and q_shape[-1] > 256:
            logger.debug(
                "Skipping FP8 replacement: head_dim=%d > 256 for %s",
                q_shape[-1],
                sdpa_node.name,
            )
            continue

        _replace_sdpa_with_fp8(
            graph=graph,
            sdpa_node=sdpa_node,
            q_node=q_node,
            k_node=k_node,
            v_node=v_node,
            is_causal=is_causal,
            scale=scale,
            enable_gqa=enable_gqa,
        )
        fp8_count += 1

    replaced_count = fused_count + fp8_count
    print(
        f"[fusion_pass] Found {len(sdpa_nodes)} SDPA node(s): "
        f"{fused_count} RoPE-fused, {fp8_count} FP8-replaced"
    )
    logger.info(
        "Found %d SDPA node(s): %d RoPE-fused, %d FP8-replaced",
        len(sdpa_nodes),
        fused_count,
        fp8_count,
    )

    if replaced_count > 0:
        # Remove nodes that are no longer used. The old RoPE ops (slice, neg,
        # cat, mul, add), transpose nodes, and SDPA nodes that were replaced
        # should now have zero users and will be eliminated.
        graph.eliminate_dead_code()
        logger.info(
            "Fusion pass complete: %d RoPE-fused, %d FP8-replaced",
            fused_count,
            fp8_count,
        )
