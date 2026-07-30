# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD 3-Clause license found in the
# LICENSE file in the root directory of this source tree.

"""FX pass that duplicates shared dequantize nodes so each consumer owns one.

When a single ``dequantize`` node feeds several downstream users, those users
cannot be annotated and lowered independently. :class:`DuplicateDQPass` clones
such a ``dequantize`` node once per annotated user (skipping the dynamic
quantization ``choose_qparams -> getitem -> q -> dq`` pattern) so that the
subsequent quantization lowering can rewrite each branch on its own.
"""

import logging
import operator

import torch
from torch.fx.node import map_arg
from torch.fx.passes.infra.pass_base import PassBase, PassResult

from torchao.quantization.pt2e.quantizer.quantizer import Q_ANNOTATION_KEY
from torchao.quantization.pt2e.utils import _filter_sym_size_users

from .utils import is_valid_annotation

logger = logging.getLogger(__name__)
logger.setLevel(logging.WARNING)

__all__ = ["DuplicateDQPass"]

_QUANTIZE_OPS = [
    torch.ops.quantized_decomposed.quantize_per_tensor.default,
    torch.ops.quantized_decomposed.quantize_per_tensor.tensor,
    torch.ops.quantized_decomposed.quantize_per_channel.default,
]

_DEQUANTIZE_OPS = [
    torch.ops.quantized_decomposed.dequantize_per_tensor.default,
    torch.ops.quantized_decomposed.dequantize_per_tensor.tensor,
    torch.ops.quantized_decomposed.dequantize_per_channel.default,
]


def _maybe_duplicate_dq(
    gm: torch.fx.GraphModule, dq_node: torch.fx.Node, user: torch.fx.Node
) -> None:
    """Clone ``dq_node`` for ``user`` if ``user`` carries a valid annotation.

    The freshly copied dequantize node is inserted right after ``dq_node`` and
    ``user``'s args/kwargs are rewired to reference the copy, leaving the
    original ``dq_node`` for its remaining consumers. Users without a valid
    quantization annotation are left untouched.
    """
    annotation = user.meta.get(Q_ANNOTATION_KEY, None)
    if not is_valid_annotation(annotation):
        return
    with gm.graph.inserting_after(dq_node):
        new_node = gm.graph.node_copy(dq_node)

        def maybe_replace_node(n: torch.fx.Node) -> torch.fx.Node:
            if n == dq_node:
                return new_node
            else:
                return n

        new_args = map_arg(user.args, maybe_replace_node)
        new_kwargs = map_arg(user.kwargs, maybe_replace_node)
        user.args = new_args  # type: ignore[assignment]
        user.kwargs = new_kwargs  # type: ignore[assignment]


class DuplicateDQPass(PassBase):
    """Duplicate shared dequantize nodes so each user owns a private copy."""

    def call(self, graph_module: torch.fx.GraphModule) -> PassResult:
        """Run the duplication pass over ``graph_module`` and return the result.

        Each ``dequantize`` node with more than one (non ``sym_size``) user is
        cloned per user, except for the dynamic-quantization
        ``choose_qparams -> getitem -> q -> dq`` pattern which is left shared.
        Dead code is eliminated and the module is recompiled before returning.
        """
        for node in graph_module.graph.nodes:
            if node.op == "call_function" and node.target in _DEQUANTIZE_OPS:
                dq_users = _filter_sym_size_users(node)
                if len(dq_users) <= 1:
                    continue
                # Do not duplicate dq for dynamic quantization
                # Pattern: choose_qparam - getitem - q - dq
                q_node = node.args[0]
                if q_node.op == "call_function" and q_node.target in _QUANTIZE_OPS:
                    getitem_node = q_node.args[1]
                    if (
                        isinstance(getitem_node, torch.fx.node.Node)
                        and getitem_node.op == "call_function"
                        and getitem_node.target == operator.getitem
                    ):
                        choose_qparam_node = getitem_node.args[0]
                        if (
                            isinstance(choose_qparam_node, torch.fx.node.Node)
                            and choose_qparam_node.op == "call_function"
                            and choose_qparam_node.target
                            == torch.ops.quantized_decomposed.choose_qparams.tensor
                        ):
                            continue
                for user in dq_users:
                    _maybe_duplicate_dq(graph_module, node, user)
        graph_module.graph.eliminate_dead_code()
        graph_module.recompile()
        return PassResult(graph_module, True)
