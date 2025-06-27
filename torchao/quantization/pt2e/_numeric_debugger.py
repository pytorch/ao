# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD 3-Clause license found in the
# LICENSE file in the root directory of this source tree.

import copy
import logging
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Callable, Optional

import torch
from torch.ao.ns.fx.utils import compute_sqnr
from torch.export import ExportedProgram
from torch.fx import GraphModule, Node
from torch.nn import functional as F

from torchao.utils import TORCH_VERSION_AT_LEAST_2_6

if TORCH_VERSION_AT_LEAST_2_6:
    from torch.fx.traceback import NodeSource

from .graph_utils import bfs_trace_with_node_process

NUMERIC_DEBUG_HANDLE_KEY = "numeric_debug_handle"
CUSTOM_KEY = "custom"
FROM_NODE_KEY = "from_node"

log = logging.getLogger(__name__)


@dataclass(frozen=True)
class NodeSourceDebugInfo:
    """
    Contains node source information for locating the node in the original graph.
    This replaces the numeric debug handle approach with direct node source info.
    """

    # The name of the node in the graph, e.g. "conv2d"
    name: str

    # The unique id of the graph that the node belongs to.
    graph_id: int


# This function is no longer used for torchao debug flow, but is kept here for backward compatibility.
def generate_numeric_debug_handle(ep: ExportedProgram) -> None:
    """
    Attach numeric_debug_handle_id for all nodes in the graph module of the given
    ExportedProgram, like conv2d, squeeze, conv1d, etc, except for placeholder.
    Notice that nodes like getattr are out of scope since they are not in the graph.

    The graph nodes of input exported program are modified inplace.

    Here's an example of using debug handle quantize flow::

        ep = export_for_training(eager_model, example_inputs)
        generate_numeric_debug_handle(ep)

        m = ep.module()
        quantizer = XNNPACKQuantizer()
        m = prepare_pt2e(m, quantizer)
        m = convert_pt2e(m)
    """

    # Sanity check the input data type
    if not isinstance(ep, ExportedProgram):
        raise ValueError(
            f"Expected ep to be ExportedProgram, got {type(ExportedProgram)}"
        )

    unique_id = 0

    def _find_max_id(node: torch.fx.Node) -> None:
        nonlocal unique_id
        unique_id = max(
            unique_id, node.meta.get(CUSTOM_KEY, {}).get(NUMERIC_DEBUG_HANDLE_KEY, 0)
        )

    def _assign_debug_handle(node: torch.fx.Node) -> None:
        nonlocal unique_id
        if CUSTOM_KEY not in node.meta:
            node.meta[CUSTOM_KEY] = {}

        if NUMERIC_DEBUG_HANDLE_KEY not in node.meta[CUSTOM_KEY]:
            node.meta[CUSTOM_KEY][NUMERIC_DEBUG_HANDLE_KEY] = unique_id
            unique_id += 1

    # Find the max ID that exists in the graph first, in case part of the graph
    # has already been annotated. This way we guarantee there are no duplicate
    # handle IDs.
    bfs_trace_with_node_process(ep, _find_max_id)

    unique_id += 1

    # Assign debug handles to all nodes in the graph that don't have one based on the
    # max ID found in the previous step.
    bfs_trace_with_node_process(ep, _assign_debug_handle)


def _extract_node_source_debug_info(node: Node) -> Optional[NodeSourceDebugInfo]:
    """
    Extract node source debug info from a node, or return None if the node
    does not need to be traced.

    Returns NodeSourceDebugInfo containing the name and graph_id from the
    node's greatest ancestor node source, or None if the node is not in
    the original graph.
    """

    def _get_greatest_ancestor_node_source(node: Node) -> "NodeSource":
        node_source = node.meta.get(FROM_NODE_KEY)[-1]

        while len(node_source.from_node) > 0:
            node_source = node_source.from_node[-1]

        return node_source

    def _is_node_in_original_graph(node: Node) -> bool:
        if (
            FROM_NODE_KEY not in node.meta
            or node.meta[FROM_NODE_KEY] is None
            or node.meta[FROM_NODE_KEY][-1].pass_name
            == "ExportedProgram.module().unlift()"
        ):
            # This node is not part of the ExportedProgram.module().graph, so it doesn't have a debug handle
            return False

        return True

    if node.op == "placeholder" or node.op == "output":
        # placeholder and output nodes don't have debug info
        return None

    if not _is_node_in_original_graph(node):
        return None

    greatest_ancestor_node_source = _get_greatest_ancestor_node_source(node)

    return NodeSourceDebugInfo(
        name=greatest_ancestor_node_source.name,
        graph_id=greatest_ancestor_node_source.graph_id,
    )


def _detach(x: object) -> object:
    detached: object = None
    if isinstance(x, torch.Tensor):
        detached = x.detach()
    elif isinstance(x, (list, tuple)):
        detached = type(x)([_detach(e) for e in x])
    elif isinstance(x, dict):
        detached = {k: _detach(e) for k, e in x.items()}
    else:
        detached = x
    return detached


def _tensor_shape_equals(x: object, y: object) -> bool:
    if isinstance(x, torch.Tensor) and isinstance(y, torch.Tensor):
        return x.shape == y.shape
    elif isinstance(x, (list, tuple)) and isinstance(y, (list, tuple)):
        return all(_tensor_shape_equals(e1, e2) for e1, e2 in zip(x, y))
    elif isinstance(x, dict) and isinstance(y, dict):
        all_equal = True
        for k in x:
            all_equal = all_equal and k in y and (_tensor_shape_equals(x[k], y[k]))
        return all_equal
    else:
        log.debug("Comparing non Tensors: %s and %s, they must be equal", x, y)
        return type(x) == type(y) and x == y


def _loss_fn(
    loss: Callable[[torch.Tensor, torch.Tensor], torch.Tensor], x: object, y: object
) -> object:
    """The returned loss will have the same structure as `x` and `y`, e.g.
    if both are Tensor, we'll return a Tensor
    if both are list, we'll return a list of Tensors
    if both are dict, we'll return a dict with the same key, and value being the loss between the
    two Tensors
    """
    if isinstance(x, torch.Tensor) and isinstance(y, torch.Tensor):
        return loss(x.to(torch.float32), y.to(torch.float32))
    elif isinstance(x, (list, tuple)) and isinstance(y, (list, tuple)):
        return type(x)([_loss_fn(loss, e1, e2) for e1, e2 in zip(x, y)])
    elif isinstance(x, dict) and isinstance(y, dict):
        return {k: _loss_fn(loss, e, y[k]) for k, e in x.items()}
    else:
        return None


class OutputLogger(torch.nn.Module):
    """
    Base class for capturing output values for nodes in a GraphModule, it only captures
    Tensor output currently, but we can extend it to work for other types of inputs later if needed
    """

    # Mark as impure so that calls to it will not be removed during DCE.
    _is_impure = True

    def __init__(
        self,
        debug_info: NodeSourceDebugInfo,
        node_name: Optional[str] = None,
        nn_module_stack: Optional[object] = None,
    ) -> None:
        super().__init__()
        self.node_name = node_name
        self.nn_module_stack = nn_module_stack
        self.debug_info = debug_info
        self.stats: list[object] = []

    def forward(self, x: object) -> object:
        self.stats.append(_detach(x))
        return x

    def __extra_repr__(self) -> str:
        return (
            f"debug_info={self.debug_info}, node_name={self.node_name}, "
            "nn_module_stack={self.nn_module_stack}, num_stats={len(self.stats)})"
        )


def _insert_logger(
    model: GraphModule, node: Node, debug_info: NodeSourceDebugInfo
) -> Node:
    """For a given node, adds an OutputLogger that observes the output of that node,
    and all its users use the OutputLogger output instead.
    The OutputLogger will contain the debug_info which can be used to compare
    graphs after transforms"""

    # to avoid circular dep
    from torchao.quantization.pt2e.utils import get_new_attr_name_with_prefix

    # add a logger after the node
    with model.graph.inserting_after(node):
        get_new_attr_name = get_new_attr_name_with_prefix(f"{node.name}_logger")
        logger_name = get_new_attr_name(model)
        setattr(
            model,
            logger_name,
            OutputLogger(debug_info, node.name, node.meta.get("nn_module_stack")),
        )
        logger_node = model.graph.call_module(logger_name, (node,), {})

    orig_users = list(node.users.keys())
    for user_node in orig_users:
        if user_node is logger_node:
            continue
        user_node.replace_input_with(node, logger_node)

    return logger_node


def prepare_for_propagation_comparison(model: GraphModule) -> GraphModule:
    """Add output loggers to unlifted node

    Args:
        model (GraphModule): original model
    Returns:
        a model with output loggers for all unlifted nodes
    """
    if not TORCH_VERSION_AT_LEAST_2_6:
        log.warning(
            "prepare_for_propagation_comparison is only supported for PyTorch 2.6+"
        )
        return model

    # don't change the original model
    model = copy.deepcopy(model)
    for n in model.graph.nodes:
        if (debug_info := _extract_node_source_debug_info(n)) is not None:
            _insert_logger(model, n, debug_info)

    model.recompile()
    return model


@dataclass(frozen=True)
class QuantizationComparisonResult:
    actual: torch.Tensor
    ref: torch.Tensor

    @property
    def mse_loss(self) -> object:
        return self.loss(F.mse_loss)

    @property
    def sqnr(self) -> object:
        return self.loss(compute_sqnr)

    def loss(
        self, loss_function: Callable[[torch.Tensor, torch.Tensor], torch.Tensor]
    ) -> object:
        return _loss_fn(loss_function, self.actual, self.ref)

    def __repr__(self) -> str:
        # Don't include the tensors themselves as they are quite large to print
        # out.
        return (
            f"QuantizationComparisonResult(mse_loss={self.mse_loss}, sqnr={self.sqnr})"
        )

    def __post_init__(self) -> None:
        if not isinstance(self.actual, (torch.Tensor, list, tuple, dict)):
            raise ValueError(
                f"`self.actual` value must be a Tensor, list, tuple or dict, got: {self.actual}"
            )

        if not isinstance(self.ref, (torch.Tensor, list, tuple, dict)):
            raise ValueError(
                f"`self.ref` value must be a Tensor, list, tuple or dict, got: {self.ref}"
            )

        if not _tensor_shape_equals(self.ref, self.actual):
            raise ValueError(
                f"Cannot compare tensors with different shapes: ref={self.ref} vs actual={self.actual}"
            )


@dataclass(frozen=True)
class NodeAccuracySummary:
    debug_info: NodeSourceDebugInfo
    actual_node_name: str
    actual_module_stack: str
    ref_node_name: str
    ref_module_stack: str
    results: Sequence[QuantizationComparisonResult]


def _module_stack_to_str(module_stack: object) -> str:
    """Simplifies the stack from ("mod", "mod.foo", "mod.foo.0", "mod.foo.0.linear")
    to "mod.foo.0.linear"
    """
    if not isinstance(module_stack, dict):
        return str(module_stack)
    module_values_list = list(module_stack.values())
    if len(module_values_list) > 0:
        owning_module = module_values_list[-1][0]
        return str(owning_module)
    else:
        return str(module_stack)


def extract_results_from_loggers(
    model: GraphModule,
) -> dict[NodeSourceDebugInfo, tuple[Optional[str], object, list[object]]]:
    """For a given model, extract the tensors stats and related information for each debug info.
    The reason we have a list of object, instead of Tensor is because the output of node may not be
    a Tensor, it could be (nested) list, tuple or dict as well.

    Returns:
        A dict is keyed by the NodeSourceDebugInfo and the values are a list of object recorded
        in loggers

    """
    # Results maps debug info to a tensor list for each model being compared.
    handles: dict[NodeSourceDebugInfo, tuple[Optional[str], object, list[object]]] = {}
    for _, module in model.named_children():
        if isinstance(module, OutputLogger) and len(module.stats) > 0:
            handles[module.debug_info] = (
                module.node_name,
                module.nn_module_stack,
                module.stats,
            )

    return handles


def compare_results(
    ref_results: dict[
        NodeSourceDebugInfo, tuple[Optional[str], object, list[torch.Tensor]]
    ],
    actual_results: dict[
        NodeSourceDebugInfo, tuple[Optional[str], object, list[torch.Tensor]]
    ],
) -> dict[NodeSourceDebugInfo, NodeAccuracySummary]:
    """Given two dict mapping from `NodeSourceDebugInfo` to list of tensors
    return a map from `NodeSourceDebugInfo` to `NodeAccuracySummary` that contains
    comparison information like SQNR, MSE etc.

    Args:
        ref_results (Dict[NodeSourceDebugInfo, Tuple[str, object, List[torch.Tensor]]]): reference results for each debug info
        actual_results (Dict[NodeSourceDebugInfo, Tuple[str, object, List[torch.Tensor]]]): actual results for each debug info

    Returns:
        Dict[NodeSourceDebugInfo, NodeAccuracySummary]
    """
    comparisons = {}
    for debug_info, (ref_name, ref_stack, ref_stats) in ref_results.items():
        if debug_info not in actual_results:
            log.debug(
                "Cannot compare for debug info %s because it wasn't found in the transformed model",
                debug_info,
            )
            continue
        actual_name, actual_stack, actual_stats = actual_results[debug_info]
        try:
            results = [
                QuantizationComparisonResult(actual=a, ref=b)
                for a, b in zip(actual_stats, ref_stats)
            ]
        except Exception as e:
            # Add extra information for an exception from QuantizationComparisonResult
            # if the shapes didn't match, to include the debug info and the node names.
            raise ValueError(
                f"For debug_info={debug_info} from ref node {ref_name} and actual node {actual_name}"
            ) from e

        comparisons[debug_info] = NodeAccuracySummary(
            debug_info=debug_info,
            actual_node_name=actual_name or "",
            actual_module_stack=_module_stack_to_str(actual_stack),
            ref_node_name=ref_name or "",
            ref_module_stack=_module_stack_to_str(ref_stack),
            results=results,
        )

    return comparisons
