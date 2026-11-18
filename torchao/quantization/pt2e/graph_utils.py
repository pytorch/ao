# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD 3-Clause license found in the
# LICENSE file in the root directory of this source tree.

import itertools
import operator
from collections import OrderedDict
from collections.abc import Sequence
from typing import Any, Callable, Optional, Union

import torch
from torch.export import ExportedProgram
from torch.fx import Node
from torch.fx.passes.utils.source_matcher_utils import (
    SourcePartition,
    check_subgraphs_connected,
    get_source_partitions,
)

__all__ = [
    "find_sequential_partitions",
    "get_equivalent_types",
    "update_equivalent_types_dict",
    "bfs_trace_with_node_process",
    "collect_producer_nodes",
]

_EQUIVALENT_TYPES: list[set[Any]] = [
    {torch.nn.Conv1d, torch.nn.functional.conv1d},
    {torch.nn.Conv2d, torch.nn.functional.conv2d},
    {torch.nn.AdaptiveAvgPool2d, torch.nn.functional.adaptive_avg_pool2d},
    {torch.nn.ReLU, torch.nn.functional.relu, torch.nn.functional.relu_},
    {torch.nn.BatchNorm2d, torch.nn.functional.batch_norm},
    {torch.nn.Hardtanh, torch.nn.functional.hardtanh, torch.nn.functional.hardtanh_},
    {torch.add, operator.add, operator.iadd, "add", "add_"},
    {torch.mul, operator.mul, operator.imul, "mul", "mul_"},
]


def _create_equivalent_types_dict() -> dict[Any, list[Any]]:
    """Create a mapping from each type or operation to its list of equivalent types or operations."""
    _DICT = {}
    for values in _EQUIVALENT_TYPES:
        for v in values:
            _DICT[v] = list(values)
    return _DICT


_EQUIVALENT_TYPES_DICT = _create_equivalent_types_dict()


def get_equivalent_types() -> list[set[Any]]:
    """Return the default list of equivalent type sets used for pattern matching."""
    return _EQUIVALENT_TYPES


def update_equivalent_types_dict(
    customized_equivalent_types: Optional[list[set[Any]]] = None,
) -> None:
    """Helper function for users who want to customize ``_EQUIVALENT_TYPES`` and ``_EQUIVALENT_TYPES_DICT``.

    When ``customized_equivalent_types`` is passed, re-generates ``_EQUIVALENT_TYPES``
    and ``_EQUIVALENT_TYPES_DICT``.

    Args:
        customized_equivalent_types: List of sets specifying custom equivalent types.

    Raises:
        ValueError: If ``customized_equivalent_types`` is None.
    """
    if customized_equivalent_types is None:
        raise ValueError("customized_equivalent_types should not be None")
    global _EQUIVALENT_TYPES
    global _EQUIVALENT_TYPES_DICT
    _EQUIVALENT_TYPES = customized_equivalent_types
    _EQUIVALENT_TYPES_DICT = _create_equivalent_types_dict()


def _partitions_sequential(partitions: Sequence[SourcePartition]) -> bool:
    """Check if a sequence of SourcePartition subgraphs are connected in order."""
    prev_partition = None
    for partition in partitions:
        if prev_partition is not None and not check_subgraphs_connected(
            prev_partition, partition
        ):
            return False
        prev_partition = partition
    return True


def _get_matching_types(partition_type: Any) -> list[Any]:
    """Get all equivalent types matching the given partition type."""
    matching_types = [partition_type]
    if partition_type in _EQUIVALENT_TYPES_DICT:
        matching_types.extend(_EQUIVALENT_TYPES_DICT[partition_type])
    return matching_types


def _valid_type_sequence(partition_types: list[Any]) -> bool:
    """Check if all partition types in the sequence are distinct across equivalence sets."""
    partition_types_set: set[Any] = set()
    for partition_type in partition_types:
        matching_types = _get_matching_types(partition_type)
        matching_types_set = set(matching_types)
        if len(partition_types_set & matching_types_set) > 0:
            return False
        partition_types_set |= matching_types_set
    return True


def find_sequential_partitions(
    gm: torch.fx.GraphModule,
    partition_types: list[Any],
    include_functional_equivalent: bool = True,
    filter_fn: Optional[Callable[[Node], bool]] = None,
) -> list[tuple[SourcePartition, ...]]:
    """Find sequential subgraphs matching the given sequence of partition types.

    Args:
        gm: Target FX GraphModule to search within.
        partition_types: Ordered list of types/operations to match sequentially.
        include_functional_equivalent: Whether to match functionally equivalent types.
        filter_fn: Optional filter function applied to nodes.

    Returns:
        List of tuples of SourcePartitions representing sequential matches.
    """
    if not _valid_type_sequence(partition_types):
        raise ValueError(
            f"Invalid partition types: {partition_types}. Each type in the sequence must be unique"
        )

    typed_partitions: OrderedDict[Any, list[SourcePartition]] = OrderedDict()
    for partition_type in partition_types:
        types_to_match = _get_matching_types(partition_type)
        partitions = get_source_partitions(gm.graph, types_to_match, filter_fn)
        typed_partitions[partition_type] = list(
            itertools.chain.from_iterable(partitions.values())
        )

    typed_partitions_list = list(typed_partitions.values())
    fusion_candidates = itertools.product(*typed_partitions_list)
    fused_partitions = [
        candidate
        for candidate in fusion_candidates
        if _partitions_sequential(candidate)
    ]
    return fused_partitions


def _get_submodule(
    graph_module: torch.fx.GraphModule, node: torch.fx.Node, arg_index: int
) -> tuple[str, torch.nn.Module, torch.fx.Node]:
    """Retrieve a control flow submodule target name, Module instance, and user Node."""
    submod_node = node.args[arg_index]
    assert isinstance(submod_node, torch.fx.Node)
    assert submod_node.op == "get_attr"
    assert isinstance(submod_node.target, str)
    submodule = graph_module.get_submodule(submod_node.target)
    # pyre-ignore
    return submod_node.target, submodule, node


def _get_control_flow_submodules(
    graph_module: torch.fx.GraphModule,
) -> list[tuple[str, torch.nn.Module, torch.fx.Node]]:
    """Returns a list of submodules used for control flow operations
    (torch.ops.higher_order.cond/map) that are in the given toplevel graph (does not look
    into submodules). Specifically, the returned value is a list containing a
    tuple of (name of the submodule that's stored in the graph module, the
    submodule itself, and the fx node that uses this submodule).
    """
    control_flow_submodules = []
    for node in graph_module.graph.nodes:
        if node.op != "call_function":
            continue

        if node.target is torch.ops.higher_order.cond:
            control_flow_submodules.append(_get_submodule(graph_module, node, 1))
            control_flow_submodules.append(_get_submodule(graph_module, node, 2))
        if node.target is torch.ops.higher_order.map_impl:
            control_flow_submodules.append(_get_submodule(graph_module, node, 0))
        if node.target is torch.ops.higher_order.scan:
            control_flow_submodules.append(_get_submodule(graph_module, node, 0))
        if node.target is torch.ops.higher_order.while_loop:
            control_flow_submodules.append(_get_submodule(graph_module, node, 1))

    return control_flow_submodules


def bfs_trace_with_node_process(
    model: Union[ExportedProgram, torch.fx.GraphModule],
    node_op: Callable[[Node], Any],
) -> None:
    """Traverse the graph module via BFS and apply ``node_op`` to each node.

    Args:
        model: An ExportedProgram or FX GraphModule to traverse.
        node_op: Callable executed on each non-output and non-placeholder FX node.
    """
    assert isinstance(model, (ExportedProgram, torch.fx.GraphModule)), (
        f"Expected GraphModule or ExportedProgram, got {type(model)}"
    )
    gm = model.graph_module if isinstance(model, ExportedProgram) else model
    queue = [gm]
    while queue:
        current_graph_module = queue.pop(0)
        for node in current_graph_module.graph.nodes:
            if node.op in ["output", "placeholder"]:
                continue

            node_op(node)

        control_flow_submodules = [
            submodule
            for _, submodule, _ in _get_control_flow_submodules(current_graph_module)
        ]
        queue.extend(control_flow_submodules)


def collect_producer_nodes(node: Node) -> Optional[list[Node]]:
    """Trace a node's producer chain until input or getattr is reached.

    Args:
        node: Starting FX Node for backward producer tracing.

    Returns:
        List of producer FX Nodes, or None if a graph input (placeholder) was reached.
    """
    nodes = [node]
    frontier = [node]
    while frontier:
        node = frontier.pop()
        all_args = list(node.args) + list(node.kwargs.values())
        for arg in all_args:
            if not isinstance(arg, Node):
                continue
            if arg.op == "placeholder":
                # hit input, can't fold in this case
                return None
            nodes.append(arg)
            if not (arg.op == "call_function" and arg.target is getattr):
                frontier.append(arg)
    return nodes

