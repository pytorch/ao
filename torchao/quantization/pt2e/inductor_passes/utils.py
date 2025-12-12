# Copyright (c) Meta Platforms...
# BSD 3-Clause

# This module hosts shared pre-grad passes used by multiple backends.
# NOTE: copied from x86 to avoid backend-to-backend dependencies.


import torch
from torch.fx.node import map_arg

aten = torch.ops.aten
prims = torch.ops.prims
quantized_decomposed = torch.ops.quantized_decomposed
quantized = torch.ops.quantized

_PER_TENSOR_QUANTIZE_OPS = [
    quantized_decomposed.quantize_per_tensor.default,
    quantized_decomposed.quantize_per_tensor.tensor,
]

_VIEW_OPS = [
    aten.transpose.int,
    aten.permute.default,
    aten.view.default,
]


# ----taken from torchao/quantization/pt2e/inductor_passes/x86.py------
def quant_lift_up(module_graph: torch.fx.graph.Graph):
    """
    Lift up the quant node before view like nodes. It can benefit performance
    of Attention like block. For example, we have the pattern as:

             DQ
    DQ       LINEAR
    LINEAR   VIEW
    VIEW     PERMUTE
    PERMUTE  TRANSPOSE
    Q        Q
    DQ       DQ
       Matmul
        DIV
        ADD
      SOFTMAX

    We want to lift up the the quant nodes from matmul before view like nodes
    as the output of Linear node.

             DQ
    DQ       LINEAR
    LINEAR   Q
    Q        VIEW
    VIEW     PERMUTE
    PERMUTE  TRANSPOSE
    DQ       DQ
       Matmul
        DIV
        ADD
      SOFTMAX

    It produces a DQ->LINEAR->Q pattern which can be fused by backend.
    """

    def is_view_op(node):
        return node.op == "call_function" and node.target in _VIEW_OPS

    for node in module_graph.nodes:
        # <TODO> Leslie: Here we verify that the quant node has exactly
        # one input FX node, with constant scalar value for scale and zero point.
        # For the case input of quant node has more than one input FX nodes,
        # extend the implementation to lift up all the connected nodes
        # before the view nodes to keep the topological order.
        if (
            node.op == "call_function"
            and node.target in _PER_TENSOR_QUANTIZE_OPS
            and len(node.all_input_nodes) == 1
            and is_view_op(node.all_input_nodes[0])
        ):
            quant_node = node
            input_node_of_quant = quant_node.args[0]

            # Check the nodes along lift up path has only 1 user node
            # Propagate view like node to find where to insert the new quant node
            could_lift_up = True
            current_node = quant_node
            input_node = current_node.args[0]
            while is_view_op(input_node):
                if len(input_node.users) != 1:
                    could_lift_up = False
                    break
                current_node = input_node
                input_node = current_node.args[0]

            # Further check the input node of the first view node has only 1 user node
            if could_lift_up and len(input_node.users) == 1:
                # Replace dequant's input from quant to quant's input
                quant_node.replace_all_uses_with(input_node_of_quant)
                # Insert the new quant node
                with module_graph.inserting_before(current_node):
                    new_quant_node = module_graph.node_copy(quant_node)
                    input_node.replace_all_uses_with(new_quant_node)

                    # Update inputs of new_quant_node
                    def maybe_replace_node(n: torch.fx.Node) -> torch.fx.Node:
                        if n == input_node_of_quant:
                            return input_node
                        else:
                            return n

                    new_args = map_arg(new_quant_node.args, maybe_replace_node)
                    new_kwargs = map_arg(new_quant_node.kwargs, maybe_replace_node)
                    new_quant_node.args = new_args  # type: ignore[assignment]
                    new_quant_node.kwargs = new_kwargs  # type: ignore[assignment]
                    module_graph.erase_node(quant_node)
