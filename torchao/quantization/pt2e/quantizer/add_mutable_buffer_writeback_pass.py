import torch
from torch.fx import GraphModule, Node
from torch.fx.passes.infra.pass_base import PassBase, PassResult

from torchao.quantization.pt2e.graph_utils import collect_producer_nodes
from torchao.quantization.quant_primitives import quant_lib  # noqa: F401

__all__ = ["AddMutableBufferWritebackPass"]

_QUANTIZE_OPS = [
    torch.ops.quantized_decomposed.quantize_per_tensor.default,
    torch.ops.quantized_decomposed.quantize_per_tensor.tensor,
    torch.ops.quantized_decomposed.quantize_per_channel.default,
    torch.ops.torchao.quantize_affine,
]

_DEQUANTIZE_OPS = [
    torch.ops.quantized_decomposed.dequantize_per_tensor.default,
    torch.ops.quantized_decomposed.dequantize_per_tensor.tensor,
    torch.ops.quantized_decomposed.dequantize_per_channel.default,
    torch.ops.torchao.dequantize_affine,
]


def _writes_to_arg0(node: Node) -> bool:
    if (
        node.op != "call_function"
        or not hasattr(node.target, "_schema")
        or len(node.args) == 0
        or not isinstance(node.args[0], Node)
    ):
        return False

    schema = node.target._schema
    return (
        bool(schema.arguments)
        and schema.arguments[0].alias_info is not None
        and schema.arguments[0].alias_info.is_write
    )


def _get_original_arg(node: Node) -> Node | None:
    if node.op != "call_function" or node.target not in _DEQUANTIZE_OPS:
        return None

    quantize_node = node.args[0]
    if (
        not isinstance(quantize_node, Node)
        or quantize_node.op != "call_function"
        or quantize_node.target not in _QUANTIZE_OPS
    ):
        return None

    original_arg = quantize_node.args[0]
    return original_arg if isinstance(original_arg, Node) else None


def _is_buffer_backed_node(node: Node, buffer_names: set[str]) -> bool:
    producer_nodes = collect_producer_nodes(node)
    if producer_nodes is None:
        return False

    return any(
        producer.op == "get_attr" and str(producer.target) in buffer_names
        for producer in producer_nodes
    )


class AddMutableBufferWritebackPass(PassBase):
    """Write back mutations performed on QDQ inputs to the original buffer."""

    def call(self, graph_module: GraphModule) -> PassResult:
        buffer_names = {
            name for name, _ in graph_module.named_buffers(remove_duplicate=False)
        }
        modified = False

        for node in list(graph_module.graph.nodes):
            if not _writes_to_arg0(node):
                continue

            qdq_dest = node.args[0]
            original_dest = _get_original_arg(qdq_dest)
            if original_dest is None or not _is_buffer_backed_node(
                original_dest, buffer_names
            ):
                continue

            with graph_module.graph.inserting_after(node):
                copy_node = graph_module.graph.call_function(
                    torch.ops.aten.copy_.default,
                    args=(original_dest, node),
                    kwargs={},
                )
                copy_node.meta.update(node.meta)
                modified = True

        if modified:
            graph_module.graph.eliminate_dead_code()
            graph_module.graph.lint()
            graph_module.recompile()

        return PassResult(graph_module, modified)
