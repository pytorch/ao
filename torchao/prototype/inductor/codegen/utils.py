from typing import Any, List

from torch._inductor import lowering as L
from torch._inductor.codegen.cpp_template_kernel import (
    parse_expr_with_index_symbols,
    wrap_with_tensorbox,
)


def expand(node, sizes: List[Any]):
    node = wrap_with_tensorbox(node)
    sizes = parse_expr_with_index_symbols(sizes)
    return L.expand(node, sizes).data
