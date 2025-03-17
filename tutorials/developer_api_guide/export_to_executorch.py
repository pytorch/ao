# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD 3-Clause license found in the
# LICENSE file in the root directory of this source tree.
"""
This tutorial shows how to preserve higher level operators in the model in order to be used in executorch

Specifically we define and preserved `torch.ops.quant.embedding_byte` op that works with quantized weights
through `torch.export.export`, we can follow Executorch tutorials: https://pytorch.org/executorch/stable/tutorials/export-to-executorch-tutorial.html#lowering-to-edge-dialect to lower the model to executorch
or rely on https://github.com/pytorch/executorch/tree/main/examples/models/llama and https://github.com/pytorch/torchchat libraries to export to target device.

This can also support exporting the model to other platforms like ONNX as well.
"""

from typing import List

import torch
from my_dtype_tensor_subclass import MyDTypeTensor

import torchao
from torchao.quantization.quant_primitives import dequantize_affine
from torchao.utils import _register_custom_op

quant_lib = torch.library.Library("quant", "FRAGMENT")
register_custom_op = _register_custom_op(quant_lib)


class MyDTypeTensorExtended(MyDTypeTensor):
    pass


implements = MyDTypeTensorExtended.implements
to_my_dtype_extended = MyDTypeTensorExtended.from_float

aten = torch.ops.aten


# NOTE: the op must start with `_`
# NOTE: typing must be compatible with infer_schema (https://github.com/pytorch/pytorch/blob/main/torch/_library/infer_schema.py)
# This will register a torch.ops.quant.embedding
@register_custom_op
def _embedding_byte(
    int_data: torch.Tensor,
    block_size: List[int],
    weight_scales: torch.Tensor,
    indices: torch.Tensor,
) -> torch.Tensor:
    weight = dequantize_affine(
        int_data,
        block_size,
        weight_scales,
        None,
        int_data.dtype,
    )
    return torch.ops.aten.embedding.default(weight, indices)


@implements(torch.nn.functional.embedding)
def _(func, types, args, kwargs):
    indices = args[0]
    weight = args[1]
    tensor_impl = weight.tensor_impl
    int_data, scale = tensor_impl.get_plain()
    block_size = (1, int_data.shape[-1])
    return torch.ops.quant.embedding_byte(int_data, block_size, scale, indices)


def main():
    m = torch.nn.Sequential(torch.nn.Embedding(4096, 128))
    input = torch.randint(0, 4096, (1, 6))

    m[0].weight = torch.nn.Parameter(
        to_my_dtype_extended(m[0].weight), requires_grad=False
    )
    y_ref = m[0].weight.dequantize()[input]
    y_q = m(input)
    from torchao.quantization.utils import compute_error

    sqnr = compute_error(y_ref, y_q)
    assert sqnr > 45.0

    # export
    m_unwrapped = torchao.utils.unwrap_tensor_subclass(m)
    m_exported = torch.export.export(m_unwrapped, (input,), strict=True).module()
    y_q_exported = m_exported(input)

    assert torch.equal(y_ref, y_q_exported)
    ops = [n.target for n in m_exported.graph.nodes]
    print(m_exported)
    assert torch.ops.quant.embedding_byte.default in ops


if __name__ == "__main__":
    main()
