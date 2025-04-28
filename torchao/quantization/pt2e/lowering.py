# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD 3-Clause license found in the
# LICENSE file in the root directory of this source tree.

from typing import Optional

import torch
from torch._inductor.constant_folding import constant_fold
from torch._inductor.fx_passes.freezing_patterns import freezing_passes

__all__ = [
    "lower_pt2e_quantized_to_x86",
]

FUSION_PATH_REGISTERED = False


def lower_pt2e_quantized_to_x86(
    model: torch.fx.GraphModule,
    example_inputs: Optional[tuple[torch.Tensor, ...]] = None,
    compile: bool = True,
    **compile_options: Optional[dict],
) -> torch.fx.GraphModule:
    """Lower a PT2E-qantized model to x86 backend.

    Args:
    * `model` (torch.fx.GraphModule): a model quantized by PT2E quantization flow.
    * `example_inputs` (tuple[torch.Tensor, ...]): example inputs for the model.
    * `compile` (bool): whether to torch.compile the model. Default is True.
        Torch.compile brings more performance improvement.
    * `compile_options` (dict): options for torch.compile.

    Return:
    A module lowered to x86 backend.
    """

    if compile:
        global FUSION_PATH_REGISTERED
        if not FUSION_PATH_REGISTERED:
            global torch
            import torch._inductor.config

            from torchao.prototype.inductor.fx_passes.quantization import (
                _register_quantization_weight_pack_pass,
                quant_lift_up,
            )

            torch._inductor.config.pre_grad_custom_pass = quant_lift_up
            _register_quantization_weight_pack_pass()
            FUSION_PATH_REGISTERED = True
        return torch.compile(model, **compile_options)

    assert example_inputs is not None, (
        "example_inputs should not be None when compile is False"
    )

    def _post_autograd_decomp_table():  # type: ignore[no-untyped-def]
        decomp_table = torch.export.default_decompositions()

        # if we are post-autograd, we shouldn't
        # decomp prim ops.
        for k in list(decomp_table.keys()):
            if not torch._export.utils._is_cia_op(k):
                del decomp_table[k]

        return decomp_table

    def _node_replace(m):  # type: ignore[no-untyped-def]
        # Replace aten.t(x) with aten.permute(x, [1, 0])
        aten = torch.ops.aten
        g = m.graph
        for node in g.nodes:
            if node.target == aten.t.default:
                with g.inserting_before(node):
                    x = node.args[0]
                    dims = [1, 0]
                    perm_node = g.call_function(aten.permute.default, args=(x, dims))
                    node.replace_all_uses_with(perm_node)
                    g.erase_node(node)

        g.lint()
        m.recompile()

    lowered_model = (
        torch.export.export_for_training(model, example_inputs, strict=True)
        .run_decompositions(_post_autograd_decomp_table())
        .module()
    )
    _node_replace(lowered_model)
    freezing_passes(lowered_model, example_inputs)
    constant_fold(lowered_model)
    return lowered_model
