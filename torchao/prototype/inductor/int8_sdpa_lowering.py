from collections.abc import Sequence
from typing import Optional

import sympy
import torch
from torch._inductor import realize_inputs
from torch._inductor.ir import ChoiceCaller, FixedLayout, TensorBox, get_fill_order
from torch._inductor.lowering import register_lowering
from torch._inductor.select_algorithm import (
    ExternKernelChoice,
    autotune_select_algorithm,
)
from torch.utils._pytree import tree_map

from .codegen.cpp_int8_sdpa_template import CppInt8SdpaTemplate


# Copied directly from https://github.com/pytorch/pytorch/commit/e221a1c853b425b8d70b36d545ccb32ddc8176bd
def maybe_realize(args):
    """Accepts a list of optional IRNodes and returns a list of realized IRNodes"""
    return tree_map(
        lambda x: (
            realize_inputs(x)
            if x is not None and not isinstance(x, sympy.Symbol)
            else x
        ),
        args,
    )


# Copied directly from https://github.com/pytorch/pytorch/commit/e221a1c853b425b8d70b36d545ccb32ddc8176bd
def construct_strides(
    sizes: Sequence[int],
    fill_order: Sequence[int],
) -> Sequence[int]:
    """From a list of sizes and a fill order, construct the strides of the permuted tensor."""
    # Initialize strides
    assert len(sizes) == len(fill_order), (
        "Length of sizes must match the length of the fill order"
    )
    strides = [0] * len(sizes)

    # Start with stride 1 for the innermost dimension
    current_stride = 1

    # Iterate through the fill order populating strides
    for dim in fill_order:
        strides[dim] = current_stride
        current_stride *= sizes[dim]

    return strides


op_int8_sdpa = ExternKernelChoice(
    torch.ops.torchao.qscaled_dot_product.default,
    "torchao::qscaled_dot_product",
    has_out_variant=False,
    use_fallback_kernel=True,
    op_overload=torch.ops.torchao.qscaled_dot_product.default,
)


def register_int8_sdpa():
    @register_lowering(
        torch.ops.torchao.qscaled_dot_product.default, type_promotion_kind=None
    )
    def int8_sdpa(
        query: TensorBox,
        key: TensorBox,
        value: TensorBox,
        attn_mask: Optional[TensorBox],
        dropout: float,
        is_causal: bool,
        scale: Optional[float] = None,
        q_scale: Optional[float] = 1.0,
        q_zp: Optional[int] = 0,
        k_scale: Optional[float] = 1.0,
        k_zp: Optional[int] = 0,
        v_scale: Optional[float] = 1.0,
        v_zp: Optional[int] = 0,
        a_scale: Optional[float] = 1.0,
        a_zp: Optional[int] = 0,
        o_scale: Optional[float] = 1.0,
        o_zp: Optional[int] = 0,
    ) -> TensorBox:
        choices: list[ChoiceCaller] = []

        (
            query,
            key,
            value,
            attn_mask,
        ) = maybe_realize(
            [
                query,
                key,
                value,
                attn_mask,
            ]
        )

        if (
            query.get_dtype() is not torch.uint8
            or key.get_dtype() is not torch.uint8
            or value.get_dtype() is not torch.uint8
        ):
            raise NotImplementedError(
                "Only `torch.uint8` is supported in Int8 SDPA template for CPU device. "
                f"Found input tensors are `{query.get_dtype()}`,`{key.get_dtype()}`,`{value.get_dtype()}`."
            )

        # Construct output layout with strides matching the query.
        out_size = query.get_size()
        fill_order = get_fill_order(query.get_stride())
        out_strides = construct_strides(out_size, fill_order)

        layout = FixedLayout(
            query.get_device(),
            query.get_dtype(),
            out_size,
            stride=[sympy.sympify(s) for s in out_strides],
        )
        input_nodes = [query, key, value]
        if attn_mask is not None:
            input_nodes.append(attn_mask)

        # use template if machine has amx
        if torch._C._cpu._is_amx_tile_supported():
            CppInt8SdpaTemplate.add_choices(
                choices=choices,
                input_nodes=input_nodes,
                layout=layout,
                scale=scale,
                q_scale=q_scale,
                q_zp=q_zp,
                k_scale=k_scale,
                k_zp=k_zp,
                v_scale=v_scale,
                v_zp=v_zp,
                a_scale=a_scale,
                a_zp=a_zp,
                o_scale=o_scale,
                o_zp=o_zp,
            )

        if len(choices) == 0:
            choices.append(
                op_int8_sdpa.bind(
                    input_nodes=input_nodes,
                    layout=layout,
                    scale=scale,
                    q_scale=q_scale,
                    q_zp=q_zp,
                    k_scale=k_scale,
                    k_zp=k_zp,
                    v_scale=v_scale,
                    v_zp=v_zp,
                    a_scale=a_scale,
                    a_zp=a_zp,
                    o_scale=o_scale,
                    o_zp=o_zp,
                )
            )

        inputs_for_autotuning = [
            query,
            key,
            value,
        ]

        return autotune_select_algorithm(
            "int8_sdpa",
            choices,
            inputs_for_autotuning,
            layout,
        )


register_int8_sdpa()
