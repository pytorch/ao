from typing import Optional

import sympy
import torch
from torch._inductor.ir import ChoiceCaller, FixedLayout, TensorBox, get_fill_order

try:
    # use the directory after refactor
    from torch._inductor.kernel.flex.common import construct_strides, maybe_realize
except ImportError:
    # use the old path for compatibility
    from torch._inductor.kernel.flex_attention import construct_strides, maybe_realize
from torch._inductor.lowering import register_lowering
from torch._inductor.select_algorithm import (
    ExternKernelChoice,
    autotune_select_algorithm,
)

from .codegen.cpp_int8_sdpa_template import CppInt8SdpaTemplate

op_qsdpa = ExternKernelChoice(
    torch.ops.torchao.qscaled_dot_product.default,
    "torchao::qscaled_dot_product",
    has_out_variant=False,
    use_fallback_kernel=True,
    op_overload=torch.ops.torchao.qscaled_dot_product.default,
)
quantize_dtypes = [torch.uint8, torch.float8_e4m3fn]


def register_qsdpa():
    @register_lowering(
        torch.ops.torchao.qscaled_dot_product.default, type_promotion_kind=None
    )
    def qsdpa(
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
            query.get_dtype() not in quantize_dtypes
            or key.get_dtype() not in quantize_dtypes
            or value.get_dtype() not in quantize_dtypes
        ):
            raise NotImplementedError(
                "Only `torch.uint8` or `torch.float8_e4m3fn` is supported in Quantized SDPA template for CPU device. "
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

        # use template if machine has amx, only support uint8 for now
        if torch._C._cpu._is_amx_tile_supported() and query.get_dtype() is torch.uint8:
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
                op_qsdpa.bind(
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
            "qsdpa",
            choices,
            inputs_for_autotuning,
            layout,
        )


register_qsdpa()
