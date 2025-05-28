from . import x86 as _x86        # the single leading dot is mandatory
import copy
import functools
import itertools
from typing import Any

import torch
from torch._dynamo.utils import counters
from torch._inductor.fx_passes.freezing_patterns import register_freezing_graph_pattern
from torch._inductor.pattern_matcher import (
    Arg,
    CallFunction,
    KeywordArg,
    Match,
    filter_nodes,
)
from torch.fx.experimental.symbolic_shapes import has_free_symbols
from torch.fx.node import map_arg


aten = torch.ops.aten
prims = torch.ops.prims
quantized_decomposed = torch.ops.quantized_decomposed
quantized = torch.ops.quantized


def _generate_linear_t_pattern(
    dequant_pattern,
    dtype,
):
    assert dtype in [torch.float32, torch.bfloat16]
    t_pattern = CallFunction(
        aten.permute.default,
        _x86._may_generate_pattern_with_dtype_convert(
            dequant_pattern,
            KeywordArg("autocast_wgt_dtype"),
            dtype == torch.bfloat16,
        ),
        KeywordArg("permute_axes"),
    )
    return t_pattern


dequantize_per_tensor_weight_pattern = CallFunction(
    quantized_decomposed.dequantize_per_tensor.default,
    KeywordArg("q_weight"),
    KeywordArg("w_scale"),
    KeywordArg("w_zp"),
    KeywordArg("w_quant_min"),
    KeywordArg("w_quant_max"),
    KeywordArg("w_dtype"),
)

dequantize_per_tensor_to_bf16_weight_pattern = (
    _x86._may_generate_pattern_with_dtype_convert(
        dequantize_per_tensor_weight_pattern,
        KeywordArg("autocast_wgt_dtype"),
    )
)

dequantize_per_tensor_clone_weight_pattern = CallFunction(
    aten.clone.default,
    dequantize_per_tensor_weight_pattern,
    memory_format=KeywordArg("memory_format"),
)

dequantize_per_tensor_to_bf16_clone_weight_pattern = CallFunction(
    aten.clone.default,
    dequantize_per_tensor_to_bf16_weight_pattern,
    memory_format=KeywordArg("memory_format"),
)


def _register_qconv_weight_prepack_pass(pattern, pass_number, dtype=torch.float32, per_channel=True,):
    expected_dequant_op = (
        quantized_decomposed.dequantize_per_channel.default
        if per_channel
        else quantized_decomposed.dequantize_per_tensor.default
    )
    @register_freezing_graph_pattern(
        pattern,
        extra_check=_x86._is_valid_dequant_conv_pattern(dtype),
        pass_number=pass_number,
    )
    def qconv_weight_prepack(match: Match, *args, **kwargs):
        """
        Match the pattern:
        int8 activation
          |
        dequant_per_tensor
          |
        Conv2d <- optional(aten.clone.default) <- expected_dequant_op <- int8_weight

        Insert weight prepack node and change the pattern to:
        int8 activation
          |
        onednn.qconv2d_pointwise <- onednn.qconv_prepack <- int8_weight
        """
        assert dtype in [torch.float32, torch.bfloat16]
        conv_node = match.output_node()
        assert conv_node.target is aten.convolution.default
        if dtype == torch.float32:
            dequant_node = conv_node.args[0]
        else:
            convert_to_bf16 = conv_node.args[0]
            dequant_node = convert_to_bf16.args[0]  # type: ignore[union-attr]
        has_clone_to_channel_last_node_in_pattern = (
            conv_node.args[1].target is aten.clone.default  # type: ignore[union-attr]
        )
        clone_node = (
            conv_node.args[1] if has_clone_to_channel_last_node_in_pattern else None
        )

        if dtype == torch.float32:
            dq_weight = (
                clone_node.args[0]  # type: ignore[union-attr]
                if has_clone_to_channel_last_node_in_pattern
                else conv_node.args[1]
            )
        else:
            weight_to_bf16_node = (
                clone_node.args[0]  # type: ignore[union-attr]
                if has_clone_to_channel_last_node_in_pattern
                else conv_node.args[1]
            )
            dq_weight = weight_to_bf16_node.args[0]  # type: ignore[union-attr]

        assert (
            dq_weight.target  # type: ignore[union-attr]
            is expected_dequant_op
        )

        # Activation QParams
        qx, x_zp, x_scale = (
            kwargs["x"],
            kwargs["x_zp"],
            kwargs["x_scale"],
        )

        # Weight QParams
        qw, w_scale, w_zp = (
            kwargs["q_weight"],
            kwargs["w_scale"],
            kwargs["w_zp"],
        )

        # Conv Params
        bias, stride, padding, dilation, groups = (
            kwargs["b"],
            kwargs["stride"],
            kwargs["padding"],
            kwargs["dilation"],
            kwargs["groups"],
        )
        if not isinstance(w_scale, torch.fx.Node):
            w_scale = match.graph.call_function(
        torch.tensor,
        args=([float(w_scale)],),          # <- wrap in list 
        kwargs=dict(dtype=torch.float32),  # ensure f32
        )
        if not isinstance(w_zp, torch.fx.Node):
            w_zp = match.graph.call_function(
        torch.tensor,
        args=([int(w_zp)],),
        kwargs=dict(dtype=torch.int32),    # oneDNN expects s32 ZP
        )
        x_shape = qx.meta.get("tensor_meta").shape
        if has_free_symbols(x_shape):
            # For dynamic shape case, we can't get activation shape ahead of runtime.
            x_shape = None
        graph = match.graph
        with graph.inserting_before(conv_node):
            # Insert weight prepack node and the QConv node
            packed_weight_inputs = (
                qw,
                w_scale,
                x_scale,
                x_zp,
                stride,
                padding,
                dilation,
                groups,
                x_shape,
            )
            packed_weight_op = torch.ops.onednn.qconv_prepack
            prepack_weight_node = graph.call_function(
                packed_weight_op, args=packed_weight_inputs
            )

            new_args: tuple[Any, ...] = (
                qx,
                x_scale,
                x_zp,
                prepack_weight_node,
                w_scale,
                w_zp,
                bias,
                stride,
                padding,
                dilation,
                groups,
                1.0,  # output_scale
                0,  # output_zero_point
                dtype,  # output_dtype
                "none",  # attr
                [],  # scalars
                "",  # algorithm
            )
            new_conv_node = graph.call_function(
                torch.ops.onednn.qconv2d_pointwise.default, args=new_args
            )
            conv_node.replace_all_uses_with(new_conv_node)
            new_conv_node.meta.update(conv_node.meta)

            # Erase the original conv node
            graph.erase_node(conv_node)
            # Erase the dequant pattern
            if dtype == torch.bfloat16:
                graph.erase_node(convert_to_bf16)  # type: ignore[possibly-undefined, arg-type]
            graph.erase_node(dequant_node)  # type: ignore[arg-type]
            # Erase the dequant per channel pattern
            if clone_node is not None:
                graph.erase_node(clone_node)  # type: ignore[arg-type]
            if dtype == torch.bfloat16:
                graph.erase_node(weight_to_bf16_node)  # type: ignore[possibly-undefined, arg-type]
            graph.erase_node(dq_weight)  # type: ignore[arg-type]
            counters["inductor"]["qconv2d_weight_prepack_matcher_count"] += 1
            counters["inductor"]["qconv2d_weight_prepack_matcher_nodes"] += len(
                match.nodes
            )


def _generate_qconv_weight_prepack_patterns(dtype=torch.float32, per_channel=True,):
    """
    per_channel = False → per-tensor weight de-quant
    per_channel = True  → per-channel weight de-quant
    """
    assert dtype in [torch.float32, torch.bfloat16]
    return (
        _x86._generate_dequant_convolution_node_pattern(
            _x86.dequantize_per_channel_weight_pattern if per_channel else dequantize_per_tensor_weight_pattern
            if dtype == torch.float32
            else _x86.dequantize_per_channel_to_bf16_weight_pattern if per_channel else dequantize_per_tensor_to_bf16_weight_pattern,
            dtype,
        ),
        # There is another pattern due to the pass of convert_conv_weights_to_channels_last
        # https://github.com/pytorch/pytorch/blob/07107919297db3f8ab37f11c12666b6d6d5f692e/torch/_inductor/freezing.py#L338-L362.
        # Depend on some heuristics, it may or may not insert to(channel_last) node
        # between convolution and dequant_per_channel node
        _x86._generate_dequant_convolution_node_pattern(
            _x86.dequantize_per_channel_weight_pattern if per_channel else dequantize_per_tensor_weight_pattern
            if dtype == torch.float32
            else _x86.dequantize_per_channel_to_bf16_weight_pattern if per_channel else dequantize_per_tensor_to_bf16_weight_pattern,
            dtype,
        ),
    )


def _register_qlinear_weight_prepack_pass(
    pattern,
    pass_number,
    dtype=torch.float32,
    input_dim_exceeds_two=False,
    input_contiguous=True,
    per_channel=True,
):
    
    expected_dq_op = (
        quantized_decomposed.dequantize_per_channel.default
        if per_channel else
        quantized_decomposed.dequantize_per_tensor.default
    )
    @register_freezing_graph_pattern(
        pattern,
        extra_check=_x86._is_valid_dequant_linear_pattern(
            dtype, input_dim_exceeds_two, input_contiguous
        ),
        pass_number=pass_number,
    )
    def qlinear_weight_prepack(match: Match, *args, **kwargs):
        """
        Match the pattern:
        int8 activation
          |
        dequant_per_tensor
          |
        mm/addmm <- t <- expected_dq_op  <- int8_weight

        Insert weight prepack node and change the pattern to:
        int8 activation
          |
        onednn.qlinear_pointwise <- onednn.qlinear_prepack <- int8_weight
        """
        assert dtype in [torch.float32, torch.bfloat16]
        (
            linear_node,
            output_reshape_node,
        ) = _x86._get_linear_node(match, input_dim_exceeds_two, input_contiguous)
        input_index = 1 if linear_node.target is aten.addmm.default else 0
        weight_index = input_index + 1

        (
            dequant_node,
            act_reshape_node,
            activation_to_bf16_node,
            act_expand_node,
        ) = _x86._get_linear_dq_node(
            linear_node, input_index, dtype, input_dim_exceeds_two, input_contiguous
        )

        if input_dim_exceeds_two and not input_contiguous:
            wgt_expand_node = linear_node.args[weight_index]
            assert wgt_expand_node.target is aten.expand.default
            t_node = wgt_expand_node.args[0]
        else:
            t_node = linear_node.args[weight_index]

        if dtype == torch.float32:
            dq_weight = t_node.args[0]
        else:
            weight_to_bf16_node = t_node.args[0]
            dq_weight = weight_to_bf16_node.args[0]

        assert (dq_weight.target is expected_dq_op)

        # Activation QParams
        qx, x_zp, x_scale = (
            kwargs["x"],
            kwargs["x_zp"],
            kwargs["x_scale"],
        )

        # Weight QParams
        qw, w_scale, w_zp = (
            kwargs["q_weight"],
            kwargs["w_scale"],
            kwargs["w_zp"],
        )

        # Params
        bias = kwargs["b"] if "b" in kwargs else None

        if not isinstance(w_scale, torch.fx.Node):
            w_scale = match.graph.call_function(
        torch.tensor,
        args=([float(w_scale)],),          # <- wrap in list 
        kwargs=dict(dtype=torch.float32),  # ensure f32
        )
            
        if not isinstance(w_zp, torch.fx.Node):
            w_zp = match.graph.call_function(
        torch.tensor,
        args=([int(w_zp)],),
        kwargs=dict(dtype=torch.int32),    # oneDNN expects s32 ZP
        )
            
        x_shape = qx.meta.get("tensor_meta").shape
        if has_free_symbols(x_shape):
            # For dynamic shape case, we can't get activation shape ahead of runtime.
            x_shape = None
        graph = match.graph
        with graph.inserting_before(linear_node):
            # Insert weight prepack node and the qlinear node
            packed_weight_inputs = (
                qw,
                x_shape,
            )
            packed_weight_op = torch.ops.onednn.qlinear_prepack
            prepack_weight_node = graph.call_function(
                packed_weight_op, args=packed_weight_inputs
            )

            new_args: tuple[Any, ...] = (
                qx,
                x_scale,
                x_zp,
                prepack_weight_node,
                w_scale,
                w_zp,
                bias,
                1.0,  # output_scale
                0,  # output_zero_point
                dtype,  # output_dtype
                "none",  # post op name
                [],  # post op args
                "",  # post op algorithm
            )
            Node = torch.fx.node.Node
            if isinstance(x_scale, Node) and isinstance(x_zp, Node):
                new_linear_node = graph.call_function(
                    torch.ops.onednn.qlinear_pointwise.tensor, args=new_args
                )
            else:
                new_linear_node = graph.call_function(
                    torch.ops.onednn.qlinear_pointwise.default, args=new_args
                )
            if input_dim_exceeds_two:
                if input_contiguous:
                    output_reshape_node.replace_all_uses_with(new_linear_node)
                    new_linear_node.meta.update(output_reshape_node.meta)
                else:
                    if bias:
                        output_add_node_for_bias = match.output_node()
                        assert output_add_node_for_bias.target is aten.add.Tensor
                        output_add_node_for_bias.replace_all_uses_with(new_linear_node)
                        new_linear_node.meta.update(output_add_node_for_bias.meta)
                    else:
                        linear_node.replace_all_uses_with(new_linear_node)
                        new_linear_node.meta.update(linear_node.meta)
            else:
                linear_node.replace_all_uses_with(new_linear_node)
                new_linear_node.meta.update(linear_node.meta)

            # Erase the original linear node
            if input_dim_exceeds_two:
                if input_contiguous:
                    graph.erase_node(output_reshape_node)
                elif not input_contiguous and bias:
                    graph.erase_node(output_add_node_for_bias)  # type: ignore[possibly-undefined]
            graph.erase_node(linear_node)
            if input_dim_exceeds_two:
                if input_contiguous:
                    graph.erase_node(act_reshape_node)
                else:
                    graph.erase_node(act_expand_node)
                    graph.erase_node(wgt_expand_node)  # type: ignore[possibly-undefined]
            if dtype == torch.bfloat16:
                graph.erase_node(activation_to_bf16_node)
            # Erase the dequant pattern
            graph.erase_node(dequant_node)
            # Erase the dequant per channel pattern
            graph.erase_node(t_node)
            if dtype == torch.bfloat16:
                graph.erase_node(weight_to_bf16_node)  # type: ignore[possibly-undefined]
            graph.erase_node(dq_weight)

            counters["inductor"]["qlinear_weight_prepack_matcher_count"] += 1
            counters["inductor"]["qlinear_weight_prepack_matcher_nodes"] += len(
                match.nodes
            )


def _generate_qlinear_weight_prepack_patterns(
    dtype=torch.float32,
    input_dim_exceeds_two=False,
    input_contiguous=True,
    with_bias=False,
    is_tensor_overload=False,
    per_channel = True, 
):
    weight_dq_quant = _x86.dequantize_per_channel_weight_pattern if per_channel else dequantize_per_tensor_weight_pattern
    if input_dim_exceeds_two and not input_contiguous:
        return _x86._generate_dequant_bmm_node_pattern(
            weight_dq_quant,
            dtype,
            with_bias,
            is_tensor_overload,
        )
    else:
        return _x86._generate_dequant_linear_node_pattern(
            weight_dq_quant,
            dtype,
            input_dim_exceeds_two,
            is_tensor_overload,
        )


def _register_qconv_weight_prepack(per_channel=True):
    for dtype in [torch.float32, torch.bfloat16]:
        weight_prepack_patterns = _generate_qconv_weight_prepack_patterns(dtype=dtype, per_channel=per_channel)
        for weight_prepack_pattern in weight_prepack_patterns:
            # Register to pass_number 1, so we can do dequant promotion in pass_number 0.
            _register_qconv_weight_prepack_pass(
                weight_prepack_pattern, pass_number=1, dtype=dtype, per_channel=per_channel
            )


def _register_qlinear_weight_prepack(per_channel=True):
    # 6 Linear related patterns will be matched based on the dtype, input dimension size and input contiguous.
    # Then convert the pattern into a QLinear node with int8_fp32/bf16.
    # Case 1: int8-mixed-fp32, input dim size is 2
    # Case 2: int8-mixed-fp32, input dim size exceeds 2 and contiguous
    # Case 3: int8-mixed-bf16, input dim size is 2
    # Case 4: int8-mixed-bf16, input dim size exceeds 2 and contiguous

    #   + - - - - | - - - - - - | - - - - - +
    #   |    dq_per_tensor  dq_per_channel  |
    #   |         |              |          |
    #   |    OPT(to_bf16)    OPT(to_bf16)   |
    #   |         |              |          |
    #   |     OPT(reshape)   permute        |
    #   |            \        /             |
    #   |             addmm/mm              |
    #   |                |                  |
    #   |           OPT(reshape)            |

    # Case 5: int8-mixed-fp32, input dim size exceeds 2 and not contiguous
    # Case 6: int8-mixed-bf16, input dim size exceeds 2 and not contiguous

    #   + - - - - | - - - - - - | - - - - - +
    #   |    dq_per_tensor  dq_per_channel  |
    #   |         |              |          |
    #   |    OPT(to_bf16)    OPT(to_bf16)   |
    #   |         |              |          |
    #   |       expand       permute        |
    #   |          \             |          |
    #   |                    expand         |
    #   |                    /              |
    #   |               bmm                 |
    #   |                |                  |
    #   |            OPT(add)               |

    linear_weight_prepack_cases = itertools.product(
        [torch.float32, torch.bfloat16], [True, False], [True, False]
    )

    # Step 1: register patterns from mm and addmm
    for dtype, input_dim_exceeds_two, is_tensor_overload in linear_weight_prepack_cases:
        weight_prepack_patterns = _generate_qlinear_weight_prepack_patterns(
            dtype,
            input_dim_exceeds_two,
            is_tensor_overload=is_tensor_overload,
            per_channel=per_channel,
        )
        for weight_prepack_pattern in weight_prepack_patterns:
            # Register to pass_number 1, so we can do dequant promotion in pass_number 0.
            _register_qlinear_weight_prepack_pass(
                weight_prepack_pattern,
                pass_number=1,
                dtype=dtype,
                input_dim_exceeds_two=input_dim_exceeds_two,
                per_channel=per_channel,
            )

    # Step 2: register patterns from bmm
    # Linear might be decomposed into bmm when input dim exceeds 2 and not contiguous
    # refer to:
    # https://github.com/pytorch/pytorch/blob/
    # 80c07df659362a95da7cd4f3ec367abfdace38c4/torch/_decomp/decompositions.py#L3965-L3968
    # in this case, we can convert it back to qlinear
    for dtype, with_bias, is_tensor_overload in itertools.product(
        [torch.float32, torch.bfloat16], [True, False], [True, False]
    ):
        bmm_pattern = _generate_qlinear_weight_prepack_patterns(
            dtype=dtype,
            input_dim_exceeds_two=True,
            input_contiguous=False,
            with_bias=with_bias,
            is_tensor_overload=is_tensor_overload,
            per_channel=per_channel,
        )
        _register_qlinear_weight_prepack_pass(
            bmm_pattern,
            pass_number=1
            if with_bias
            else 2,  # if with_bias, there is an output add, so we should try to match it firstly
            dtype=dtype,
            input_dim_exceeds_two=True,
            input_contiguous=False,
            per_channel=per_channel,
        )


@functools.lru_cache(None)
def _register_quantization_weight_pack_pass(per_channel=True):
    # Step 1: Dequant promotion for int8-mixed-fp32/bf16
    _x86._register_dequant_promotion()
    # Step 1: QConv weight prepack
    _register_qconv_weight_prepack(per_channel)

    # Step 2: QLinear weight prepack
    _register_qlinear_weight_prepack(per_channel)

    # Step 4: weight prepack for SmoothQuant from Torchao
    _x86._register_smooth_quant_int_mm_pattern()

