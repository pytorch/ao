import functools
import itertools

import torch
from torch._dynamo.utils import counters
from torch._inductor import config
from torch._inductor.fx_passes.post_grad import register_lowering_pattern
from torch._inductor.lowering import lowerings as L
from torch._inductor.lowering import make_fallback
from torch._inductor.pattern_matcher import (
    Arg,
    CallFunction,
    KeywordArg,
    Match,
    PatternMatcherPass,
)

__all__ = [
    "_int8_sdpa_init",
]

make_fallback(torch.ops.torchao.scaled_dot_product_int8.default)

aten = torch.ops.aten
patterns = PatternMatcherPass()


def _is_valid_int8_sdpa_pattern():
    def fn(match):
        assert all(k in match.kwargs for k in ("query", "key", "value"))
        query = match.kwargs["query"].meta["val"]
        key = match.kwargs["key"].meta["val"]
        value = match.kwargs["value"].meta["val"]
        return (
            query.dtype == torch.uint8
            and key.dtype == torch.uint8
            and value.dtype == torch.uint8
            and query.device.type == "cpu"
            and key.device == query.device
            and value.device == query.device
        )

    return fn


def _register_int8_sdpa_pattern(pattern):
    @register_lowering_pattern(
        pattern,
        extra_check=_is_valid_int8_sdpa_pattern(),
    )
    def int8_sdpa(match: Match, *args, **kwargs):
        query = kwargs["query"]
        key = kwargs["key"]
        value = kwargs["value"]
        inv_scale = kwargs["inv_scale"]
        attn_mask = kwargs["attn_mask"] if "attn_mask" in kwargs else None
        q_zp = kwargs["q_zp"]
        q_scale = kwargs["q_scale"]
        k_zp = kwargs["k_zp"]
        k_scale = kwargs["k_scale"]
        v_zp = kwargs["v_zp"]
        v_scale = kwargs["v_scale"]
        a_zp = kwargs["a_zp"]
        a_scale = kwargs["a_scale"]
        o_zp = kwargs["o_zp"]
        o_scale = kwargs["o_scale"]
        counters["inductor"]["int8_fuse_attention"] += 1
        counters["inductor"]["int8_sdpa_nodes"] += len(match.nodes)

        trans_query = L[aten.permute.default](query, [0, 2, 1, 3])
        trans_key = L[aten.permute.default](key, [0, 2, 1, 3])
        trans_value = L[aten.permute.default](value, [0, 2, 1, 3])
        output = L[torch.ops.torchao.scaled_dot_product_int8.default](
            trans_query,
            trans_key,
            trans_value,
            attn_mask,
            0.0,  # dropout
            False,  # is_causal
            1.0 / inv_scale,  # scale
            q_zp,
            q_scale,
            k_zp,
            k_scale,
            v_zp,
            v_scale,
            a_zp,
            a_scale,
            o_zp,
            o_scale,
        )
        trans_output = L[aten.permute.default](output, [0, 2, 1, 3])
        return L[aten.clone.default](
            trans_output, memory_format=torch.contiguous_format
        )

    return int8_sdpa


def _get_int8_sdpa_qkv_pattern(
    is_batch_size_1: bool, has_convert: bool, input_name: str
):
    assert input_name in ["query", "key", "value"]
    int8_sdpa_qkv_pattern_before_dequant = CallFunction(
        aten.permute.default,
        KeywordArg(input_name),
        Arg(),
    )
    if input_name == "key":
        # do transpose
        int8_sdpa_qkv_pattern_before_dequant = CallFunction(
            aten.permute.default,
            int8_sdpa_qkv_pattern_before_dequant,
            Arg(),
        )
    int8_sdpa_qkv_basic_pattern = CallFunction(
        torch.ops.quantized_decomposed.dequantize_per_tensor.default,
        int8_sdpa_qkv_pattern_before_dequant,
        KeywordArg(input_name[0] + "_scale"),
        KeywordArg(input_name[0] + "_zp"),
        Arg(),
        Arg(),
        Arg(),
    )
    if has_convert:
        int8_sdpa_qkv_basic_pattern = CallFunction(
            torch.ops.prims.convert_element_type.default,
            int8_sdpa_qkv_basic_pattern,
            Arg(),
        )
    int8_sdpa_qkv_basic_pattern = CallFunction(
        aten.expand.default,
        int8_sdpa_qkv_basic_pattern,
        Arg(),
    )
    if is_batch_size_1:
        # pattern is different for bs=1
        return CallFunction(
            aten.reshape.default,
            int8_sdpa_qkv_basic_pattern,
            Arg(),
        )
    else:
        return CallFunction(
            aten.reshape.default,
            CallFunction(
                aten.clone.default,
                int8_sdpa_qkv_basic_pattern,
                memory_format=Arg(),
            ),
            Arg(),
        )


def _get_int8_sdpa_score_pattern(
    has_mask: bool, is_batch_size_1: bool, is_reduced_type: bool, has_convert: bool
):
    int8_sdpa_q_pattern = _get_int8_sdpa_qkv_pattern(
        is_batch_size_1, has_convert, "query"
    )
    int8_sdpa_k_pattern = _get_int8_sdpa_qkv_pattern(
        is_batch_size_1, has_convert, "key"
    )
    int8_sdpa_score_basic_pattern = CallFunction(
        aten.reshape.default,
        CallFunction(
            aten.bmm.default,
            int8_sdpa_q_pattern,
            int8_sdpa_k_pattern,
        ),
        Arg(),
    )
    if is_reduced_type and not has_mask:
        int8_sdpa_score_basic_pattern = CallFunction(
            torch.ops.prims.convert_element_type.default,
            int8_sdpa_score_basic_pattern,
            Arg(),
        )
    if has_mask:
        return CallFunction(
            aten.add.Tensor,
            CallFunction(
                aten.div.Tensor,
                int8_sdpa_score_basic_pattern,
                KeywordArg("inv_scale"),
            ),
            KeywordArg("attn_mask"),
            _users=2,
        )
    else:
        return CallFunction(
            aten.mul.Tensor,
            int8_sdpa_score_basic_pattern,
            Arg(),
            _users=2,
        )


def _get_int8_sdpa_exp_pattern(
    has_mask: bool, is_batch_size_1: bool, is_reduced_type: bool, has_convert: bool
):
    int8_sdpa_score_pattern = _get_int8_sdpa_score_pattern(
        has_mask, is_batch_size_1, is_reduced_type, has_convert
    )
    int8_sdpa_exp_basic_pattern = CallFunction(
        aten.sub.Tensor,
        int8_sdpa_score_pattern,
        CallFunction(
            aten.amax.default,
            int8_sdpa_score_pattern,
            Arg(),
            Arg(),
        ),
    )
    if has_mask:
        return CallFunction(
            aten.exp.default,
            int8_sdpa_exp_basic_pattern,
            _users=2,
        )
    else:
        return CallFunction(
            aten.exp.default,
            CallFunction(
                aten.div.Tensor,
                int8_sdpa_exp_basic_pattern,
                KeywordArg("inv_scale"),
            ),
            _users=2,
        )


def _get_int8_sdpa_attn_pattern(
    has_mask: bool, is_batch_size_1: bool, is_reduced_type: bool, has_convert: bool
):
    int8_sdpa_exp_pattern = _get_int8_sdpa_exp_pattern(
        has_mask, is_batch_size_1, is_reduced_type, has_convert
    )
    int8_sdpa_div_pattern = CallFunction(
        aten.div.Tensor,
        int8_sdpa_exp_pattern,
        CallFunction(
            aten.sum.dim_IntList,
            int8_sdpa_exp_pattern,
            Arg(),
            Arg(),
        ),
    )
    int8_sdpa_softmax_pattern = CallFunction(
        torch.ops.quantized_decomposed.dequantize_per_tensor.default,
        CallFunction(
            torch.ops.quantized_decomposed.quantize_per_tensor.default,
            int8_sdpa_div_pattern,
            KeywordArg("a_scale"),
            KeywordArg("a_zp"),
            Arg(),
            Arg(),
            Arg(),
        ),
        KeywordArg("a_scale"),
        KeywordArg("a_zp"),
        Arg(),
        Arg(),
        Arg(),
    )
    if is_reduced_type:
        if has_mask:
            int8_sdpa_softmax_pattern = CallFunction(
                torch.ops.prims.convert_element_type.default,
                int8_sdpa_softmax_pattern,
                Arg(),
            )
        else:
            int8_sdpa_softmax_pattern = CallFunction(
                torch.ops.quantized_decomposed.dequantize_per_tensor.default,
                CallFunction(
                    torch.ops.quantized_decomposed.quantize_per_tensor.default,
                    CallFunction(
                        torch.ops.prims.convert_element_type.default,
                        int8_sdpa_div_pattern,
                        Arg(),
                    ),
                    KeywordArg("a_scale"),
                    KeywordArg("a_zp"),
                    Arg(),
                    Arg(),
                    Arg(),
                ),
                KeywordArg("a_scale"),
                KeywordArg("a_zp"),
                Arg(),
                Arg(),
                Arg(),
            )
            if has_convert:
                int8_sdpa_softmax_pattern = CallFunction(
                    torch.ops.prims.convert_element_type.default,
                    int8_sdpa_softmax_pattern,
                    Arg(),
                )
    return CallFunction(
        aten.reshape.default,
        CallFunction(
            aten.expand.default,
            int8_sdpa_softmax_pattern,
            Arg(),
        ),
        Arg(),
    )


# Parameters to generate various patterns:
#   has_mask: if SDPA has attention mask
#   is_batch_size_1: if the batch size is 1
#   is_reduced_type: if autocast is enabled
#   has_convert: convert type if dequant out dtype is assigned
def _get_int8_sdpa_final_pattern(
    has_mask: bool, is_batch_size_1: bool, is_reduced_type: bool, has_convert: bool
):
    int8_sdpa_v_pattern = _get_int8_sdpa_qkv_pattern(
        is_batch_size_1, has_convert, "value"
    )
    int8_sdpa_attn_pattern = _get_int8_sdpa_attn_pattern(
        has_mask, is_batch_size_1, is_reduced_type, has_convert
    )
    return CallFunction(
        torch.ops.quantized_decomposed.quantize_per_tensor.default,
        CallFunction(
            aten.clone.default,
            CallFunction(
                aten.permute.default,
                CallFunction(
                    aten.reshape.default,
                    CallFunction(
                        aten.bmm.default,
                        int8_sdpa_attn_pattern,
                        int8_sdpa_v_pattern,
                    ),
                    Arg(),
                ),
                Arg(),
            ),
            memory_format=Arg(),
        ),
        KeywordArg("o_scale"),
        KeywordArg("o_zp"),
        Arg(),
        Arg(),
        Arg(),
    )


def _register_int8_sdpa_lowerings():
    for has_mask, is_batch_size_1, is_reduced_type, has_convert in itertools.product(
        [True, False], [True, False], [True, False], [True, False]
    ):
        _register_int8_sdpa_pattern(
            _get_int8_sdpa_final_pattern(
                has_mask=has_mask,
                is_batch_size_1=is_batch_size_1,
                is_reduced_type=is_reduced_type,
                has_convert=has_convert,
            )
        )


@functools.lru_cache(None)
def _int8_sdpa_init():
    _register_int8_sdpa_lowerings()
    config.post_grad_custom_pre_pass = patterns.apply
