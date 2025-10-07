import functools
import itertools

import torch
from torch._dynamo.utils import counters
from torch._inductor import config
from torch._inductor.lowering import lowerings as L
from torch._inductor.lowering import make_fallback
from torch._inductor.pattern_matcher import (
    Arg,
    CallFunction,
    KeywordArg,
    Match,
    PatternMatcherPass,
    register_lowering_pattern,
)

from torchao.utils import torch_version_at_least

if torch_version_at_least("2.7.0"):
    # PyTorch 2.7+ is needed for functions in qsdpa lowering
    from ..qsdpa_lowering import register_qsdpa  # noqa: F401
else:
    make_fallback(torch.ops.torchao.qscaled_dot_product.default)

__all__ = [
    "_qsdpa_init",
]

aten = torch.ops.aten
quantize_dtypes = [torch.uint8]


def _is_valid_qsdpa_pattern():
    def fn(match):
        assert all(k in match.kwargs for k in ("query", "key", "value"))
        query = match.kwargs["query"].meta["val"]
        key = match.kwargs["key"].meta["val"]
        value = match.kwargs["value"].meta["val"]
        return (
            query.dtype in quantize_dtypes
            and key.dtype in quantize_dtypes
            and value.dtype in quantize_dtypes
            and query.device.type == "cpu"
            and key.device == query.device
            and value.device == query.device
        )

    return fn


def _register_qsdpa_pattern(pattern, custom_pass_dict):
    @register_lowering_pattern(
        pattern, extra_check=_is_valid_qsdpa_pattern(), pass_dict=custom_pass_dict
    )
    def qsdpa(match: Match, *args, **kwargs):
        query = kwargs["query"]
        key = kwargs["key"]
        value = kwargs["value"]
        scale = 1.0 / kwargs["inv_scale"] if "inv_scale" in kwargs else None
        if scale is None:
            scale = kwargs["scale"] if "scale" in kwargs else None
        attn_mask = kwargs["attn_mask"] if "attn_mask" in kwargs else None
        q_zp = 0
        k_zp = 0
        v_zp = 0
        a_zp = 0
        o_zp = 0
        if query.dtype == torch.uint8:
            q_scale = kwargs["q_scale"]
            q_zp = kwargs["q_zp"]
            k_scale = kwargs["k_scale"]
            k_zp = kwargs["k_zp"]
            v_scale = kwargs["v_scale"]
            v_zp = kwargs["v_zp"]
            a_scale = kwargs["a_scale"]
            a_zp = kwargs["a_zp"]
            o_scale = kwargs["o_scale"]
            o_zp = kwargs["o_zp"]
        else:
            assert match.kwargs["q_scale"].target == aten.full.default
            q_scale = match.kwargs["q_scale"].args[1]
            k_scale = match.kwargs["k_scale"].args[1]
            v_scale = match.kwargs["v_scale"].args[1]
            a_scale = match.kwargs["a_scale"].args[1]
            o_scale = match.kwargs["o_scale"].args[1]

        counters["inductor"]["qsdpa_fuse_attention"] += 1
        counters["inductor"]["qsdpa_nodes"] += len(match.nodes)

        trans_query = L[aten.permute.default](query, [0, 2, 1, 3])
        trans_key = L[aten.permute.default](key, [0, 2, 1, 3])
        trans_value = L[aten.permute.default](value, [0, 2, 1, 3])
        output = L[torch.ops.torchao.qscaled_dot_product.default](
            trans_query,
            trans_key,
            trans_value,
            attn_mask,
            0.0,  # dropout
            False,  # is_causal
            scale,
            q_scale,
            q_zp,
            k_scale,
            k_zp,
            v_scale,
            v_zp,
            a_scale,
            a_zp,
            o_scale,
            o_zp,
        )
        trans_output = L[aten.permute.default](output, [0, 2, 1, 3])
        return L[aten.clone.default](
            trans_output, memory_format=torch.contiguous_format
        )

    return qsdpa


def _generate_dequant_pattern(
    input_pattern, qtype, is_reduced_type, scale: str, zp: str = None
):
    assert qtype is torch.uint8, "QSDPA expects type to be uint8"
    assert zp is not None, "Zero point must be provided for uint8 dequantization"
    return CallFunction(
        torch.ops.quantized_decomposed.dequantize_per_tensor.default,
        input_pattern,
        KeywordArg(scale),
        KeywordArg(zp),
        Arg(),
        Arg(),
        Arg(),
    )


def _generate_quant_pattern(input_pattern, qtype, scale: str, zp: str = None):
    assert qtype is torch.uint8, "QSDPA expects type to be uint8"
    assert zp is not None, "Zero point must be provided for uint8 quantization"
    return CallFunction(
        torch.ops.quantized_decomposed.quantize_per_tensor.default,
        input_pattern,
        KeywordArg(scale),
        KeywordArg(zp),
        Arg(),
        Arg(),
        Arg(),
    )


def _get_qsdpa_qkv_pattern(
    qtype,
    is_batch_size_1: bool,
    is_reduced_type: bool,
    has_convert: bool,
    input_name: str,
):
    assert input_name in ["query", "key", "value"]
    qsdpa_qkv_pattern_before_dequant = CallFunction(
        aten.permute.default,
        KeywordArg(input_name),
        Arg(),
    )
    if input_name == "key":
        # do transpose
        qsdpa_qkv_pattern_before_dequant = CallFunction(
            aten.permute.default,
            qsdpa_qkv_pattern_before_dequant,
            Arg(),
        )
    qsdpa_qkv_basic_pattern = _generate_dequant_pattern(
        qsdpa_qkv_pattern_before_dequant,
        qtype,
        is_reduced_type,
        input_name[0] + "_scale",
        input_name[0] + "_zp" if qtype is torch.uint8 else None,
    )
    if has_convert:
        qsdpa_qkv_basic_pattern = CallFunction(
            torch.ops.prims.convert_element_type.default,
            qsdpa_qkv_basic_pattern,
            Arg(),
        )
    qsdpa_qkv_basic_pattern = CallFunction(
        aten.expand.default,
        qsdpa_qkv_basic_pattern,
        Arg(),
    )
    if is_batch_size_1:
        # pattern is different for bs=1
        return CallFunction(
            aten.reshape.default,
            qsdpa_qkv_basic_pattern,
            Arg(),
        )
    else:
        return CallFunction(
            aten.reshape.default,
            CallFunction(
                aten.clone.default,
                qsdpa_qkv_basic_pattern,
                memory_format=Arg(),
            ),
            Arg(),
        )


def _get_qsdpa_score_pattern(
    qtype,
    has_mask: bool,
    is_batch_size_1: bool,
    is_reduced_type: bool,
    has_convert: bool,
    is_inv_scale: bool,
):
    qsdpa_q_pattern = _get_qsdpa_qkv_pattern(
        qtype, is_batch_size_1, is_reduced_type, has_convert, "query"
    )
    qsdpa_k_pattern = _get_qsdpa_qkv_pattern(
        qtype, is_batch_size_1, is_reduced_type, has_convert, "key"
    )
    qsdpa_score_basic_pattern = CallFunction(
        aten.reshape.default,
        CallFunction(
            aten.bmm.default,
            qsdpa_q_pattern,
            qsdpa_k_pattern,
        ),
        Arg(),
    )
    if is_reduced_type and not has_mask:
        qsdpa_score_basic_pattern = CallFunction(
            torch.ops.prims.convert_element_type.default,
            qsdpa_score_basic_pattern,
            Arg(),
        )
    if not has_mask:
        return CallFunction(
            aten.mul.Tensor,
            qsdpa_score_basic_pattern,
            Arg(),
            _users=2,
        )
    elif is_inv_scale:
        return CallFunction(
            aten.add.Tensor,
            CallFunction(
                aten.div.Tensor,
                qsdpa_score_basic_pattern,
                KeywordArg("inv_scale"),
            ),
            KeywordArg("attn_mask"),
            _users=2,
        )
    else:
        return CallFunction(
            aten.add.Tensor,
            CallFunction(
                aten.mul.Tensor,
                qsdpa_score_basic_pattern,
                KeywordArg("scale"),
            ),
            KeywordArg("attn_mask"),
            _users=2,
        )


def _get_qsdpa_exp_pattern(
    qtype,
    has_mask: bool,
    is_batch_size_1: bool,
    is_reduced_type: bool,
    has_convert: bool,
    is_inv_scale: bool,
):
    qsdpa_score_pattern = _get_qsdpa_score_pattern(
        qtype, has_mask, is_batch_size_1, is_reduced_type, has_convert, is_inv_scale
    )
    qsdpa_exp_basic_pattern = CallFunction(
        aten.sub.Tensor,
        qsdpa_score_pattern,
        CallFunction(
            aten.amax.default,
            qsdpa_score_pattern,
            Arg(),
            Arg(),
        ),
    )
    if has_mask:
        return CallFunction(
            aten.exp.default,
            qsdpa_exp_basic_pattern,
            _users=2,
        )
    elif is_inv_scale:
        return CallFunction(
            aten.exp.default,
            CallFunction(
                aten.div.Tensor,
                qsdpa_exp_basic_pattern,
                KeywordArg("inv_scale"),
            ),
            _users=2,
        )
    else:
        return CallFunction(
            aten.exp.default,
            CallFunction(
                aten.mul.Tensor,
                qsdpa_exp_basic_pattern,
                KeywordArg("scale"),
            ),
            _users=2,
        )


def _get_qsdpa_attn_pattern(
    qtype,
    has_mask: bool,
    is_batch_size_1: bool,
    is_reduced_type: bool,
    has_convert: bool,
    is_inv_scale: bool,
):
    qsdpa_exp_pattern = _get_qsdpa_exp_pattern(
        qtype, has_mask, is_batch_size_1, is_reduced_type, has_convert, is_inv_scale
    )
    qsdpa_div_pattern = CallFunction(
        aten.div.Tensor,
        qsdpa_exp_pattern,
        CallFunction(
            aten.sum.dim_IntList,
            qsdpa_exp_pattern,
            Arg(),
            Arg(),
        ),
    )
    qsdpa_softmax_pattern = _generate_dequant_pattern(
        _generate_quant_pattern(
            qsdpa_div_pattern,
            qtype,
            "a_scale",
            "a_zp" if qtype is torch.uint8 else None,
        ),
        qtype,
        is_reduced_type,
        "a_scale",
        "a_zp" if qtype is torch.uint8 else None,
    )
    if is_reduced_type:
        if has_mask:
            qsdpa_softmax_pattern = CallFunction(
                torch.ops.prims.convert_element_type.default,
                qsdpa_softmax_pattern,
                Arg(),
            )
        else:
            qsdpa_softmax_pattern = _generate_dequant_pattern(
                _generate_quant_pattern(
                    CallFunction(
                        torch.ops.prims.convert_element_type.default,
                        qsdpa_div_pattern,
                        Arg(),
                    ),
                    qtype,
                    "a_scale",
                    "a_zp" if qtype is torch.uint8 else None,
                ),
                qtype,
                is_reduced_type,
                "a_scale",
                "a_zp" if qtype is torch.uint8 else None,
            )
            if has_convert:
                qsdpa_softmax_pattern = CallFunction(
                    torch.ops.prims.convert_element_type.default,
                    qsdpa_softmax_pattern,
                    Arg(),
                )
    return CallFunction(
        aten.reshape.default,
        CallFunction(
            aten.expand.default,
            qsdpa_softmax_pattern,
            Arg(),
        ),
        Arg(),
    )


# Parameters to generate various patterns:
#   qdtype: quantized dtypes are uint8, float8_e4m3fn for now
#   has_mask: if SDPA has attention mask
#   is_batch_size_1: if the batch size is 1
#   is_reduced_type: if autocast is enabled
#   has_convert: convert type if dequant out dtype is assigned
#   is_inv_scale: if the scale in SDPA is inversed, in which case it is multiplied instead of divided
def _get_qsdpa_final_pattern(
    qtype,
    has_mask: bool,
    is_batch_size_1: bool,
    is_reduced_type: bool,
    has_convert: bool,
    is_inv_scale: bool,
):
    qsdpa_v_pattern = _get_qsdpa_qkv_pattern(
        qtype, is_batch_size_1, is_reduced_type, has_convert, "value"
    )
    qsdpa_attn_pattern = _get_qsdpa_attn_pattern(
        qtype, has_mask, is_batch_size_1, is_reduced_type, has_convert, is_inv_scale
    )
    return _generate_quant_pattern(
        CallFunction(
            aten.clone.default,
            CallFunction(
                aten.permute.default,
                CallFunction(
                    aten.reshape.default,
                    CallFunction(
                        aten.bmm.default,
                        qsdpa_attn_pattern,
                        qsdpa_v_pattern,
                    ),
                    Arg(),
                ),
                Arg(),
            ),
            memory_format=Arg(),
        ),
        qtype,
        "o_scale",
        "o_zp" if qtype is torch.uint8 else None,
    )


def _register_qsdpa_lowerings(custom_pass_dict):
    for (
        qtype,
        has_mask,
        is_batch_size_1,
        is_reduced_type,
        has_convert,
        is_inv_scale,
    ) in itertools.product(
        quantize_dtypes,
        [True, False],
        [True, False],
        [True, False],
        [True, False],
        [True, False],
    ):
        _register_qsdpa_pattern(
            _get_qsdpa_final_pattern(
                qtype=qtype,
                has_mask=has_mask,
                is_batch_size_1=is_batch_size_1,
                is_reduced_type=is_reduced_type,
                has_convert=has_convert,
                is_inv_scale=is_inv_scale,
            ),
            custom_pass_dict,
        )


custom_pass = None
if torch_version_at_least("2.7.0"):
    # PyTorch 2.7+ is needed for custom graph pass
    from torch._inductor.custom_graph_pass import CustomGraphPass, get_hash_for_files

    # define the custom pass
    class _CustomPass(PatternMatcherPass, CustomGraphPass):
        def __init__(self) -> None:
            super().__init__()

        def __call__(self, g: torch.fx.graph.Graph):
            self.apply(g)

        def uuid(self) -> bytes:
            return get_hash_for_files((__file__,))

    custom_pass = _CustomPass()


@functools.lru_cache(None)
def _qsdpa_init():
    if torch_version_at_least("2.7.0"):
        _register_qsdpa_lowerings(config.post_grad_custom_pre_pass)
    else:
        pass
