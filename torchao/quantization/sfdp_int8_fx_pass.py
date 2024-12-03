import functools
from typing import Callable

import torch
from torch._inductor import config
from torch._inductor.pattern_matcher import (
    filter_nodes,
    fwd_only,
    register_replacement,
    gen_register_replacement,
    PatternMatcherPass,
)
from torch._dynamo.utils import counters
from torch._inductor.fx_passes.fuse_attention import (
    partialize_and_update_signature
)
from torchao.ops import scaled_dot_product_int8

__all__ = [
    # "_sfdp_pattern_int8",
    # "_sfdp_replacement_int8",
    # "_gen_sfdp_patterns_int8",
    "_sfdp_init_int8",
]

aten = torch.ops.aten
# scaled_dot_product_int8 = torch.ops.torchao.scaled_dot_product_int8
patterns = PatternMatcherPass()

# def _sfdp_pattern_int8(query, key, value, inv_scale):
#     return (
#         torch.matmul(query, key.transpose(-2, -1))
#         .div(inv_scale)
#         .softmax(dim=-1)
#         .matmul(value)
#     )

# def _sfdp_replacement_int8(query, key, value, inv_scale):
#     print("*** enter _sfdp_replacement in torchao ***")
#     counters["inductor"]["fuse_attention_int8"] += 1
#     return torch.nn.functional.scaled_dot_product_attention(
#         query,
#         key,
#         value,
#         attn_mask=None,
#         dropout_p=0.0,
#         is_causal=False,
#         scale=1.0 / inv_scale,
#     )

def _sfdp_pattern_int8_1(
    query,
    key,
    value,
    attn_mask,
    inv_scale,
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
    dropout,
):
    # int8-mix-fp32 QUANTIZED SDPA with mask
    q = query.permute([0, 2, 1, 3])
    q = torch.ops.quantized_decomposed.dequantize_per_tensor.default(
        q, float(q_scale), int(q_zp), 0, 255, torch.uint8
    )
    k = key.permute([0, 2, 1, 3]).transpose(-2, -1)
    k = torch.ops.quantized_decomposed.dequantize_per_tensor.default(
        k, float(k_scale), int(k_zp), 0, 255, torch.uint8
    )
    v = value.permute([0, 2, 1, 3])
    v = torch.ops.quantized_decomposed.dequantize_per_tensor.default(
        v, float(v_scale), int(v_zp), 0, 255, torch.uint8
    )
    a = torch.nn.functional.dropout(
        (torch.matmul(q, k).div(inv_scale) + attn_mask).softmax(dim=-1),
        dropout,
    )
    qa = torch.ops.quantized_decomposed.quantize_per_tensor.default(
        a, float(a_scale), int(a_zp), 0, 255, torch.uint8
    )
    a = torch.ops.quantized_decomposed.dequantize_per_tensor.default(
        qa, float(a_scale), int(a_zp), 0, 255, torch.uint8
    )
    o = a.matmul(v)
    o = o.permute(0, 2, 1, 3).contiguous()
    return torch.ops.quantized_decomposed.quantize_per_tensor.default(
        o, float(o_scale), int(o_zp), 0, 255, torch.uint8
    )


def _sfdp_replacement_int8_1(
    query,
    key,
    value,
    attn_mask,
    inv_scale,
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
    dropout,
):
    print("hit _sfdp_replacement_int8_1")
    counters["inductor"]["fuse_attention_int8"] += 1
    res = scaled_dot_product_int8(
        query.transpose(1, 2),
        key.transpose(1, 2),
        value.transpose(1, 2),
        attn_mask=attn_mask,
        dropout_p=dropout,
        is_causal=False,
        scale=1.0 / inv_scale,
        q_zp=q_zp,
        q_scale=q_scale,
        k_zp=k_zp,
        k_scale=k_scale,
        v_zp=v_zp,
        v_scale=v_scale,
        a_zp=a_zp,
        a_scale=a_scale,
        o_zp=o_zp,
        o_scale=o_scale,
    )
    return res.permute(0, 2, 1, 3).contiguous()


def _sfdp_pattern_int8_2(
    query,
    key,
    value,
    attn_mask,
    inv_scale,
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
    dropout,
):
    # int8-mix-reduce QUANTIZED SDPA with mask
    q = query.permute([0, 2, 1, 3])
    q = torch.ops.quantized_decomposed.dequantize_per_tensor.default(
        q, float(q_scale), int(q_zp), 0, 255, torch.uint8
    ).to(torch.float16)
    k = key.permute([0, 2, 1, 3]).transpose(-2, -1)
    k = torch.ops.quantized_decomposed.dequantize_per_tensor.default(
        k, float(k_scale), int(k_zp), 0, 255, torch.uint8
    ).to(torch.float16)
    v = value.permute([0, 2, 1, 3])
    v = torch.ops.quantized_decomposed.dequantize_per_tensor.default(
        v, float(v_scale), int(v_zp), 0, 255, torch.uint8
    ).to(torch.float16)
    a = torch.nn.functional.dropout(
        (torch.matmul(q, k).div(inv_scale) + attn_mask).softmax(dim=-1),
        dropout,
    )
    qa = torch.ops.quantized_decomposed.quantize_per_tensor.default(
        a, float(a_scale), int(a_zp), 0, 255, torch.uint8
    )
    a = torch.ops.quantized_decomposed.dequantize_per_tensor.default(
        qa, float(a_scale), int(a_zp), 0, 255, torch.uint8
    ).to(torch.float16)
    o = a.matmul(v)
    o = o.permute(0, 2, 1, 3).contiguous()
    return torch.ops.quantized_decomposed.quantize_per_tensor.default(
        o, float(o_scale), int(o_zp), 0, 255, torch.uint8
    )


def _sfdp_replacement_int8_2(
    query,
    key,
    value,
    attn_mask,
    inv_scale,
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
    dropout,
):
    print("hit _sfdp_replacement_int8_2")
    counters["inductor"]["fuse_attention_int8"] += 1
    res = scaled_dot_product_int8(
        query.transpose(1, 2),
        key.transpose(1, 2),
        value.transpose(1, 2),
        attn_mask=attn_mask,
        dropout_p=dropout,
        is_causal=False,
        scale=1.0 / inv_scale,
        q_zp=q_zp,
        q_scale=q_scale,
        k_zp=k_zp,
        k_scale=k_scale,
        v_zp=v_zp,
        v_scale=v_scale,
        a_zp=a_zp,
        a_scale=a_scale,
        o_zp=o_zp,
        o_scale=o_scale,
    )
    return res.permute(0, 2, 1, 3).contiguous()


def _sfdp_pattern_int8_3(
    query,
    key,
    value,
    inv_scale,
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
    dropout,
):
    # int8-mix-fp32 QUANTIZED SDPA without mask
    q = query.permute([0, 2, 1, 3])
    q = torch.ops.quantized_decomposed.dequantize_per_tensor.default(
        q, float(q_scale), int(q_zp), 0, 255, torch.uint8
    )
    k = key.permute([0, 2, 1, 3]).transpose(-2, -1)
    k = torch.ops.quantized_decomposed.dequantize_per_tensor.default(
        k, float(k_scale), int(k_zp), 0, 255, torch.uint8
    )
    v = value.permute([0, 2, 1, 3])
    v = torch.ops.quantized_decomposed.dequantize_per_tensor.default(
        v, float(v_scale), int(v_zp), 0, 255, torch.uint8
    )
    a = torch.nn.functional.dropout(
        torch.matmul(q, k).div(inv_scale).softmax(dim=-1),
        dropout,
    )
    qa = torch.ops.quantized_decomposed.quantize_per_tensor.default(
        a, float(a_scale), int(a_zp), 0, 255, torch.uint8
    )
    a = torch.ops.quantized_decomposed.dequantize_per_tensor.default(
        qa, float(a_scale), int(a_zp), 0, 255, torch.uint8
    )
    o = a.matmul(v)
    o = o.permute(0, 2, 1, 3).contiguous()
    return torch.ops.quantized_decomposed.quantize_per_tensor.default(
        o, float(o_scale), int(o_zp), 0, 255, torch.uint8
    )


def _sfdp_replacement_int8_3(
    query,
    key,
    value,
    inv_scale,
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
    dropout,
):
    print("hit _sfdp_replacement_int8_3")
    counters["inductor"]["fuse_attention_int8"] += 1
    res = scaled_dot_product_int8(
        query.transpose(1, 2),
        key.transpose(1, 2),
        value.transpose(1, 2),
        dropout_p=dropout,
        is_causal=False,
        scale=1.0 / inv_scale,
        q_zp=q_zp,
        q_scale=q_scale,
        k_zp=k_zp,
        k_scale=k_scale,
        v_zp=v_zp,
        v_scale=v_scale,
        a_zp=a_zp,
        a_scale=a_scale,
        o_zp=o_zp,
        o_scale=o_scale,
    )
    return res.permute(0, 2, 1, 3).contiguous()


def _sfdp_pattern_int8_4(
    query,
    key,
    value,
    inv_scale,
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
    dropout,
):
    # int8-mix-reduce QUANTIZED SDPA without mask
    q = query.permute([0, 2, 1, 3])
    q = torch.ops.quantized_decomposed.dequantize_per_tensor.default(
        q, float(q_scale), int(q_zp), 0, 255, torch.uint8
    ).to(torch.float16)
    k = key.permute([0, 2, 1, 3]).transpose(-2, -1)
    k = torch.ops.quantized_decomposed.dequantize_per_tensor.default(
        k, float(k_scale), int(k_zp), 0, 255, torch.uint8
    ).to(torch.float16)
    v = value.permute([0, 2, 1, 3])
    v = torch.ops.quantized_decomposed.dequantize_per_tensor.default(
        v, float(v_scale), int(v_zp), 0, 255, torch.uint8
    ).to(torch.float16)
    a = torch.nn.functional.dropout(
        torch.matmul(q, k).div(inv_scale).softmax(dim=-1),
        dropout,
    )
    qa = torch.ops.quantized_decomposed.quantize_per_tensor.default(
        a, float(a_scale), int(a_zp), 0, 255, torch.uint8
    )
    a = torch.ops.quantized_decomposed.dequantize_per_tensor.default(
        qa, float(a_scale), int(a_zp), 0, 255, torch.uint8
    ).to(torch.float16)
    o = a.matmul(v)
    o = o.permute(0, 2, 1, 3).contiguous()
    return torch.ops.quantized_decomposed.quantize_per_tensor.default(
        o, float(o_scale), int(o_zp), 0, 255, torch.uint8
    )


def _sfdp_replacement_int8_4(
    query,
    key,
    value,
    inv_scale,
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
    dropout,
):
    print("hit _sfdp_replacement_int8_4")
    counters["inductor"]["fuse_attention_int8"] += 1
    res = scaled_dot_product_int8(
        query.transpose(1, 2),
        key.transpose(1, 2),
        value.transpose(1, 2),
        dropout_p=dropout,
        is_causal=False,
        scale=1.0 / inv_scale,
        q_zp=q_zp,
        q_scale=q_scale,
        k_zp=k_zp,
        k_scale=k_scale,
        v_zp=v_zp,
        v_scale=v_scale,
        a_zp=a_zp,
        a_scale=a_scale,
        o_zp=o_zp,
        o_scale=o_scale,
    )
    return res.permute(0, 2, 1, 3).contiguous()


def _sfdp_params_check_int8(match):
    assert all(k in match.kwargs for k in ("query", "key", "value"))
    query = match.kwargs["query"].meta["val"]
    key = match.kwargs["key"].meta["val"]
    value = match.kwargs["value"].meta["val"]
    if not (query.dtype == key.dtype == value.dtype) or not (
        query.device == key.device == value.device
    ):
        return False
    add_nodes = filter_nodes(match.nodes, aten.add.Tensor)
    # Has attn_mask add.
    add_mask_node = [n for n in add_nodes if n.prev.target == torch.ops.aten.div.Tensor]
    if len(add_mask_node) > 0:
        attn_mask_node = add_mask_node[0].args[1]
        # attn_mask_node may be a float/int number.
        if not hasattr(attn_mask_node, "meta"):
            return False
        attn_mask = attn_mask_node.meta["val"]  # type: ignore[union-attr]
        # Make sure attn_mask.dtype == query.dtype or attn_mask.dtype == torch.bool
        # attn_mask.dtype == torch.float for models like albert.
        if (
            not isinstance(attn_mask, torch.Tensor)
            or not (
                attn_mask.dtype == query.dtype
                or attn_mask.dtype == torch.bool
                or attn_mask.dtype == torch.float
            )
            or query.device != attn_mask.device
        ):
            return False
    return True


def _sfdp_extra_check_int8(scale_factor_op=None, disable_cuda=False):
    def fn(match):
        if (
            disable_cuda
            and "query" in match.kwargs
            and "cuda" in str(match.kwargs["query"].meta["val"].device)
        ):
            return False
        if scale_factor_op is not None:
            scale_factor_node = filter_nodes(match.nodes, scale_factor_op)[0]
            # Note: args[1] of the scale_factor_node is always the scale_factor for the current patterns.
            scale_factor = scale_factor_node.args[1]
            # make sure the scale_factor a float/int. SymInt?
            if not isinstance(scale_factor, (float, int)):
                return False
        return _sfdp_params_check_int8(match)

    return fn


def _gen_sfdp_patterns_int8():
    if torch.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"
    g_inp = functools.partial(
        torch.empty, (2, 4, 8, 16), device=device, requires_grad=True
    )
    m_inp = functools.partial(torch.empty, (2, 1, 1, 4), device=device) # attn_mask
    c_inp = functools.partial(torch.tensor, 2.0, device=device) # inv_scale
    zp_inp = functools.partial(torch.tensor, 127, device=device) # quant_zero_point
    scale_inp = functools.partial(torch.tensor, 0.018, device=device) # quant_scale
    
    # reshape in matmul decomposition generates a clone when batch_size>1 due to the memory layout change.
    # however when batch_size=1, reshape does not change the memory layout, so clone would not be generated.
    # here we need to trace with input of batch_size=1 to generate a pattern graph without clone.
    g_bs1_inp = functools.partial(
        torch.empty, (1, 4, 8, 16), device=device, requires_grad=True
    )
    m_bs1_inp = functools.partial(torch.empty, (1, 1, 1, 4), device=device)
    for dtype in [torch.float, torch.half]:
        # g = functools.partial(g_inp, dtype=dtype)
        # c = functools.partial(c_inp, dtype=dtype)
        # candidates = [
        #     (
        #         _sfdp_pattern_int8,
        #         _sfdp_replacement_int8,
        #         [g(), g(), g(), c()],
        #         {},
        #         _sfdp_extra_check_int8(aten.div.Tensor),
        #     ),
        # ]
        g_u8 = functools.partial(g_inp, dtype=torch.uint8, requires_grad=False)
        g_u8_bs1 = functools.partial(g_bs1_inp, dtype=torch.uint8, requires_grad=False)
        m = functools.partial(m_inp, dtype=dtype)
        m_bs1 = functools.partial(m_bs1_inp, dtype=dtype)
        c = functools.partial(c_inp, dtype=dtype)
        zp = functools.partial(zp_inp, dtype=torch.int)
        scale = functools.partial(scale_inp, dtype=torch.float)
        d_u8 = {
            "dropout": 0.113377,
            "q_zp": 23,
            "q_scale": 0.0111541,
            "k_zp": 14,
            "k_scale": 0.0256212,
            "v_zp": 28,
            "v_scale": 0.0164518,
            "a_zp": 12,
            "a_scale": 0.0572114,
            "o_zp": 36,
            "o_scale": 0.0235489,
        }
        int8_candidates = [
            (
                _sfdp_pattern_int8_1,
                _sfdp_replacement_int8_1,
                [
                    g_u8(),
                    g_u8(),
                    g_u8(),
                    m(),
                    c(),
                    zp(),
                    scale(),
                    zp(),
                    scale(),
                    zp(),
                    scale(),
                    zp(),
                    scale(),
                    zp(),
                    scale(),
                ],
                d_u8,
                _sfdp_extra_check_int8(aten.div.Tensor),
            ),
            (
                _sfdp_pattern_int8_1,
                _sfdp_replacement_int8_1,
                [
                    g_u8_bs1(),
                    g_u8_bs1(),
                    g_u8_bs1(),
                    m_bs1(),
                    c(),
                    zp(),
                    scale(),
                    zp(),
                    scale(),
                    zp(),
                    scale(),
                    zp(),
                    scale(),
                    zp(),
                    scale(),
                ],
                d_u8,
                _sfdp_extra_check_int8(aten.div.Tensor),
            ),
            (
                _sfdp_pattern_int8_2,
                _sfdp_replacement_int8_2,
                [
                    g_u8(),
                    g_u8(),
                    g_u8(),
                    m(),
                    c(),
                    zp(),
                    scale(),
                    zp(),
                    scale(),
                    zp(),
                    scale(),
                    zp(),
                    scale(),
                    zp(),
                    scale(),
                ],
                d_u8,
                _sfdp_extra_check_int8(aten.div.Tensor),
            ),
            (
                _sfdp_pattern_int8_2,
                _sfdp_replacement_int8_2,
                [
                    g_u8_bs1(),
                    g_u8_bs1(),
                    g_u8_bs1(),
                    m_bs1(),
                    c(),
                    zp(),
                    scale(),
                    zp(),
                    scale(),
                    zp(),
                    scale(),
                    zp(),
                    scale(),
                    zp(),
                    scale(),
                ],
                d_u8,
                _sfdp_extra_check_int8(aten.div.Tensor),
            ),
            (
                _sfdp_pattern_int8_3,
                _sfdp_replacement_int8_3,
                [
                    g_u8(),
                    g_u8(),
                    g_u8(),
                    c(),
                    zp(),
                    scale(),
                    zp(),
                    scale(),
                    zp(),
                    scale(),
                    zp(),
                    scale(),
                    zp(),
                    scale(),
                ],
                d_u8,
                _sfdp_extra_check_int8(aten.div.Tensor),
            ),
            (
                _sfdp_pattern_int8_3,
                _sfdp_replacement_int8_3,
                [
                    g_u8_bs1(),
                    g_u8_bs1(),
                    g_u8_bs1(),
                    c(),
                    zp(),
                    scale(),
                    zp(),
                    scale(),
                    zp(),
                    scale(),
                    zp(),
                    scale(),
                    zp(),
                    scale(),
                ],
                d_u8,
                _sfdp_extra_check_int8(aten.div.Tensor),
            ),
            (
                _sfdp_pattern_int8_4,
                _sfdp_replacement_int8_4,
                [
                    g_u8(),
                    g_u8(),
                    g_u8(),
                    c(),
                    zp(),
                    scale(),
                    zp(),
                    scale(),
                    zp(),
                    scale(),
                    zp(),
                    scale(),
                    zp(),
                    scale(),
                ],
                d_u8,
                _sfdp_extra_check_int8(aten.div.Tensor),
            ),
            (
                _sfdp_pattern_int8_4,
                _sfdp_replacement_int8_4,
                [
                    g_u8_bs1(),
                    g_u8_bs1(),
                    g_u8_bs1(),
                    c(),
                    zp(),
                    scale(),
                    zp(),
                    scale(),
                    zp(),
                    scale(),
                    zp(),
                    scale(),
                    zp(),
                    scale(),
                ],
                d_u8,
                _sfdp_extra_check_int8(aten.div.Tensor),
            ),
        ]
    for pattern, replacement, args, workaround, extra_check in int8_candidates:
        # XXX: when adding a new pattern, re-run `gen_attention_patterns` so the pattern
        # gets serialized to a python file and does not require tracing at runtime.
        assert isinstance(workaround, dict)
        name = pattern.__name__

        if len(workaround) >= 1:
            # if "dropout_p" in workaround:
            #     # functools.partial insufficient because we look at signature downstream
            #     pattern = partialize_and_update_signature(pattern, dropout_p=0.0)
            #     replacement = partialize_and_update_signature(
            #         replacement, dropout_p=0.0
            #     )
            #     workaround = {}
            # else:
            # for uint8 pattern with more workarounds other than dropout,
            # we need to rename it to avoid influcing other patterns
            pattern = partialize_and_update_signature(pattern, dropout=0.0)
            replacement = partialize_and_update_signature(
                replacement, dropout=0.0
            )
            if "dropout" in workaround:
                del workaround["dropout"] 

        inference_name = name + "_inference"
        yield inference_name, {
            "search_fn": pattern,
            "replace_fn": replacement,
            "example_inputs": args,
            "trace_fn": fwd_only,
            "pass_dicts": patterns,
            "extra_check": extra_check,
            "scalar_workaround": workaround,
        }


@functools.lru_cache(None)
def _sfdp_init_int8():
    for key, register_replacement_kwargs in _gen_sfdp_patterns_int8():
        register_replacement(**register_replacement_kwargs)
    config.joint_custom_pre_pass = patterns.apply
    # print("\n\njoint_custom_pre_pass:", config.joint_custom_pre_pass)
