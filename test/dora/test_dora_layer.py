import sys

import pytest

if sys.version_info < (3, 11):
    pytest.skip("requires Python >= 3.11", allow_module_level=True)

bnbnn = pytest.importorskip("bitsandbytes.nn", reason="requires bitsandbytes")
hqq_core = pytest.importorskip("hqq.core.quantize", reason="requires hqq")

import itertools

import torch

# Import modules as opposed to classes directly, otherwise pytest.importorskip always skips
Linear4bit = bnbnn.Linear4bit
BaseQuantizeConfig = hqq_core.BaseQuantizeConfig
HQQLinear = hqq_core.HQQLinear
from torchao.prototype.dora.dora_layer import BNBDoRALinear, DoRALinear, HQQDoRALinear


def check(expected, actual, dtype):
    if dtype not in [torch.float32, torch.float16, torch.bfloat16]:
        raise ValueError(f"Unsupported dtype: {dtype}")
    diff = (expected - actual).abs().max()
    print(f"diff: {diff}")
    # assert diff < atol
    return diff


def _arg_to_id(arg):
    if isinstance(arg, (tuple, list)):
        return "x".join([str(x) for x in arg])
    return str(arg)


BATCH_SIZES = [1]
SEQ_LENS = [512]
DTYPES = [torch.float32, torch.float16, torch.bfloat16]
IN_FEATURES = [4096]
OUT_FEATURES = [4096, 11008]
LORA_RANKS = [16]
MODEL_TYPES = ["DoRALinear", "BNBDoRALinear", "HQQDoRALinear"]

TEST_CONFIGS = list(
    itertools.product(
        BATCH_SIZES,
        SEQ_LENS,
        IN_FEATURES,
        OUT_FEATURES,
        LORA_RANKS,
        DTYPES,
        MODEL_TYPES,
    )
)


@pytest.mark.parametrize(
    "bs, seqlen, in_features, out_features, lora_rank, dtype, model_type",
    TEST_CONFIGS,
    ids=_arg_to_id,
)
def test_dora_layer(
    bs, seqlen, in_features, out_features, lora_rank, dtype, model_type
):
    x = torch.randn(bs, seqlen, in_features, dtype=dtype).cuda()

    if model_type == "DoRALinear":
        base_layer = torch.nn.Linear(
            in_features, out_features, dtype=dtype, bias=False
        ).cuda()
        dora_cls = DoRALinear

    elif model_type == "BNBDoRALinear":
        base_layer = Linear4bit(
            input_features=in_features,
            output_features=out_features,
            bias=False,
            quant_type="nf4",
            compute_dtype=dtype,
        ).cuda()
        base_layer.quant_state.dtype = base_layer.compute_dtype
        dora_cls = BNBDoRALinear

    elif model_type == "HQQDoRALinear":
        quant_config = BaseQuantizeConfig(
            nbits=4,
            group_size=64,
            quant_zero=False,
            quant_scale=False,
            offload_meta=True,
            view_as_float=True,
        )
        torch_base = torch.nn.Linear(in_features, out_features, dtype=dtype, bias=False)
        base_layer = HQQLinear(
            torch_base,
            quant_config,
            compute_dtype=dtype,
        )
        dora_cls = HQQDoRALinear
    dora_layer = dora_cls(base_layer, lora_rank).cuda()

    ref = dora_layer.forward(x)
    test = dora_layer.forward_fused(x)
    check(ref, test, dtype)
