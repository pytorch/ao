import torch
from bitsandbytes.nn import Linear4bit
from hqq.core.quantize import BaseQuantizeConfig, HQQLinear
from prototypes.dora.dora_layer import BNBDoRALinear, HQQDoRALinear
from prototypes.dora.kernels.matmul import triton_mm
from prototypes.dora.kernels.smallk import triton_mm_small_k


def make_lora_weights(ranks, in_features, out_features, dtype):
    As = [torch.randn(rank, in_features, device="cuda", dtype=dtype) for rank in ranks]
    Bs = [torch.randn(out_features, rank, device="cuda", dtype=dtype) for rank in ranks]
    return As, Bs


def make_dora_source_and_magnitude(in_features, out_features, dtype):
    source = torch.randn(out_features, in_features, device="cuda", dtype=dtype)
    magnitude = torch.randn(out_features, device="cuda", dtype=dtype)
    return source, magnitude


def make_inputs(batch_sizes, seqlen, in_features, dtype):
    xs = [
        torch.randn(bs * seqlen, in_features, device="cuda", dtype=dtype)
        for bs in batch_sizes
    ]
    return xs


def make_weights(batch_sizes, in_features, out_features, dtype):
    weights = [
        torch.randn(in_features, out_features, device="cuda", dtype=dtype)
        for _ in range(len(batch_sizes))
    ]
    return weights


def make_epilogue_sources(batch_sizes, seqlen, out_features, dtype):
    epilogue_sources = [
        torch.randn(bs * seqlen, out_features, device="cuda", dtype=dtype)
        for bs in batch_sizes
    ]
    return epilogue_sources


def make_epilogue_scales(batch_sizes, out_features, dtype):
    epilogue_scales = [
        torch.randn(out_features, device="cuda", dtype=dtype)
        for _ in range(len(batch_sizes))
    ]
    return epilogue_scales


def dora_colnorm_ref(
    A: torch.Tensor,
    B: torch.Tensor,
    base_weight: torch.Tensor,
    magnitude_vector: torch.Tensor,
):
    column_norm = (base_weight + B @ A).norm(p=2, dim=1)
    return magnitude_vector / column_norm


def dora_mm_epilogue_ref(
    A: torch.Tensor,
    B: torch.Tensor,
    epilogue_source: torch.Tensor,
    epilogue_scale: torch.Tensor,
):
    out = (A @ B + epilogue_source) * epilogue_scale[None, :]
    return out


def dora_ref(x, w, lora_A, lora_B, magnitude_vector):
    # (bs x seq_len x out_features) = (bs x seq_len x in_features) @ (in_features x rank) @ (rank x out_features)
    lora_out = (x @ lora_A.T) @ lora_B.T
    # (out_features)
    magnitude_scale = dora_colnorm_ref(lora_A, lora_B, w, magnitude_vector)
    # (bs x seq_len x out_features)
    dora_out_ref = dora_mm_epilogue_ref(x, w, lora_out, magnitude_scale)
    return dora_out_ref


def dora_triton(x, w, lora_A, lora_B, magnitude_vector):
    lora_out = (x @ lora_A.T) @ lora_B.T
    magnitude_scale = triton_mm_small_k(
        lora_B,
        lora_A,
        epilogue_norm=True,
        source=w,
        magnitude=magnitude_vector,
        store_acc=False,
    )
    dora_out = triton_mm(x, w, epilogue_source=lora_out, epilogue_scale=magnitude_scale)
    return dora_out


def setup_dora_base_layers(layer_type, in_features, out_features, dtype):
    if "bnb" in layer_type:
        # BitsandBytes
        base_layer = Linear4bit(
            input_features=in_features,
            output_features=out_features,
            bias=False,
            quant_type="nf4",
            compute_dtype=dtype,
        ).cuda()
        base_layer.quant_state.dtype = base_layer.compute_dtype
        dora_cls = BNBDoRALinear
    elif "hqq" in layer_type:
        # HQQ
        quant_config = BaseQuantizeConfig(
            nbits=4,
            group_size=64,
            quant_zero=False,
            quant_scale=False,
            offload_meta=True,
            view_as_float=True,
        )
        linear = torch.nn.Linear(
            in_features, out_features, dtype=dtype, bias=False
        ).cuda()
        base_layer = HQQLinear(
            linear,
            quant_config,
            compute_dtype=dtype,
        )
        dora_cls = HQQDoRALinear
    else:
        raise ValueError(f"Unknown layer type: {layer_type}")
    return base_layer, dora_cls
