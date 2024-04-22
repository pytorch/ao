import torch
from termcolor import colored

import pandas as pd
from hqq.core.quantize import HQQLinear, BaseQuantizeConfig
from torchao.prototype.hqq.hqq_tinygemm_linear import HQQLinearTorchWeightOnlyInt4
from torchao.prototype.hqq import triton_mixed_mm, pack_2xint4

from triton.testing import do_bench


BASE_QUANT_CONFIG = {
    "optimize": True,
    "view_as_float": False,
    "nbits": 4,
    "bitpack": False,
    "axis": 1,
}


def bench_custom_kernel(x, W_q, scales, zeros, group_size, kernel_type="max_autotune", fp8_fast_accum=False):
    packed_w = pack_2xint4(W_q.T)

    def fn():
        _ = triton_mixed_mm(
            x,
            packed_w,
            scales.T,
            zeros.T,
            group_size=group_size,
            fp8_fast_accum=fp8_fast_accum,
            kernel_type=kernel_type,
        )

    t = do_bench(fn)
    return t


def bench_hqq(x, hqq_linear: HQQLinear):
    def fn():
        _ = hqq_linear.forward(x)

    t = do_bench(fn)
    return t


def run_benchmark(shape, group_size, dtype, axis=1, quant_dtype=torch.uint8):
    qcfg = {
        **BASE_QUANT_CONFIG,
        **dict(group_size=group_size, axis=axis),
    }
    M, N, K = shape

    x = torch.randn(M, K, dtype=dtype, device="cuda")
    linear = torch.nn.Linear(K, N, bias=False, dtype=dtype, device="cuda")

    quant_config = BaseQuantizeConfig(
        quant_zero=False, quant_scale=False, offload_meta=False, view_as_float=False
    )
    quant_config.update({"weight_quant_params": qcfg})

    hqq_linear = HQQLinear(linear, quant_config, compute_dtype=dtype, del_orig=False)

    # Reference
    ref_time = bench_hqq(x, hqq_linear)

    # Custom kernel
    W_q, meta = hqq_linear.W_q, hqq_linear.meta
    scales, zeros = meta["scale"], meta["zero"]

    W_q = (
        W_q.reshape(meta["shape"])
        if quant_config["weight_quant_params"]["bitpack"] == False
        else W_q
    )
    W_q = W_q.to(dtype=quant_dtype)
    scales = scales.reshape(N, -1)
    zeros = zeros.reshape(N, -1)
    tt_time = bench_custom_kernel(x, W_q, scales, zeros, group_size)

    if dtype == torch.bfloat16:
        _ = quant_config["weight_quant_params"].pop("bitpack")
        hqq_int4mm = HQQLinearTorchWeightOnlyInt4(
            linear, quant_config, compute_dtype=dtype, del_orig=False
        )
        int4_time = bench_hqq(x, hqq_int4mm)

    print(colored(f"{shape=} {group_size=} {dtype=}:", attrs=["bold"]))

    print(
        colored(f"Ref: {ref_time:.4f}", "blue"),
        colored(f"Triton: {tt_time:.4f}", "green"),
        colored(f"Torch int4mm: {int4_time:.4f}", "yellow")
        if dtype == torch.bfloat16
        else "",
    )
    print()
    return ref_time, tt_time, int4_time if dtype == torch.bfloat16 else None


SHAPES = [
    [16, 4096, 4096],
    [32, 4096, 4096],
    [128, 4096, 4096],
    [256, 4096, 4096],
    [512, 4096, 4096],
    [1024, 4096, 4096],
]

DTYPES = [torch.bfloat16]  # , torch.float16]
GROUP_SIZES = [128]

print(torch.cuda.get_device_properties(0))

HEADERS = [
    "M",
    "N",
    "K",
    "group_size",
    "dtype",
    "ref",
    "triton",
    "tinygemm",
]
data = []
for shape in SHAPES:
    for group_size in GROUP_SIZES:
        for dtype in DTYPES:
            timings = run_benchmark(shape, group_size, dtype)
            data.append((*shape, group_size, dtype, *timings))


df = pd.DataFrame(data, columns=HEADERS)
df.to_csv("benchmark_triton.csv", index=False)
