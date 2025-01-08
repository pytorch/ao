import pandas as pd
import torch
import torch.nn.functional as F
from tqdm import tqdm

from torchao.dtypes import to_affine_quantized_fpx
from torchao.dtypes.floatx import FloatxTensorCoreLayout
from torchao.utils import benchmark_torch_function_in_microseconds


def benchmark(m: int, k: int, n: int):
    float_data_fp16 = torch.randn(n, k, dtype=torch.float16, device="cuda")
    float_data_bf16 = torch.randn(n, k, dtype=torch.bfloat16, device="cuda")
    fp6_weight_fp16 = to_affine_quantized_fpx(
        float_data_fp16, FloatxTensorCoreLayout(3, 2)
    )
    fp6_weight_bf16 = to_affine_quantized_fpx(
        float_data_bf16, FloatxTensorCoreLayout(3, 2)
    )
    fp16_weight = fp6_weight_fp16.dequantize(torch.float16)
    bf16_weight = fp6_weight_bf16.dequantize(torch.bfloat16)

    fp16_act = torch.randn(m, k, dtype=torch.float16, device="cuda")
    bf16_act = fp16_act.to(torch.bfloat16)
    fp6_output_fp16 = F.linear(fp16_act, fp6_weight_fp16)
    fp6_output_bf16 = F.linear(bf16_act, fp6_weight_bf16)
    fp16_output = F.linear(fp16_act, fp16_weight)
    bf16_output = F.linear(bf16_act, bf16_weight)

    fp16_time = benchmark_torch_function_in_microseconds(
        F.linear, fp16_act, fp16_weight
    )
    bf16_time = benchmark_torch_function_in_microseconds(
        F.linear, bf16_act, bf16_weight
    )
    fp6_time_fp16 = benchmark_torch_function_in_microseconds(
        F.linear, fp16_act, fp6_weight_fp16
    )
    fp6_time_bf16 = benchmark_torch_function_in_microseconds(
        F.linear, bf16_act, fp6_weight_bf16
    )

    # follow https://github.com/usyd-fsalab/fp6_llm/blob/ce76774bcfc26b325c1b558abcf1935026d9abbc/tests/python/kernel_test.py
    # doesn't seem to be the right way to check for correctness
    correct_fp16 = (
        fp6_output_fp16 - fp16_output
    ).abs().mean() / fp16_output.abs().mean() < 1e-3
    correct_bf16 = (
        fp6_output_bf16 - bf16_output
    ).abs().mean() / bf16_output.abs().mean() < 1e-2

    return {
        "m": m,
        "k": k,
        "n": n,
        "fp6-fp16 latency (ms)": fp6_time_fp16,
        "fp16 latency (ms)": fp16_time,
        "speedup fp16": fp16_time / fp6_time_fp16,
        "correct fp16": correct_fp16,
        "fp6-bf16 latency (ms)": fp6_time_bf16,
        "bf16 latency (ms)": bf16_time,
        "speedup bf16": bf16_time / fp6_time_bf16,
        "correct bf16": correct_bf16,
    }


if __name__ == "__main__":
    # from https://github.com/usyd-fsalab/fp6_llm/blob/ce76774bcfc26b325c1b558abcf1935026d9abbc/tests/python/run.sh
    k_vals = (8192, 8192, 8192, 28672)
    n_vals = (8192, 10240, 57344, 8192)

    results = []

    for m in tqdm([1 << i for i in range(10)]):
        for n, k in zip(n_vals, k_vals):
            results.append(benchmark(m, k, n))

    df = pd.DataFrame(results)
    df.to_csv("fp6_llm_benchmark_results.csv", index=False)
    print(df.to_markdown(index=False))
