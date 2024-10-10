import torch
import pandas as pd
import torch.nn.functional as F
from torchao.dtypes import to_affine_quantized_fpx
from torchao.dtypes.floatx import FloatxTensorCoreAQTTensorImpl, FloatxTensorCoreLayout
from torchao.utils import benchmark_torch_function_in_microseconds
from tqdm import tqdm


def benchmark(m: int, k: int, n: int, use_bf16=False):
    dtype = torch.bfloat16 if use_bf16 else torch.half
    float_data = torch.randn(n, k, dtype=dtype, device="cuda")
    fp6_weight = to_affine_quantized_fpx(float_data, FloatxTensorCoreLayout(3, 2))
    fp16_weight = fp6_weight.dequantize(dtype)

    fp16_act = torch.randn(m, k, dtype=dtype, device="cuda")
    fp6_output = F.linear(fp16_act, fp6_weight)
    fp16_output = F.linear(fp16_act, fp16_weight)

    fp6_time = benchmark_torch_function_in_microseconds(F.linear, fp16_act, fp6_weight)
    fp16_time = benchmark_torch_function_in_microseconds(F.linear, fp16_act, fp16_weight)

    # follow https://github.com/usyd-fsalab/fp6_llm/blob/ce76774bcfc26b325c1b558abcf1935026d9abbc/tests/python/kernel_test.py
    # doesn't seem to be the right way to check for correctness
    correct = (fp6_output - fp16_output).abs().mean() / fp16_output.abs().mean() < 1e-3

    return {
        "m": m,
        "k": k,
        "n": n,
        "fp6_latency (ms)": fp6_time,
        f"{'bf16' if use_bf16 else 'fp16'}_latency (ms)": fp16_time,
        "speedup (d/s)": fp16_time / fp6_time,
        "correct": correct,
    }


if __name__ == "__main__":
    # from https://github.com/usyd-fsalab/fp6_llm/blob/ce76774bcfc26b325c1b558abcf1935026d9abbc/tests/python/run.sh
    k_vals = (8192, 8192, 8192, 28672)
    n_vals = (8192, 10240, 57344, 8192)

    use_bf16 = True

    results = []

    for m in tqdm([1 << i for i in range(10)]):
        for n, k in zip(n_vals, k_vals):
            results.append(benchmark(m, k, n, use_bf16=use_bf16))

    df = pd.DataFrame(results)
    df.to_csv("fp6_llm_benchmark_results.csv", index=False)
    print(df.to_markdown(index=False))
