import torch
import pandas as pd
import torchao
from torchao.dtypes.floatx import from_scaled_tc_floatx
from torchao.utils import benchmark_torch_function_in_microseconds
from tqdm import tqdm


def benchmark(m: int, k: int, n: int):
    ebits = 3
    mbits = 2
    nbits = 1 + ebits + mbits

    fp6_weight = torch.randint(256, (n, k // 8 * nbits), dtype=torch.uint8, device="cuda")
    scale = torch.rand(n, device="cuda").half() + 0.5
    fp16_act = torch.randn(m, k, dtype=torch.half, device="cuda") + 0.5

    fp6_output = torchao.ops.quant_llm_linear(ebits, mbits, fp16_act, fp6_weight, scale, splitK=1)

    fp16_weight = from_scaled_tc_floatx(fp6_weight, ebits, mbits, scale).half()
    fp16_output = torch.matmul(fp16_act, fp16_weight.T)

    fp6_time = benchmark_torch_function_in_microseconds(torchao.ops.quant_llm_linear, ebits, mbits, fp16_act, fp6_weight, scale, splitK=1)
    fp16_time = benchmark_torch_function_in_microseconds(torch.matmul, fp16_act, fp16_weight.T)

    # follow https://github.com/usyd-fsalab/fp6_llm/blob/ce76774bcfc26b325c1b558abcf1935026d9abbc/tests/python/kernel_test.py
    # doesn't seem to be the right way to check for correctness
    correct = (fp6_output - fp16_output).abs().mean() / fp16_output.abs().mean() < 1e-3

    return {
        "m": m,
        "k": k,
        "n": n,
        "fp6_latency (ms)": fp6_time,
        "fp16_latency (ms)": fp16_time,
        "speedup (d/s)": fp16_time / fp6_time,
        "correct": correct,
    }


if __name__ == "__main__":
    # from https://github.com/usyd-fsalab/fp6_llm/blob/ce76774bcfc26b325c1b558abcf1935026d9abbc/tests/python/run.sh
    k_vals = (8192, 8192, 8192, 28672)
    n_vals = (8192, 10240, 57344, 8192)

    results = []

    for m in tqdm([1 << i for i in range(10)]):
        for n, k in zip(n_vals, k_vals):
            results.append(benchmark(m, n, k))

    df = pd.DataFrame(results)
    df.to_csv("fp6_llm_benchmark_results.csv", index=False)
    print(df.to_markdown(index=False))
