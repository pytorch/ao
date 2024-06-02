import torch
from torch import nn
from torchao.quantization.fp6_llm import Fp6LlmLinear
from torch.utils.benchmark import Timer
import pandas as pd
from tqdm import tqdm


def benchmark(m: int, k: int, n: int):
    fp16_act = torch.randn(m, k, device="cuda", dtype=torch.half)
    fp16_linear = nn.Linear(k, n, bias=False, device="cuda", dtype=torch.half)
    fp6_linear = Fp6LlmLinear.from_float(fp16_linear)

    fp6_output = fp6_linear(fp16_act)
    fp16_output = fp16_linear(fp16_act)

    fp6_measurement = Timer(stmt="fp6_linear(fp16_act)", globals=locals()).blocked_autorange()
    fp16_measurement = Timer(stmt="fp16_linear(fp16_act)", globals=locals()).blocked_autorange()

    # follow https://github.com/usyd-fsalab/fp6_llm/blob/ce76774bcfc26b325c1b558abcf1935026d9abbc/tests/python/kernel_test.py
    # doesn't seem to be the right way to check for correctness
    correct = (fp6_output - fp16_output).abs().mean() / fp16_output.abs().mean() < 1e-3

    return {
        "m": m,
        "k": k,
        "n": n,
        "fp6_latency (ms)": fp6_measurement.median * 1000,
        "fp16_latency (ms)": fp16_measurement.median * 1000,
        "speedup (d/s)": fp16_measurement.median / fp6_measurement.median,
        "correct": correct,
    }


if __name__ == "__main__":
    # from https://github.com/usyd-fsalab/fp6_llm/blob/ce76774bcfc26b325c1b558abcf1935026d9abbc/tests/python/run.sh
    k_vals = (8192, 8192, 8192, 28672)
    n_vals = (10240, 8192, 57344, 8192)

    results = []

    for m in tqdm([1 << i for i in range(10)]):
        for n, k in zip(n_vals, k_vals):
            results.append(benchmark(m, n, k))

    df = pd.DataFrame(results)
    df.to_csv("fp6_benchmark_results.csv", index=False)
    print(df.to_markdown(index=False))
