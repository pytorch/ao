import torch
from torch import nn
from torchao.prototype.fp6_llm.fp6_llm import Fp6LlmLinear, from_tc_float6_e3m2
from torch.utils.benchmark import Timer
import pandas as pd
from tqdm import tqdm


def benchmark(m: int, k: int, n: int):
    fp6_weight = torch.randint(256, size=(n, k // 4 * 3), dtype=torch.uint8, device="cuda")
    scales = torch.rand(n, dtype=torch.half, device="cuda") + 0.5
    fp6_linear = Fp6LlmLinear(fp6_weight.view(torch.int32), scales)

    fp16_linear = nn.Linear(k, n, bias=True, dtype=torch.half, device="cuda")
    fp16_linear.weight.data = from_tc_float6_e3m2(fp6_weight.view(-1), n, k, dtype=torch.half) * scales[:, None]

    fp16_act = torch.randn(m, k, dtype=torch.half, device="cuda")
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
    n_vals = (8192, 10240, 57344, 8192)

    results = []

    for m in tqdm([1 << i for i in range(10)]):
        for n, k in zip(n_vals, k_vals):
            results.append(benchmark(m, n, k))

    df = pd.DataFrame(results)
    df.to_csv("fp6_llm_benchmark_results.csv", index=False)
    print(df.to_markdown(index=False))
