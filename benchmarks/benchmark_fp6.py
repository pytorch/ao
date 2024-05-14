import torch
import torchao
from torch.utils.benchmark import Timer
import pandas as pd
from tqdm import tqdm


def benchmark(m, k, n, splitK):
    # Randomly initialize each bytes. The highest value for randint() is set the the max value of uint32_t.
    fp6_weight = torch.randint(4294967295, (n, k // 16 * 3)).to(torch.int)
    fp16_scale = torch.rand(n).half() + 0.5
    fp16_activation = torch.rand(m, k).half() + 0.5

    fp6_weight_packed = torchao.ops.prepack_fp6_weight(fp6_weight)
    act_cuda = fp16_activation.cuda()
    weight_cuda = fp6_weight_packed.cuda()
    scale_cuda = fp16_scale.cuda()

    # need to do this since Timer cannot see torchao
    def fp6_linear(act_cuda, weight_cuda, scale_cuda, splitK):
        return torchao.ops.fp16act_fp6weight_linear(act_cuda, weight_cuda, scale_cuda, splitK)

    fp6_output = fp6_linear(act_cuda, weight_cuda, scale_cuda, splitK)

    fp6_measurement = Timer(
        stmt="fp6_linear(act_cuda, weight_cuda, scale_cuda, splitK)",
        globals=locals(),
    ).blocked_autorange()

    fp16_weight = torchao.ops.fp6_weight_dequant(fp6_weight, fp16_scale).cuda()
    fp16_output = act_cuda @ fp16_weight.T

    fp16_measurement = Timer(
        stmt="act_cuda @ fp16_weight.T",
        globals=locals(),
    ).blocked_autorange()

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

    # splitK can be tuned based on m, k, n
    for m, splitK_vals in tqdm([
        (1, (5, 6, 7, 6)),
        (2, (5, 6, 7, 6)),
        (4, (5, 6, 7, 6)),
        (8, (5, 6, 7, 6)),
        # (16, (5, 6, 7, 6)),
        # (64, (5, 6, 7, 6)),
        # (128, (5, 3, 3, 3)),
        # (256, (4, 3, 2, 3)),
        # (512, (2, 5, 2, 4)),
        (1024, (1, 2, 1, 2)),
        (2048, (1, 1, 1, 1)),
        (4096, (1, 1, 1, 1)),
        # (8192, (1, 1, 1, 1)),
        # (16384, (1, 1, 1, 1)),
    ]):
        for n, k, splitK in zip(n_vals, k_vals, splitK_vals):
            results.append(benchmark(m, n, k, splitK))

    df = pd.DataFrame(results)
    df.to_csv("fp6_benchmark_results.csv", index=False)
    print(df.to_markdown(index=False))
