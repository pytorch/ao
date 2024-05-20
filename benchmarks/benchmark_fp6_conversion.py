from functools import partial

import torch
import torchao
import pandas as pd
from torch.utils.benchmark import Timer


def benchmark(f, weight):
    measurement = Timer(
        stmt="f(weight)",
        globals={"f": f, "weight": weight},
    ).blocked_autorange()
    return measurement.median * 1000


if __name__ == "__main__":
    M = 8192
    N = 8192

    fp32_weight = torch.randn(M, N)
    fp32_weight_cuda = fp32_weight.cuda()
    fp16_weight = fp32_weight.half()
    fp16_weight_cuda = fp16_weight.cuda()

    functions = [
        ("original (FP6 packed)", torchao.ops.fp16_to_fp6_original),
        # ("custom C++/CUDA (FP6 unpacked)", torchao.ops.to_fp6_unpacked),
        ("custom C++/CUDA (FP6 packed)", torchao.ops.to_fp6_packed),
        # ("PyTorch + torch.compile (FP6 unpacked)", partial(torch.compile(torchao.ops.to_fp6_pt), unpacked=True)),
        ("PyTorch + torch.compile (FP6 packed)", partial(torch.compile(torchao.ops.to_fp6_pt), unpacked=False)),
    ]

    results = []
    for name, f in functions:
        results.append([name, "CPU", "FP32->FP6", benchmark(f, fp32_weight)])
        results.append([name, "CPU", "FP16->FP6", benchmark(f, fp16_weight)])
        if name != "original (FP6 packed)":
            results.append([name, "CUDA", "FP32->FP6", benchmark(f, fp32_weight_cuda)])
            results.append([name, "CUDA", "FP16->FP6", benchmark(f, fp16_weight_cuda)])

    df = pd.DataFrame(results, columns=["op", "device", "dtype", "time (m/s)"])
    df["op"] = df["op"].str.removesuffix(" (FP6 packed)")
    print(df.to_markdown(index=False))
