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
        ("original", torchao.ops.fp16_to_fp6_original),
        ("C++/CUDA extension", torchao.ops.to_fp6_packed),
        ("PyTorch + torch.compile (default)", torch.compile(torchao.ops.to_fp6_pt)),
        ("PyTorch + torch.compile (max-autotune)", torch.compile(torchao.ops.to_fp6_pt, mode="max-autotune")),

        # ("C++/CUDA extension (no bit-packing)", torchao.ops.to_fp6_unpacked),
        # ("PyTorch + torch.compile (no bit-packing)", partial(torch.compile(torchao.ops.to_fp6_pt), unpacked=True)),
    ]

    results = []
    for name, f in functions:
        results.append(["CPU", "FP32->FP6", name, benchmark(f, fp32_weight)])
        results.append(["CPU", "FP16->FP6", name, benchmark(f, fp16_weight)])

        if name != "original":
            results.append(["CUDA", "FP32->FP6", name, benchmark(f, fp32_weight_cuda)])
            results.append(["CUDA", "FP16->FP6", name, benchmark(f, fp16_weight_cuda)])

    df = pd.DataFrame(results, columns=["device", "dtype", "op", "time (m/s)"])
    df = df.sort_values(["device", "dtype"])
    print(df.to_markdown(index=False))
