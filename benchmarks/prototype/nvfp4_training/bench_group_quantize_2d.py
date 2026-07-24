# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD 3-Clause license found in the
# LICENSE file in the root directory of this source tree.

from dataclasses import dataclass

import torch
import triton
from tabulate import tabulate
from tqdm import tqdm

from benchmarks.prototype.nvfp4_training.deepseek_v3_shapes import (
    get_deepseek_v3_weight_shapes,
)
from benchmarks.utils import benchmark_cuda_function_in_microseconds
from torchao.prototype.moe_training.nvfp4_training.group_quantize_2d_triton import (
    BLOCK_M,
    BLOCK_N,
    _group_weight_quantize_2d_kernel,
)
from torchao.utils import is_sm_at_least_100


@dataclass(frozen=True)
class ExperimentConfig:
    experts: int
    m: int
    n: int
    model: str = ""
    projection: str = ""


@dataclass(frozen=True)
class ExperimentResult:
    time_us: float
    gbps: float


def run_experiment(config: ExperimentConfig) -> ExperimentResult:
    E, M, N = config.experts, config.m, config.n
    weights = torch.randn((E, M, N), dtype=torch.bfloat16, device="cuda")
    global_amax = weights.float().abs().amax(dim=(1, 2))

    qa = torch.empty((E, M, N // 2), dtype=torch.uint8, device="cuda")
    sfa = torch.empty(
        (E, M // 128, N // 64, 32, 16),
        dtype=torch.float8_e4m3fn,
        device="cuda",
    )
    qa_t = torch.empty((E, N, M // 2), dtype=torch.uint8, device="cuda")
    sfa_t = torch.empty(
        (E, N // 128, M // 64, 32, 16),
        dtype=torch.float8_e4m3fn,
        device="cuda",
    )

    grid = (triton.cdiv(M, BLOCK_M), triton.cdiv(N, BLOCK_N), E)

    def run_kernel():
        _group_weight_quantize_2d_kernel[grid](
            weights,
            global_amax,
            qa,
            sfa,
            qa_t,
            sfa_t,
            M,
            N,
            BLOCK_M=BLOCK_M,
            BLOCK_N=BLOCK_N,
        )

    time_us = benchmark_cuda_function_in_microseconds(run_kernel)

    elements = E * M * N
    read_bytes = elements * 2
    write_fp4_bytes = elements
    write_scale_bytes = 2 * elements // 16
    gbps = ((read_bytes + write_fp4_bytes + write_scale_bytes) / 1e9) / (time_us / 1e6)
    return ExperimentResult(time_us=time_us, gbps=gbps)


def main() -> None:
    if not torch.cuda.is_available() or not is_sm_at_least_100():
        raise RuntimeError("Grouped NVFP4 2D quantization requires SM100+")

    configs = [
        ExperimentConfig(
            shape.experts,
            shape.m,
            shape.n,
            model=shape.model,
            projection=shape.projection,
        )
        for shape in get_deepseek_v3_weight_shapes()
    ]
    rows = []
    for config in tqdm(configs):
        result = run_experiment(config)
        rows.append(
            [
                config.model,
                config.projection,
                config.experts,
                config.m,
                config.n,
                round(result.time_us, 3),
                round(result.gbps, 3),
            ]
        )
    print(
        tabulate(
            rows,
            headers=["model", "projection", "E", "M", "N", "time_us", "gbps"],
        )
    )


if __name__ == "__main__":
    main()
