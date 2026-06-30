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
from torchao.prototype.moe_training.nvfp4_training.group_hadamard_amax_triton import (
    _group_rht_amax_triton_kernel,
)
from torchao.prototype.moe_training.nvfp4_training.group_hadamard_utils import (
    BLOCK_M,
    BLOCK_N,
    SAME_BOTH_DIMS,
)
from torchao.prototype.moe_training.nvfp4_training.hadamard_utils import (
    get_rht_matrix,
)
from torchao.utils import is_sm_at_least_100

RHT_SIGN_VECTOR = (
    1,
    1,
    1,
    -1,
    1,
    -1,
    -1,
    -1,
    -1,
    -1,
    -1,
    1,
    -1,
    1,
    -1,
    -1,
)


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
    total_m = E * M
    weights = torch.randn((total_m, N), dtype=torch.bfloat16, device="cuda")
    rht = get_rht_matrix(RHT_SIGN_VECTOR, weights.device, torch.bfloat16, 16)
    offsets = torch.arange(E + 1, dtype=torch.int64, device="cuda") * M * N
    row_amax = torch.zeros((E,), dtype=torch.float32, device="cuda")
    col_amax = torch.zeros((E,), dtype=torch.float32, device="cuda")

    grid = (triton.cdiv(total_m, BLOCK_M) * triton.cdiv(N, BLOCK_N),)

    def run_kernel():
        _group_rht_amax_triton_kernel[grid](
            weights,
            rht,
            offsets,
            row_amax,
            col_amax,
            total_m,
            N,
            num_tensors=E,
            SHAPE_REP=SAME_BOTH_DIMS,
            BLOCK_M=BLOCK_M,
            BLOCK_N=BLOCK_N,
            RHT_SIZE=16,
            num_warps=8,
            num_stages=3,
        )

    time_us = benchmark_cuda_function_in_microseconds(run_kernel)
    total_bytes = weights.numel() * weights.element_size() + 2 * E * 4
    gbps = (total_bytes / 1e9) / (time_us / 1e6)
    return ExperimentResult(time_us=time_us, gbps=gbps)


def main() -> None:
    if not torch.cuda.is_available() or not is_sm_at_least_100():
        raise RuntimeError("Grouped NVFP4 amax requires SM100+")

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
