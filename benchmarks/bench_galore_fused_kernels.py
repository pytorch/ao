# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD 3-Clause license found in the
# LICENSE file in the root directory of this source tree.
import argparse
import os

import torch
from fused_benchmark_utils import get_benchmark  # , make_data


def run(args):
    dtype = getattr(torch, args.dtype)
    allow_tf32 = args.allow_tf32
    torch.backends.cuda.matmul.allow_tf32 = allow_tf32
    M, N = args.M, args.N
    rank = args.rank

    # exp_avg, exp_avg2, grad, proj_matrix, params = make_data(M, N, rank, dtype)

    benchmark = get_benchmark(M, N, dtype, allow_tf32=allow_tf32)
    save_path = (
        f'benchmark_{M}x{N}_{rank}_{args.dtype}_{"tf32" if allow_tf32 else "no-tf32"}'
    )
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    print(
        f"Running benchmark for {M}x{N}, dtype {args.dtype}, allow_tf32 {allow_tf32}",
        flush=True,
    )
    benchmark.run(show_plots=False, print_data=True, save_path=save_path)
    print(f"Finished benchmark, results saved to {save_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        "--kernel",
        choices=["hybrid", "fused", "compiled"],
        default="hybrid",
        type=str,
        help="Kernel to test",
    )

    parser.add_argument(
        "--allow_tf32", action="store_true", help="Allow tf32 for matmuls"
    )
    parser.add_argument("--M", type=int, default=4096, help="Grad (param) shape M")
    parser.add_argument("--N", type=int, default=4096, help="Grad (param) shape N")
    parser.add_argument(
        "--rank", type=int, default=128, help="Rank of GaLore projection"
    )
    parser.add_argument(
        "--dtype",
        type=str,
        choices=["float32", "float16", "bfloat16"],
        default="float32",
        help="Data type of grad (param) tensors",
    )

    args = parser.parse_args()
    run(args)
