import argparse
import random

import pandas as pd
import torch
import torch.utils.benchmark as benchmark
import torch.nn.functional as F
from torch import nn
from torch.sparse import SparseSemiStructuredTensor, to_sparse_semi_structured

from torch.sparse._triton_ops_meta import optimize_bsr_dense_addmm
from torchao.utils import benchmark_model
from torchao.sparsity.utils import create_semi_structured_tensor, create_block_sparse_tensor

torch.set_printoptions(
    precision=2,
    threshold=None,
    edgeitems=16,
    linewidth=480,
    profile=None,
    sci_mode=False,
)


def run_gpu_sparse_benchmark(m, k, n, args):
    dtype = getattr(torch, args.dtype)

    x = torch.randn(n, k).to(dtype).cuda()
    b = torch.randn(m, dtype=dtype).cuda()

    # handle sparsity types
    if args.sparsity == "semi-structured":
        SparseSemiStructuredTensor._FORCE_CUTLASS = args.backend == "cutlass"
        A = create_semi_structured_tensor(m, k, dtype)
        A_sparse = to_sparse_semi_structured(A)
    elif args.sparsity == "block-sparse":
        A = create_block_sparse_tensor(m, k, args.block_size, args.sparsity_level, dtype)
        A_sparse = A.to_sparse_bsr(blocksize=args.block_size)
        # BSR kernel tuning
        if args.bsr_autotune:
            print("Tuning kernel params")
            optimize_bsr_dense_addmm(m, k, n, args.block_size, args.block_size,
                                     dtype=dtype, sparsity=args.sparsity_level, verbose=True)
    else:
        raise ValueError(f"Unknown sparsity: {args.sparsity}")

    if args.eval_fn == "linear":
        dense_output = F.linear(x, A, b)
        sparse_output = F.linear(x, A_sparse, b)
        # warmup
        benchmark_model(F.linear, 10, args=(x, A, b), device_type="cuda")
        dense_time = benchmark_model(F.linear, 100, args=(x, A, b), device_type="cuda")

        benchmark_model(F.linear, 10, args=(x, A_sparse, b), device_type="cuda")
        sparse_time = benchmark_model(F.linear, 100, args=(x, A_sparse, b), device_type="cuda")
    elif args.eval_fn == "mm":
        dense_output = torch.mm(A, x.t())
        sparse_output = torch.mm(A_sparse, x.t())
        dense_time = benchmark_in_us(torch.mm, A, x.t())
        sparse_time = benchmark_in_us(torch.mm, A_sparse, x.t())
    else:
        raise ValueError(f"Unknown eval_fn: {args.eval_fn}")


    return {
        "test_function": args.eval_fn,
        "m": m,
        "k": k,
        "n": n,
        "dtype": args.dtype,
        "sparse_latency (ms)": sparse_time,
        "dense_latency (ms)": dense_time,
        "speedup (d/s)": dense_time / sparse_time,
        "contiguous": sparse_output.is_contiguous(),
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="GPU Sparsity Microbenchmarks")
    parser.add_argument(
        "--mode",
        type=str,
        choices=[
            "bert-large",
            "vit-mlp-shapes",
            "nvidia-fixed-k",
            "nvidia-fixed-mn",
        ],
    )
    parser.add_argument(
        "--sparsity",
        type=str,
        choices=[
            "semi-structured",
            "block-sparse",
        ],
    )
    parser.add_argument(
        "--sparsity-level",
        type=float,
    )
    parser.add_argument(
        "--block-size",
        type=int,
        choices=[
            16,
            32,
            64,
        ]
    )
    parser.add_argument(
        "--dtype",
        type=str,
        choices=[
            "int8",
            "float16",
            "bfloat16",
            "float32",
        ],
        default="bfloat16",
    )
    parser.add_argument(
        "--backend", type=str, choices=["cutlass", "cusparselt"], default="cusparselt"
    )
    parser.add_argument("--eval-fn", type=str, choices=["linear", "mm"], default="linear")
    parser.add_argument("-contiguous", action="store_true")
    parser.add_argument("-save", action="store_true")
    parser.add_argument("-bsr-autotune", action="store_true", help="Tune BSR kernel parameters")
    args = parser.parse_args()

    print(f"Started benchmark: {args}")

    if args.mode == "bert-large-shapes":
        bert_shapes = [
            (3072, 1024, 16384),
            (4096, 1024, 16384),
            (1024, 1024, 16384),
            (1024, 4096, 16384),
        ]
        results = (
            run_gpu_sparse_benchmark(m, k, n, args)
            for (m, k, n) in bert_shapes
        )
    elif args.mode == "vit-mlp-shapes":
        vit_shapes= [
            (768, 3072, 50432),
            (3072, 768, 50432),
            (1280, 5120, 65792),
            (5120, 1280, 65792),
        ]
        results = (
            run_gpu_sparse_benchmark(m, k, n, args)
            for (m, k, n) in vit_shapes
        )
    elif args.mode == "nvidia-fixed-k":
        mn_vals = [
            3072,
            4096,
            5120,
            6144,
            7168,
            8192,
            9216,
            10240,
            11264,
            12288,
            13312,
            14336,
            15360,
            16384,
            17408,
            18432,
            19456,
            20480,
        ]
        results = (
            run_gpu_sparse_benchmark(mn, 10240, mn, args)
            for mn in mn_vals
        )
    elif args.mode == "nvidia-fixed-mn":
        k_vals = [
            2560,
            3840,
            5120,
            6400,
            7680,
            8960,
            10240,
            11520,
            12800,
            14080,
            15360,
            16640,
            17920,
            19200,
            20480,
        ]
        results = (
            run_gpu_sparse_benchmark(10240, k, 10240, args)
            for k in k_vals
        )

    else:
        raise ValueError(f"Unknown mode: {args.mode}")


    df = pd.DataFrame.from_records(results)
    if args.save:
        save_file = f"{args.mode}_{args.dtype}_{args.backend}.csv"
        df.to_csv(save_file)
        print(f"Finished benchmark: {args.mode} saved results to {save_file}")
    print(df)
