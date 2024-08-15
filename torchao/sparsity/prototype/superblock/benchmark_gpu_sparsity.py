import argparse
import random

import pandas as pd
import torch
import torch.utils.benchmark as benchmark
import torch.nn.functional as F
from torch import nn
from torch.sparse import SparseSemiStructuredTensor, to_sparse_semi_structured
from tqdm import tqdm


torch.set_printoptions(
    precision=2,
    threshold=None,
    edgeitems=16,
    linewidth=480,
    profile=None,
    sci_mode=False,
)

def create_block_sparse_tensor(M, N, blocksize, sparsity, dtype):
    assert sparsity <= 1.0 and sparsity >= 0.0, \
        "sparsity should be a value between 0 and 1"
    A = torch.bernoulli(torch.full((M//blocksize, N//blocksize),
                        1 - sparsity, dtype=dtype))
    A = torch.repeat_interleave(A, blocksize, dim=0)
    A = torch.repeat_interleave(A, blocksize, dim=1)
    return A.to(dtype).contiguous().cuda()

def create_semi_structured_tensor(
    r, c, dtype
):
    """
    This function returns a 1:2 sparse matrix of size (r, c).
    Note that this means this matrix will also be 2:4 and 4:8 sparse as well.
    """

    choices = [[0, 1], [1, 0]]
    mask_entries = [random.choice(choices) for i in range(r * c // 2)]

    mask = (
        torch.tensor(mask_entries, dtype=dtype)
        .reshape(r, c)
        .contiguous()
    ).cuda()
    sparse_weight = torch.rand(r, c).to(dtype).cuda() * mask
    return sparse_weight


def benchmark_in_us(f, *args, **kwargs):
    t0 = benchmark.Timer(
        stmt="f(*args, **kwargs)",
        globals={"args": args, "kwargs": kwargs, "f": f}
    )
    return t0.blocked_autorange()


def run_gpu_sparse_benchmark(m, k, n, args):
    dtype = DTYPE_LOOKUP[args.dtype]

    x = torch.randn(n, k).to(dtype).cuda()
    b = torch.randn(m, dtype=dtype).cuda()


    if args.sparsity == "semi-structured":
        SparseSemiStructuredTensor._FORCE_CUTLASS = args.backend == "cutlass"
        A = create_semi_structured_tensor(m, k, dtype)
        A_sparse = to_sparse_semi_structured(A)

    elif args.sparsity == "block-sparse":
        A = create_block_sparse_tensor(m, k, args.block_size, args.sparsity_level, dtype)
        A_sparse = A.to_sparse_bsr(blocksize=args.block_size)
    else:
        raise ValueError(f"Unknown sparsity: {args.sparsity}")

    if args.eval_fn == "linear":
        dense_output = F.linear(x, A, b)
        sparse_output = F.linear(x, A_sparse, b)
        correct = torch.allclose(dense_output, sparse_output, rtol=1e-3, atol=1e-3)
        dense_time = benchmark_in_us(F.linear, x, A, b)
        sparse_time = benchmark_in_us(F.linear, x, A_sparse, b)

    elif args.eval_fn == "mm":
        dense_output = torch.mm(x, A)
        sparse_output = torch.mm(x, A_sparse)
        correct = torch.allclose(dense_output, sparse_output, rtol=1e-3, atol=1e-3)

        dense_time = benchmark_in_us(torch.mm, x, A)
        sparse_time = benchmark_in_us(torch.mm, x, A_sparse)
    else:
        raise ValueError(f"Unknown eval_fn: {args.eval_fn}")


    return {
        "test_function": "linear",
        "m": m,
        "k": k,
        "n": n,
        "dtype": args.dtype,
        # "backend": args.backend,
        "sparse_latency (ms)": sparse_time.median * 1000,
        "dense_latency (ms)": dense_time.median * 1000,
        "speedup (d/s)": dense_time.median / sparse_time.median,
        # "correct": correct,
        # "contiguous": sparse_output.is_contiguous(),
    }


if __name__ == "__main__":
    DTYPE_LOOKUP = {
        "int8": torch.int8,
        "fp16": torch.float16,
        "bf16": torch.bfloat16,
        "fp32": torch.float32,
    }

    parser = argparse.ArgumentParser(description="GPU Sparsity Microbenchmarks")
    parser.add_argument(
        "--mode",
        type=str,
        choices=[
            "bert-large-shapes",
            "sam-vitb-shapes",
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
        choices=DTYPE_LOOKUP.keys(),
        default="bf16",
    )
    parser.add_argument(
        "--backend", type=str, choices=["cutlass", "cusparselt"], default="cusparselt"
    )
    parser.add_argument("-eval-fn", type=str, choices=["linear", "mm"], default="linear")
    parser.add_argument("-contiguous", action="store_true")
    parser.add_argument("-save", action="store_true")
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
            for (m, k, n) in tqdm(bert_shapes)
        )
    elif args.mode == "sam-vitb-shapes":
        batch_size = 256
        sam_vitb_shapes = [
            (768, 3072, 50432),
            (3072, 768, 50432),
        ]
        results = (
            run_gpu_sparse_benchmark(m, k, n, args)
            for (m, k, n) in tqdm(sam_vitb_shapes)
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
            for mn in tqdm(mn_vals)
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
            for k in tqdm(k_vals)
        )

    else:
        raise ValueError(f"Unknown mode: {args.mode}")


    df = pd.DataFrame.from_records(results)
    if args.save:
        save_file = f"{args.mode}_{args.dtype}_{args.backend}.csv"
        df.to_csv(save_file)
        print(f"Finished benchmark: {args.mode} saved results to {save_file}")
    print(df)
