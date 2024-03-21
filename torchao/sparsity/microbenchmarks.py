import argparse
import random

import pandas as pd
import torch
import torch.utils.benchmark as benchmark
from torch import nn
from torch.sparse import SparseSemiStructuredTensor, to_sparse_semi_structured
from torch.ao.pruning import WeightNormSparsifier
from tqdm import tqdm

import math

import torch
import torch.nn.functional as F
import itertools
import torch.utils.benchmark as benchmark
import math

dtype = torch.float16
device = "cuda"
torch.manual_seed(42)


torch.set_printoptions(
    precision=2,
    threshold=None,
    edgeitems=16,
    linewidth=480,
    profile=None,
    sci_mode=False,
)

def create_blocked_tensor(M, N, blocksize, sparsity):
    assert sparsity <= 1.0 and sparsity >= 0.0, \
        "sparsity should be a value between 0 and 1"
    A = torch.bernoulli(torch.full((M//blocksize, N//blocksize),
                        1 - sparsity, dtype=torch.bfloat16, device=device))
    A = torch.repeat_interleave(A, blocksize, dim=0)
    A = torch.repeat_interleave(A, blocksize, dim=1)
    return A.contiguous()


def create_24_tensor(M, N):
    A = torch.randn(weight_shape, device="cuda")

    choices = [[0, 1], [1, 0]]
    mask_entries = [random.choice(choices) for i in range(M * N // 2)]

    mask = torch.tensor(mask_entries).cuda().bool().reshape(M, N)

    A.masked_fill_(~mask, 0)

    return A.contiguous()


def benchmark_in_us(f, *args, **kwargs):
    t0 = benchmark.Timer(
        stmt="f(*args, **kwargs)",
        globals={"args": args, "kwargs": kwargs, "f": f}
    )
    return int(t0.blocked_autorange().mean * 1e6)


def run_benchmark(input_shape, weight_shape, dtype, sparsity=None, backend=None, blocksize=None, sparsity_level=None):

    m, k = weight_shape
    n, k = math.prod(input_shape[:-1]), input_shape[-1]

    if sparsity == "blocksparse":
        A = create_blocked_tensor(m, k, blocksize=blocksize, sparsity=sparsity_level).to(dtype)
        A_sparse = A.to_sparse_bsr(blocksize=blocksize)

    elif sparsity == "24":
        # blocksize = 4
        # sparsity_level = 0.5
        if backend == "cutlass":
            SparseSemiStructuredTensor._FORCE_CUTLASS = True
        elif backend == "cusparselt":
            SparseSemiStructuredTensor._FORCE_CUTLASS = False
        else:
            raise ValueError("Wrong value for backend")

        A = create_24_tensor(m, k).to(dtype)
        A_sparse = to_sparse_semi_structured(A)

    # b = torch.randn(m, device="cuda").to(dtype)
    x = torch.randn(n, k).to(dtype).cuda()


    # get timing speedups
    # handle int_mm custom
    if dtype == torch.int8:
        dense_time = benchmark_in_us(torch._int_mm, A, x.t())
        dense_output = torch._int_mm(A, x.t()).to(torch.float32).t()
    else:
        dense_time = benchmark_in_us(F.linear, x, A)
        dense_output = F.linear(x, A).to(torch.float32)

    sparse_time = benchmark_in_us(F.linear, x, A_sparse)
    sparse_output = F.linear(x, A_sparse).to(torch.float32)

    ratio = dense_time / sparse_time


    if backend == "cusparselt":
        # grab optimal alg id for cusparselt
        padded = A_sparse._pad_tensor_for_matmul(x)
        if dtype is torch.int8:
            out_dtype = torch.bfloat16
        optimal_alg_id = torch._cslt_sparse_mm_search(A_sparse.compressed_tensor_cusparselt, padded.t())
        # print("optimal alg_id", optimal_alg_id)
    else:
        optimal_alg_id = None

    # sanity check correctness
    correct = torch.allclose(dense_output, sparse_output, rtol=1e-3, atol=1e-3)

    # # in depth checks
    # dense_output = F.linear(x.to(torch.float32), A.to(torch.float32))

    # diff = ~torch.isclose(dense_output, sparse_output)

    # dense_output_diff = dense_output[diff]
    # sparse_output_diff = sparse_output[diff]

    # sparse_output_diff_nonzero = sparse_output_diff.nonzero()
    # dense_output_diff = dense_output_diff[sparse_output_diff_nonzero]
    # sparse_output_diff = sparse_output_diff[sparse_output_diff_nonzero]

    # outside_atol = ~((dense_output_diff - sparse_output_diff).abs() < 1e-3)

    # larger_dense_output_diff = dense_output_diff[outside_atol]
    # larger_sparse_output_diff = sparse_output_diff[outside_atol]

    # pos = (1 - (larger_dense_output_diff / larger_sparse_output_diff)).abs().argmax().item()

    return {
        "dtype": str(dtype),
        "m": m,
        "k": k,
        "n": n,
        "sparse_latency (us)": sparse_time,
        "dense_latency (us)": dense_time,
        "speedup (d/s)": f"{ratio:.3f}",
        "correct": correct,
        # "sparse v dense diff": f"{larger_dense_output_diff[pos]:+11.7f} vs. {larger_sparse_output_diff[pos]:+11.7f}",
        "sparsity type": sparsity,
        "backend": backend,
        "blocksize": blocksize,
        "sparsity level": sparsity_level,
        "optimal_alg_id": optimal_alg_id,
    }

if __name__ == "__main__":
    dtype_lookup = {
        "int8": torch.int8,
        "fp16": torch.float16,
        "bf16": torch.bfloat16,
        "fp32": torch.float32,
    }

    parser = argparse.ArgumentParser(description="GPU Sparsity Kernel Microbenchmarks")
    parser.add_argument(
        "--mode",
        type=str,
        choices=[
            "nvidia-bert",
            "sam-shapes",
            "nvidia-fixed-k",
            "nvidia-fixed-mn",
            "optimize-matmul-block-sparse",
        ],
    )
    parser.add_argument(
        "--dtype",
        type=str,
        choices=dtype_lookup.keys(),
        default="fp16",
    )
    parser.add_argument("--backend", type=str, choices=["cutlass", "cusparselt"], default="cusparselt")
    parser.add_argument("--function", type=str, choices=["linear", "mm"], default="linear")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("-contiguous", action="store_true")
    parser.add_argument("-save", action="store_true")
    args = parser.parse_args()

    eval_fn = run_benchmark

    print(f"Started benchmark: {args.mode} | dtype: {args.dtype}")
    dtype = dtype_lookup[args.dtype]

    if args.mode == "nvidia-bert":
        bert_shapes = [
            (3072, 1024, 16384),
            (4096, 1024, 16384),
            (1024, 1024, 16384),
            (1024, 4096, 16384),
        ]
        results = [
            eval_fn(m, k, n, dtype, sparsity="blocksparse", blocksize=64, sparsity_level=0.8)
            for (m, k, n) in tqdm(bert_shapes)
        ]

        results += [
            eval_fn(m, k, n, dtype, sparsity="24", backend="cusparselt")
            for (m, k, n) in tqdm(bert_shapes)
        ]

    if args.mode == "optimize-matmul-block-sparse":
        batch_size = args.batch_size

        sam_shapes = [
            (torch.Size([batch_size, 64, 64, 1280]), torch.Size([5120, 1280])),
        ]

        from collections import defaultdict
        results = []
        total_runtime = defaultdict(int)

        for (activation_shape, weight_shape) in tqdm(sam_shapes):
            for blocksize in [64]:
                for sparsity_level in range(0, 100):
                    sparsity_level = float(sparsity_level) / 100
                    result = run_benchmark(
                        activation_shape,
                        weight_shape,
                        dtype,
                        sparsity="blocksparse",
                        blocksize=blocksize,
                        sparsity_level=sparsity_level)
                    total_runtime[f"{blocksize}_{sparsity_level}"] += 32 * result["sparse_latency (us)"]
                    results.append(result)

    if args.mode == "sam-shapes":
        batch_size = args.batch_size

        sam_shapes = [
            (torch.Size([batch_size, 256, 3072]), torch.Size([768, 3072])),
            (torch.Size([batch_size, 256, 768]), torch.Size([3072, 768])),
        ]

        from collections import defaultdict
        results = []
        total_runtime = defaultdict(int)

        for (activation_shape, weight_shape) in tqdm(sam_shapes):
            # for backend in ["cutlass", "cusparselt"]:
                # result = run_benchmark(
                    # activation_shape,
                    # weight_shape,
                    # dtype,
                    # sparsity="24",
                    # backend=backend)

                # blocksize = None
                # sparsity_level = 0.5
                # total_runtime[f"{backend}"] += 32 * result["sparse_latency (us)"]
                # results.append(result)
            for blocksize in [8, 16, 32, 64]:
                for sparsity_level in [0.8, 0.9]:
                    result = run_benchmark(
                        activation_shape,
                        weight_shape,
                        dtype,
                        sparsity="blocksparse",
                        blocksize=blocksize,
                        sparsity_level=sparsity_level)
                    # total_runtime[f"{blocksize}_{sparsity_level}"] += 32 * result["sparse_latency (us)"]
                    results.append(result)

            # total_runtime["dense"] += 32 * result["dense_latency (us)"]

        # for line in total_runtime:
            # print(line, total_runtime[line], sep="\t")

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
            eval_fn(mn, 10240, mn, dtype, args.contiguous, args.backend)
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
            eval_fn(10240, k, 10240, dtype, args.contiguous, args.backend)
            for k in tqdm(k_vals)
        )

    df = pd.DataFrame.from_records(results)
    if args.save:
        save_file = f"{args.mode}_{args.dtype}_{args.backend}.csv"
        df.to_csv(save_file)
        print(f"Finished benchmark: {args.mode} saved results to {save_file}")
    print(df)
