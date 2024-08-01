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
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    mem_begin = torch.cuda.max_memory_allocated() / 2**20
    t0 = benchmark.Timer(
        stmt="f(*args, **kwargs)",
        globals={"args": args, "kwargs": kwargs, "f": f}
    )
    measurement = t0.timeit(number=100)
    torch.cuda.synchronize()
    name = measurement.task_spec.description
    memory = torch.cuda.max_memory_allocated() / 2**20 - mem_begin
    measurement.mem_use = memory
    return measurement


def run_gpu_sparse_benchmark(m, k, n, args):
    dtype = getattr(torch, args.dtype)

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
        dense_time = benchmark_in_us(F.linear, x, A, b)
        sparse_time = benchmark_in_us(F.linear, x, A_sparse, b)

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
        "sparse_latency (ms)": sparse_time.median * 1000,
        "sparse_latency iqr": sparse_time.iqr* 1000,
        "sparse_mem": sparse_time.mem_use,
        "dense_latency (ms)": dense_time.median* 1000,
        "dense_latenct iqr": dense_time.iqr* 1000,
        "dense_mem": dense_time.mem_use,
        "speedup (d/s)": dense_time.median / sparse_time.median,
        "contiguous": sparse_output.is_contiguous(),
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="GPU Sparsity Microbenchmarks")
    parser.add_argument(
        "--mode",
        type=str,
        choices=[
            "bert-large",
            "sam-vit-b",
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
            (768, 3072, 1024 * batch_size),
            (2304, 768, 1024 * batch_size),
            (3072, 768, 1024 * batch_size),
            (768, 768, 1024 * batch_size),
            (2304, 768, 1225 * batch_size),
            (768, 768, 1225 * batch_size),
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
    import torch
    print(torch.utils.collect_env.get_pretty_env_info())

    import matplotlib.pyplot as plt
    import numpy as np
    plt.figure(figsize=(10, 5))

    data = [
            [ 0.971323, 1.114185,  1.103950,  0.963362],
            [ 1.793014, 1.291936, 2.169129, 2.033742],
            [ 1.851897, 1.656134,  2.020207, 1.884365],
            [ 3.298112, 3.555021, 3.927926, 2.656046]]

    columns = ('Attention QKV', 'Attention Output', 'MLP Lin1', 'MLP Lin2')
    rows = ["sparsity_level=0.8, block_size=32", "sparsity_level=0.8, block_size=64", "sparsity_level=0.9, block_size=32", "sparsity_level=0.9, block_size=64"]

    values = np.arange(0, 4, 0.5)

    # Get some pastel shades for the colors
    colors = plt.cm.Paired(np.linspace(0, 1, 7))
    n_rows = len(data)

    index = np.arange(len(columns))
    bar_width = 0.15

    # Initialize the vertical-offset for the grouped bar chart.
    y_offset = np.zeros((n_rows, len(columns)))

    # Plot bars and create text labels for the table
    cell_text = []
    for row in range(n_rows):
        plt.bar(index + row * bar_width, data[row], bar_width, bottom=y_offset[row], color=colors[row])
        y_offset[row] = y_offset[row] + data[row]
        cell_text.append(['%1.2fx' % (x) for x in y_offset[row]])

    # Add a table at the bottom of the axes
    the_table = plt.table(cellText=cell_text,
                        rowLabels=rows,
                        rowColours=colors,
                        colLabels=columns,
                        loc='bottom')

    # Adjust layout to make room for the table:
    plt.subplots_adjust(left=0.2, bottom=0)
    plt.axhline(y=1, linestyle='--', color='gray')
    plt.ylabel(f"Speedup (Dense/Sparse)")
    plt.yticks()
    plt.xticks([], [])
    plt.title(f'{args.mode} Microbenchmarks (dtype={args.dtype})')

    plt.show()
    plt.savefig('foo.png', bbox_inches='tight')
