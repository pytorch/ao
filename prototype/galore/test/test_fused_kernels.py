import argparse
import logging
import os
import torch
from test_utils import get_kernel, make_copy, make_data
from triton.testing import do_bench


def run_test(kernel, exp_avg, exp_avg2, grad, proj_matrix, params, allow_tf32):
    # Copy to use for first run -- needed because of autotuning and inplace ops
    (
        exp_avg_autotune_copy,
        exp_avg2_autotune_copy,
        grad_autotune_copy,
        proj_matrix_autotune_copy,
        params_autotune_copy,
    ) = make_copy(exp_avg, exp_avg2, grad, proj_matrix, params)

    # Copy to use for second run to check accuracy
    (
        exp_avg_test_copy,
        exp_avg2_test_copy,
        grad_test_copy,
        proj_matrix_test_copy,
        params_test_copy,
    ) = make_copy(exp_avg, exp_avg2, grad, proj_matrix, params)

    print(
        f"Running with {grad.shape[0]} x {grad.shape[1]} grad (param) shape, GaLore orthogonal matrix {list(proj_matrix.shape)}, dtype {grad.dtype} and allow_tf32 {allow_tf32}\n"
        f"Kernel: {kernel}",
        flush=True,
    )

    ref_op = get_kernel("ref")
    test_op = get_kernel(kernel)

    # Reference run
    ref_out = ref_op(
        grad,
        proj_matrix,
        exp_avg,
        exp_avg2,
        params,
    )

    # Autotune
    _ = test_op(
        grad_autotune_copy,
        proj_matrix_autotune_copy,
        exp_avg_autotune_copy,
        exp_avg2_autotune_copy,
        params_autotune_copy,
        store=False,
        allow_tf32=allow_tf32,
    )

    # Accuracy run
    test_out = test_op(
        grad_test_copy,
        proj_matrix_test_copy,
        exp_avg_test_copy,
        exp_avg2_test_copy,
        params_test_copy,
        store=True,
        allow_tf32=allow_tf32,
    )
    print("Accuracy:")

    output_names = [
        "adam state - running grad mean",
        "adam state - running grad var",
        "params (after update)",
    ]
    for name, ref, tt in zip(output_names, ref_out, test_out):
        print(
            f"-> {name}:\n  Max err: {(ref- tt).abs().max():.6f} Relative err: {(ref- tt).abs().max() / ref.abs().mean():.6f}"
        )

    # Turn off autotune logging during benchmarking
    from galore_fused.triton.custom_autotune import logger

    logger.setLevel(logging.WARNING)

    ref_perf = do_bench(lambda: ref_op(grad, proj_matrix, exp_avg, exp_avg2, params))
    test_perf = do_bench(
        lambda: test_op(
            grad_test_copy,
            proj_matrix_test_copy,
            exp_avg_test_copy,
            exp_avg2_test_copy,
            params_test_copy,
            store=True,
            allow_tf32=allow_tf32,
        )
    )
    print(
        f"Performance, torch vs test: {ref_perf:.4f}ms vs {test_perf:.4f}ms, {ref_perf / test_perf:1.2f}x"
    )


def run(args):
    dtype = getattr(torch, args.dtype)
    allow_tf32 = args.allow_tf32
    fp8_fast_accum = False
    torch.backends.cuda.matmul.allow_tf32 = allow_tf32
    kernel = args.kernel
    M, N = args.M, args.N
    rank = args.rank

    exp_avg, exp_avg2, grad, proj_matrix, params = make_data(M, N, rank, dtype)
    if args.mode.lower() == "test":
        run_test(kernel, exp_avg, exp_avg2, grad, proj_matrix, params, allow_tf32)
    elif args.mode.lower() == "benchmark":
        from test_utils import get_benchmark

        benchmark = get_benchmark(M, N, dtype, allow_tf32=allow_tf32)
        save_path = f'benchmark_{M}x{N}_{rank}_{args.dtype}_{"tf32" if allow_tf32 else "no-tf32"}'
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
        "--mode",
        choices=["test", "benchmark"],
        default="test",
        type=str,
        help="If test, runs kernel vs torch, comparing accuracy first then performance using triton `do_bench` for given config {M, N, rank, dtype, allow_tf32}."
        "If benchmark, runs all kernels across range of ranks for given config {M, N, dtype, allow_tf32}",
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
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="If true, prints autotuning output",
    )
    args = parser.parse_args()
    if args.verbose:
        logging.basicConfig(level=logging.INFO)
    run(args)
