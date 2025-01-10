import argparse

import pandas as pd
import torch
from bench_utils import (
    dora_colnorm_ref,
    dora_mm_epilogue_ref,
    dora_ref,
    dora_triton,
    make_dora_source_and_magnitude,
    make_epilogue_scales,
    make_epilogue_sources,
    make_inputs,
    make_lora_weights,
    make_weights,
    setup_dora_base_layers,
)
from triton.testing import do_bench

from torchao.prototype.common.profiling_tools import pivot_df
from torchao.prototype.dora.kernels.matmul import triton_mm
from torchao.prototype.dora.kernels.smallk import triton_mm_small_k


def run_colnorm_bench(args):
    in_features, out_features = args.in_features, args.out_features

    dtype = getattr(torch, args.dtype)

    # Inputs
    As, Bs = make_lora_weights(args.dora_ranks, in_features, out_features, dtype)
    source, magnitude = make_dora_source_and_magnitude(in_features, out_features, dtype)

    # torch.compile
    dora_colnorm_compiled = torch.compile(dora_colnorm_ref, mode=args.compile_mode)
    compiled_key = f"compiled_{args.compile_mode}"

    # Benchmark
    timings = []

    for a, b in zip(As, Bs):
        ref_t = do_bench(lambda: dora_colnorm_ref(a, b, source, magnitude))
        compiled_t = do_bench(lambda: dora_colnorm_compiled(a, b, source, magnitude))

        test_t = do_bench(
            lambda: triton_mm_small_k(
                b,
                a,
                epilogue_norm=True,
                source=source,
                magnitude=magnitude,
                store_acc=False,
            ),
        )
        common_args = [a.shape[0], a.shape[1], b.shape[0], args.dtype]
        timings.append([*common_args, "ref", ref_t])
        timings.append([*common_args, compiled_key, compiled_t])
        timings.append([*common_args, "triton", test_t])

    # Group results for kernel type
    headers = ["rank", "in_features", "out_features", "dtype", "kernel", "time(ms)"]
    df = pd.DataFrame(timings, columns=headers)
    id_cols = ["rank", "in_features", "out_features"]
    pivot_df(
        df,
        id_cols=id_cols,
        columns="kernel",
        values="time(ms)",
        column_order=[*id_cols, "ref", compiled_key, "triton"],
        show=True,
    )


def run_epilogue_bench(args):
    in_features, out_features = args.in_features, args.out_features
    seqlen = args.seqlen
    batch_sizes = (
        args.batch_sizes if isinstance(args.batch_sizes, list) else [args.batch_sizes]
    )
    dtype = getattr(torch, args.dtype)

    # Inputs
    xs = make_inputs(batch_sizes, seqlen, in_features, dtype)
    weights = make_weights(batch_sizes, in_features, out_features, dtype)
    epilogue_sources = make_epilogue_sources(batch_sizes, seqlen, out_features, dtype)
    epilogue_scales = make_epilogue_scales(batch_sizes, out_features, dtype)

    # torch.compile
    dora_mm_epilogue_compiled = torch.compile(
        dora_mm_epilogue_ref, mode=args.compile_mode
    )
    compiled_key = f"compiled_{args.compile_mode}"

    # Benchmark
    timings = []
    for bs, x, w, e1, e2 in zip(
        batch_sizes, xs, weights, epilogue_sources, epilogue_scales
    ):
        ref_t = do_bench(lambda: dora_mm_epilogue_ref(x, w, e1, e2))
        compiled_t = do_bench(lambda: dora_mm_epilogue_compiled(x, w, e1, e2))

        test_t = do_bench(
            lambda: triton_mm(
                x,
                w,
                epilogue_source=e1,
                epilogue_scale=e2,
            )
        )
        common_args = [bs, seqlen, w.shape[0], w.shape[1], args.dtype]
        timings.append([*common_args, "ref", ref_t])
        timings.append([*common_args, compiled_key, compiled_t])
        timings.append([*common_args, "triton", test_t])

    # Group results for kernel type
    headers = [
        "bs",
        "seqlen",
        "in_features",
        "out_features",
        "dtype",
        "kernel",
        "time(ms)",
    ]
    df = pd.DataFrame(timings, columns=headers)
    id_cols = ["bs", "seqlen", "in_features", "out_features", "dtype"]

    pivot_df(
        df,
        id_cols=id_cols,
        columns="kernel",
        values="time(ms)",
        column_order=[*id_cols, "ref", compiled_key, "triton"],
        show=True,
    )


def run_full_dora(args):
    """Dora Layer

    out = (x @ base_weight + lora_out) * magnitude_scale
    where:
    `lora_out = lora_B(lora_A(x)`
    `magnitude_scale = (base_weight + lora_B @ lora_A).norm(p=2, dim=1) * magnitude_vector`
    """

    dtype = getattr(torch, args.dtype)
    xs = make_inputs(args.batch_sizes, args.seqlen, args.in_features, dtype)
    weights = make_weights(args.batch_sizes, args.in_features, args.out_features, dtype)
    lora_As, lora_Bs = make_lora_weights(
        args.dora_ranks, args.in_features, args.out_features, dtype
    )
    _, magnitude_vector = make_dora_source_and_magnitude(
        args.in_features, args.out_features, dtype
    )

    # torch.compile
    dora_compiled = torch.compile(dora_ref, mode=args.compile_mode)
    # triton_compiled = torch.compile(dora_triton, mode=args.compile_mode)

    compiled_key = f"compiled_{args.compile_mode}"
    # triton_compiled_key = f"triton_compiled_{args.compile_mode}"

    # Benchmark
    timings = []
    for lora_A, lora_B in zip(lora_As, lora_Bs):
        for bs, x, w in zip(args.batch_sizes, xs, weights):
            # ref = dora_ref(x, w, lora_A, lora_B, magnitude_vector)
            # test = dora_triton(x, w, lora_A, lora_B, magnitude_vector)
            # compiled = dora_compiled(x, w, lora_A, lora_B, magnitude_vector)
            #       test_compiled = triton_compiled(x, w, lora_A, lora_B, magnitude_vector)
            # print(f"triton diff: {(ref - test).abs().max()}")
            # print(f"compiled diff: {(ref - compiled).abs().max()}")
            # print(f"triton compiled diff: {(ref - test_compiled).abs().max()}")
            ref_t = do_bench(lambda: dora_ref(x, w, lora_A, lora_B, magnitude_vector))
            compiled_t = do_bench(
                lambda: dora_compiled(x, w, lora_A, lora_B, magnitude_vector)
            )
            triton_t = do_bench(
                lambda: dora_triton(x, w, lora_A, lora_B, magnitude_vector)
            )
            # triton_compiled_t = do_bench(
            #     lambda: triton_compiled(x, w, lora_A, lora_B, magnitude_vector)
            # )

            # batch_size, seq_len, rank, in_features, out_features, dtype
            common_args = [
                bs,
                args.seqlen,
                lora_A.shape[0],
                args.in_features,
                args.out_features,
                args.dtype,
            ]
            timings.append([*common_args, "ref", ref_t])
            timings.append([*common_args, compiled_key, compiled_t])
            timings.append([*common_args, "triton", triton_t])
            # timings.append([*common_args, triton_compiled_key, triton_compiled_t])

    headers = [
        "bs",
        "seqlen",
        "rank",
        "in_features",
        "out_features",
        "dtype",
        "kernel",
        "time(ms)",
    ]
    df = pd.DataFrame(timings, columns=headers)
    id_cols = ["bs", "seqlen", "rank", "in_features", "out_features", "dtype"]

    pivot_df(
        df,
        id_cols=id_cols,
        columns="kernel",
        values="time(ms)",
        column_order=[
            *id_cols,
            "ref",
            compiled_key,
            "triton",
        ],  # , triton_compiled_key],
        show=True,
    )


def run_dora_layer_bench(args):
    dtype = getattr(torch, args.dtype)
    in_features, out_features = args.in_features, args.out_features
    xs = make_inputs(args.batch_sizes, args.seqlen, args.in_features, dtype)
    base_layer, dora_cls = setup_dora_base_layers(
        args.kernel, in_features, out_features, dtype
    )

    timings = []
    layer_key = f"{args.kernel}"
    layer_key_fused = f"{args.kernel}-fused"

    for bs, x in zip(args.batch_sizes, xs):
        for rank in args.dora_ranks:
            dora_layer = dora_cls(base_layer, rank).cuda()
            common_args = [
                bs,
                args.seqlen,
                rank,
                args.in_features,
                args.out_features,
                args.dtype,
            ]
            ref_t = do_bench(lambda: dora_layer.forward(x))
            fused_t = do_bench(lambda: dora_layer.forward_fused(x))
            timings.append([*common_args, layer_key, ref_t])
            timings.append([*common_args, layer_key_fused, fused_t])

    headers = [
        "bs",
        "seqlen",
        "rank",
        "in_features",
        "out_features",
        "dtype",
        "layer",
        "time(ms)",
    ]
    df = pd.DataFrame(timings, columns=headers)
    id_cols = ["bs", "seqlen", "rank", "in_features", "out_features", "dtype"]

    pivot_df(
        df,
        id_cols=id_cols,
        columns="layer",
        values="time(ms)",
        column_order=[
            *id_cols,
            layer_key,
            layer_key_fused,
        ],
        show=True,
    )


def run_bench(args):
    print(f"""Running {args.kernel} benchmark with dtype={args.dtype}, batch_sizes={args.batch_sizes}, seqlen={args.seqlen},
          in_features={args.in_features}, out_features={args.out_features}, dora_ranks={args.dora_ranks}""")
    if args.kernel == "dora-colnorm":
        return run_colnorm_bench(args)
    elif args.kernel == "dora-mm-epilogue":
        return run_epilogue_bench(args)
    elif args.kernel == "dora-full":
        return run_full_dora(args)
    elif args.kernel == "dora-bnb" or args.kernel == "dora-hqq":
        return run_dora_layer_bench(args)
    else:
        raise ValueError(f"Unknown kernel: {args.kernel}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        "--kernel",
        type=str,
        default="dora-mm-epilogue",
        choices=(
            "dora-colnorm",
            "dora-mm-epilogue",
            "dora-full",
            "dora-bnb",
            "dora-hqq",
        ),
        help="""The kernel to benchmark
        
            dora-colnorm: Small K GEMM with fused column-norm and magnitude vector multiplication
            dora-mm-epilogue: GEMM with fused epilogue elementwise addition and broadcasted scale
            dora-full: Full DORA kernel (dora-colnorm + dora-mm-epilogue)
            dora-bnb: BNBDoRALinear layer with fused kernels
            dora-hqq: HQQDoRALinear layer with fused kernels
        """,
    )
    parser.add_argument("--seqlen", type=int, default=512)
    parser.add_argument(
        "--batch_sizes", type=int, nargs="+", default=[1, 2, 4, 8, 16, 32]
    )
    parser.add_argument("--dora_ranks", type=int, nargs="+", default=[16, 32, 64])
    parser.add_argument("--in_features", type=int, default=4096)
    parser.add_argument("--out_features", type=int, default=4096)
    parser.add_argument(
        "--dtype",
        type=str,
        default="float16",
        choices=("float16", "bfloat16", "float32"),
    )
    parser.add_argument(
        "--compile_mode",
        type=str,
        default="default",
        choices=(
            "default",
            "reduce-overhead",
            "max-autotune-no-cudagraphs",
            "max-autotune",
        ),
    )

    args = parser.parse_args()
    run_bench(args)
