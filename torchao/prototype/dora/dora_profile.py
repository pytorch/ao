import argparse

import torch
from bitsandbytes.nn import Linear4bit
from hqq.core.quantize import BaseQuantizeConfig, HQQBackend, HQQLinear

from torchao.prototype.common.profiling_tools import (
    CudaProfilerCtx,
    TorchProfilerCtx,
    get_annotation_ctx,
)
from torchao.prototype.dora.dora_layer import BNBDoRALinear, DoRALinear, HQQDoRALinear


def run_profile(args, dora_forward):
    if args.profiler == "nsys":
        profiler = CudaProfilerCtx()
    else:
        profiler = TorchProfilerCtx.profiler(
            f"dora_layer-{args.layer_type}",
            active=max(5, args.num_iterations),
            warmup=0,
            out_dir=args.outdir,
        )

    annotation_ctx = get_annotation_ctx(args.profiler)

    x = torch.randn(
        args.bs, args.seqlen, args.in_features, dtype=getattr(torch, args.dtype)
    ).cuda()
    for _ in range(args.warmup):
        _ = dora_forward(x, annotation_ctx=annotation_ctx)

    with profiler as prof:
        for _ in range(args.num_iterations):
            _ = dora_forward(x, annotation_ctx=annotation_ctx)
            prof.step()
    print(f"Finished profiling, saving results to {args.outdir}")


def run(args):
    in_features, out_features = args.in_features, args.out_features
    dora_rank = args.dora_rank
    dtype = getattr(torch, args.dtype)

    base_layer = torch.nn.Linear(
        in_features, out_features, dtype=dtype, bias=False
    ).cuda()

    if args.layer_type == "torch":
        dora_layer = DoRALinear(base_layer=base_layer, lora_rank=dora_rank)
    elif args.layer_type == "bnb":
        base_layer = Linear4bit(
            input_features=in_features,
            output_features=out_features,
            bias=False,
            quant_type="nf4",
            compute_dtype=dtype,
        )
        base_layer.quant_state.dtype = base_layer.compute_dtype
        dora_layer = BNBDoRALinear(base_layer=base_layer, lora_rank=dora_rank)
    elif args.layer_type == "hqq":
        quant_config = BaseQuantizeConfig(
            nbits=4,
            group_size=64,
            quant_zero=False,
            quant_scale=False,
            offload_meta=True,
            view_as_float=True,
        )

        base_layer = HQQLinear(
            base_layer,
            quant_config,
            compute_dtype=dtype,
        )

        base_layer.set_backend(HQQBackend.PYTORCH)
        dora_layer = HQQDoRALinear(base_layer=base_layer, lora_rank=dora_rank)

    run_profile(args, dora_layer.forward_instrumented)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        "--profiler",
        type=str,
        default="torch",
        choices=("nsys", "torch"),
        help="""
        Which profiler to use
        
        Default is the torch.profiler
        
        If using `nsys`, run the nsys profiler as so, substituting with other desired nsys options: 
        `nsys profile --capture-range=cudaProfilerApi ... python dora_profile.py --profiler=nsys`
        
        Note that `--capture-range=cudaProfilerApi` is required
        """,
    )
    parser.add_argument(
        "--layer_type",
        type=str,
        default="torch",
        choices=("torch", "bnb", "hqq"),
    )
    parser.add_argument("--in_features", type=int, default=4096)
    parser.add_argument("--out_features", type=int, default=4096)
    parser.add_argument("--dora_rank", type=int, default=16)
    parser.add_argument("--bs", type=int, default=1)
    parser.add_argument("--seqlen", type=int, default=512)
    parser.add_argument(
        "--dtype",
        type=str,
        default="float16",
        choices=("float16", "bfloat16", "float32"),
    )
    parser.add_argument("--num_iterations", type=int, default=10)
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--outdir", type=str, default="./dora_profiles")
    run(parser.parse_args())
