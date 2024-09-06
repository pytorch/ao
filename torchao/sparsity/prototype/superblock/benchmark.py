#  Copyright (c) Meta Platforms, Inc. and affiliates.

import os
import time
import sys
import warnings
import hashlib
import torchvision

import presets
import torch
import torch.utils.data
import utils
from torch import nn
from torch.sparse._triton_ops_meta import optimize_bsr_dense_addmm
from torchao.sparsity.prototype.superblock.utils import accelerate_with_sparsity, simulate_sparsity
from torchao.utils import benchmark_model, profiler_runner

torch.sparse.SparseSemiStructuredTensor._FORCE_CUTLASS = False

@torch.inference_mode
def main(args):
    print(args)
    device = torch.device(args.device)

    # We disable the cudnn benchmarking because it can noticeably affect the accuracy
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    num_classes = 1000

    dtype = getattr(torch, args.dtype)
    print(f"Using dtype: {dtype}")

    # BSR kernel tuning
    if args.bsr and args.tune_kernel_params:
        print("Tuning kernel params")
        if args.model == "vit_b_16":
            optimize_bsr_dense_addmm(3072, 768, 50432, args.bsr, args.bsr, dtype=dtype, sparsity=args.sparsity_linear, verbose=True)
            optimize_bsr_dense_addmm(768, 3072, 50432, args.bsr, args.bsr, dtype=dtype, sparsity=args.sparsity_linear, verbose=True)
        elif args.model == "vit_h_14":
            optimize_bsr_dense_addmm(5120, 1280, 65792, args.bsr, args.bsr, dtype=dtype, sparsity=args.sparsity_linear, verbose=True)
            optimize_bsr_dense_addmm(1280, 5120, 65792, args.bsr, args.bsr, dtype=dtype, sparsity=args.sparsity_linear, verbose=True)
        else:
            raise NotImplementedError("Tuning kernel params for this model is not supported yet.")

    print("Creating model")
    model = torchvision.models.get_model(args.model, weights=args.weights, num_classes=num_classes)

    # Fake sparsity necessary for BSR
    simulate_sparsity(model, args)

    if args.weights_path:
        try:
            checkpoint = torch.load(args.weights_path, map_location="cpu")
            model.load_state_dict(checkpoint["model"])
            print(f"Loaded checkpoint successfully from: {args.weights_path}")
        except FileNotFoundError:
            raise FileNotFoundError(f"No checkpoint found at {args.weights_path}.")

    model.to(device).to(dtype)

    # Fake sparsity necessary for BSR
    accelerate_with_sparsity(model, args)

    # compile 
    model = torch.compile(model, mode='max-autotune', fullgraph=True)

    # define image
    image = torch.randn(args.batch_size, 3, args.val_crop_size, args.val_crop_size, dtype=dtype, device=device)

    # warmup
    benchmark_model(model, 10, args=(image,)) 
    if args.profile:
        return profiler_runner("test.json.gz", benchmark_model, model, 10, (image,)) 
    else:
        return benchmark_model(model, 100, args=(image,)) 



def get_args_parser(add_help=True):
    import argparse

    parser = argparse.ArgumentParser(description="PyTorch Classification Training", add_help=add_help)
    parser.add_argument("--model", default="resnet18", type=str, help="model name")
    parser.add_argument("--device", default="cuda", type=str, help="device (Use cuda or cpu Default: cuda)")
    parser.add_argument(
        "-b", "--batch-size", default=32, type=int, help="images per gpu, the total batch size is $NGPU x batch_size"
    )
    parser.add_argument(
        "--val-crop-size", default=224, type=int, help="the central crop size used for validation (default: 224)"
    )
    parser.add_argument("--weights", default=None, type=str, help="the weights enum name to load")
    parser.add_argument("--weights-path", type=str, help="path of pretrained weights to load")
    # NOTE: sparsity args
    parser.add_argument("--sparsity", choices=["bsr", "semi_structured"], default=None, help='weight sparsification to apply')
    parser.add_argument("--sparsity-linear", type=float, default=0.0)
    parser.add_argument("--sp-linear-tile-size", type=int, default=1)
    parser.add_argument("--sparsity-conv1x1", type=float, default=0.0)
    parser.add_argument("--sp-conv1x1-tile-size", type=int, default=1)
    parser.add_argument("--sparsity-conv", type=float, default=0.0)
    parser.add_argument("--sp-conv-tile-size", type=int, default=1)
    parser.add_argument("--skip-last-layer-sparsity", action="store_true", help="Skip applying sparsity to the last linear layer (for vit only)")
    parser.add_argument("--skip-first-transformer-sparsity", action="store_true", help="Skip applying sparsity to the first transformer layer (for vit only)")
    parser.add_argument('--bsr', type=int, nargs='?', const=256, default=None, help='Convert sparsified weights to BSR format with optional block size (default: 256)')
    parser.add_argument("--dtype", choices=["float32", "bfloat16", "float16"], help="data type", default="bfloat16")
    parser.add_argument("--float16", action="store_true", help="Use float16")
    parser.add_argument("--tune-kernel-params", action="store_true", help="Tune kernel params")
    parser.add_argument("--profile", action="store_true", help="Profile the run and dump Prefetto trace")   
    parser.add_argument("--quantization", action="store_true", help="Profile the run and dump Prefetto trace")   

    return parser


if __name__ == "__main__":
    args = get_args_parser().parse_args()
    result = main(args)
    print(f"{result:.3f} ms", file=sys.stderr)
    print(f"{1000/result:.3f} img/s")
