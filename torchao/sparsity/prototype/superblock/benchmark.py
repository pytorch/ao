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

from torchao.quantization import quantize_, int8_dynamic_activation_int8_weight, int4_weight_only
from torchao.sparsity import sparsify_, apply_fake_sparsity, int8_dynamic_activation_int8_semi_sparse_weight, semi_sparse_weight

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from supermask import apply_supermask, SupermaskLinear
from blocksparse import BlockSparseTensor
from torchao.utils import benchmark_model, profiler_runner


def apply_sparsity(model):
    for name, module in model.named_modules():
        if isinstance(module, SupermaskLinear) and "mlp" in name:
            module.sparsify_offline()


def apply_bsr(model, blocksize):
    for name, module in model.named_modules():
        if isinstance(module, torch.nn.Linear) and "mlp" in name:
            try:
                module.weight = torch.nn.Parameter(BlockSparseTensor.from_dense(module.weight.data, blocksize))
                print(f"Converted {name} to bsr format.")
            except ValueError as e:
                print(f"Unable to convert weight of {name} to bsr format: {e}")


def verify_sparsity(model):
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear):
            total_weights = module.weight.numel()
            sparse_weights = (module.weight == 0).sum().item()
            sparsity_percentage = (sparse_weights / total_weights) * 100
            print(f"Sparsity verified in layer {name}: {sparsity_percentage:.2f}%")

@torch.inference_mode
def main(args):
    print(args)
    device = torch.device(args.device)

    # We disable the cudnn benchmarking because it can noticeably affect the accuracy
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    num_classes = 1000

    dtype = None
    if args.bfloat16:
        print("Using bfloat16")
        dtype = torch.bfloat16
    elif args.float16:
        print("Using float16")
        dtype = torch.float16

    if args.bsr and args.tune_kernel_params:
        print("Tuning kernel params")
        assert args.model == "vit_b_16", "--tune-kernel-params only supported for vit-b-16!"
        optimize_bsr_dense_addmm(3072, 768, 50432, args.bsr, args.bsr, dtype=dtype, sparsity=args.sparsity_linear, verbose=True)
        optimize_bsr_dense_addmm(768, 3072, 50432, args.bsr, args.bsr, dtype=dtype, sparsity=args.sparsity_linear, verbose=True)

    print("Creating model")
    model = torchvision.models.get_model(args.model, weights=args.weights, num_classes=num_classes)


    if args.sparsity == "bsr":
        apply_supermask(
            model,
            linear_sparsity=args.sparsity_linear,
            linear_sp_tilesize=args.sp_linear_tile_size,
            conv1x1_sparsity=args.sparsity_conv1x1,
            conv1x1_sp_tilesize=args.sp_conv1x1_tile_size,
            conv_sparsity=args.sparsity_conv,
            conv_sp_tilesize=args.sp_conv_tile_size,
            skip_last_layer_sparsity=args.skip_last_layer_sparsity,
            skip_first_transformer_sparsity=args.skip_first_transformer_sparsity,
            device=device,
            verbose=False,
        )

    elif args.sparsity == "semi_structured":
        sparse_config = []
        from torch.ao.pruning import WeightNormSparsifier
        for name, mod in model.named_modules():
            if args.skip_last_layer_sparsity and "heads.head" in name:
                continue
            if args.skip_first_transformer_sparsity and "encoder.layers.encoder_layer_0" in name:
                continue
            if isinstance(mod, torch.nn.Linear): 
                sparse_config.append({"tensor_fqn": f"{name}.weight"})

        sparsifier = WeightNormSparsifier(
            sparsity_level=1.0, sparse_block_shape=(1, 4), zeros_per_block=2
        )
        sparsifier.prepare(model, sparse_config)
        sparsifier.step()
        sparsifier.squash_mask()


    if args.weights_path:
        try:
            checkpoint = torch.load(args.weights_path, map_location="cpu")
            model.load_state_dict(checkpoint["model"])
            print(f"Loaded checkpoint successfully from: {args.weights_path}")
        except FileNotFoundError:
            raise FileNotFoundError(f"No checkpoint found at {args.weights_path}.")

    model.to(device)
    if dtype:
        model = model.to(dtype)

    if args.sparsity == "bsr":
        apply_sparsity(model)
        verify_sparsity(model)
        if args.bsr:
            apply_bsr(model, blocksize=args.bsr)
        

    if args.sparsity == "semi_structured":
        torch.sparse.SparseSemiStructuredTensor._FORCE_CUTLASS = False
        def mlp_only(mod, name):
            return isinstance(mod, torch.nn.Linear) and 'mlp' in name
        sparsify_(model,
                  semi_sparse_weight(),
                  mlp_only)

    model = torch.compile(model, mode='max-autotune', fullgraph=True)

    image = torch.randn(args.batch_size, 3, args.val_crop_size, args.val_crop_size, dtype=dtype, device=device)

    # warmup
    benchmark_model(model, 10, args=(image,)) 
    if args.profile:
        return profiler_runner("test.json.gz", benchmark_model, model, 10, (image,)) 
    else:
        return benchmark_model(model, 10, args=(image,)) 



def get_args_parser(add_help=True):
    import argparse

    parser = argparse.ArgumentParser(description="PyTorch Classification Training", add_help=add_help)
    parser.add_argument("--model", default="resnet18", type=str, help="model name")
    parser.add_argument("--device", default="cuda", type=str, help="device (Use cuda or cpu Default: cuda)")
    parser.add_argument(
        "-b", "--batch-size", default=32, type=int, help="images per gpu, the total batch size is $NGPU x batch_size"
    )

    # Mixed precision training parameters
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
    parser.add_argument("--bfloat16", action="store_true", help="Use bfloat16")
    parser.add_argument("--float16", action="store_true", help="Use float16")
    parser.add_argument("--tune-kernel-params", action="store_true", help="Tune kernel params")
    parser.add_argument("--profile", action="store_true", help="Profile the run and dump Prefetto trace")   

    return parser


if __name__ == "__main__":
    args = get_args_parser().parse_args()
    result = main(args)
    print(f"{result:.3f} ms", file=sys.stderr)
