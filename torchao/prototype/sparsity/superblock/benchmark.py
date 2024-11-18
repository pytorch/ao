#  Copyright (c) Meta Platforms, Inc. and affiliates.

import torch
import torchvision

from torch.sparse._triton_ops_meta import (
    dump as store_tuned_kernel_params,
    optimize_bsr_dense_addmm,
)
from torchao.prototype.sparsity.superblock.utils import (
    accelerate_with_sparsity,
    get_args_parser,
    simulate_sparsity,
)
from torchao.utils import benchmark_model, profiler_runner

torch.sparse.SparseSemiStructuredTensor._FORCE_CUTLASS = False
torch.backends.mha.set_fastpath_enabled(False)


@torch.inference_mode
def main(args):
    device = torch.device(args.device)

    # We disable the cudnn benchmarking because it can noticeably affect the accuracy
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    num_classes = 1000

    dtype = getattr(torch, args.dtype)

    # BSR kernel tuning
    if args.bsr and args.tune_kernel_params:
        kwargs = dict(
            dtype=torch.int8 if args.quantization else dtype,
            sparsity=args.sparsity_linear,
            verbose=True,
            # per blocksparse_int_addmm:
            alpha=1,
            beta=0,
            use_left_alpha=True,
            use_right_alpha=True,
            # force tuning because existing tuning parameters are
            # computed for use_left/right_alpha=False, however, it
            # turns out that re-tuning for use_left/right_alpha=False
            # leads to the same set of tuning parametes:
            # force=True
        )
        if args.model == "vit_b_16":
            optimize_bsr_dense_addmm(3072, 768, 50432, args.bsr, args.bsr, **kwargs)
            optimize_bsr_dense_addmm(768, 3072, 50432, args.bsr, args.bsr, **kwargs)
        elif args.model == "vit_h_14":
            optimize_bsr_dense_addmm(5120, 1280, 65792, args.bsr, args.bsr, **kwargs)
            optimize_bsr_dense_addmm(1280, 5120, 65792, args.bsr, args.bsr, **kwargs)
        else:
            raise NotImplementedError(
                "Tuning kernel params for this model is not supported yet."
            )
        # Warning: the following call will overwrite the source code
        # of torch.sparse._triton_ops_meta (hence it is commented out
        # by default) but when used, it'll enables reusing the tuned
        # parameters in subsequent runs of this script:
        # store_tuned_kernel_params()
    model = torchvision.models.get_model(
        args.model, weights=args.weights, num_classes=num_classes
    ).eval()

    # Fake sparsity necessary for BSR, since we find based on SuperBlock
    sparsifier_or_none = simulate_sparsity(model, args)
    if sparsifier_or_none is not None:
        sparsifier_or_none.squash_mask()

    if args.weights_path:
        try:
            checkpoint = torch.load(args.weights_path, map_location="cpu")
            model.load_state_dict(checkpoint["model"])
        except FileNotFoundError:
            raise FileNotFoundError(f"No checkpoint found at {args.weights_path}.")

    model.to(device).to(dtype)

    # With quantization, we must use cuSPARSELt to fuse one of the scalar matmuls.
    # Otherwise, we observe the CUTLASS kernels to be faster, so we use those instead.
    accelerate_with_sparsity(model, args)

    # compile
    model = torch.compile(model, mode="max-autotune", fullgraph=True)

    # define image
    image = torch.randn(
        args.batch_size,
        3,
        args.val_crop_size,
        args.val_crop_size,
        dtype=dtype,
        device=device,
    )

    # warmup
    benchmark_model(model, 10, args=(image,))
    if args.profile:
        return profiler_runner("test.json.gz", benchmark_model, model, 10, (image,))
    else:
        return benchmark_model(model, 100, args=(image,))


if __name__ == "__main__":
    args = get_args_parser(benchmark=True).parse_args()
    result = main(args)
    header = [
        "model",
        "batch_size",
        "dtype",
        "sparsity",
        "bsr",
        "sparsity_level",
        "quantization",
        "tune_kernel_params",
        "latency",
        "img/s",
    ]
    result_string = ",".join(
        str(_)
        for _ in [
            args.model,
            args.batch_size,
            args.dtype,
            args.sparsity,
            args.bsr,
            args.sparsity_linear,
            args.quantization,
            args.tune_kernel_params,
            result,
            1000 / result,
        ]
    )
    with open("benchmark_results.txt", "a") as f:
        if args.header:
            f.write(",".join(header) + "\n")
        f.write(result_string + "\n")
    print(result_string)
