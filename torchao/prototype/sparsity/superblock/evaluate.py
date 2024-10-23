# Copyright (c) Meta Platforms, Inc. and affiliates.
import os

import torch
import torchvision

from torchao.prototype.sparsity.superblock.train import evaluate, load_data
from torchao.prototype.sparsity.superblock.utils import (
    accelerate_with_sparsity,
    apply_sparsity,
    get_args_parser,
    init_distributed_mode,
    simulate_sparsity,
)

torch.sparse.SparseSemiStructuredTensor._FORCE_CUTLASS = False
torch.backends.mha.set_fastpath_enabled(False)


def main(args):
    init_distributed_mode(args)
    print(args)

    device = torch.device(args.device)
    # We disable the cudnn benchmarking because it can noticeably affect the accuracy
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

    # Load validation data
    val_dir = os.path.join(args.data_path, "val")
    dataset_test, test_sampler = load_data(None, val_dir, args)
    data_loader_test = torch.utils.data.DataLoader(
        dataset_test,
        batch_size=args.batch_size,
        sampler=test_sampler,
        num_workers=args.workers,
        pin_memory=True,
        drop_last=True,
    )
    num_classes = len(dataset_test.classes)

    # Create Model
    print("Creating model")
    model = torchvision.models.get_model(
        args.model, weights=args.weights, num_classes=num_classes
    )

    sparsifier_or_none = simulate_sparsity(model, args)

    if args.weights_path:
        try:
            checkpoint = torch.load(args.weights_path, map_location="cpu")
            model.load_state_dict(checkpoint["model"])
            print(f"Loaded checkpoint successfully from: {args.weights_path}")
        except FileNotFoundError:
            raise FileNotFoundError(f"No checkpoint found at {args.weights_path}")

    model.to(device).bfloat16()

    if sparsifier_or_none is not None:
        sparsifier_or_none.squash_mask()
    accelerate_with_sparsity(model, args)
    model = torch.compile(model, mode="max-autotune", fullgraph=True)

    criterion = torch.nn.CrossEntropyLoss(label_smoothing=args.label_smoothing)
    return evaluate(
        model, criterion, data_loader_test, device=device, dtype=torch.bfloat16
    )


if __name__ == "__main__":
    args = get_args_parser(evaluate=True).parse_args()
    accuracy, throughput, max_mem = main(args)
    header = [
        "model",
        "batch_size",
        "dtype",
        "sparsity",
        "bsr",
        "sparsity_level",
        "quantization",
        "top-1_acc",
        "encoder img/s",
        "max_mem (MB)",
    ]
    result_string = ",".join(
        str(_)
        for _ in [
            args.model,
            args.batch_size,
            "bfloat16",
            args.sparsity,
            args.bsr,
            args.sparsity_linear,
            args.quantization,
            accuracy,
            throughput,
            max_mem,
        ]
    )
    with open("evaluation_results.txt", "a") as f:
        if args.header:
            f.write(",".join(header) + "\n")
        f.write(result_string + "\n")
    print(result_string)
