# Copyright (c) Meta Platforms, Inc. and affiliates.
import torch
import torchvision

from torchao.sparsity.prototype.superblock.train import evaluate, load_data
from torchao.sparsity.prototype.superblock.utils import (
    accelerate_with_sparsity,
    apply_sparsity,
    get_args_parser,
    init_distributed_mode,
    simulate_sparsity,
)

torch.sparse.SparseSemiStructuredTensor._FORCE_CUTLASS = False


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

    criterion = torch.nn.CrossEntropyLoss(label_smoothing=args.label_smoothing)
    evaluate(model, criterion, data_loader_test, device=device, dtype=torch.bfloat16)


if __name__ == "__main__":
    args = get_args_parser(evaluate=True).parse_args()
    main(args)
