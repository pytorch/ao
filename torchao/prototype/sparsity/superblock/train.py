#  Copyright (c) Meta Platforms, Inc. and affiliates.

import datetime
import glob
import os
import time
import warnings

import torch
import torch.utils.data
import torchvision
import utils
from torch import nn
from torch.utils.data.dataloader import default_collate
from torchvision.transforms.functional import InterpolationMode
from utils import RASampler

from torchao.prototype.sparsity.superblock.utils import simulate_sparsity


def train_one_epoch(
    model,
    criterion,
    optimizer,
    data_loader,
    device,
    epoch,
    args,
    model_ema=None,
    scaler=None,
):
    model.train()
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter("lr", utils.SmoothedValue(window_size=1, fmt="{value}"))
    metric_logger.add_meter("img/s", utils.SmoothedValue(window_size=10, fmt="{value}"))

    header = f"Epoch: [{epoch}]"
    accumulation_counter = 0  # Counter for tracking accumulated gradients

    for i, (image, target) in enumerate(
        metric_logger.log_every(data_loader, args.print_freq, header)
    ):
        start_time = time.time()
        image, target = image.to(device), target.to(device)

        with torch.cuda.amp.autocast(enabled=scaler is not None):
            output = model(image)
            loss = criterion(output, target) / args.accumulation_steps  # Scale loss

        if scaler is not None:
            scaler.scale(loss).backward()
        else:
            loss.backward()

        accumulation_counter += 1

        if accumulation_counter % args.accumulation_steps == 0:
            if scaler is not None:
                if args.clip_grad_norm is not None:
                    scaler.unscale_(optimizer)  # Unscale gradients before clipping
                    nn.utils.clip_grad_norm_(model.parameters(), args.clip_grad_norm)
                scaler.step(optimizer)
                scaler.update()
            else:
                if args.clip_grad_norm is not None:
                    nn.utils.clip_grad_norm_(model.parameters(), args.clip_grad_norm)
                optimizer.step()

            optimizer.zero_grad()  # Zero out gradients after optimization step

        if model_ema and i % args.model_ema_steps == 0:
            model_ema.update_parameters(model)
            if epoch < args.lr_warmup_epochs:
                model_ema.n_averaged.fill_(0)

        acc1, acc5 = utils.accuracy(output, target, topk=(1, 5))
        batch_size = image.shape[0]
        metric_logger.update(
            loss=loss.item() * args.accumulation_steps,
            lr=optimizer.param_groups[0]["lr"],
        )  # Scale back up for logging
        metric_logger.meters["acc1"].update(acc1.item(), n=batch_size)
        metric_logger.meters["acc5"].update(acc5.item(), n=batch_size)
        metric_logger.meters["img/s"].update(batch_size / (time.time() - start_time))


def evaluate(
    model,
    criterion,
    data_loader,
    device,
    print_freq=100,
    log_suffix="",
    dtype=torch.float32,
):
    model.eval()
    metric_logger = utils.MetricLogger(delimiter="  ")
    header = f"Test: {log_suffix}"
    encoder_time = 0
    num_processed_samples = 0
    with torch.inference_mode():
        for image, target in metric_logger.log_every(data_loader, print_freq, header):
            image = image.to(device, non_blocking=True).to(dtype)
            target = target.to(device, non_blocking=True).to(dtype)
            # intialize encoder measurements
            torch.cuda.reset_max_memory_allocated()
            torch.cuda.synchronize()
            start_event = torch.cuda.Event(enable_timing=True)
            end_event = torch.cuda.Event(enable_timing=True)
            start_event.record()

            # run encoder
            output = model(image)

            # measure time in encoder
            end_event.record()
            torch.cuda.synchronize()
            encoder_time += start_event.elapsed_time(end_event)
            max_mem = torch.cuda.max_memory_allocated() / (1024**2)

            acc1, acc5 = utils.accuracy(output, target, topk=(1, 5))
            # FIXME need to take into account that the datasets
            # could have been padded in distributed setup
            batch_size = image.shape[0]
            # metric_logger.update(loss=loss.item())
            metric_logger.meters["acc1"].update(acc1.item(), n=batch_size)
            metric_logger.meters["acc5"].update(acc5.item(), n=batch_size)
            metric_logger.meters["batch_time"].update(encoder_time, n=batch_size)
            num_processed_samples += batch_size
    # gather the stats from all processes

    num_processed_samples = utils.reduce_across_processes(num_processed_samples)
    if (
        hasattr(data_loader.dataset, "__len__")
        and len(data_loader.dataset) != num_processed_samples
    ):
        # See FIXME above
        warnings.warn(
            f"It looks like the dataset has {len(data_loader.dataset)} samples, but {num_processed_samples} "
            "samples were used for the validation, which might bias the results. "
            "Try adjusting the batch size and / or the world size. "
            "Setting the world size to 1 is always a safe bet."
        )

    metric_logger.synchronize_between_processes()

    print(
        f"{header} Acc@1 {metric_logger.acc1.global_avg:.3f} Acc@5 {metric_logger.acc5.global_avg:.3f}"
    )
    total_time = encoder_time / 1000.0
    return (
        metric_logger.acc1.global_avg,
        num_processed_samples.item() / total_time,
        max_mem,
    )


def _get_cache_path(filepath):
    import hashlib

    h = hashlib.sha1(filepath.encode()).hexdigest()
    cache_path = os.path.join(
        "~", ".torch", "vision", "datasets", "imagefolder", h[:10] + ".pt"
    )
    cache_path = os.path.expanduser(cache_path)
    return cache_path


def load_data(traindir, valdir, args):
    # Data loading code
    print("Loading data")
    (
        val_resize_size,
        val_crop_size,
    ) = (
        args.val_resize_size,
        args.val_crop_size,
    )
    interpolation = InterpolationMode(args.interpolation)
    if traindir is not None:
        train_crop_size = args.train_crop_size
        print("Loading training data")
        st = time.time()
        cache_path = _get_cache_path(traindir)
        if args.cache_dataset and os.path.exists(cache_path):
            # Attention, as the transforms are also cached!
            print(f"Loading dataset_train from {cache_path}")
            dataset, _ = torch.load(cache_path)
        else:
            auto_augment_policy = getattr(args, "auto_augment", None)
            random_erase_prob = getattr(args, "random_erase", 0.0)
            ra_magnitude = args.ra_magnitude
            augmix_severity = args.augmix_severity
            preprocessing = utils.ClassificationPresetTrain(
                crop_size=train_crop_size,
                interpolation=interpolation,
                auto_augment_policy=auto_augment_policy,
                random_erase_prob=random_erase_prob,
                ra_magnitude=ra_magnitude,
                augmix_severity=augmix_severity,
            )
            dataset = torchvision.datasets.ImageFolder(traindir, preprocessing)
            # ) if args.meta else torchvision.datasets.ImageNet(
            #     traindir,
            #     split="train",
            #     transform=preprocessing,
            # )
            if args.cache_dataset:
                print(f"Saving dataset_train to {cache_path}")
                utils.mkdir(os.path.dirname(cache_path))
                utils.save_on_master((dataset, traindir), cache_path)
        print("Took", time.time() - st)
        print(f"Number of training images: {len(dataset)}")
        if args.distributed:
            if hasattr(args, "ra_sampler") and args.ra_sampler:
                train_sampler = RASampler(
                    dataset, shuffle=True, repetitions=args.ra_reps
                )
            else:
                train_sampler = torch.utils.data.distributed.DistributedSampler(dataset)
        else:
            train_sampler = torch.utils.data.RandomSampler(dataset)

    print("Loading validation data")
    cache_path = _get_cache_path(valdir)
    if args.cache_dataset and os.path.exists(cache_path):
        # Attention, as the transforms are also cached!
        print(f"Loading dataset_test from {cache_path}")
        dataset_test, test_sampler = torch.load(cache_path)
    else:
        if args.weights:
            weights = torchvision.models.get_weight(args.weights)
            preprocessing = weights.transforms()
        else:
            preprocessing = utils.ClassificationPresetEval(
                crop_size=val_crop_size,
                resize_size=val_resize_size,
                interpolation=interpolation,
            )
        dataset_test = (
            torchvision.datasets.ImageFolder(
                valdir,
                preprocessing,
            )
            if args.meta
            else torchvision.datasets.ImageNet(
                valdir, split="val", transform=preprocessing
            )
        )
        if args.cache_dataset:
            print(f"Saving dataset_test to {cache_path}")
            utils.mkdir(os.path.dirname(cache_path))
            utils.save_on_master((dataset_test, valdir), cache_path)

        print(f"Number of validation images: {len(dataset_test)}")
        test_sampler = (
            torch.utils.data.distributed.DistributedSampler(dataset_test, shuffle=False)
            if args.distributed
            else torch.utils.data.SequentialSampler(dataset_test)
        )

    # for evaluation
    if traindir is None:
        return dataset_test, test_sampler

    return dataset, dataset_test, train_sampler, test_sampler


def main(args):
    if args.output_dir:
        utils.mkdir(args.output_dir)

    utils.init_distributed_mode(args)
    print(args)

    device = torch.device(args.device)

    if args.use_deterministic_algorithms:
        torch.backends.cudnn.benchmark = False
        torch.use_deterministic_algorithms(True)
    else:
        torch.backends.cudnn.benchmark = True

    train_dir = os.path.join(args.data_path, "train_blurred")
    val_dir = os.path.join(args.data_path, "val")
    dataset, dataset_test, train_sampler, test_sampler = load_data(
        train_dir, val_dir, args
    )

    collate_fn = None
    num_classes = len(dataset.classes)
    mixup_transforms = []
    if args.mixup_alpha > 0.0:
        mixup_transforms.append(
            utils.RandomMixup(num_classes, p=1.0, alpha=args.mixup_alpha)
        )
    if args.cutmix_alpha > 0.0:
        mixup_transforms.append(
            utils.RandomCutmix(num_classes, p=1.0, alpha=args.cutmix_alpha)
        )
    if mixup_transforms:
        mixupcutmix = torchvision.transforms.RandomChoice(mixup_transforms)

        def collate_fn(batch):
            return mixupcutmix(*default_collate(batch))

    data_loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=args.batch_size,
        sampler=train_sampler,
        num_workers=args.workers,
        pin_memory=True,
        collate_fn=collate_fn,
    )
    data_loader_test = torch.utils.data.DataLoader(
        dataset_test,
        batch_size=args.batch_size,
        sampler=test_sampler,
        num_workers=args.workers,
        pin_memory=True,
    )

    print("Creating model")
    model = torchvision.models.get_model(
        args.model, weights=args.weights, num_classes=num_classes
    )

    if args.weights_path is not None:
        sd = torch.load(args.weights_path, map_location="cpu")
        model.load_state_dict(sd)

    model.to(device)
    if args.distributed and args.sync_bn:
        model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)

    sparsifier = simulate_sparsity(model, args)
    criterion = nn.CrossEntropyLoss(label_smoothing=args.label_smoothing)

    custom_keys_weight_decay = []
    if args.bias_weight_decay is not None:
        custom_keys_weight_decay.append(("bias", args.bias_weight_decay))
    if args.transformer_embedding_decay is not None:
        for key in [
            "class_token",
            "position_embedding",
            "relative_position_bias_table",
        ]:
            custom_keys_weight_decay.append((key, args.transformer_embedding_decay))
    parameters = utils.set_weight_decay(
        model,
        args.weight_decay,
        norm_weight_decay=args.norm_weight_decay,
        custom_keys_weight_decay=(
            custom_keys_weight_decay if len(custom_keys_weight_decay) > 0 else None
        ),
    )

    opt_name = args.opt.lower()
    if opt_name.startswith("sgd"):
        optimizer = torch.optim.SGD(
            parameters,
            lr=args.lr,
            momentum=args.momentum,
            weight_decay=args.weight_decay,
            nesterov="nesterov" in opt_name,
        )
    elif opt_name == "rmsprop":
        optimizer = torch.optim.RMSprop(
            parameters,
            lr=args.lr,
            momentum=args.momentum,
            weight_decay=args.weight_decay,
            eps=0.0316,
            alpha=0.9,
        )
    elif opt_name == "adamw":
        optimizer = torch.optim.AdamW(
            parameters, lr=args.lr, weight_decay=args.weight_decay
        )
    else:
        raise RuntimeError(
            f"Invalid optimizer {args.opt}. Only SGD, RMSprop and AdamW are supported."
        )

    scaler = torch.cuda.amp.GradScaler() if args.amp else None

    args.lr_scheduler = args.lr_scheduler.lower()
    if args.lr_scheduler == "steplr":
        main_lr_scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer, step_size=args.lr_step_size, gamma=args.lr_gamma
        )
    elif args.lr_scheduler == "cosineannealinglr":
        main_lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=args.epochs - args.lr_warmup_epochs, eta_min=args.lr_min
        )
    elif args.lr_scheduler == "exponentiallr":
        main_lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(
            optimizer, gamma=args.lr_gamma
        )
    else:
        raise RuntimeError(
            f"Invalid lr scheduler '{args.lr_scheduler}'. Only StepLR, CosineAnnealingLR and ExponentialLR "
            "are supported."
        )

    if args.lr_warmup_epochs > 0:
        if args.lr_warmup_method == "linear":
            warmup_lr_scheduler = torch.optim.lr_scheduler.LinearLR(
                optimizer,
                start_factor=args.lr_warmup_decay,
                total_iters=args.lr_warmup_epochs,
            )
        elif args.lr_warmup_method == "constant":
            warmup_lr_scheduler = torch.optim.lr_scheduler.ConstantLR(
                optimizer,
                factor=args.lr_warmup_decay,
                total_iters=args.lr_warmup_epochs,
            )
        else:
            raise RuntimeError(
                f"Invalid warmup lr method '{args.lr_warmup_method}'. Only linear and constant are supported."
            )
        lr_scheduler = torch.optim.lr_scheduler.SequentialLR(
            optimizer,
            schedulers=[warmup_lr_scheduler, main_lr_scheduler],
            milestones=[args.lr_warmup_epochs],
        )
    else:
        lr_scheduler = main_lr_scheduler

    model_without_ddp = model
    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(
            model, device_ids=[args.gpu], find_unused_parameters=True
        )
        model_without_ddp = model.module

    model_ema = None
    if args.model_ema:
        # Decay adjustment that aims to keep the decay independent from other hyper-parameters originally proposed at:
        # https://github.com/facebookresearch/pycls/blob/f8cd9627/pycls/core/net.py#L123
        #
        # total_ema_updates = (Dataset_size / n_GPUs) * epochs / (batch_size_per_gpu * EMA_steps)
        # We consider constant = Dataset_size for a given dataset/setup and ommit it. Thus:
        # adjust = 1 / total_ema_updates ~= n_GPUs * batch_size_per_gpu * EMA_steps / epochs
        adjust = args.world_size * args.batch_size * args.model_ema_steps / args.epochs
        alpha = 1.0 - args.model_ema_decay
        alpha = min(1.0, alpha * adjust)
        model_ema = utils.ExponentialMovingAverage(
            model_without_ddp, device=device, decay=1.0 - alpha
        )

    # TODO: need to test resume functionality
    if args.resume:
        checkpoint_pattern = os.path.join(args.output_dir, "model_*.pth")
        checkpoint_files = glob.glob(checkpoint_pattern)
        epochs = [int(f.split("_")[-1].split(".")[0]) for f in checkpoint_files]
        if epochs:
            latest_epoch = max(epochs)
            latest_checkpoint = os.path.join(
                args.output_dir, f"model_{latest_epoch}.pth"
            )
            try:
                checkpoint = torch.load(latest_checkpoint, map_location="cpu")
                model_without_ddp.load_state_dict(checkpoint["model"])
                optimizer.load_state_dict(checkpoint["optimizer"])
                lr_scheduler.load_state_dict(checkpoint["lr_scheduler"])
                args.start_epoch = checkpoint["epoch"] + 1
                if model_ema:
                    model_ema.load_state_dict(checkpoint["model_ema"])
                if scaler:
                    scaler.load_state_dict(checkpoint["scaler"])
                print(f"Resumed training from epoch {args.start_epoch}.")
            except FileNotFoundError:
                print(
                    f"No checkpoint found at {latest_checkpoint}. Starting training from scratch."
                )
                args.start_epoch = 0
        else:
            print("No checkpoint found. Starting training from scratch.")
            args.start_epoch = 0
    else:
        args.start_epoch = 0
        print("Zero-shot evaluation")
        if model_ema:
            evaluate(
                model_ema, criterion, data_loader_test, device=device, log_suffix="EMA"
            )
        else:
            evaluate(model, criterion, data_loader_test, device=device)

    print("Start training")
    start_time = time.time()
    for epoch in range(args.start_epoch, args.epochs):
        if args.distributed:
            train_sampler.set_epoch(epoch)
        train_one_epoch(
            model,
            criterion,
            optimizer,
            data_loader,
            device,
            epoch,
            args,
            model_ema,
            scaler,
        )
        lr_scheduler.step()
        evaluate(model, criterion, data_loader_test, device=device)
        if model_ema:
            evaluate(
                model_ema, criterion, data_loader_test, device=device, log_suffix="EMA"
            )
        if args.output_dir:
            checkpoint = {
                "model": model_without_ddp.state_dict(),
                "optimizer": optimizer.state_dict(),
                "lr_scheduler": lr_scheduler.state_dict(),
                "epoch": epoch,
                "args": args,
            }
            if sparsifier:
                checkpoint["sparsifier"] = sparsifier.state_dict()
            if model_ema:
                checkpoint["model_ema"] = model_ema.state_dict()
            if scaler:
                checkpoint["scaler"] = scaler.state_dict()
            utils.save_on_master(
                checkpoint, os.path.join(args.output_dir, f"model_{epoch}.pth")
            )
            utils.save_on_master(
                checkpoint, os.path.join(args.output_dir, "checkpoint.pth")
            )

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print(f"Training time {total_time_str}")


if __name__ == "__main__":
    args = utils.get_args_parser(train=True).parse_args()
    main(args)
