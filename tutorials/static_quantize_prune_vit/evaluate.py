import os
import sys
import warnings
import hashlib

import torch
import torch.utils.data
import torchvision
from torch import nn
from torchvision.transforms.functional import InterpolationMode

def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.inference_mode():
        maxk = max(topk)
        batch_size = target.size(0)
        if target.ndim == 2:
            target = target.max(dim=1)[1]

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target[None])

        res = []
        for k in topk:
            correct_k = correct[:k].flatten().sum(dtype=torch.float32)
            res.append(correct_k * (100.0 / batch_size))
        return res

def load_data(valdir, preprocessing):
    # Data loading code
    print(f"Loading validation data from {valdir}")
    # for META internal
    dataset_test = torchvision.datasets.ImageFolder(
        valdir,
        preprocessing,
    )
    # for OSS
    # dataset_test = torchvision.datasets.ImageNet(
    #     valdir,
    #     split='val',
    #     transform=preprocessing
    # )

    print(f"Number of validation images: {len(dataset_test)}")

    test_sampler = torch.utils.data.SequentialSampler(dataset_test)
    return dataset_test, test_sampler

def evaluate(model, preprocessing, val_dir, batch_size, print_freq=100, log_suffix="", verbose=True, limit=None):
    device = "cuda"
    dataset_test, test_sampler = load_data(val_dir, preprocessing)
    data_loader = torch.utils.data.DataLoader(
        dataset_test, batch_size=batch_size, sampler=test_sampler, num_workers=16, pin_memory=True,
        drop_last=True, # Change in batch size triggers recompilation, so we drop it.
    )
    model.eval()
    header = f"Test: {log_suffix}"

    num_processed_samples = 0
    count = 0
    acc1_total = 0.0
    acc5_total = 0.0
    with torch.no_grad():
        for i, (image, target) in enumerate(data_loader):
            image = image.to(device, non_blocking=True, dtype=torch.bfloat16)
            target = target.to(device, non_blocking=True)
            output = model(image)

            acc1, acc5 = accuracy(output, target, topk=(1, 5))
            acc1_total += acc1
            acc5_total += acc5
            count += 1
            # FIXME need to take into account that the datasets
            # could have been padded in distributed setup
            batch_size = image.shape[0]
            num_processed_samples += batch_size
            if verbose:
                print(f"\racc1: {acc1_total / count:5.2f} acc5: {acc5_total / count:5.2f} i: {i+1}/{len(data_loader)}", end='')
            if limit is not None and i > limit:
                break
    return acc1_total / count, acc5_total / count
