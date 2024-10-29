import fire
import torch
import json
from sam2.utils.amg import rle_to_mask

"""
Script to calculate mIoU given two lists of rles from upload_rle endpoint
of server.
"""


def iou(mask1, mask2):
    assert mask1.dim() == 2
    assert mask2.dim() == 2
    intersection = torch.logical_and(mask1, mask2)
    union = torch.logical_or(mask1, mask2)
    return (intersection.sum(dim=(-1, -2)) / union.sum(dim=(-1, -2)))


def main(path0, path1):
    fail_count = 0
    miou_sum = 0.0
    miou_count = 0
    with open(path0, 'r') as f0, open(path1, 'r') as f1:
        for line0, line1 in zip(f0, f1):
            masks0 = json.loads(line0)
            masks1 = json.loads(line1)
            if masks0.keys() != masks1.keys():
                fail_count += 1
                continue
            for mask0, mask1 in zip(masks0.values(), masks1.values()):
                mask0 = torch.from_numpy(rle_to_mask(mask0))
                mask1 = torch.from_numpy(rle_to_mask(mask1))
                miou_sum += iou(mask0, mask1).item()
                miou_count += 1

    print(f"fail_count: {fail_count} mIoU: {miou_sum / miou_count}")


if __name__ == "__main__":
    fire.Fire(main)
