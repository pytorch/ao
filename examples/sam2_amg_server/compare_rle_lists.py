import fire
import torch
import json
from torchao._models.sam2.utils.amg import rle_to_mask

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


def compare_masks(masks, ref_masks, order_by_area=False, verbose=False):
    v0_areas = []
    v1_areas = []
    v0_masks = []
    v1_masks = []
    for k0 in ref_masks:
        assert k0 in masks, f"Expected {k0} to be in return data"
        from torchao._models.sam2.utils.amg import area_from_rle
        v0_area = area_from_rle(ref_masks[k0])
        v1_area = area_from_rle(masks[k0])
        v0_areas.append(v0_area)
        v1_areas.append(v1_area)
        if (v0_area != v1_area) and verbose:
            print(f"v0 area {v0_area} doesn't match v1 area {v1_area}")
        v0_mask = torch.from_numpy(rle_to_mask(ref_masks[k0]))
        v1_mask = torch.from_numpy(rle_to_mask(masks[k0]))
        v0_masks.append((v0_mask, v0_area))
        v1_masks.append((v1_mask, v1_area))

    if order_by_area:
        v0_masks = sorted(v0_masks, key=(lambda x: x[1]), reverse=True)
        v1_masks = sorted(v1_masks, key=(lambda x: x[1]), reverse=True)
    miou_sum = 0.0
    miou_count = 0.0
    equal_count = 0
    for ((v0_mask, _), (v1_mask, _)) in zip(v0_masks, v1_masks):
        miou_sum += iou(v0_mask, v1_mask)
        miou_count += 1
        equal_count += torch.equal(v0_mask, v1_mask)
        if verbose:
            print(f"Masks don't match for key {k0}. IoU is {iou(v0_mask, v1_mask)}")

    return miou_sum / miou_count, equal_count


def main(path0, path1, strict=False):
    # path0 are candidates and path1 the ground truth
    fail_count = 0
    miou_sum = 0.0
    miou_count = 0
    with open(path0, 'r') as f0, open(path1, 'r') as f1:
        for line0, line1 in zip(f0, f1):
            masks0 = json.loads(line0)
            masks1 = json.loads(line1)
            if masks0.keys() != masks1.keys():
                if strict:
                    fail_count += 1
                    continue

            m, e = compare_masks(masks0, masks1, order_by_area=True)
            miou_sum += m
            miou_count += 1

    print(f"fail_count: {fail_count} mIoU: {miou_sum / miou_count}")


if __name__ == "__main__":
    fire.Fire(main)
