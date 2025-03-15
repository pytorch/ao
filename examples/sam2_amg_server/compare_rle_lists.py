# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD 3-Clause license found in the
# LICENSE file in the root directory of this source tree.
import json
from pathlib import Path
from typing import Any, Dict

import fire
import numpy as np
import torch


# from torchao._models.sam2.utils.amg import rle_to_mask
def rle_to_mask(rle: Dict[str, Any]) -> np.ndarray:
    """Compute a binary mask from an uncompressed RLE."""
    h, w = rle["size"]
    mask = np.empty(h * w, dtype=bool)
    idx = 0
    parity = False
    for count in rle["counts"]:
        mask[idx : idx + count] = parity
        idx += count
        parity ^= True
    mask = mask.reshape(w, h)
    return mask.transpose()  # Put in C order


"""
Script to calculate mIoU given two lists of rles from upload_rle endpoint
of server.
"""


def iou(mask1, mask2):
    assert mask1.dim() == 2
    assert mask2.dim() == 2
    intersection = torch.logical_and(mask1, mask2)
    union = torch.logical_or(mask1, mask2)
    return intersection.sum(dim=(-1, -2)) / union.sum(dim=(-1, -2))


def area_from_rle(rle: Dict[str, Any]) -> int:
    return sum(rle["counts"][1::2])


def compare_masks(masks, ref_masks, order_by_area=False, verbose=False):
    v0_areas = []
    v1_areas = []
    v0_masks = []
    v1_masks = []
    for k0 in ref_masks:
        assert k0 in masks, f"Expected {k0} to be in return data"
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
    for i, ((v0_mask, _), (v1_mask, _)) in enumerate(zip(v0_masks, v1_masks)):
        miou_sum += iou(v0_mask, v1_mask)
        miou_count += 1
        equal_count += torch.equal(v0_mask, v1_mask)
        if verbose:
            # If sorted we don't map back to the original key
            # TODO: Could recover the indices for this
            if order_by_area:
                print(f"IoU is {iou(v0_mask, v1_mask)}")
            else:
                print(f"mask {i} IoU is iou(v0_mask, v1_mask)")

    return float((miou_sum / miou_count).item()), equal_count


def compare_masks_str(str0, str1, strict):
    masks0 = json.loads(str0)
    masks1 = json.loads(str1)
    if masks0.keys() != masks1.keys():
        if strict:
            return None, None, True

    # TODO: We might not want to order_by_area when comparing
    # masks from specific input points.
    m, e = compare_masks(masks0, masks1, order_by_area=True)
    return m, e, False


def compare(path0, path1, strict=False, compare_folders=False):
    # path0 are candidates and path1 the ground truth
    fail_count = 0
    miou_sum = 0.0
    miou_count = 0
    if compare_folders:
        path0, path1 = Path(path0), Path(path1)
        assert path0.is_dir()
        assert path1.is_dir()
        mask_files0 = [f.relative_to(path0) for f in list(path0.rglob("*.json"))]
        mask_files1 = [f.relative_to(path1) for f in list(path1.rglob("*.json"))]
        assert all(m0 == m1 for (m0, m1) in zip(mask_files0, mask_files1))
        for m0, m1 in zip(mask_files0, mask_files1):
            with open(path0 / m0, "r") as f0, open(path1 / m1, "r") as f1:
                m, e, fail = compare_masks_str(f0.read(), f1.read(), strict)
                if fail:
                    fail_count += 1
                else:
                    miou_sum += m
                    miou_count += 1

    else:
        with open(path0, "r") as f0, open(path1, "r") as f1:
            for line0, line1 in zip(f0, f1):
                m, e, fail = compare_masks_str(line0, line1, strict)
                if fail:
                    fail_count += 1
                else:
                    miou_sum += m
                    miou_count += 1

    return miou_count, miou_sum, fail_count


def main(path0, path1, strict=False, compare_folders=False):
    miou_count, miou_sum, fail_count = compare(
        path0, path1, strict=strict, compare_folders=compare_folders
    )
    print(f"fail_count: {fail_count} mIoU: {miou_sum / miou_count}")


if __name__ == "__main__":
    fire.Fire(main)
