# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import math
from copy import deepcopy
from dataclasses import dataclass
from itertools import product
from typing import Any, Dict, Generator, ItemsView, List, Tuple

import numpy as np
import torch


@dataclass
class RLEData:
    alt_lens_nt: torch.Tensor
    counts_init: torch.Tensor
    b: int
    h: int
    w: int

    def __len__(self):
        return self.b


def nt_index_select_dim_0(nt, index):
    values = nt.values()
    offsets = nt.offsets()
    lengths = offsets.diff()
    offsets_index = offsets[index].cpu()
    lengths_index = lengths[index].cpu()
    indices = [o + torch.arange(l) for (o, l) in zip(offsets_index, lengths_index)]
    indices = torch.cat(indices).to(values.device)
    values_index = torch.index_select(values, 0, indices)
    lengths_index = lengths_index.to(values.device)
    return torch.nested.nested_tensor_from_jagged(values_index, lengths=lengths_index)


def nt_cat_dim_0(nts: List[torch.Tensor]):
    all_values = []
    all_lengths = []
    for nt in nts:
        all_values.append(nt.values())
        all_lengths.append(nt.offsets().diff())
    new_values = torch.cat(all_values)
    new_lengths = torch.cat(all_lengths)
    return torch.nested.nested_tensor_from_jagged(new_values, lengths=new_lengths)


# Very lightly adapted from https://github.com/facebookresearch/segment-anything/blob/main/segment_anything/utils/amg.py
class MaskData:
    """
    A structure for storing masks and their related data in batched format.
    Implements basic filtering and concatenation.
    """

    def __init__(self, **kwargs) -> None:
        for v in kwargs.values():
            assert isinstance(
                v, (list, np.ndarray, torch.Tensor, RLEData)
            ), "MaskData only supports list, numpy arrays, and torch tensors."
        self._stats = dict(**kwargs)

    def __setitem__(self, key: str, item: Any) -> None:
        assert isinstance(
            item, (list, np.ndarray, torch.Tensor, RLEData)
        ), "MaskData only supports list, numpy arrays, and torch tensors."
        self._stats[key] = item

    def __delitem__(self, key: str) -> None:
        del self._stats[key]

    def __getitem__(self, key: str) -> Any:
        return self._stats[key]

    def items(self) -> ItemsView[str, Any]:
        return self._stats.items()

    def filter(self, keep: torch.Tensor) -> None:
        for k, v in self._stats.items():
            if v is None:
                self._stats[k] = None
            elif isinstance(v, torch.Tensor):
                # self._stats[k] = v[torch.as_tensor(keep, device=v.device)]
                self._stats[k] = v[keep]
            elif isinstance(v, np.ndarray):
                self._stats[k] = v[keep.detach().cpu().numpy()]
            elif isinstance(v, list) and keep.dtype == torch.bool:
                self._stats[k] = [a for i, a in enumerate(v) if keep[i]]
            elif isinstance(v, list):
                self._stats[k] = [v[i] for i in keep]
            elif isinstance(v, RLEData):
                new_alt_lens_nt = nt_index_select_dim_0(v.alt_lens_nt, keep)
                self._stats[k] = RLEData(
                    alt_lens_nt=new_alt_lens_nt,
                    counts_init=v.counts_init[keep],
                    b=new_alt_lens_nt.size(0),
                    h=v.h,
                    w=v.w,
                )
            else:
                raise TypeError(f"MaskData key {k} has an unsupported type {type(v)}.")

    def cat(self, new_stats: "MaskData") -> None:
        for k, v in new_stats.items():
            if k not in self._stats or self._stats[k] is None:
                self._stats[k] = deepcopy(v)
            elif isinstance(v, torch.Tensor):
                self._stats[k] = torch.cat([self._stats[k], v], dim=0)
            elif isinstance(v, np.ndarray):
                self._stats[k] = np.concatenate([self._stats[k], v], axis=0)
            elif isinstance(v, list):
                self._stats[k] = self._stats[k] + deepcopy(v)
            elif isinstance(v, RLEData):
                assert self._stats[k].h == v.h
                assert self._stats[k].w == v.w
                self._stats[k] = RLEData(
                    alt_lens_nt=nt_cat_dim_0(
                        [self._stats[k].alt_lens_nt, v.alt_lens_nt]
                    ),
                    counts_init=torch.cat([self._stats[k].counts_init, v.counts_init]),
                    b=self._stats[k].b + v.b,
                    h=v.h,
                    w=v.w,
                )
            else:
                raise TypeError(f"MaskData key {k} has an unsupported type {type(v)}.")

    def to_numpy(self) -> None:
        for k, v in self._stats.items():
            if isinstance(v, torch.Tensor):
                self._stats[k] = v.float().detach().cpu().numpy()


def is_box_near_crop_edge_torch(
    boxes: torch.Tensor,
    crop_box: List[int],
    crop_box_torch: torch.Tensor,
    orig_box_torch: torch.Tensor,
    atol: float = 20.0,
) -> torch.Tensor:
    """Filter masks at the edge of a crop, but not at the edge of the original image."""
    boxes = uncrop_boxes_xyxy(boxes, crop_box).float()
    near_crop_edge = torch.isclose(boxes, crop_box_torch[None, :], atol=atol, rtol=0)
    near_image_edge = torch.isclose(boxes, orig_box_torch[None, :], atol=atol, rtol=0)
    near_crop_edge = torch.logical_and(near_crop_edge, ~near_image_edge)
    return torch.any(near_crop_edge, dim=1)


def is_box_near_crop_edge(
    boxes: torch.Tensor, crop_box: List[int], orig_box: List[int], atol: float = 20.0
) -> torch.Tensor:
    crop_box_torch = torch.as_tensor(crop_box, dtype=torch.float, device=boxes.device)
    orig_box_torch = torch.as_tensor(orig_box, dtype=torch.float, device=boxes.device)
    return is_box_near_crop_edge_torch(
        boxes, crop_box, crop_box_torch, orig_box_torch, atol
    )


def box_xyxy_to_xywh(box_xyxy: torch.Tensor) -> torch.Tensor:
    box_xywh = deepcopy(box_xyxy)
    box_xywh[2] = box_xywh[2] - box_xywh[0]
    box_xywh[3] = box_xywh[3] - box_xywh[1]
    return box_xywh


def batch_iterator(batch_size: int, *args) -> Generator[List[Any], None, None]:
    assert len(args) > 0 and all(
        len(a) == len(args[0]) for a in args
    ), "Batched iteration must have inputs of all the same size."
    n_batches = len(args[0]) // batch_size + int(len(args[0]) % batch_size != 0)
    for b in range(n_batches):
        yield [arg[b * batch_size : (b + 1) * batch_size] for arg in args]


def mask_to_rle_pytorch(tensor: torch.Tensor) -> List[Dict[str, Any]]:
    """
    Encodes masks to an uncompressed RLE, in the format expected by
    pycoco tools.
    """
    # Put in fortran order and flatten h,w
    b, h, w = tensor.shape
    tensor = tensor.permute(0, 2, 1).flatten(1)

    # Compute change indices
    diff = tensor[:, 1:] ^ tensor[:, :-1]
    change_indices = diff.nonzero()

    # Encode run length
    out = []
    for i in range(b):
        cur_idxs = change_indices[change_indices[:, 0] == i, 1]
        cur_idxs = torch.cat(
            [
                torch.tensor([0], dtype=cur_idxs.dtype, device=cur_idxs.device),
                cur_idxs + 1,
                torch.tensor([h * w], dtype=cur_idxs.dtype, device=cur_idxs.device),
            ]
        )
        btw_idxs = cur_idxs[1:] - cur_idxs[:-1]
        counts = [] if tensor[i, 0] == 0 else [0]
        counts.extend(btw_idxs.detach().cpu().tolist())
        out.append({"size": [h, w], "counts": counts})
    return out


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


# @torch.compile(fullgraph=True, dynamic=True)
def _mask_to_rle_pytorch_2_0_0(tensor: torch.Tensor) -> (torch.Tensor, torch.Tensor):
    """
    Encodes masks to an uncompressed RLE, in the format expected by
    pycoco tools.
    """
    # Put in fortran order and flatten h,w
    b, h, w = tensor.shape
    tensor = tensor.permute(0, 2, 1).flatten(1)

    # Compute change indices
    diff = tensor[:, 1:] ^ tensor[:, :-1]
    # a = torch.tensor([[True]])
    a = torch.ones((1, 1), dtype=bool, device=diff.device)
    # if diff.is_cuda:
    #     a = a.pin_memory().cuda()
    #     # a = a.to(diff.device)
    a = a.expand_as(diff.narrow(1, 0, 1))
    diff = torch.cat([a, diff, a], dim=1)
    return diff


# @torch.compile(fullgraph=True, dynamic=True)
def _mask_to_rle_pytorch_2_0_1(
    tensor: torch.Tensor, diff: torch.Tensor, change_indices: torch.Tensor
) -> (torch.Tensor, torch.Tensor):
    tensor = tensor.permute(0, 2, 1).flatten(1)

    alt_lens = diff.sum(dim=1)

    all_cur_idx = change_indices[:, 1]
    if all_cur_idx.numel() == 0:
        all_cur_idx_0 = all_cur_idx
        all_cur_idx_1 = all_cur_idx
    else:
        all_cur_idx_0 = all_cur_idx.narrow(0, 1, all_cur_idx.size(0) - 1)
        all_cur_idx_1 = all_cur_idx.narrow(0, 0, 1)
    all_btw_idx = torch.cat([all_cur_idx_0, all_cur_idx_1])
    all_btw_idx = all_btw_idx - all_cur_idx

    alt_lens_nt = torch.nested.nested_tensor_from_jagged(all_btw_idx, lengths=alt_lens)
    # Encode run length
    counts_init = tensor[:, 0] == 0
    return alt_lens_nt, counts_init


def _mask_to_rle_pytorch_2_0(tensor: torch.Tensor) -> RLEData:
    b, h, w = tensor.shape
    with torch.autograd.profiler.record_function(
        "mask_to_rle_pytorch_2: _mask_to_rle_pytorch_2_0_0"
    ):
        diff = _mask_to_rle_pytorch_2_0_0(tensor)
    with torch.autograd.profiler.record_function("mask_to_rle_pytorch_2: nonzero"):
        # NOTE: While we could operate on less chunks, a set number of chunks prevents recompilations
        # if diff.numel() > 2147483646:
        #     num_chunks = (diff.numel() + 2147483646) // 2147483646
        #     change_indices = torch.cat([d.nonzero() for d in diff.chunk(num_chunks)])
        # else:
        #     change_indices = diff.nonzero()
        num_chunks = 8
        assert num_chunks >= (
            (diff.numel() + 2147483646) // 2147483646
        ), "Needed more chunks than expected."
        change_indices = torch.cat([d.nonzero() for d in diff.chunk(num_chunks)])
    with torch.autograd.profiler.record_function(
        "mask_to_rle_pytorch_2: _mask_to_rle_pytorch_2_0_1"
    ):
        alt_lens_nt, counts_init = _mask_to_rle_pytorch_2_0_1(
            tensor, diff, change_indices
        )
    return RLEData(alt_lens_nt=alt_lens_nt, counts_init=counts_init, b=b, h=h, w=w)


def _mask_to_rle_pytorch_2_1(rle_data: RLEData):
    with torch.autograd.profiler.record_function(
        "mask_to_rle_pytorch_2: Encode run length"
    ):
        out = []
        alt_lens = rle_data.alt_lens_nt.offsets().diff()
        all_btw_idx = rle_data.alt_lens_nt.values()
        alt_lens = alt_lens.tolist()
        all_btw_idx = all_btw_idx.tolist()
        counts_init = rle_data.counts_init.tolist()
        offset = 0
        for i, ci in zip(range(rle_data.b), counts_init):
            btw_idxs = all_btw_idx[offset : offset + alt_lens[i]][:-1]
            offset += alt_lens[i]
            counts = [] if ci else [0]
            counts.extend(btw_idxs)
            out.append({"size": [rle_data.h, rle_data.w], "counts": counts})

    return out


def mask_to_rle_pytorch_2(tensor: torch.Tensor) -> List[Dict[str, Any]]:
    with torch.autograd.profiler.record_function("mask_to_rle_pytorch_2"):
        return _mask_to_rle_pytorch_2_1(_mask_to_rle_pytorch_2_0(tensor))


def area_from_rle(rle: Dict[str, Any]) -> int:
    return sum(rle["counts"][1::2])


# TODO: Turn this on if you can mitigate recompiles!
# @torch.compile(fullgraph=True, dynamic=True)
def calculate_stability_score(
    masks: torch.Tensor, mask_threshold: float, threshold_offset: float
) -> torch.Tensor:
    """
    Computes the stability score for a batch of masks. The stability
    score is the IoU between the binary masks obtained by thresholding
    the predicted mask logits at high and low values.
    """
    # One mask is always contained inside the other.
    # Save memory by preventing unnecessary cast to torch.int64
    intersections = (
        (masks > (mask_threshold + threshold_offset))
        .sum(-1, dtype=torch.int16)
        .sum(-1, dtype=torch.int32)
    )
    unions = (
        (masks > (mask_threshold - threshold_offset))
        .sum(-1, dtype=torch.int16)
        .sum(-1, dtype=torch.int32)
    )
    return intersections / unions


def build_point_grid(n_per_side: int) -> np.ndarray:
    """Generates a 2D grid of points evenly spaced in [0,1]x[0,1]."""
    offset = 1 / (2 * n_per_side)
    points_one_side = np.linspace(offset, 1 - offset, n_per_side)
    points_x = np.tile(points_one_side[None, :], (n_per_side, 1))
    points_y = np.tile(points_one_side[:, None], (1, n_per_side))
    points = np.stack([points_x, points_y], axis=-1).reshape(-1, 2)
    return points


def build_all_layer_point_grids(
    n_per_side: int, n_layers: int, scale_per_layer: int
) -> List[np.ndarray]:
    """Generates point grids for all crop layers."""
    points_by_layer = []
    for i in range(n_layers + 1):
        n_points = int(n_per_side / (scale_per_layer**i))
        points_by_layer.append(build_point_grid(n_points))
    return points_by_layer


def generate_crop_boxes(
    im_size: Tuple[int, ...], n_layers: int, overlap_ratio: float
) -> Tuple[List[List[int]], List[int]]:
    """
    Generates a list of crop boxes of different sizes. Each layer
    has (2**i)**2 boxes for the ith layer.
    """
    crop_boxes, layer_idxs = [], []
    im_h, im_w = im_size
    short_side = min(im_h, im_w)

    # Original image
    crop_boxes.append([0, 0, im_w, im_h])
    layer_idxs.append(0)

    def crop_len(orig_len, n_crops, overlap):
        return int(math.ceil((overlap * (n_crops - 1) + orig_len) / n_crops))

    for i_layer in range(n_layers):
        n_crops_per_side = 2 ** (i_layer + 1)
        overlap = int(overlap_ratio * short_side * (2 / n_crops_per_side))

        crop_w = crop_len(im_w, n_crops_per_side, overlap)
        crop_h = crop_len(im_h, n_crops_per_side, overlap)

        crop_box_x0 = [int((crop_w - overlap) * i) for i in range(n_crops_per_side)]
        crop_box_y0 = [int((crop_h - overlap) * i) for i in range(n_crops_per_side)]

        # Crops in XYWH format
        for x0, y0 in product(crop_box_x0, crop_box_y0):
            box = [x0, y0, min(x0 + crop_w, im_w), min(y0 + crop_h, im_h)]
            crop_boxes.append(box)
            layer_idxs.append(i_layer + 1)

    return crop_boxes, layer_idxs


def uncrop_boxes_xyxy(boxes: torch.Tensor, crop_box: List[int]) -> torch.Tensor:
    x0, y0, _, _ = crop_box
    offset = torch.tensor([[x0, y0, x0, y0]]).pin_memory()
    offset = offset.to(device=boxes.device, non_blocking=True)
    # Check if boxes has a channel dimension
    if len(boxes.shape) == 3:
        offset = offset.unsqueeze(1)
    return boxes + offset


def uncrop_points(points: torch.Tensor, crop_box: List[int]) -> torch.Tensor:
    x0, y0, _, _ = crop_box
    offset = torch.tensor([[x0, y0]]).pin_memory()
    offset = offset.to(device=points.device, non_blocking=True)
    # Check if points has a channel dimension
    if len(points.shape) == 3:
        offset = offset.unsqueeze(1)
    return points + offset


def uncrop_masks(
    masks: torch.Tensor, crop_box: List[int], orig_h: int, orig_w: int
) -> torch.Tensor:
    x0, y0, x1, y1 = crop_box
    if x0 == 0 and y0 == 0 and x1 == orig_w and y1 == orig_h:
        return masks
    # Coordinate transform masks
    pad_x, pad_y = orig_w - (x1 - x0), orig_h - (y1 - y0)
    pad = (x0, pad_x - x0, y0, pad_y - y0)
    return torch.nn.functional.pad(masks, pad, value=0)


def remove_small_regions(
    mask: np.ndarray, area_thresh: float, mode: str
) -> Tuple[np.ndarray, bool]:
    """
    Removes small disconnected regions and holes in a mask. Returns the
    mask and an indicator of if the mask has been modified.
    """
    import cv2  # type: ignore

    assert mode in ["holes", "islands"]
    correct_holes = mode == "holes"
    working_mask = (correct_holes ^ mask).astype(np.uint8)
    n_labels, regions, stats, _ = cv2.connectedComponentsWithStats(working_mask, 8)
    sizes = stats[:, -1][1:]  # Row 0 is background label
    small_regions = [i + 1 for i, s in enumerate(sizes) if s < area_thresh]
    if len(small_regions) == 0:
        return mask, False
    fill_labels = [0] + small_regions
    if not correct_holes:
        fill_labels = [i for i in range(n_labels) if i not in fill_labels]
        # If every region is below threshold, keep largest
        if len(fill_labels) == 0:
            fill_labels = [int(np.argmax(sizes)) + 1]
    mask = np.isin(regions, fill_labels)
    return mask, True


def coco_encode_rle(uncompressed_rle: Dict[str, Any]) -> Dict[str, Any]:
    from pycocotools import mask as mask_utils  # type: ignore

    h, w = uncompressed_rle["size"]
    rle = mask_utils.frPyObjects(uncompressed_rle, h, w)
    rle["counts"] = rle["counts"].decode("utf-8")  # Necessary to serialize with json
    return rle


# TODO: Turn this on if you can mitigate recompiles!
# @torch.compile(fullgraph=True, dynamic=True)
def batched_mask_to_box(masks: torch.Tensor) -> torch.Tensor:
    """
    Calculates boxes in XYXY format around masks. Return [0,0,0,0] for
    an empty mask. For input shape C1xC2x...xHxW, the output shape is C1xC2x...x4.
    """
    # # torch.max below raises an error on empty inputs, just skip in this case
    # if torch.numel(masks) == 0:
    #     return torch.zeros(*masks.shape[:-2], 4, device=masks.device)

    # Normalize shape to CxHxW
    shape = masks.shape
    h, w = shape[-2:]
    if len(shape) > 2:
        masks = masks.flatten(0, -3)
    else:
        masks = masks.unsqueeze(0)

    # Get top and bottom edges
    in_height, _ = torch.max(masks, dim=-1)
    in_height_coords = in_height * torch.arange(h, device=in_height.device)[None, :]
    bottom_edges, _ = torch.max(in_height_coords, dim=-1)
    in_height_coords = in_height_coords + h * (~in_height)
    top_edges, _ = torch.min(in_height_coords, dim=-1)

    # Get left and right edges
    in_width, _ = torch.max(masks, dim=-2)
    in_width_coords = in_width * torch.arange(w, device=in_width.device)[None, :]
    right_edges, _ = torch.max(in_width_coords, dim=-1)
    in_width_coords = in_width_coords + w * (~in_width)
    left_edges, _ = torch.min(in_width_coords, dim=-1)

    # If the mask is empty the right edge will be to the left of the left edge.
    # Replace these boxes with [0, 0, 0, 0]
    empty_filter = (right_edges < left_edges) | (bottom_edges < top_edges)
    out = torch.stack([left_edges, top_edges, right_edges, bottom_edges], dim=-1)
    out = out * (~empty_filter).unsqueeze(-1)

    # Return to original shape
    if len(shape) > 2:
        out = out.reshape(*shape[:-2], 4)
    else:
        out = out[0]

    return out
