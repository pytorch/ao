# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD 3-Clause license found in the
# LICENSE file in the root directory of this source tree.
import json
from datetime import datetime
from io import BytesIO
from pathlib import Path

import fire
import matplotlib.pyplot as plt
import numpy as np
from scipy import ndimage
from server import (
    MODEL_TYPES_TO_MODEL,
    file_bytes_to_image_tensor,
    show_anns,
)
from tqdm import tqdm

from torchao._models.sam2.utils.amg import area_from_rle, rle_to_mask


def timestamped_print(*args, **kwargs):
    # Get the current timestamp
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")
    # Prepend the timestamp to the original print arguments
    print(f"[{timestamp}]", *args, **kwargs)


# From https://github.com/pytorch-labs/segment-anything-fast/blob/e6aadeb86f3ae1f58c3f98e2a91e251716e0f2aa/experiments/data.py
# All credit to vkuzo
def _get_center_point(mask):
    """
    This is a rudimentary version of https://arxiv.org/pdf/2304.02643.pdf,
    section D.1.Point Sampling

    From the paper: "The first point is chosen deterministically as the point
    farthest from the object boundary."

    The code below is an approximation of this.

    First, we try to calculate the center of mass. If it's inside the mask, we
    stop here.

    The centroid may be outside of the mask for some mask shapes. In this case
    we do a slow hack, specifically, we check for the
    minumum of the maximum distance from the boundary in four directions
    (up, right, down, left), and take the point with the maximum of these
    minimums. Note: this is not performant for large masks.

    Returns the center point in (x, y) format
    """

    # try the center of mass, keep it if it's inside the mask
    com_y, com_x = ndimage.center_of_mass(mask)
    com_y, com_x = int(round(com_y, 0)), int(round(com_x, 0))
    if mask[com_y][com_x]:
        return (com_x, com_y)

    # if center of mass didn't work, do the slow manual approximation

    # up, right, down, left
    # TODO(future): approximate better by adding more directions
    distances_to_check_deg = [0, 90, 180, 270]

    global_min_max_distance = float("-inf")
    global_coords = None
    # For now, terminate early to speed up the calculation as long as
    # the point sample is gooe enough. This sacrifices the quality of point
    # sampling for speed. In the future we can make this more accurate.
    DISTANCE_GOOD_ENOUGH_THRESHOLD = 20

    # Note: precalculating the bounding box could be somewhat
    #   helpful, but checked the performance gain and it's not much
    #   so leaving it out to keep the code simple.
    # Note: tried binary search instead of incrementing by one to
    #   travel up/right/left/down, but that does not handle masks
    #   with all shapes properly (there could be multiple boundaries).
    for row_idx in range(mask.shape[0]):
        for col_idx in range(mask.shape[1]):
            cur_point = mask[row_idx, col_idx]

            # skip points inside bounding box but outside mask
            if not cur_point:
                continue

            max_distances = []
            for direction in distances_to_check_deg:
                # TODO(future) binary search instead of brute forcing it if we
                # need a speedup, with a cache it doesn't really matter though
                if direction == 0:
                    # UP
                    cur_row_idx = row_idx

                    while cur_row_idx >= 0 and mask[cur_row_idx, col_idx]:
                        cur_row_idx = cur_row_idx - 1
                    cur_row_idx += 1
                    distance = row_idx - cur_row_idx
                    max_distances.append(distance)

                elif direction == 90:
                    # RIGHT
                    cur_col_idx = col_idx

                    while (
                        cur_col_idx <= mask.shape[1] - 1 and mask[row_idx, cur_col_idx]
                    ):
                        cur_col_idx += 1
                    cur_col_idx -= 1
                    distance = cur_col_idx - col_idx
                    max_distances.append(distance)

                elif direction == 180:
                    # DOWN
                    cur_row_idx = row_idx
                    while (
                        cur_row_idx <= mask.shape[0] - 1 and mask[cur_row_idx, col_idx]
                    ):
                        cur_row_idx = cur_row_idx + 1
                    cur_row_idx -= 1
                    distance = cur_row_idx - row_idx
                    max_distances.append(distance)

                elif direction == 270:
                    # LEFT
                    cur_col_idx = col_idx
                    while cur_col_idx >= 0 and mask[row_idx, cur_col_idx]:
                        cur_col_idx -= 1
                    cur_col_idx += 1
                    distance = col_idx - cur_col_idx
                    max_distances.append(distance)

            min_max_distance = min(max_distances)
            if min_max_distance > global_min_max_distance:
                global_min_max_distance = min_max_distance
                global_coords = (col_idx, row_idx)
            if global_min_max_distance >= DISTANCE_GOOD_ENOUGH_THRESHOLD:
                break

    return global_coords


# TODO: Create prompts
# Get prompts for each mask and prompt for largest mask
# Use those prompts as input for generate data

# Create 3 images for each task type
# amg: all masks without center point
# sps: one mask with center point
# mps: multiple masks with center points


def main_docstring():
    return f"""
    Args:
        checkpoint_path (str): Path to folder containing checkpoints from https://github.com/facebookresearch/sam2?tab=readme-ov-file#download-checkpoints
        model_type (str): Choose from one of {", ".join(MODEL_TYPES_TO_MODEL.keys())}
        input_path (str): Path to input image
        output_path (str): Path to output image
    """


def main(
    checkpoint_path,
    model_type,
    input_paths,
    amg_mask_folder,
    output_folder,
    output_format="png",
    verbose=False,
    fast=False,
    furious=False,
    load_fast="",
    overwrite=False,
    store_image=False,
    baseline=False,
):
    # Input path validation
    input_paths = [
        Path(input_path.strip())
        for input_path in Path(input_paths).read_text().splitlines()
    ]
    # We include parent folder to reduce possible duplicates
    filenames = [
        Path(input_path.parent.name) / Path(input_path.name)
        for input_path in input_paths
    ]
    if len(filenames) != len(set(filenames)):
        raise ValueError("Expected input_paths to have unique filenames.")
    if any(not input_path.is_file() for input_path in input_paths):
        raise ValueError("One of the input paths does not point to a file.")
    if not Path(amg_mask_folder).is_dir():
        raise ValueError(f"Expected {amg_mask_folder} to be a directory.")
    rle_json_paths = [
        Path(amg_mask_folder)
        / Path(filename.parent)
        / Path(filename.stem + "_masks.json")
        for filename in filenames
    ]
    for p in rle_json_paths:
        if not p.exists():
            raise ValueError(f"Expected mask {p} to exist.")

    # Output path validation
    if not Path(output_folder).is_dir():
        raise ValueError(f"Expected {output_folder} to be a directory.")

    output_image_paths = [
        (Path(output_folder) / filename).with_suffix("." + output_format)
        for filename in filenames
    ]
    if not overwrite and any(p.exists() for p in output_image_paths):
        raise ValueError(
            "Output image path already exists, but --overwrite was not specified."
        )

    output_json_paths = [
        Path(output_folder) / Path(filename.parent) / Path(filename.stem + "_meta.json")
        for filename in filenames
    ]
    if not overwrite and any(p.exists() for p in output_json_paths):
        raise ValueError(
            "Output json path already exists, but --overwrite was not specified."
        )

    for (
        input_path,
        filename,
        output_image_path,
        rle_json_path,
        output_json_path,
    ) in tqdm(
        zip(
            input_paths,
            filenames,
            output_image_paths,
            rle_json_paths,
            output_json_paths,
        ),
        total=len(input_paths),
    ):
        input_bytes = bytearray(open(input_path, "rb").read())
        image_tensor = file_bytes_to_image_tensor(input_bytes)
        if verbose:
            timestamped_print(f"Loading rle from {rle_json_path}")
        with open(rle_json_path, "r") as file:
            rle_dict = json.load(file)
            masks = {}
            for key in rle_dict:
                masks[key] = {
                    "segmentation": rle_dict[key],
                    "area": area_from_rle(rle_dict[key]),
                    "center_point": _get_center_point(rle_to_mask(rle_dict[key])),
                }

        if verbose:
            timestamped_print(
                f"Generating mask annotations for input image {filename}."
            )
        plt.figure(
            figsize=(image_tensor.shape[1] / 100.0, image_tensor.shape[0] / 100.0),
            dpi=100,
        )
        plt.imshow(image_tensor)
        # seed for consistent coloring
        # Converts segmentation to binary mask for annotations
        show_anns(list(masks.values()), rle_to_mask, seed=42)
        plt.axis("off")
        plt.tight_layout()

        points = np.array([mask["center_point"] for mask in masks.values()])
        ax = plt.gca()
        marker_size = 375
        ax.scatter(
            points[:, 0],
            points[:, 1],
            color="green",
            marker="*",
            s=marker_size,
            edgecolor="white",
            linewidth=1.25,
        )

        buf = BytesIO()
        plt.savefig(buf, format=output_format)
        buf.seek(0)
        output_bytes = buf.getvalue()
        output_image_path.parent.mkdir(parents=False, exist_ok=True)

        if verbose:
            timestamped_print(f"Storing result image under {output_image_path}")
        with open(output_image_path, "wb") as file:
            file.write(output_bytes)

        # Back to RLE representation
        for key in masks:
            masks[key]["segmentation"] = rle_dict[key]

        if verbose:
            timestamped_print(f"Storing meta under {output_json_path}")

        with open(output_json_path, "w") as file:
            file.write(json.dumps(masks, indent=4))

        plt.close()


main.__doc__ = main_docstring()
if __name__ == "__main__":
    fire.Fire(main)
