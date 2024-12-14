from pathlib import Path
from tqdm import tqdm
import json
import fire
import logging
import matplotlib.pyplot as plt
from datetime import datetime
from server import file_bytes_to_image_tensor
from server import show_anns
from server import model_type_to_paths
from server import MODEL_TYPES_TO_MODEL
from server import set_fast
from server import set_aot_fast
from server import load_aot_fast
from server import set_furious
from server import masks_to_rle_dict
from io import BytesIO


def timestamped_print(*args, **kwargs):
    # Get the current timestamp
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")
    # Prepend the timestamp to the original print arguments
    print(f"[{timestamp}]", *args, **kwargs)


# TODO: Generate baseline data
# Do this based on a file with ~1000 paths
# AMG: Automatic mask generation
# for each image: prompt, RLE Masks, annotated image with mask overlay
# SPS: Single point segmentation
# for each image: take largest AMG mask, find center point for prompt, RLE Mask, annotated image with prompt and mask overlay
# MPS: Multi point segmentation
# for each image: take AMG mask, find all center points for prompte, RLE Masks, annotated image with prompts from AMG and mask overlay

# If done right this could also build the basis for the benchmark script
# The first step is running AMG and then the subsequent steps are based on prompts taken from the AMG output
# The modified variants compare RLE data using a separate script.
# - We only need to run baseline, AO, AO + Fast, AO + Fast + Furious

# Create separate script to
# - produce prompts from AMG masks
# - calculate mIoU from output masks
# - annotate images with rle json


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
    output_folder,
    points_per_batch=1024,
    output_format="png",
    verbose=False,
    fast=False,
    furious=False,
    load_fast="",
    overwrite=False,
    store_image=False,
    baseline=False,
):
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
    if not Path(output_folder).is_dir():
        raise ValueError("Expected {output_folder} to be a directory.")
    output_image_paths = [
        (Path(output_folder) / filename).with_suffix("." + output_format)
        for filename in filenames
    ]
    output_rle_json_paths = [
        Path(output_folder)
        / Path(filename.parent)
        / Path(filename.stem + "_masks.json")
        for filename in filenames
    ]
    if not overwrite and any(
        output_image_path.exists() for output_image_path in output_image_paths
    ):
        raise ValueError(
            "Output image path already exists, but --overwrite was not specified."
        )
    if not overwrite and any(
        output_rle_json_path.exists() for output_rle_json_path in output_rle_json_paths
    ):
        raise ValueError(
            "Output image path already exists, but --overwrite was not specified."
        )

    if baseline:
        from sam2.build_sam import build_sam2
        from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator
        from sam2.utils.amg import rle_to_mask
    else:
        from torchao._models.sam2.build_sam import build_sam2
        from torchao._models.sam2.automatic_mask_generator import (
            SAM2AutomaticMaskGenerator,
        )
        from torchao._models.sam2.utils.amg import rle_to_mask
    device = "cuda"
    sam2_checkpoint, model_cfg = model_type_to_paths(checkpoint_path, model_type)
    if verbose:
        timestamped_print(f"Loading model {sam2_checkpoint} with config {model_cfg}")
    sam2 = build_sam2(
        model_cfg, sam2_checkpoint, device=device, apply_postprocessing=False
    )
    mask_generator = SAM2AutomaticMaskGenerator(
        sam2, points_per_batch=points_per_batch, output_mode="uncompressed_rle"
    )
    if furious:
        set_furious(mask_generator)
    if load_fast:
        load_aot_fast(mask_generator, load_fast)
    if fast:
        set_fast(mask_generator, load_fast)

    for input_path, filename, output_image_path, output_rle_json_path in tqdm(
        zip(input_paths, filenames, output_image_paths, output_rle_json_paths)
    ):
        input_bytes = bytearray(open(input_path, "rb").read())
        image_tensor = file_bytes_to_image_tensor(input_bytes)
        if verbose:
            timestamped_print(
                f"Loaded image {input_path} of size {tuple(image_tensor.shape)} and generating mask."
            )
        masks = mask_generator.generate(image_tensor)
        rle_dict = masks_to_rle_dict(masks)

        if verbose:
            timestamped_print(f"Storing rle under {output_rle_json_path}")
        output_rle_json_path.parent.mkdir(parents=False, exist_ok=True)
        with open(output_rle_json_path, "w") as file:
            file.write(json.dumps(rle_dict, indent=4))

        if not store_image:
            continue

        if verbose:
            timestamped_print(
                f"Generating mask annotations for input image {filename}."
            )
        plt.figure(
            figsize=(image_tensor.shape[1] / 100.0, image_tensor.shape[0] / 100.0),
            dpi=100,
        )
        plt.imshow(image_tensor)
        show_anns(masks, rle_to_mask)
        plt.axis("off")
        plt.tight_layout()
        buf = BytesIO()
        plt.savefig(buf, format=output_format)
        buf.seek(0)
        output_bytes = buf.getvalue()
        output_image_path.parent.mkdir(parents=False, exist_ok=True)

        if verbose:
            timestamped_print(f"Storing result image under {output_image_path}")
        with open(output_image_path, "wb") as file:
            file.write(output_bytes)
        plt.close()


main.__doc__ = main_docstring()
if __name__ == "__main__":
    fire.Fire(main)
