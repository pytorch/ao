from pathlib import Path
import torch
from tqdm import tqdm
import time
import json
import fire
import logging
import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime
from server import file_bytes_to_image_tensor
from server import show_anns
from server import model_type_to_paths
from server import MODEL_TYPES_TO_MODEL
from compile_export_utils import set_fast
# from compile_export_utils import set_aot_fast
from compile_export_utils import set_furious
from server import masks_to_rle_dict
from server import max_memory_allocated
from server import profiler_runner
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


TASK_TYPES = ["amg", "sps", "mps"]


# TODO: Add task type argument next to model_type
# Task types: amg, mps, sps (largest)
# mps and sps require _meta.json files
# sps picks largest area for prediction
def main(
    checkpoint_path,
    model_type,
    task_type,
    input_paths,
    output_folder,
    points_per_batch=1024,
    output_format="png",
    verbose=False,
    fast=False,
    furious=False,
    overwrite=False,
    baseline=False,
    meta_folder=None,
    export_model="",
    load_exported_model="",
    num_images=None,
):
    start_time = time.time()
    if task_type not in TASK_TYPES:
        raise ValueError(f"Expected task_type to be one of {','.join(TASK_TYPES)}, but got {task_type}")
    if task_type != "amg" and meta_folder is None:
        raise ValueError(f"Task type {task_type} requires a path for --meta-folder")

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
        raise ValueError(f"Expected {output_folder} to be a directory.")
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
    if not overwrite and any(p.exists() for p in output_image_paths):
        raise ValueError(
            "Output image path already exists, but --overwrite was not specified."
        )
    if not overwrite and any(p.exists() for p in output_rle_json_paths):
        raise ValueError(
            "Output image path already exists, but --overwrite was not specified."
        )

    if task_type == "amg":
        meta_paths = len(output_rle_json_paths) * [None]
    else:
        meta_paths = [
            Path(meta_folder)
            / Path(filename.parent)
            / Path(filename.stem + "_meta.json")
            for filename in filenames
        ]
        if any(not p.exists() for p in meta_paths):
            raise ValueError(
                "--meta-folder was specified, but one of the files doesn't exist."
            )

    if baseline:
        from sam2.build_sam import build_sam2
        from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator
        from sam2.sam2_image_predictor import SAM2ImagePredictor
        from sam2.utils.amg import rle_to_mask
        from sam2.utils.amg import mask_to_rle_pytorch
    else:
        from torchao._models.sam2.build_sam import build_sam2
        from torchao._models.sam2.automatic_mask_generator import (
            SAM2AutomaticMaskGenerator,
        )
        from torchao._models.sam2.sam2_image_predictor import SAM2ImagePredictor
        from torchao._models.sam2.utils.amg import rle_to_mask
        from torchao._models.sam2.utils.amg import mask_to_rle_pytorch_2 as mask_to_rle_pytorch
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
    if export_model != "":
        if not Path(output_folder).is_dir():
            raise ValueError(f"Expected {export_model} to be a directory.")
        print(f"Exporting model to {export_model}.")
        from compile_export_utils import export_model as export_model_fn
        export_model_fn(mask_generator,
                        export_model,
                        task_type,
                        furious=furious,
                        fast=fast,
                        batch_size=1,
                        points_per_batch=points_per_batch)
    if load_exported_model == "":
        if furious:
            set_furious(mask_generator)
        if fast:
            set_fast(mask_generator, task_type)
    else:
        from compile_export_utils import load_exported_model as load_exported_model_fn
        load_exported_model_fn(mask_generator,
                               load_exported_model,
                               task_type,
                               furious,
                               fast,
                               batch_size=1,
                               points_per_batch=points_per_batch)

    num_images = len(input_paths) if num_images is None else num_images
    input_paths = input_paths[:num_images]
    for input_path, filename, output_image_path, output_rle_json_path, meta_path in tqdm(
        zip(input_paths, filenames, output_image_paths, output_rle_json_paths, meta_paths),
        total=num_images,
    ):
        if task_type != "amg":
            if verbose:
                timestamped_print(f"Loading meta from {meta_path}")
            with open(meta_path, 'r') as file:
                amg_masks = list(json.load(file).values())
                amg_masks = sorted(amg_masks, key=(lambda x: x['area']), reverse=True)
                # center points for biggest area first.
                center_points = [mask['center_point'] for mask in amg_masks]
                center_points = np.array(center_points)
                center_points_label = np.array(len(center_points) * [1])
                if task_type == "sps":
                    center_points = center_points[:1]
                    center_points_label = center_points_label[:1]

        if baseline:
            input_bytes = bytearray(open(input_path, "rb").read())
            image_tensor = file_bytes_to_image_tensor(input_bytes)
            if verbose:
                timestamped_print(
                    f"Generating mask for image {input_path} of size {tuple(image_tensor.shape)}."
                )
            if task_type == "amg":
                masks = mask_generator.generate(image_tensor)
            elif task_type == "sps":
                mask_generator.predictor.set_image(image_tensor)
                masks, scores, _ = mask_generator.predictor.predict(
                    point_coords=center_points,
                    point_labels=center_points_label,
                    multimask_output=True,
                    return_logits=False,
                )
                masks = torch.from_numpy(masks[np.argmax(scores).item()]).to(torch.bool)
            elif task_type == "mps":
                mask_generator.predictor.set_image(image_tensor)
                masks = []
                for i in range(len(center_points)):
                    mask, score, _ = mask_generator.predictor.predict(
                        point_coords=center_points[i:i+1],
                        point_labels=center_points_label[i:i+1],
                        multimask_output=True,
                        return_logits=False,
                    )
                    mask = torch.from_numpy(mask[np.argmax(score).item()]).to(torch.bool)
                    masks.append(mask)
                masks = torch.stack(masks)
        else:
            if task_type == "amg":
                masks = mask_generator.generate_from_path(input_path)
            elif task_type == "sps":
                from torchvision import io as tio
                img_bytes_tensor = tio.read_file(input_path)
                image_tensor = tio.decode_jpeg(img_bytes_tensor, device='cuda')
                mask_generator.predictor.set_image(image_tensor)
                masks, scores, _ = mask_generator.predictor.predict(
                    point_coords=center_points,
                    point_labels=center_points_label,
                    multimask_output=True,
                    return_logits=False,
                    return_type="torch",
                )
                masks = masks.index_select(0, torch.argmax(scores))[0]
            elif task_type == "mps":
                # NOTE: There are multiple opportunities for batching here
                # Batching of images
                # Batching of prompts
                # First we do batching of prompts
                # Use MapTensor to create pseudobatches of points and labels
                from torchvision import io as tio
                img_bytes_tensor = tio.read_file(input_path)
                image_tensor = tio.decode_jpeg(img_bytes_tensor, device='cuda')
                mask_generator.predictor.set_image(image_tensor)

                center_points_torch = torch.from_numpy(center_points).unsqueeze(1)
                center_points_label_torch = torch.from_numpy(center_points_label).unsqueeze(1)
                from torchao._models.sam2.map_tensor import to_map_tensor
                center_points_torch = to_map_tensor(center_points_torch)
                center_points_label_torch = to_map_tensor(center_points_label_torch)
                masks, scores, _ = mask_generator.predictor.predict(
                    point_coords=center_points_torch,
                    point_labels=center_points_label_torch,
                    multimask_output=True,
                    return_logits=False,
                    return_type="torch",
                )
                # Unwrapping MapTensor
                masks = masks.elems
                scores = scores.elems
                # TODO: This isn't exactly efficient
                masks = torch.stack([mask[i] for (mask, i) in zip(masks.unbind(), torch.argmax(scores, dim=1).tolist())])

                # TODO: NEXT!!
                # TODO: export the model at the end to include recompilations.
                # Could export the predict method and the mask_to_rle_pytorch_2 function
                # I think mask_to_rle_pytorch_2 recompiles

        with torch.autograd.profiler.record_function("mask_to_rle_pytorch"):
            if task_type == "sps":
                masks = mask_to_rle_pytorch(masks.unsqueeze(0))[0]
                masks = [{'segmentation': masks}]
            elif task_type == "mps":
                masks = mask_to_rle_pytorch(masks)
                masks = [{'segmentation': mask} for mask in masks]

        with torch.autograd.profiler.record_function("masks_to_rle_dict"):
            rle_dict = masks_to_rle_dict(masks)

        with torch.autograd.profiler.record_function("json.dumps"):
            if verbose:
                timestamped_print(f"Storing rle under {output_rle_json_path}")
            output_rle_json_path.parent.mkdir(parents=False, exist_ok=True)
            with open(output_rle_json_path, "w") as file:
                file.write(json.dumps(rle_dict, indent=4))
    end_time = time.time()
    total_time = end_time - start_time
    print(f"This took {total_time}s with {len(input_paths) / total_time}img/s or {total_time / len(input_paths) * 1000}ms per image")
    max_memory_allocated()


main.__doc__ = main_docstring()
if __name__ == "__main__":
    # profiler_runner("asdf.json.gz", fire.Fire, main)
    fire.Fire(main)
