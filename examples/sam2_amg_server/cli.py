from io import BytesIO

import fire
import matplotlib.pyplot as plt
from server import (
    MODEL_TYPES_TO_MODEL,
    file_bytes_to_image_tensor,
    load_aot_fast,
    model_type_to_paths,
    set_fast,
    set_furious,
    show_anns,
)

from torchao._models.sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator
from torchao._models.sam2.build_sam import build_sam2
from torchao._models.sam2.utils.amg import rle_to_mask


def main_docstring():
    return f"""
    Args:
        checkpoint_path (str): Path to folder containing checkpoints from https://github.com/facebookresearch/sam2?tab=readme-ov-file#download-checkpoints
        model_type (str): Choose from one of {", ".join(MODEL_TYPES_TO_MODEL.keys())}
        input_path (str): Path to input image
        output_path (str): Path to output image
    """


def main_headless(
    checkpoint_path,
    model_type,
    input_bytes,
    points_per_batch=1024,
    output_format="png",
    verbose=False,
    fast=False,
    furious=False,
    load_fast="",
):
    device = "cuda"
    sam2_checkpoint, model_cfg = model_type_to_paths(checkpoint_path, model_type)
    if verbose:
        print(f"Loading model {sam2_checkpoint} with config {model_cfg}")
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

    image_tensor = file_bytes_to_image_tensor(input_bytes)
    if verbose:
        print(f"Loaded image of size {tuple(image_tensor.shape)} and generating mask.")
    masks = mask_generator.generate(image_tensor)

    if verbose:
        print("Generating mask annotations for input image.")
    plt.figure(
        figsize=(image_tensor.shape[1] / 100.0, image_tensor.shape[0] / 100.0), dpi=100
    )
    plt.imshow(image_tensor)
    show_anns(masks, rle_to_mask)
    plt.axis("off")
    plt.tight_layout()
    buf = BytesIO()
    plt.savefig(buf, format=output_format)
    buf.seek(0)
    return buf.getvalue()


def main(
    checkpoint_path,
    model_type,
    input_path,
    output_path,
    points_per_batch=1024,
    output_format="png",
    verbose=False,
    fast=False,
    furious=False,
    load_fast="",
):
    input_bytes = bytearray(open(input_path, "rb").read())
    output_bytes = main_headless(
        checkpoint_path,
        model_type,
        input_bytes,
        points_per_batch=points_per_batch,
        output_format=output_format,
        verbose=verbose,
        fast=fast,
        furious=furious,
        load_fast=load_fast,
    )
    with open(output_path, "wb") as file:
        file.write(output_bytes)


main.__doc__ = main_docstring()
if __name__ == "__main__":
    fire.Fire(main)
