import fire
import logging
import matplotlib.pyplot as plt
from server import file_bytes_to_image_tensor
from server import show_anns
from server import model_type_to_paths
from server import MODEL_TYPES_TO_MODEL
from torchao._models.sam2.build_sam import build_sam2
from torchao._models.sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator
from torchao._models.sam2.utils.amg import rle_to_mask
from io import BytesIO

def main_docstring():
    return f"""
    Args:
        checkpoint_path (str): Path to folder containing checkpoints from https://github.com/facebookresearch/sam2?tab=readme-ov-file#download-checkpoints
        model_type (str): Choose from one of {", ".join(MODEL_TYPES_TO_MODEL.keys())}
        input_path (str): Path to input image
        output_path (str): Path to output image
    """

def main(checkpoint_path, model_type, input_path, output_path, points_per_batch=1024, output_format='png', verbose=False):
    device = "cuda"
    sam2_checkpoint, model_cfg = model_type_to_paths(checkpoint_path, model_type)
    if verbose:
        print(f"Loading model {sam2_checkpoint} with config {model_cfg}")
    sam2 = build_sam2(model_cfg, sam2_checkpoint, device=device, apply_postprocessing=False)
    mask_generator = SAM2AutomaticMaskGenerator(sam2, points_per_batch=points_per_batch, output_mode="uncompressed_rle")
    image_tensor = file_bytes_to_image_tensor(bytearray(open(input_path, 'rb').read()))
    if verbose:
        print(f"Loaded image of size {tuple(image_tensor.shape)} and generating mask.")
    masks = mask_generator.generate(image_tensor)
    
    # Save an example
    plt.figure(figsize=(image_tensor.shape[1]/100., image_tensor.shape[0]/100.), dpi=100)
    plt.imshow(image_tensor)
    show_anns(masks, rle_to_mask)
    plt.axis('off')
    plt.tight_layout()
    buf = BytesIO()
    plt.savefig(buf, format=output_format)
    buf.seek(0)
    with open(output_path, "wb") as file:
        file.write(buf.getvalue())

main.__doc__ = main_docstring()
if __name__ == "__main__":
    fire.Fire(main)
