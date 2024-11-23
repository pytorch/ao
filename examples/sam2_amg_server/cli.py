import fire
import logging
import matplotlib.pyplot as plt
from server import file_bytes_to_image_tensor
from server import show_anns
from torchao._models.sam2.build_sam import build_sam2
from torchao._models.sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator
from torchao._models.sam2.utils.amg import rle_to_mask
from io import BytesIO

def main(checkpoint_path, input_path, output_path, points_per_batch=1024):
    device = "cuda"
    from pathlib import Path
    sam2_checkpoint = Path(checkpoint_path) / Path("sam2.1_hiera_large.pt")
    model_cfg = "configs/sam2.1/sam2.1_hiera_l.yaml"
    
    logging.info(f"Loading model {sam2_checkpoint} with config {model_cfg}")
    sam2 = build_sam2(model_cfg, sam2_checkpoint, device=device, apply_postprocessing=False)
    mask_generator = SAM2AutomaticMaskGenerator(sam2, points_per_batch=points_per_batch, output_mode="uncompressed_rle")
    image_tensor = file_bytes_to_image_tensor(bytearray(open(input_path, 'rb').read()))
    masks = mask_generator.generate(image_tensor)
    
    # Save an example
    plt.figure(figsize=(image_tensor.shape[1]/100., image_tensor.shape[0]/100.), dpi=100)
    plt.imshow(image_tensor)
    show_anns(masks, rle_to_mask)
    plt.axis('off')
    plt.tight_layout()
    buf = BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)
    with open(output_path, "wb") as file:
        file.write(buf.getvalue())
    print(f"Wrote output image to {output_path}")

if __name__ == "__main__":
    fire.Fire(main)
