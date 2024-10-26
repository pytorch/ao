import itertools
import uvicorn
import fire
import tempfile
import logging
import sys
import time
from pathlib import Path
from typing import List, Optional

import torch
import torch._dynamo.config
import torch._inductor.config
from fastapi.responses import Response
from fastapi import FastAPI, File, UploadFile
from fastapi.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from io import BytesIO
import shutil
from pydantic import BaseModel
import cv2

import matplotlib.pyplot as plt
import numpy as np

# from torch._inductor import config as inductorconfig
# inductorconfig.triton.unique_kernel_names = True
# inductorconfig.coordinate_descent_tuning = True
# inductorconfig.coordinate_descent_check_all_directions = True

# torch.set_float32_matmul_precision('high')

def show_anns(anns):
    if len(anns) == 0:
        return
    sorted_anns = sorted(anns, key=(lambda x: x['area']), reverse=True)
    ax = plt.gca()
    ax.set_autoscale_on(False)

    img = np.ones((sorted_anns[0]['segmentation'].shape[0], sorted_anns[0]['segmentation'].shape[1], 4))
    img[:,:,3] = 0
    ms = []
    for ann in sorted_anns:
        m = ann['segmentation']
        ms.append(torch.as_tensor(m))
        color_mask = np.concatenate([np.random.random(3), [0.35]])
        img[m] = color_mask
    ax.imshow(img)
    return torch.stack(ms)


def main(checkpoint_path, fast=False, furious=False, benchmark=False, verbose=False, points_per_batch=64):
    if verbose:
        logging.basicConfig(level=logging.INFO)
    logging.info(f"Running with fast set to {fast} and furious set to {furious}")

    if fast:
        from torchao._models.sam2.build_sam import build_sam2
        from torchao._models.sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator
    else:
        from sam2.build_sam import build_sam2
        from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator
    
    device = "cuda"
    from pathlib import Path
    sam2_checkpoint = Path(checkpoint_path) / Path("sam2.1_hiera_large.pt")
    model_cfg = "configs/sam2.1/sam2.1_hiera_l.yaml"
    
    logging.info(f"Loading model {sam2_checkpoint} with config {model_cfg}")
    sam2 = build_sam2(model_cfg, sam2_checkpoint, device=device, apply_postprocessing=False)

    logging.info(f"Using {points_per_batch} points_per_batch")
    mask_generator = SAM2AutomaticMaskGenerator(sam2, points_per_batch=points_per_batch)

    if furious:
        torch.set_float32_matmul_precision('high')
        torch.autocast("cuda", dtype=torch.bfloat16).__enter__()

    if fast:
        # TODO: Using CUDA graphs can cause numerical differences?
        mask_generator.predictor.model.image_encoder = torch.compile(
            mask_generator.predictor.model.image_encoder,
            # mode="max-autotune-no-cudagraphs",
            mode="max-autotune",
            fullgraph=True,
            dynamic=False,
        )

        # torch._dynamo.config.capture_dynamic_output_shape_ops = True
        # mask_generator._process_batch = torch.compile(
        #     mask_generator._process_batch,
        #     # mode="max-autotune-no-cudagraphs",
        #     # fullgraph=True,
        #     dynamic=True,
        # )
        mask_generator.predictor._predict = torch.compile(
            mask_generator.predictor._predict,
            # mode="max-autotune-no-cudagraphs",
            fullgraph=True,
            dynamic=True,
        )

    example_image = cv2.imread('dog.jpg')
    example_image = cv2.cvtColor(example_image, cv2.COLOR_BGR2RGB)
    with torch.backends.cuda.sdp_kernel(enable_cudnn=True):
        t = time.time()
        logging.info(f"Running one iteration to compile.")
        masks = mask_generator.generate(example_image)
        logging.info(f"First iteration took {time.time() - t}s.")
        if benchmark:
            logging.info(f"Running 3 warumup iterations.")
            for _ in range(3):
                masks = mask_generator.generate(example_image)
            logging.info(f"Running 10 benchmark iterations, then exit.")
            t = time.time()
            for _ in range(10):
                masks = mask_generator.generate(example_image)
            print(f"Benchmark took {(time.time() - t)/10.0}s per iteration.")
            max_memory_allocated_bytes = torch.cuda.max_memory_allocated()
            _, total_memory = torch.cuda.mem_get_info()
            max_memory_allocated_percentage = int(100 * (max_memory_allocated_bytes / total_memory))
            max_memory_allocated_bytes = max_memory_allocated_bytes >> 20
            print(f"max_memory_allocated_bytes: {max_memory_allocated_bytes}MiB or {max_memory_allocated_percentage}%")
            return

    app = FastAPI()

    # Allow all origins (you can restrict it in production)
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    
    @app.post("/upload")
    async def upload_image(image: UploadFile = File(...)):
        # Save the uploaded image to a temporary location
        temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=f"_{image.filename}")
        with open(temp_file.name, "wb") as b:
            shutil.copyfileobj(image.file, b)
    
        # Read the image back into memory to send as response
        example_image = cv2.imread(temp_file.name)
        t = time.time()
        with torch.backends.cuda.sdp_kernel(enable_cudnn=True):
            masks = mask_generator.generate(example_image)
        print(f"Took {time.time() - t} to generate a mask for input image.")
        # Save an example
        plt.figure(figsize=(example_image.shape[1]/100., example_image.shape[0]/100.), dpi=100)
        plt.imshow(example_image)
        show_anns(masks)
        plt.axis('off')
        plt.tight_layout()
        plt.savefig(temp_file.name, format='png')

        # Read the image back into memory to send as response
        with open(temp_file.name, "rb") as f:
            image_data = f.read()
    
        # Return the image as a StreamingResponse
        return StreamingResponse(BytesIO(image_data), media_type="image/png")
    

    uvicorn.run(app, host="127.0.0.1", port=5000, log_level="info")

if __name__ == "__main__":
    fire.Fire(main)
