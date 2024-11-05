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

import asyncio
from contextlib import asynccontextmanager
import contextlib

from torch._inductor import config as inductorconfig
inductorconfig.triton.unique_kernel_names = True
inductorconfig.coordinate_descent_tuning = True
inductorconfig.coordinate_descent_check_all_directions = True

# torch.set_float32_matmul_precision('high')

def iou(mask1, mask2):
    assert mask1.dim() == 2
    assert mask2.dim() == 2
    intersection = torch.logical_and(mask1, mask2)
    union = torch.logical_or(mask1, mask2)
    return (intersection.sum(dim=(-1, -2)) / union.sum(dim=(-1, -2)))


def show_anns(anns, rle_to_mask):
    if len(anns) == 0:
        return
    sorted_anns = sorted(anns, key=(lambda x: x['area']), reverse=True)
    ax = plt.gca()
    ax.set_autoscale_on(False)

    for ann in sorted_anns:
        ann['segmentation'] = rle_to_mask(ann['segmentation'])

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


def profiler_runner(path, fn, *args, **kwargs):
    with torch.profiler.profile(
            activities=[torch.profiler.ProfilerActivity.CPU,
                        torch.profiler.ProfilerActivity.CUDA],
            record_shapes=True) as prof:
        result = fn(*args, **kwargs)
    prof.export_chrome_trace(path)
    return result


def image_tensor_to_masks(example_image, mask_generator):
    masks = mask_generator.generate(example_image)
    return masks


def image_tensors_to_masks(example_images, mask_generator):
    return mask_generator.generate_batch(example_images)


def file_bytes_to_image_tensor(file_bytes):
    image_array = np.asarray(file_bytes, dtype=np.uint8)
    example_image = cv2.imdecode(image_array, cv2.IMREAD_COLOR)
    example_image = cv2.cvtColor(example_image, cv2.COLOR_BGR2RGB)
    return example_image


def masks_to_rle_dict(masks):
    ret_data = {}
    for mask_id in range(len(masks)):
        ret_data[f"mask_{mask_id}"] = masks[mask_id]["segmentation"]
    return ret_data


# Queue to hold incoming requests
request_queue = asyncio.Queue()
batch_interval = 1  # Time interval to wait before processing a batch


def process_batch(batch, mask_generator):
    print(f"Processing batch of len {len(batch)}")
    t = time.time()
    image_tensors = [image_tensor for (image_tensor, _) in batch]
    masks = mask_generator.generate_batch(image_tensors)
    print(f"Took avg. {(time.time() - t) / len(batch)}s per batch entry")
    return masks


async def batch_worker(mask_generator, batch_size, *, pad_batch=True, furious=False):
    # cm = torch.autocast("cuda", dtype=torch.bfloat16) if furious else contextlib.nullcontext()
    # cm.__enter__()
    while True:
        batch = []
        while len(batch) < batch_size and not request_queue.empty():
            batch.append(await request_queue.get())

        if batch:

            padded_batch = batch
            if pad_batch:
                padded_batch = batch + ([batch[-1]] * (batch_size - len(batch)))
            print(f"len(padded_batch): {len(padded_batch)}")
            results = process_batch(padded_batch, mask_generator)
            for i, (_, response_future) in enumerate(batch):
                response_future.set_result(results[i])

        await asyncio.sleep(batch_interval)


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup logic
    mask_generator = app.state.mask_generator
    batch_size = app.state.batch_size
    furious = app.state.furious
    task = asyncio.create_task(batch_worker(mask_generator, batch_size, furious=furious))
    yield
    # Shutdown logic (if needed)
    task.cancel()


def benchmark_fn(func, inp, mask_generator):
    torch.cuda.reset_peak_memory_stats()
    logging.info("Running 3 warumup iterations.")
    for _ in range(3):
        func(inp, mask_generator)
    logging.info("Running 10 benchmark iterations.")
    t = time.time()
    for _ in range(10):
        func(inp, mask_generator)
    print(f"Benchmark took {(time.time() - t)/10.0}s per iteration.")
    max_memory_allocated_bytes = torch.cuda.max_memory_allocated()
    _, total_memory = torch.cuda.mem_get_info()
    max_memory_allocated_percentage = int(100 * (max_memory_allocated_bytes / total_memory))
    max_memory_allocated_bytes = max_memory_allocated_bytes >> 20
    print(f"max_memory_allocated_bytes: {max_memory_allocated_bytes}MiB or {max_memory_allocated_percentage}%")


def unittest_fn(masks, ref_masks, order_by_area=False):
    from torchao._models.sam2.utils.amg import rle_to_mask
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
        if v0_area != v1_area:
            print(f"v0 area {v0_area} doesn't match v1 area {v1_area}")
        v0_mask = torch.from_numpy(rle_to_mask(ref_masks[k0]))
        v1_mask = torch.from_numpy(rle_to_mask(masks[k0]))
        v0_masks.append((v0_mask, v0_area))
        v1_masks.append((v1_mask, v1_area))

    v0_masks = sorted(v0_masks, key=(lambda x: x[1]), reverse=True)
    v1_masks = sorted(v1_masks, key=(lambda x: x[1]), reverse=True)
    miou_sum = 0.0
    miou_count = 0
    for ((v0_mask, v0_area), (v1_mask, v1_area)) in zip(v0_masks, v1_masks):
        if not torch.allclose(v0_mask, v1_mask):
            miou_sum += iou(v0_mask, v1_mask)
            miou_count += 1
            print(f"Masks don't match for key {k0}. IoU is {iou(v0_mask, v1_mask)}")
    if miou_count == 0:
        print("Masks exactly match reference.")
    else:
        print(f"mIoU is {miou_sum / miou_count}")


def main(checkpoint_path,
         baseline=False,
         fast=False,
         furious=False,
         unittest=False,
         benchmark=False,
         profile=None,
         verbose=False,
         points_per_batch=64,
         port=5000,
         host="127.0.0.1",
         dry=False,
         batch_size=1):
    if verbose:
        logging.basicConfig(level=logging.INFO,
                            format='%(asctime)s - %(levelname)s - %(message)s',
                            datefmt='%Y-%m-%d %H:%M:%S')
    logging.info(f"Running with fast set to {fast} and furious set to {furious}")
    logging.info(f"Running with port {port} and host {host}")
    logging.info(f"Running with batch size {batch_size}")

    if baseline:
        assert batch_size == 1, "baseline only supports batch size 1."
        logging.info(f"Importing sam2 from outside of torchao. If this errors, install https://github.com/facebookresearch/sam2")
        from sam2.build_sam import build_sam2
        from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator
        from sam2.utils.amg import rle_to_mask
    else:
        from torchao._models.sam2.build_sam import build_sam2
        from torchao._models.sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator
        from torchao._models.sam2.utils.amg import rle_to_mask
    
    device = "cuda"
    from pathlib import Path
    sam2_checkpoint = Path(checkpoint_path) / Path("sam2.1_hiera_large.pt")
    model_cfg = "configs/sam2.1/sam2.1_hiera_l.yaml"
    
    logging.info(f"Loading model {sam2_checkpoint} with config {model_cfg}")
    sam2 = build_sam2(model_cfg, sam2_checkpoint, device=device, apply_postprocessing=False)

    logging.info(f"Using {points_per_batch} points_per_batch")
    mask_generator = SAM2AutomaticMaskGenerator(sam2, points_per_batch=points_per_batch, output_mode="uncompressed_rle")

    # if furious:
    #     torch.set_float32_matmul_precision('high')
    #     torch.autocast("cuda", dtype=torch.bfloat16).__enter__()

    if fast:
        assert not baseline, "--fast cannot be combined with baseline. code to be torch.compile(fullgraph=True) compatible."
        # TODO: Using CUDA graphs can cause numerical differences?
        mask_generator.predictor.model.image_encoder = torch.compile(
            mask_generator.predictor.model.image_encoder,
            # mode="max-autotune-no-cudagraphs",
            mode="max-autotune",
            fullgraph=True,
            dynamic=False,
        )

        mask_generator.predictor._predict = torch.compile(
            mask_generator.predictor._predict,
            # mode="max-autotune-no-cudagraphs",
            fullgraph=True,
            dynamic=True,
        )

    # if furious:
    #     from torchao.quantization import autoquant
    #     mask_generator.predictor.model.image_encoder = autoquant(mask_generator.predictor.model.image_encoder)

    if furious:
        mask_generator.predictor.model.image_encoder = mask_generator.predictor.model.image_encoder.to(torch.float16)
        torch.set_float32_matmul_precision('high')
        mask_generator.predictor.model.sam_mask_decoder = mask_generator.predictor.model.sam_mask_decoder.to(torch.float16)


    with open('dog.jpg', 'rb') as f:
        image_tensor = file_bytes_to_image_tensor(bytearray(f.read()))

    t = time.time()
    logging.info("Running three iterations to compile and warmup.")
    image_tensor_to_masks(image_tensor, mask_generator)
    image_tensor_to_masks(image_tensor, mask_generator)
    image_tensor_to_masks(image_tensor, mask_generator)
    logging.info(f"Three iterations took {time.time() - t}s.")

    if unittest:
        logging.info("batch size 1 unittest")
        masks = image_tensor_to_masks(image_tensor, mask_generator)
        masks = masks_to_rle_dict(masks)
        import json
        ref_masks = json.loads(open("dog_rle.json").read())
        unittest_fn(masks, ref_masks)

        if batch_size > 1:
            # TODO: Transpose dog image to create diversity in input image shape
            logging.info(f"batch size {batch_size} unittest")
            all_masks = image_tensors_to_masks([image_tensor] * batch_size, mask_generator)
            all_masks = [masks_to_rle_dict(masks) for masks in all_masks]
            for masks in all_masks:
                unittest_fn(masks, ref_masks)

    if benchmark:
        print("batch size 1 test")
        benchmark_fn(image_tensor_to_masks, image_tensor, mask_generator)
        if batch_size > 1:
            print(f"batch size {batch_size} test")
            benchmark_fn(image_tensors_to_masks, [image_tensor] * batch_size, mask_generator)

    if profile is not None:
        print(f"Saving profile under {profile}")
        profiler_runner(profile, image_tensors_to_masks, [image_tensor] * batch_size, mask_generator)

    if dry:
        return

    app = FastAPI(lifespan=lifespan)
    app.state.mask_generator = mask_generator
    app.state.batch_size = batch_size
    app.state.furious = furious

    # Allow all origins (you can restrict it in production)
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    @app.post("/upload_rle")
    async def upload_rle(image: UploadFile = File(...)):
        image_tensor = file_bytes_to_image_tensor(bytearray(await image.read()))
        response_future = asyncio.Future()
        await request_queue.put((image_tensor, response_future))
        masks = await response_future
        return masks_to_rle_dict(masks)
    
    @app.post("/upload")
    async def upload_image(image: UploadFile = File(...)):
        image_tensor = file_bytes_to_image_tensor(bytearray(await image.read()))
        response_future = asyncio.Future()
        await request_queue.put((image_tensor, response_future))
        masks = await response_future

        # Save an example
        plt.figure(figsize=(image_tensor.shape[1]/100., image_tensor.shape[0]/100.), dpi=100)
        plt.imshow(image_tensor)
        show_anns(masks, rle_to_mask)
        plt.axis('off')
        plt.tight_layout()
        buf = BytesIO()
        plt.savefig(buf, format='png')
        buf.seek(0)
        return StreamingResponse(buf, media_type="image/png")
    

    uvicorn.run(app, host=host, port=port, log_level="info")

if __name__ == "__main__":
    fire.Fire(main)
