import itertools
import requests
import uvicorn
import fire
import tempfile
import logging
import sys
import time
import json
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
inductorconfig.allow_buffer_reuse = False

# torch._dynamo.config.capture_dynamic_output_shape_ops = True
torch._dynamo.config.capture_dynamic_output_shape_ops = True

def download_file(url, download_dir):
    # Create the directory if it doesn't exist
    download_dir = Path(download_dir)
    download_dir.mkdir(parents=True, exist_ok=True)
    # Extract the file name from the URL
    file_name = url.split('/')[-1]
    # Define the full path for the downloaded file
    file_path = download_dir / file_name
    # Download the file
    response = requests.get(url, stream=True)
    response.raise_for_status()  # Raise an error for bad responses
    # Write the file to the specified directory
    print(f"Downloading '{file_name}' to '{download_dir}'")
    with open(file_path, 'wb') as file:
        for chunk in response.iter_content(chunk_size=8192):
            file.write(chunk)
    print(f"Downloaded '{file_name}' to '{download_dir}'")

def example_shapes():
    return [(848, 480, 3),
            (720, 1280, 3),
            (848, 480, 3),
            (1280, 720, 3),
            (480, 848, 3),
            (1080, 1920, 3),
            (1280, 720, 3),
            (1280, 720, 3),
            (720, 1280, 3),
            (848, 480, 3),
            (480, 848, 3),
            (864, 480, 3),
            (1920, 1080, 3),
            (1920, 1080, 3),
            (1280, 720, 3),
            (1232, 672, 3),
            (848, 480, 3),
            (848, 480, 3),
            (1920, 1080, 3),
            (1080, 1920, 3),
            (480, 848, 3),
            (848, 480, 3),
            (480, 848, 3),
            (480, 848, 3),
            (720, 1280, 3),
            (720, 1280, 3),
            (900, 720, 3),
            (848, 480, 3),
            (864, 480, 3),
            (360, 640, 3),
            (360, 640, 3),
            (864, 480, 3)]


def example_shapes_2():
    return [(1080, 1920, 3),
            (1920, 1080, 3),
            (1920, 1080, 3),
            (1080, 1920, 3),
            (848, 480, 3),
            (864, 480, 3),
            (720, 1280, 3),
            (864, 480, 3),
            (848, 480, 3),
            (848, 480, 3),
            (848, 480, 3),
            (848, 480, 3),
            (720, 1280, 3),
            (864, 480, 3),
            (480, 848, 3),
            (1280, 720, 3),
            (720, 1280, 3),
            (1080, 1920, 3),
            (1080, 1920, 3),
            (1280, 720, 3),
            (1080, 1920, 3),
            (1080, 1920, 3),
            (720, 1280, 3),
            (720, 1280, 3),
            (1280, 720, 3),
            (360, 640, 3),
            (864, 480, 3),
            (1920, 1080, 3),
            (1080, 1920, 3),
            (1920, 1080, 3),
            (1920, 1080, 3),
            (1080, 1920, 3)]

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


def memory_runner(path, fn, *args, **kwargs):
    print("Start memory recording")
    torch.cuda.synchronize()
    torch.cuda.memory._record_memory_history(
        True,
        trace_alloc_max_entries=100000,
        trace_alloc_record_context=True
    )
    result = fn(*args, **kwargs)
    torch.cuda.synchronize()
    snapshot = torch.cuda.memory._snapshot()
    print("Finish memory recording")
    import pickle
    with open(path, 'wb') as f:
        pickle.dump(snapshot, f)
    # Use to convert pickle file into html
    # python torch/cuda/_memory_viz.py trace_plot <snapshot>.pickle -o <snapshot>.html
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
batch_interval = 0.01  # Time interval to wait before processing a batch


def process_batch(batch, mask_generator):
    t = time.time()
    image_tensors = [image_tensor for (image_tensor, _) in batch]
    if len(batch) == 1:
        print(f"Processing batch of len {len(batch)} using generate")
        masks = [mask_generator.generate(image_tensors[0])]
    else:
        print(f"Processing batch of len {len(batch)} using generate_batch")
        masks = mask_generator.generate_batch(image_tensors)
    print(f"Took avg. {(time.time() - t) / len(batch)}s per batch entry")
    max_memory_allocated()
    return masks


async def batch_worker(mask_generator, batch_size, *, pad_batch=True, furious=False):
    while True:
        batch = []
        while len(batch) < batch_size and not request_queue.empty():
            batch.append(await request_queue.get())

        if batch:

            padded_batch = batch
            if pad_batch:
                padded_batch = batch + ([batch[-1]] * (batch_size - len(batch)))
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


def benchmark_fn(func, inp, mask_generator, warmup=3, runs=10):
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    logging.info("Running {warmup} warmup iterations.")
    for _ in range(warmup):
        func(inp, mask_generator)
    logging.info("Running {runs} benchmark iterations.")
    t = time.time()
    for _ in range(runs):
        func(inp, mask_generator)
    print(f"Benchmark took {(time.time() - t)/runs}s per iteration.")
    max_memory_allocated()


def max_memory_allocated():
    max_memory_allocated_bytes = torch.cuda.max_memory_allocated()
    _, total_memory = torch.cuda.mem_get_info()
    max_memory_allocated_percentage = int(100 * (max_memory_allocated_bytes / total_memory))
    max_memory_allocated_bytes = max_memory_allocated_bytes >> 20
    print(f"max_memory_allocated_bytes: {max_memory_allocated_bytes}MiB or {max_memory_allocated_percentage}%")


def unittest_fn(masks, ref_masks, order_by_area=False, verbose=False):
    from compare_rle_lists import compare_masks
    miou, equal_count = compare_masks(masks, ref_masks, order_by_area=order_by_area, verbose=verbose)
    if equal_count == len(masks):
        print("Masks exactly match reference.")
    else:
        print(f"mIoU is {miou} with equal count {equal_count} out of {len(masks)}")


MODEL_TYPES_TO_CONFIG = {
        "tiny": "sam2.1_hiera_t.yaml",
        "small": "sam2.1_hiera_s.yaml",
        "plus": "sam2.1_hiera_b+.yaml",
        "large": "sam2.1_hiera_l.yaml",
        }

MODEL_TYPES_TO_MODEL = {
        "tiny": "sam2.1_hiera_tiny.pt",
        "small": "sam2.1_hiera_small.pt",
        "plus": "sam2.1_hiera_base_plus.pt",
        "large": "sam2.1_hiera_large.pt",
        }


MODEL_TYPES_TO_URL = {
        "tiny": "https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_tiny.pt",
        "small": "https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_small.pt",
        "plus": "https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_base_plus.pt",
        "large": "https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_large.pt",
        }


def main_docstring():
    return f"""
    Args:
        checkpoint_path (str): Path to folder containing checkpoints from https://github.com/facebookresearch/sam2?tab=readme-ov-file#download-checkpoints
        model_type (str): Choose from one of {", ".join(MODEL_TYPES_TO_MODEL.keys())}
    """


def model_type_to_paths(checkpoint_path, model_type):
    if model_type not in MODEL_TYPES_TO_CONFIG.keys():
        raise ValueError(f"Expected model_type to be one of {', '.join(MODEL_TYPES_TO_MODEL.keys())} but got {model_type}")
    sam2_checkpoint = Path(checkpoint_path) / Path(MODEL_TYPES_TO_MODEL[model_type])
    if not sam2_checkpoint.exists():
        print(f"Can't find checkpoint {sam2_checkpoint} in folder {checkpoint_path}. Downloading.")
        download_file(MODEL_TYPES_TO_URL[model_type], checkpoint_path)
    assert sam2_checkpoint.exists(), "Can't find downloaded file. Please open an issue."
    model_cfg = f"configs/sam2.1/{MODEL_TYPES_TO_CONFIG[model_type]}"
    return sam2_checkpoint, model_cfg


def aot_compile(model_directory, name, fn, sample_args):
    path = Path(model_directory) / Path(f"{name}.pt2")
    print(f"Saving at {path=}")
    options = {
        "max_autotune": True,
        "triton.cudagraphs": True,
    }

    exported = torch.export.export_for_inference(fn, sample_args)
    output_path = torch._inductor.aoti_compile_and_package(
        exported,
        package_path=str(path),
        inductor_configs=options,
    )
    return output_path


def aot_load(path):
    return torch._export.aot_load(path, "cuda")

class FunctionModel(torch.nn.Module):

    def __init__(self, module, fn_name):
        super().__init__()
        self.module = module
        self.fn_name = fn_name

    def forward(self, *args):
        return getattr(self.module, self.fn_name)(*args)


def set_aot_fast(mask_generator, model_directory):
    example_input = torch.empty(1, 3, 1024, 1024)
    example_input = example_input.to(mask_generator.predictor._image_dtype)
    example_input = (example_input.to(mask_generator.predictor.device),)
    aot_compile(model_directory,
                "sam2_image_encoder",
                mask_generator.predictor.model.image_encoder,
                example_input)

    # NOTE: THIS DOESN'T WORK YET!
    # example_input_0_0 = torch.empty(1, 32, 256, 256, dtype=torch.float16, device=mask_generator.predictor.device)
    # example_input_0_1 = torch.empty(1, 64, 128, 128, dtype=torch.float16, device=mask_generator.predictor.device)
    # example_input_1 = torch.empty(1, 256, 64, 64, dtype=torch.float32, device=mask_generator.predictor.device)
    # example_input_2 = torch.empty(1024, 1, 2, dtype=torch.float32, device=mask_generator.predictor.device)
    # example_input_3 = torch.empty(1024, 1, dtype=torch.int32, device=mask_generator.predictor.device)
    # example_input = ([example_input_0_0, example_input_0_1],
    #                  example_input_1,
    #                  example_input_2,
    #                  example_input_3,
    #                  None,
    #                  None,
    #                  True,
    #                  True,
    #                  -1)
    # mask_generator.forward = mask_generator.predictor._predict_masks_with_features
    # mask_generator(*example_input)
    # aot_compile("sam2__predict_masks_with_features",
    #             mask_generator,
    #             example_input)

    # example_input_2 = torch.empty(1024, 1, 2, dtype=torch.float32, device=mask_generator.predictor.device)
    # example_input_3 = torch.empty(1024, 1, dtype=torch.int32, device=mask_generator.predictor.device)
    # aot_compile("sam2_sam_prompt_encoder",
    #             mask_generator.predictor.model.sam_prompt_encoder,
    #             ((example_input_2, example_input_3),
    #              None,
    #              None))

    # NOTE: THIS DOESN'T WORK YET!
    # example_input_0 = torch.empty(1, 256, 64, 64, dtype=torch.float32, device=mask_generator.predictor.device)
    # example_input_1 = torch.empty(1, 256, 64, 64, dtype=torch.float32, device=mask_generator.predictor.device)
    # example_input_2 = torch.empty(1024, 2, 256, dtype=torch.float32, device=mask_generator.predictor.device)
    # example_input_3 = torch.empty(1024, 256, 64, 64, dtype=torch.float32, device=mask_generator.predictor.device)

    # example_input_4_0 = torch.empty(1, 32, 256, 256, dtype=torch.float16, device=mask_generator.predictor.device)
    # example_input_4_1 = torch.empty(1, 64, 128, 128, dtype=torch.float16, device=mask_generator.predictor.device)

    # example_input = (example_input_0,
    #                  example_input_1,
    #                  example_input_2,
    #                  example_input_3,
    #                  True,
    #                  True,
    #                  [example_input_4_0, example_input_4_1])
    # print("Example")
    # mask_generator.predictor.model.sam_mask_decoder(*example_input)
    # print("Example done")
    # aot_compile("sam2_sam_mask_decoder",
    #             mask_generator.predictor.model.sam_mask_decoder,
    #             example_input)

    # example_input_0 = torch.empty(1024, 256, 64, 64, dtype=torch.float16, device=mask_generator.predictor.device)
    # example_input_1 = torch.empty(1024, 256, 64, 64, dtype=torch.float16, device=mask_generator.predictor.device)
    # example_input_2 = torch.empty(1024, 8, 256, dtype=torch.float16, device=mask_generator.predictor.device)
    # example_input = (example_input_0, example_input_1, example_input_2)

    # mask_generator.predictor.model.sam_mask_decoder.transformer(*example_input)
    # aot_compile("sam2_sam_mask_decoder_transformer",
    #             mask_generator.predictor.model.sam_mask_decoder.transformer,
    #             example_input)




class LoadedModel(torch.nn.Module):

    def __init__(self, aoti_compiled_model):
        super().__init__()
        self.aoti_compiled_model = aoti_compiled_model

    def forward(self, *args):
        return self.aoti_compiled_model(*args)

class LoadedDecoder(torch.nn.Module):

    def __init__(self, aoti_compiled_model, other):
        super().__init__()
        self.aoti_compiled_model = aoti_compiled_model
        self.other = other

    def forward(self, *args):
        return self.aoti_compiled_model(*args)

    def get_dense_pe(self, *args, **kwargs) -> torch.Tensor:
        return self.other.get_dense_pe(*args, **kwargs)

def load_aot_fast(mask_generator, model_directory):
    t0 = time.time()
    path = Path(model_directory) / Path(f"sam2_image_encoder.pt2")
    assert path.exists(), f"Expected {path} to exist."
    print(f"Start load from {path}")
    pkg = torch._inductor.aoti_load_package(str(path))
    pkg_m = LoadedModel(pkg)
    mask_generator.predictor.model.image_encoder = pkg_m

    # NOTE: This doesn't work yet!
    # pkg = torch._inductor.aoti_load_package(os.path.join(os.getcwd(), "sam2__predict_masks_with_features.pt2"))
    # pkg_m = LoadedModel(pkg)
    # mask_generator.predictor._predict_masks_with_features = pkg_m.forward

    # pkg = torch._inductor.aoti_load_package(os.path.join(os.getcwd(), "sam2_sam_prompt_encoder.pt2"))
    # pkg_m = LoadedDecoder(pkg, mask_generator.predictor.model.sam_prompt_encoder)
    # mask_generator.predictor.model.sam_prompt_encoder = pkg_m

    # NOTE: This doesn't work yet!
    # pkg = torch._inductor.aoti_load_package(os.path.join(os.getcwd(), "sam2_sam_mask_decoder.pt2"))
    # pkg_m = LoadedModel(pkg)
    # pkg_m.conv_s0 = mask_generator.predictor.model.sam_mask_decoder.conv_s0
    # pkg_m.conv_s1 = mask_generator.predictor.model.sam_mask_decoder.conv_s1
    # mask_generator.predictor.model.sam_mask_decoder = pkg_m

    # pkg = torch._inductor.aoti_load_package(os.path.join(os.getcwd(), "sam2_sam_mask_decoder_transformer.pt2"))
    # pkg_m = LoadedModel(pkg)
    # mask_generator.predictor.model.sam_mask_decoder.transformer = pkg_m

    print(f"End load. Took {time.time() - t0}s")


def set_fast(mask_generator, load_fast=""):
    if load_fast == "":
        # TODO: Using CUDA graphs can cause numerical differences?
        mask_generator.predictor.model.image_encoder = torch.compile(
            mask_generator.predictor.model.image_encoder,
            mode="max-autotune",
            fullgraph=True,
            dynamic=False,
        )

    mask_generator.predictor._predict_masks = torch.compile(
        mask_generator.predictor._predict_masks,
        mode="max-autotune",
        fullgraph=True,
        dynamic=False,
    )

    # mask_generator.predictor._predict_masks_postprocess = torch.compile(
    #     mask_generator.predictor._predict_masks_postprocess,
    #     fullgraph=True,
    #     dynamic=True,
    # )


def set_furious(mask_generator):
    mask_generator.predictor.model.image_encoder = mask_generator.predictor.model.image_encoder.to(torch.float16)
    # NOTE: Not baseline feature
    mask_generator.predictor._image_dtype = torch.float16
    mask_generator.predictor._transforms_device = mask_generator.predictor.device
    torch.set_float32_matmul_precision('high')
    mask_generator.predictor.model.sam_mask_decoder = mask_generator.predictor.model.sam_mask_decoder.to(torch.float16)
    # NOTE: Not baseline feature
    mask_generator.predictor.model.sam_mask_decoder._src_dtype = torch.float16

def set_autoquant(mask_generator):
    from torchao import autoquant
    from torchao.quantization import DEFAULT_FLOAT_AUTOQUANT_CLASS_LIST
    # NOTE: Not baseline feature
    mask_generator.predictor.model.image_encoder = autoquant(mask_generator.predictor.model.image_encoder, qtensor_class_list=DEFAULT_FLOAT_AUTOQUANT_CLASS_LIST, min_sqnr=40)
    mask_generator.predictor._transforms_device = mask_generator.predictor.device
    torch.set_float32_matmul_precision('high')
    # NOTE: this fails when we run
    # python server.py ~/checkpoints/sam2 large --port 8000 --host localhost --fast --use_autoquant --unittest
    # https://gist.github.com/jerryzh168/d337cb5de0a1dec306069fe48ac8225e
    # mask_generator.predictor.model.sam_mask_decoder = autoquant(mask_generator.predictor.model.sam_mask_decoder, qtensor_class_list=DEFAULT_FLOAT_AUTOQUANT_CLASS_LIST, min_sqnr=40)


def main(checkpoint_path,
         model_type,
         baseline=False,
         fast=False,
         furious=False,
         use_autoquant=False,
         unittest=False,
         benchmark=False,
         profile=None,
         memory_profile=None,
         verbose=False,
         points_per_batch=64,
         port=5000,
         host="127.0.0.1",
         dry=False,
         batch_size=1,
         load_fast="",
         save_fast=""):
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
    sam2_checkpoint, model_cfg = model_type_to_paths(checkpoint_path, model_type)

    logging.info(f"Loading model {sam2_checkpoint} with config {model_cfg}")
    sam2 = build_sam2(model_cfg, sam2_checkpoint, device=device, apply_postprocessing=False)

    logging.info(f"Using {points_per_batch} points_per_batch")
    mask_generator = SAM2AutomaticMaskGenerator(sam2, points_per_batch=points_per_batch, output_mode="uncompressed_rle")

    if load_fast != "":
        load_aot_fast(mask_generator, load_fast)

    if furious:
        set_furious(mask_generator)
    # since autoquant is replicating what furious mode is doing, don't use these two together
    elif use_autoquant:
        set_autoquant(mask_generator)

    if save_fast != "":
        assert load_fast == "", "Can't save compiled models while loading them with --load-fast."
        assert not baseline, "--fast cannot be combined with baseline. code to be torch.compile(fullgraph=True) compatible."
        print(f"Saving compiled models under directory {save_fast}")
        set_aot_fast(mask_generator, save_fast)

    if fast:
        assert not baseline, "--fast cannot be combined with baseline. code to be torch.compile(fullgraph=True) compatible."
        set_fast(mask_generator, load_fast)

    with open('dog.jpg', 'rb') as f:
        image_tensor = file_bytes_to_image_tensor(bytearray(f.read()))

    if unittest:
        if batch_size == 1:
            logging.info("batch size 1 unittest")
            masks = image_tensor_to_masks(image_tensor, mask_generator)
            masks = masks_to_rle_dict(masks)
            ref_masks = json.loads(open("dog_rle.json").read())
            unittest_fn(masks, ref_masks, order_by_area=True, verbose=verbose)
        else:
            # TODO: Transpose dog image to create diversity in input image shape
            logging.info(f"batch size {batch_size} unittest")
            all_masks = image_tensors_to_masks([image_tensor] * batch_size, mask_generator)
            all_masks = [masks_to_rle_dict(masks) for masks in all_masks]
            ref_masks = json.loads(open("dog_rle.json").read())
            for masks in all_masks:
                unittest_fn(masks, ref_masks, order_by_area=True, verbose=verbose)

    if benchmark:
        print(f"batch size {batch_size} dog benchmark")
        if batch_size == 1:
            benchmark_fn(image_tensor_to_masks, image_tensor, mask_generator)
        else:
            benchmark_fn(image_tensors_to_masks, [image_tensor] * batch_size, mask_generator)

        for i, shapes in enumerate([example_shapes(), example_shapes_2()]):
            print(f"batch size {batch_size} example shapes {i} benchmark")
            random_images = [np.random.randint(0, 256, size=size, dtype=np.uint8) for size in shapes]
            if batch_size > len(random_images):
                num_repeat = (len(random_images) + batch_size) // batch_size
                random_images = num_repeat * random_images

            if batch_size == 1:
                [benchmark_fn(image_tensor_to_masks, r, mask_generator) for r in random_images]
            else:
                random_images = random_images[:batch_size]
                print("len(random_images): ", len(random_images))
                benchmark_fn(image_tensors_to_masks, random_images, mask_generator)

    if profile is not None:
        print(f"Saving profile under {profile}")
        if batch_size == 1:
            profiler_runner(profile, image_tensor_to_masks, image_tensor, mask_generator)
        else:
            profiler_runner(profile, image_tensors_to_masks, [image_tensor] * batch_size, mask_generator)

    if memory_profile is not None:
        print(f"Saving memory profile under {memory_profile}")
        if batch_size == 1:
            memory_runner(memory_profile, image_tensor_to_masks, image_tensor, mask_generator)
        else:
            memory_runner(memory_profile, image_tensors_to_masks, [image_tensor] * batch_size, mask_generator)

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


    # uvicorn.run(app, host=host, port=port, log_level="info")
    uvicorn.run(app, host=host, port=port)

main.__doc__ = main_docstring()
if __name__ == "__main__":
    fire.Fire(main)
