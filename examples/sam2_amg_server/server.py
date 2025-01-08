import asyncio
import json
import logging
import time
from contextlib import asynccontextmanager
from io import BytesIO
from pathlib import Path

import cv2
import fire
import matplotlib.pyplot as plt
import numpy as np
import requests
import torch
import torch._dynamo.config
import torch._inductor.config
import uvicorn
from compile_export_utils import (
    export_model,
    load_exported_model,
    set_fast,
    set_furious,
)
from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from torch._inductor import config as inductorconfig

from torchao._models.utils import (
    get_arch_name,
    write_json_result_local,
    write_json_result_ossci,
)

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
    file_name = url.split("/")[-1]
    # Define the full path for the downloaded file
    file_path = download_dir / file_name
    # Download the file
    response = requests.get(url, stream=True)
    response.raise_for_status()  # Raise an error for bad responses
    # Write the file to the specified directory
    print(f"Downloading '{file_name}' to '{download_dir}'")
    with open(file_path, "wb") as file:
        for chunk in response.iter_content(chunk_size=8192):
            file.write(chunk)
    print(f"Downloaded '{file_name}' to '{download_dir}'")


def example_shapes():
    return [
        (848, 480, 3),
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
        (864, 480, 3),
    ]


def example_shapes_2():
    return [
        (1080, 1920, 3),
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
        (1080, 1920, 3),
    ]


# torch.set_float32_matmul_precision('high')


def iou(mask1, mask2):
    assert mask1.dim() == 2
    assert mask2.dim() == 2
    intersection = torch.logical_and(mask1, mask2)
    union = torch.logical_or(mask1, mask2)
    return intersection.sum(dim=(-1, -2)) / union.sum(dim=(-1, -2))


def show_anns(anns, rle_to_mask, sort_by_area=True, seed=None):
    if len(anns) == 0:
        return
    if sort_by_area:
        sorted_anns = sorted(anns, key=(lambda x: x["area"]), reverse=True)
    else:
        sorted_anns = anns
    ax = plt.gca()
    ax.set_autoscale_on(False)

    for ann in sorted_anns:
        ann["segmentation"] = rle_to_mask(ann["segmentation"])

    img = np.ones(
        (
            sorted_anns[0]["segmentation"].shape[0],
            sorted_anns[0]["segmentation"].shape[1],
            4,
        )
    )
    img[:, :, 3] = 0

    np.random.seed(seed)
    ms = []
    for ann in sorted_anns:
        m = ann["segmentation"]
        ms.append(torch.as_tensor(m))
        color_mask = np.concatenate([np.random.random(3), [0.35]])
        img[m] = color_mask
    ax.imshow(img)
    return torch.stack(ms)


def profiler_runner(path, fn, *args, **kwargs):
    with torch.profiler.profile(
        activities=[
            torch.profiler.ProfilerActivity.CPU,
            torch.profiler.ProfilerActivity.CUDA,
        ],
        record_shapes=True,
    ) as prof:
        result = fn(*args, **kwargs)
    prof.export_chrome_trace(path)
    return result


def memory_runner(path, fn, *args, **kwargs):
    print("Start memory recording")
    torch.cuda.synchronize()
    torch.cuda.memory._record_memory_history(
        True, trace_alloc_max_entries=100000, trace_alloc_record_context=True
    )
    result = fn(*args, **kwargs)
    torch.cuda.synchronize()
    snapshot = torch.cuda.memory._snapshot()
    print("Finish memory recording")
    import pickle

    with open(path, "wb") as f:
        pickle.dump(snapshot, f)
    # Use to convert pickle file into html
    # python torch/cuda/_memory_viz.py trace_plot <snapshot>.pickle -o <snapshot>.html
    return result


def image_tensor_to_masks(example_image, mask_generator):
    masks = mask_generator.generate(example_image)
    return masks


def image_tensors_to_masks(example_images, mask_generator):
    return mask_generator.generate_batch(example_images)


def file_bytes_to_image_tensor(file_bytes, output_format="numpy"):
    image_array = np.asarray(file_bytes, dtype=np.uint8)
    example_image = cv2.imdecode(image_array, cv2.IMREAD_COLOR)
    example_image = cv2.cvtColor(example_image, cv2.COLOR_BGR2RGB)
    if output_format == "numpy":
        return example_image
    if output_format not in ["torch"]:
        raise ValueError(
            "Expected output_format to be numpy or torch," f" but got {output_format}"
        )
    from torchvision.transforms import ToTensor

    return ToTensor()(example_image)


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
    task = asyncio.create_task(
        batch_worker(mask_generator, batch_size, furious=furious)
    )
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
    avg_time_per_run = (time.time() - t) / runs
    print(f"Benchmark took {avg_time_per_run}s per iteration.")
    max_memory_allocated_bytes, max_memory_allocated_percentage = max_memory_allocated()
    return avg_time_per_run, max_memory_allocated_bytes, max_memory_allocated_percentage


def max_memory_allocated_stats():
    max_memory_allocated_bytes = torch.cuda.max_memory_allocated()
    _, total_memory = torch.cuda.mem_get_info()
    max_memory_allocated_percentage = int(
        100 * (max_memory_allocated_bytes / total_memory)
    )
    return {
        "bytes": max_memory_allocated_bytes,
        "percentage": max_memory_allocated_percentage,
    }


def max_memory_allocated():
    stats = max_memory_allocated_stats()
    mib = stats["bytes"] >> 20
    print(f"max_memory_allocated_bytes: {mib}MiB")
    print(f"max_memory_allocated_percentage: {stats['percentage']}%")
    return mib, stats["percentage"]


def unittest_fn(masks, ref_masks, order_by_area=False, verbose=False):
    from compare_rle_lists import compare_masks

    miou, equal_count = compare_masks(
        masks, ref_masks, order_by_area=order_by_area, verbose=verbose
    )
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
        raise ValueError(
            f"Expected model_type to be one of {', '.join(MODEL_TYPES_TO_MODEL.keys())} but got {model_type}"
        )
    sam2_checkpoint = Path(checkpoint_path) / Path(MODEL_TYPES_TO_MODEL[model_type])
    if not sam2_checkpoint.exists():
        print(
            f"Can't find checkpoint {sam2_checkpoint} in folder {checkpoint_path}. Downloading."
        )
        download_file(MODEL_TYPES_TO_URL[model_type], checkpoint_path)
    assert sam2_checkpoint.exists(), "Can't find downloaded file. Please open an issue."
    model_cfg = f"configs/sam2.1/{MODEL_TYPES_TO_CONFIG[model_type]}"
    return sam2_checkpoint, model_cfg


def set_autoquant(mask_generator, autoquant_type, min_sqnr):
    import torchao
    from torchao import autoquant

    # NOTE: Not baseline feature
    if autoquant_type == "autoquant":
        mask_generator.predictor.model.image_encoder = autoquant(
            mask_generator.predictor.model.image_encoder, min_sqnr=min_sqnr
        )
    elif autoquant_type == "autoquant-fp":
        mask_generator.predictor.model.image_encoder = autoquant(
            mask_generator.predictor.model.image_encoder,
            qtensor_class_list=torchao.quantization.DEFAULT_FLOAT_AUTOQUANT_CLASS_LIST,
            min_sqnr=min_sqnr,
        )
    elif autoquant_type == "autoquant-all":
        mask_generator.predictor.model.image_encoder = autoquant(
            mask_generator.predictor.model.image_encoder,
            qtensor_class_list=torchao.quantization.ALL_AUTOQUANT_CLASS_LIST,
            min_sqnr=min_sqnr,
        )
    else:
        raise ValueError(f"Unexpected autoquant type: {autoquant_type}")

    mask_generator.predictor._transforms_device = mask_generator.predictor.device
    torch.set_float32_matmul_precision("high")
    # NOTE: this fails when we run
    # python server.py ~/checkpoints/sam2 large --port 8000 --host localhost --fast --use_autoquant --unittest
    # https://gist.github.com/jerryzh168/d337cb5de0a1dec306069fe48ac8225e
    # mask_generator.predictor.model.sam_mask_decoder = autoquant(mask_generator.predictor.model.sam_mask_decoder, qtensor_class_list=DEFAULT_FLOAT_AUTOQUANT_CLASS_LIST, min_sqnr=40)


def main(
    checkpoint_path,
    model_type,
    baseline=False,
    fast=False,
    furious=False,
    autoquant_type=None,
    min_sqnr=None,
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
    save_fast="",
    output_json_path=None,
    output_json_local=False,
):
    if verbose:
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s - %(levelname)s - %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
    logging.info(f"Running with fast set to {fast} and furious set to {furious}")
    logging.info(f"Running with port {port} and host {host}")
    logging.info(f"Running with batch size {batch_size}")

    if baseline:
        assert batch_size == 1, "baseline only supports batch size 1."
        logging.info(
            "Importing sam2 from outside of torchao. If this errors, install https://github.com/facebookresearch/sam2"
        )
        from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator
        from sam2.build_sam import build_sam2
        from sam2.utils.amg import rle_to_mask
    else:
        from torchao._models.sam2.automatic_mask_generator import (
            SAM2AutomaticMaskGenerator,
        )
        from torchao._models.sam2.build_sam import build_sam2
        from torchao._models.sam2.utils.amg import rle_to_mask

    device = "cuda"
    sam2_checkpoint, model_cfg = model_type_to_paths(checkpoint_path, model_type)

    logging.info(f"Loading model {sam2_checkpoint} with config {model_cfg}")
    sam2 = build_sam2(
        model_cfg, sam2_checkpoint, device=device, apply_postprocessing=False
    )

    logging.info(f"Using {points_per_batch} points_per_batch")
    mask_generator = SAM2AutomaticMaskGenerator(
        sam2, points_per_batch=points_per_batch, output_mode="uncompressed_rle"
    )

    if load_fast != "":
        load_exported_model(
            mask_generator, load_fast, "amg", furious, batch_size, points_per_batch
        )

    if furious:
        set_furious(mask_generator)

    if save_fast != "":
        assert (
            load_fast == ""
        ), "Can't save compiled models while loading them with --load-fast."
        assert not baseline, "--fast cannot be combined with baseline. code to be torch.compile(fullgraph=True) compatible."
        print(f"Saving compiled models under directory {save_fast}")
        export_model(
            mask_generator,
            save_fast,
            "amg",
            furious=furious,
            batch_size=batch_size,
            points_per_batch=points_per_batch,
        )

    if fast:
        assert not baseline, "--fast cannot be combined with baseline. code to be torch.compile(fullgraph=True) compatible."
        set_fast(mask_generator, load_fast)

    # since autoquant is replicating what furious mode is doing, don't use these two together
    if autoquant_type is not None:
        assert not furious, "use autoquant can't be used together with furious"
        set_autoquant(mask_generator, autoquant_type, min_sqnr)

    with open("dog.jpg", "rb") as f:
        output_format = "numpy" if baseline else "torch"
        image_tensor = file_bytes_to_image_tensor(
            bytearray(f.read()), output_format=output_format
        )

    # from torchvision import io as tio
    # img_bytes_tensor = tio.read_file('dog.jpg')
    # image_tensor = tio.decode_jpeg(img_bytes_tensor, device='cuda', mode=tio.ImageReadMode.RGB)

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
            all_masks = image_tensors_to_masks(
                [image_tensor] * batch_size, mask_generator
            )
            all_masks = [masks_to_rle_dict(masks) for masks in all_masks]
            ref_masks = json.loads(open("dog_rle.json").read())
            for masks in all_masks:
                unittest_fn(masks, ref_masks, order_by_area=True, verbose=verbose)

    if benchmark:
        print(f"batch size {batch_size} dog benchmark")
        if batch_size == 1:
            result = benchmark_fn(image_tensor_to_masks, image_tensor, mask_generator)
        else:
            result = benchmark_fn(
                image_tensors_to_masks, [image_tensor] * batch_size, mask_generator
            )

        for i, shapes in enumerate([example_shapes(), example_shapes_2()]):
            print(f"batch size {batch_size} example shapes {i} benchmark")
            random_images = [
                np.random.randint(0, 256, size=size, dtype=np.uint8) for size in shapes
            ]
            if batch_size > len(random_images):
                num_repeat = (len(random_images) + batch_size) // batch_size
                random_images = num_repeat * random_images

            if batch_size == 1:
                [
                    benchmark_fn(image_tensor_to_masks, r, mask_generator)
                    for r in random_images
                ]
            else:
                random_images = random_images[:batch_size]
                print("len(random_images): ", len(random_images))
                benchmark_fn(image_tensors_to_masks, random_images, mask_generator)

        if output_json_path:
            headers = ["name", "dtype", "device", "arch", "metric", "actual", "target"]
            name = "sam2-" + model_type
            arch = get_arch_name()
            dtype = autoquant_type or "noquant"
            (
                avg_time_per_run,
                max_memory_allocated_bytes,
                max_memory_allocated_percentage,
            ) = result
            memory_result = [
                name,
                dtype,
                device,
                arch,
                "memory(MiB)",
                max_memory_allocated_bytes,
                None,
            ]
            memory_percent_result = [
                name,
                dtype,
                device,
                arch,
                "memory(%)",
                max_memory_allocated_percentage,
                None,
            ]
            performance_result = [
                name,
                dtype,
                device,
                arch,
                "time_s(avg)",
                avg_time_per_run,
                None,
            ]
            write_json_result = (
                write_json_result_local
                if output_json_local
                else write_json_result_ossci
            )
            write_json_result(output_json_path, headers, memory_result)
            write_json_result(output_json_path, headers, memory_percent_result)
            write_json_result(output_json_path, headers, performance_result)

    if profile is not None:
        print(f"Saving profile under {profile}")
        if batch_size == 1:
            profiler_runner(
                profile, image_tensor_to_masks, image_tensor, mask_generator
            )
        else:
            profiler_runner(
                profile,
                image_tensors_to_masks,
                [image_tensor] * batch_size,
                mask_generator,
            )

    if memory_profile is not None:
        print(f"Saving memory profile under {memory_profile}")
        if batch_size == 1:
            memory_runner(
                memory_profile, image_tensor_to_masks, image_tensor, mask_generator
            )
        else:
            memory_runner(
                memory_profile,
                image_tensors_to_masks,
                [image_tensor] * batch_size,
                mask_generator,
            )

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

        # Create figure and ensure it's closed after generating response
        fig = plt.figure(figsize=(image_tensor.shape[1]/100., image_tensor.shape[0]/100.), dpi=100)
        plt.imshow(image_tensor)
        show_anns(masks, rle_to_mask)
        plt.axis("off")
        plt.tight_layout()

        buf = BytesIO()
        plt.savefig(buf, format="png")
        buf.seek(0)
        plt.close(fig)  # Close figure after we're done with it

        return StreamingResponse(buf, media_type="image/png")

    # uvicorn.run(app, host=host, port=port, log_level="info")
    uvicorn.run(app, host=host, port=port)


main.__doc__ = main_docstring()
if __name__ == "__main__":
    fire.Fire(main)
