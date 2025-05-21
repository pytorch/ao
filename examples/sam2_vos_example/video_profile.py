# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD 3-Clause license found in the
# LICENSE file in the root directory of this source tree.
import os
import time
from datetime import datetime
from pathlib import Path

import fire
import numpy as np
import requests
import torch
from PIL import Image, ImageDraw

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
    timestamped_print(f"Downloading '{file_name}' to '{download_dir}'")
    with open(file_path, "wb") as file:
        for chunk in response.iter_content(chunk_size=8192):
            file.write(chunk)
    timestamped_print(f"Downloaded '{file_name}' to '{download_dir}'")


def model_type_to_paths(checkpoint_path, model_type):
    if model_type not in MODEL_TYPES_TO_MODEL.keys():
        raise ValueError(
            f"Expected model_type to be one of {', '.join(MODEL_TYPES_TO_MODEL.keys())} but got {model_type}"
        )
    sam2_checkpoint = Path(checkpoint_path) / Path(MODEL_TYPES_TO_MODEL[model_type])
    if not sam2_checkpoint.exists():
        timestamped_print(
            f"Can't find checkpoint {sam2_checkpoint} in folder {checkpoint_path}. Downloading."
        )
        download_file(MODEL_TYPES_TO_URL[model_type], checkpoint_path)
    assert sam2_checkpoint.exists(), "Can't find downloaded file. Please open an issue."
    model_cfg = f"configs/sam2.1/{MODEL_TYPES_TO_CONFIG[model_type]}"
    return sam2_checkpoint, model_cfg


from torch._inductor import config as inductorconfig

inductorconfig.triton.unique_kernel_names = True
inductorconfig.coordinate_descent_tuning = True
inductorconfig.coordinate_descent_check_all_directions = True


# timer.py
from collections import defaultdict


class CodeTimer:
    def __init__(self):
        self.start_times = {}
        self.elapsed_times = defaultdict(list)
        self.enabled = False

    def tic(self, section_name):
        self.start_times[section_name] = time.time()

    def toc(self, section_name):
        if section_name in self.start_times:
            elapsed_time = time.time() - self.start_times[section_name]
            self.elapsed_times[section_name].append(elapsed_time)
            del self.start_times[section_name]

    def get_average_time(self, section_name, warmup: int = 1):
        times = self.elapsed_times.get(section_name, [])
        times = times[warmup:]
        return sum(times) / len(times) if times else 0.0

    def reset(self):
        self.start_times.clear()
        self.elapsed_times.clear()

    def print_all_timings(self, warmup: int = 5):
        if not self.elapsed_times:
            timestamped_print("No timings recorded.")
            return
        timestamped_print("Average timings for all sections:")
        for section_name in self.elapsed_times:
            average_time = self.get_average_time(section_name, warmup)
            timestamped_print(f"{section_name}, {average_time * 1000.0:.6f}")


global_timer = CodeTimer()


def max_memory_allocated():
    max_memory_allocated_bytes = torch.cuda.max_memory_allocated()
    _, total_memory = torch.cuda.mem_get_info()
    max_memory_allocated_percentage = int(
        100 * (max_memory_allocated_bytes / total_memory)
    )
    max_memory_allocated_bytes = max_memory_allocated_bytes >> 20
    timestamped_print(
        f"max_memory_allocated_bytes: {max_memory_allocated_bytes}MiB or {max_memory_allocated_percentage}%"
    )


def synthesize_video_data(
    out_dir: str,
    radius: int,
    seed: int,
    speed: int,
    width: int,
    height: int,
    n_frames: int,
    x: int,
    y: int,
    synthesize_overwrite: bool,
):
    circle_color = (255, 0, 0)  # red

    os.makedirs(out_dir, exist_ok=True)

    np.random.seed(seed)
    # Initial position and velocity
    x = np.random.randint(radius, width - radius)
    y = np.random.randint(radius, height - radius)
    vx = np.random.choice([-1, 1]) * speed
    vy = np.random.choice([-1, 1]) * speed

    # TODO: If these frames exist, they will not be deleted in subsequent runs with less frames.
    timestamped_print(f"Generate {n_frames} frames under path {out_dir}")
    if not synthesize_overwrite and len(os.listdir(out_dir)) > 0:
        raise ValueError(
            f"Expected folder {out_dir} to be empty unless --synthesize-overwrite is specified."
        )
    # Generate n_frames
    for i in range(n_frames):
        # Create a new image with a black background
        img = Image.new("RGB", (width, height), (0, 0, 0))
        draw = ImageDraw.Draw(img)
        # Draw the circle at its current position
        draw.ellipse(
            [(x - radius, y - radius), (x + radius, y + radius)], fill=circle_color
        )
        # Save the image as a JPEG file
        filename = f"{i:03d}.jpg"
        img.save(os.path.join(out_dir, filename))
        # Update the circle's position for the next frame
        x += vx
        y += vy
        # Bounce off the edges
        if x - radius < 0 or x + radius > width:
            vx *= -1
        if y - radius < 0 or y + radius > height:
            vy *= -1


def profiler_runner(path, fn, *args, **kwargs):
    if path is None:
        path = os.path.join(
            os.path.expanduser("~/traces"),
            f"{datetime.now().strftime('%Y-%m-%d-%H-%M-%S')}.json.gz",
        )
    with torch.profiler.profile(
        activities=[
            torch.profiler.ProfilerActivity.CPU,
            torch.profiler.ProfilerActivity.CUDA,
        ],
        record_shapes=True,
    ) as prof:
        result = fn(*args, **kwargs)
    prof.export_chrome_trace(path)
    timestamped_print(f"Exported trace to {path}")
    return result


def main_loop(
    predictor,
    inference_state,
    time_profile=True,
    accumulate_result=False,
    count_result=False,
):
    results = []
    num_output_frames = 0
    with torch.autograd.profiler.record_function("main_loop"):
        for out_frame_idx, out_obj_ids, out_mask_logits in predictor.propagate_in_video(
            inference_state
        ):
            if accumulate_result:
                results.append(out_mask_logits)
            if count_result:
                num_output_frames += 1
    assert not (accumulate_result and count_result)
    if accumulate_result:
        return torch.cat(results)
    if count_result:
        return num_output_frames


def timestamped_print(*args, **kwargs):
    # Get the current timestamp
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")
    # Prepend the timestamp to the original print arguments
    print(f"[{timestamp}]: ", *args, **kwargs)


def main(
    checkpoint_path: str,
    model_type: str,
    video_dir="/tmp/segment-anything-2/synth_video",
    profile=None,
    radius=50,
    seed=42,
    speed=20,
    width=1024,
    height=1024,
    n_frames=200,
    use_compile=False,
    batch_size=1,
    frame_batch_size=1,
    synthesize=False,
    synthesize_overwrite=False,
    store_output="",
    compare_output="",
    print_all_timings=False,
    use_baseline=False,
    export_model="",
    load_exported_model="",
    furious=False,
):
    np.random.seed(seed)
    start_x = np.random.randint(radius, width - radius)
    start_y = np.random.randint(radius, height - radius)
    if synthesize:
        for i in range(batch_size):
            synthesize_video_data(
                out_dir=f"{video_dir}_{i}",
                radius=radius,
                seed=(seed + i),  # Make sure every video is different
                speed=speed,
                width=width,
                height=height,
                n_frames=n_frames,
                x=start_x,
                y=start_y,
                synthesize_overwrite=synthesize_overwrite,
            )

    # use bfloat16 for the entire notebook
    torch.autocast(device_type="cuda", dtype=torch.bfloat16).__enter__()
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    sam2_checkpoint, model_cfg = model_type_to_paths(checkpoint_path, model_type)

    build_sam2_video_predictor = None
    if use_baseline:
        from sam2.build_sam import build_sam2_video_predictor
    else:
        from torchao._models.sam2.build_sam import build_sam2_video_predictor

    device = "cuda:0"
    # hydra_overrides_extra = ["++model.compile_image_encoder=true"]
    predictor = build_sam2_video_predictor(
        model_cfg,
        sam2_checkpoint,
        device=device,
        # hydra_overrides_extra=hydra_overrides_extra,
    )
    predictor._frame_batch_size = frame_batch_size
    predictor.image_encoder.trunk = predictor.image_encoder.trunk.to(torch.bfloat16)
    from torchao._models.sam2.modeling.sam.transformer import RoPEAttention

    rope_attention_modules = [
        module for module in predictor.modules() if isinstance(module, RoPEAttention)
    ]
    for r in rope_attention_modules:
        r.freqs_cis = r.compute_cis(end_x=64, end_y=64, device=device)

    inference_states = []
    for i in range(batch_size):
        inference_state = predictor.init_state(
            video_path=f"{video_dir}_{i}", async_loading_frames=False
        )
        _, out_obj_ids, out_mask_logits = predictor.add_new_points(
            inference_state=inference_state,
            frame_idx=0,
            obj_id=1,
            points=np.array([[start_x, start_y]], dtype=np.float32),
            labels=np.array([1], dtype=np.int32),
        )
        inference_states.append(inference_state)
    if batch_size == 1:
        inference_state = inference_states[0]
    else:
        inference_state = predictor.batch_inference_states(inference_states)

    if export_model != "":
        if not Path(export_model).is_dir():
            raise ValueError(f"Expected {export_model} to be a directory.")
        timestamped_print(f"Exporting model to {export_model}.")
        from compile_export_utils import export_model as export_model_fn

        export_model_fn(
            predictor,
            export_model,
            furious=furious,
            batch_size=1,
            overwrite=False,
        )

    if load_exported_model != "":
        from compile_export_utils import load_exported_model as load_exported_model_fn

        load_exported_model_fn(
            predictor, load_exported_model, furious=furious, batch_size=1
        )

    if use_compile:
        from compile_export_utils import set_fast

        set_fast(predictor, (load_exported_model != ""))

    timestamped_print("Warm-up round and gather outputs.")
    global_timer.reset()
    result = main_loop(
        predictor=predictor, inference_state=inference_state, accumulate_result=True
    )
    if store_output:
        timestamped_print(f"Writing results to {store_output}")
        torch.save(result, store_output)
    if compare_output:
        timestamped_print(f"Comparing to results from {compare_output}")
        ref_result = torch.load(compare_output)
        torch.testing.assert_close(result, ref_result)
        timestamped_print("Passed comparison!")
    if print_all_timings:
        global_timer.print_all_timings()

    global_timer.reset()
    if profile is None:
        timestamped_print("Practice round")
        main_loop(predictor=predictor, inference_state=inference_state)
    else:
        timestamped_print(f"Saving profile under {profile}")
        profiler_runner(
            profile,
            main_loop,
            predictor=predictor,
            inference_state=inference_state,
        )
    if print_all_timings:
        global_timer.print_all_timings()

    timestamped_print("Final timing and memory usage round.")
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    global_timer.reset()
    t0 = time.time()
    num_output_frames = main_loop(
        predictor=predictor, inference_state=inference_state, count_result=True
    )
    t = time.time() - t0
    timestamped_print(
        f"main_loop took {t}s for {num_output_frames} frames at {num_output_frames / t}fps"
    )
    max_memory_allocated()
    if print_all_timings:
        global_timer.print_all_timings()


if __name__ == "__main__":
    fire.Fire(main)
