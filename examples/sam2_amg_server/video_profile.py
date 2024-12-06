import argparse
import time
import os
from datetime import datetime

import numpy as np
import torch
from PIL import Image, ImageDraw
from torchao._models.sam2.build_sam import build_sam2_video_predictor
from server import MODEL_TYPES_TO_MODEL
from server import model_type_to_paths
from pathlib import Path

from torch._inductor import config as inductorconfig
inductorconfig.triton.unique_kernel_names = True
inductorconfig.coordinate_descent_tuning = True
inductorconfig.coordinate_descent_check_all_directions = True

from torch.nn.attention import SDPBackend, sdpa_kernel

# timer.py
import time
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
            print("No timings recorded.")
            return
        print("Average timings for all sections:")
        for section_name in self.elapsed_times:
            average_time = self.get_average_time(section_name, warmup)
            print(f"{section_name}, {average_time*1000.0:.6f}")


global_timer = CodeTimer()


def max_memory_allocated():
    max_memory_allocated_bytes = torch.cuda.max_memory_allocated()
    _, total_memory = torch.cuda.mem_get_info()
    max_memory_allocated_percentage = int(100 * (max_memory_allocated_bytes / total_memory))
    max_memory_allocated_bytes = max_memory_allocated_bytes >> 20
    print(f"max_memory_allocated_bytes: {max_memory_allocated_bytes}MiB or {max_memory_allocated_percentage}%")


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
    print(f"Generate {n_frames} frames")
    if not synthesize_overwrite and len(os.listdir(out_dir)) > 0:
        raise ValueError("Expected folder to be empty unless --synthesize-overwrite is specified.")
    # Generate 100 frames
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
            f'{datetime.now().strftime("%Y-%m-%d-%H-%M-%S")}.json.gz',
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
    print(f"Exported trace to {path}")
    return result


def main_loop(predictor, inference_state, time_profile=True, accumulate_result=False, count_result=False):
    results = []
    num_output_frames = 0
    with sdpa_kernel([SDPBackend.CUDNN_ATTENTION, SDPBackend.FLASH_ATTENTION]):
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


def run_test(
    checkpoint_path: str,
    model_type: str,
    profile: bool,
    video_dir: str,
    radius: int,
    seed: int,
    speed: int,
    width: int,
    height: int,
    n_frames: int,
    use_compile: bool,
    frame_batch_size: int,
    synthesize: bool,
    synthesize_overwrite: bool,
    store_output: str,
    compare_output: str,
    print_all_timings: bool,
):
    np.random.seed(seed)
    start_x = np.random.randint(radius, width - radius)
    start_y = np.random.randint(radius, height - radius)
    if synthesize:
        synthesize_video_data(
            out_dir=video_dir,
            radius=radius,
            seed=seed,
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

    device = "cuda:0"
    # hydra_overrides_extra = ["++model.compile_image_encoder=true"]
    predictor = build_sam2_video_predictor(
        model_cfg,
        sam2_checkpoint,
        device=device,
        # hydra_overrides_extra=hydra_overrides_extra,
    )
    predictor._frame_batch_size = frame_batch_size

    inference_state = predictor.init_state(
        video_path=video_dir, async_loading_frames=False
    )
    _, out_obj_ids, out_mask_logits = predictor.add_new_points(
        inference_state=inference_state,
        frame_idx=0,
        obj_id=1,
        points=np.array([[start_x, start_y]], dtype=np.float32),
        labels=np.array([1], dtype=np.int32),
    )

    if use_compile:
        print("Using torch.compile")
        predictor.image_encoder.trunk.forward = torch.compile(
                predictor.image_encoder.trunk.forward,
                # mode="max-autotune-no-cudagraphs",
                mode="max-autotune",
                fullgraph=True,
                dynamic=False,
            )

        predictor.sam_prompt_encoder.forward = torch.compile(
                predictor.sam_prompt_encoder.forward,
                # mode="max-autotune-no-cudagraphs",
                mode="max-autotune",
                fullgraph=True,
                dynamic=False,
            )

        predictor.sam_mask_decoder.transformer = torch.compile(
                predictor.sam_mask_decoder.transformer,
                mode="max-autotune",
                # mode="max-autotune-no-cudagraphs",
                fullgraph=True,
                dynamic=False,
            )

        predictor._forward_sam_heads = torch.compile(
                predictor._forward_sam_heads,
                mode="max-autotune",
                # mode="max-autotune-no-cudagraphs",
                fullgraph=True,
                dynamic=False,
            )

        predictor.memory_attention = torch.compile(
                predictor.memory_attention,
                # mode="max-autotune",
                # mode="max-autotune-no-cudagraphs",
                fullgraph=True,
                dynamic=True,
            )

        predictor.memory_encoder.forward = torch.compile(
                predictor.memory_encoder.forward,
                mode="max-autotune",
                # mode="max-autotune-no-cudagraphs",
                fullgraph=True,
                dynamic=False,
            )

    print("\nWarm-up round and gather outputs.")
    global_timer.reset()
    result = main_loop(predictor=predictor, inference_state=inference_state, accumulate_result=True)
    if store_output:
        print(f"Writing results to {store_output}")
        torch.save(result, store_output)
    if compare_output:
        print(f"Comparing to results from {compare_output}")
        ref_result = torch.load(compare_output)
        torch.testing.assert_close(result, ref_result)
        print("Passed comparison!")
    if print_all_timings:
        global_timer.print_all_timings()

    global_timer.reset()
    print("\nProfile round.")
    if profile is None:
        main_loop(predictor=predictor, inference_state=inference_state)
    else:
        profiler_runner(
            profile,
            main_loop,
            predictor=predictor,
            inference_state=inference_state,
        )
    if print_all_timings:
        global_timer.print_all_timings()

    print("\nFinal timing and memory usage round.")
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    global_timer.reset()
    t0 = time.time()
    num_output_frames = main_loop(predictor=predictor, inference_state=inference_state, count_result=True)
    t = time.time() - t0
    print(f"main_loop took {t}s for {num_output_frames} frames at {num_output_frames / t}fps")
    max_memory_allocated()
    if print_all_timings:
        global_timer.print_all_timings()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "checkpoint_path",
        type=str,
        help="Path to folder containing checkpoints from https://github.com/facebookresearch/sam2?tab=readme-ov-file#download-checkpoints",
    )
    parser.add_argument(
        "model_type",
        type=str,
        help=f"Choose one of {list(MODEL_TYPES_TO_MODEL.keys())}",
    )
    parser.add_argument(
        "--video_dir",
        type=str,
        default="/tmp/segment-anything-2/synth_video",
        help="Directory to store the synthetic video",
    )
    parser.add_argument(
        "--profile",
        type=str,
        dest="profile",
        help="If specified stores profile at given path.",
    )
    parser.add_argument(
        "--radius",
        type=int,
        default=50,
        help="Radius of the circle for synthetic video",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Seed for initial position and velocity",
    )
    parser.add_argument(
        "--speed", type=int, default=20, help="Speed of the circle for synthetic video"
    )
    parser.add_argument(
        "--width", type=int, default=1024, help="Width of the synthetic video"
    )
    parser.add_argument(
        "--height", type=int, default=1024, help="Height of the synthetic video"
    )
    parser.add_argument(
        "--n_frames",
        type=int,
        default=200,
        help="Number of frames in the synthetic video",
    )
    parser.add_argument(
        "--use-compile",
        action="store_true",
        dest="use_compile",
        help="Use torch.compile to speed things up. First iteration will be much slower.",
    )
    parser.add_argument(
        "--frame_batch_size",
        type=int,
        default=1,
        help="frame_batch_size",
    )
    parser.add_argument(
        "--synthesize",
        action="store_true",
        dest="synthesize",
        help="Synthesize data for the benchmark.",
    )
    parser.add_argument(
        "--synthesize-overwrite",
        action="store_true",
        dest="synthesize_overwrite",
        help="Overwrite data if it already exists when synthesizing.",
    )
    parser.add_argument(
        "--store-output",
        type=str,
        default="",
        help="Pass a .pt file to store outputs in.",
    )
    parser.add_argument(
        "--compare-output",
        type=str,
        default="",
        help="Pass a .pt file to load for comparison.",
    )
    parser.add_argument(
        "--print-all-timings",
        action="store_true",
        dest="print_all_timings",
        help="Use torch.compile to speed things up. First iteration will be much slower.",
    )

    args = parser.parse_args()

    run_test(
        args.checkpoint_path,
        args.model_type,
        profile=args.profile,
        video_dir=args.video_dir,
        radius=args.radius,
        seed=args.seed,
        speed=args.speed,
        width=args.width,
        height=args.height,
        n_frames=args.n_frames,
        use_compile=args.use_compile,
        frame_batch_size=args.frame_batch_size,
        synthesize=args.synthesize,
        synthesize_overwrite=args.synthesize_overwrite,
        store_output=args.store_output,
        compare_output=args.compare_output,
        print_all_timings=args.print_all_timings,
    )
