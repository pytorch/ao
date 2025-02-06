import json
import time
from collections import OrderedDict
from datetime import datetime
from pathlib import Path

import fire
import numpy as np
import torch

# from compile_export_utils import set_aot_fast
from compile_export_utils import set_fast, set_furious
from server import (
    MODEL_TYPES_TO_MODEL,
    file_bytes_to_image_tensor,
    masks_to_rle_dict,
    max_memory_allocated_stats,
    model_type_to_paths,
)
from torch.autograd.profiler import record_function
from tqdm import tqdm


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


def latencies_statistics(data):
    # Convert the list to a NumPy array
    data_array = np.array(data)
    # Calculate the mean
    mean = np.mean(data_array)
    # Calculate the median
    median = np.median(data_array)
    # Calculate the 95th percentile
    p95 = np.percentile(data_array, 95)
    # Calculate the 99th percentile
    p99 = np.percentile(data_array, 99)
    # Calculate the 99.9th percentile
    p999 = np.percentile(data_array, 99.9)
    # Calculate the highest number
    max = np.max(data_array)
    # Calculate the experiment id
    argmax = int(np.argmax(data_array))
    statistics_dict = OrderedDict(
        {
            "mean": mean,
            "median": median,
            "p95": p95,
            "p99": p99,
            "p999": p999,
            "max": max,
            "argmax": argmax,
            "first": data[0] if len(data) > 0 else 0,
            "second": data[1] if len(data) > 1 else 0,
            "third": data[2] if len(data) > 2 else 0,
            "fourth": data[3] if len(data) > 3 else 0,
            "fifth": data[4] if len(data) > 4 else 0,
        }
    )
    return statistics_dict


def timestamped_print(*args, **kwargs):
    # Get the current timestamp
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")
    # Prepend the timestamp to the original print arguments
    print(f"[{timestamp}]", *args, **kwargs)


@record_function("generate_baseline")
def gen_masks_baseline(
    task_type,
    image_tensor,
    mask_generator,
    center_points=None,
    center_points_label=None,
):
    if task_type != "amg":
        assert center_points is not None
        assert center_points_label is not None
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
        masks = torch.from_numpy(masks[np.argmax(scores).item()])
        masks = masks.to(torch.bool)
    elif task_type == "mps":
        mask_generator.predictor.set_image(image_tensor)
        masks = []
        for i in range(len(center_points)):
            mask, score, _ = mask_generator.predictor.predict(
                point_coords=center_points[i : i + 1],
                point_labels=center_points_label[i : i + 1],
                multimask_output=True,
                return_logits=False,
            )
            mask = torch.from_numpy(mask[np.argmax(score).item()])
            mask = mask.to(torch.bool)
            masks.append(mask)
        masks = torch.stack(masks)
    return masks


@record_function("generate_ao_batch")
def gen_masks_ao_batch(
    task_type,
    image_tensors,
    mask_generator,
    batch_size,
    center_points_batch=None,
    center_points_label_batch=None,
):
    assert isinstance(image_tensors, list)
    assert len(image_tensors) <= batch_size
    # NOTE: We could create a smaller padding image of 0 size, but
    # image transforms will resize to full size anyway.
    image_tensors += [image_tensors[-1]] * (batch_size - len(image_tensors))
    assert len(image_tensors) == batch_size
    if center_points_batch is not None:
        center_points_batch += (center_points_batch[-1],) * (
            batch_size - len(center_points_batch)
        )
        assert len(center_points_batch) == batch_size
    if center_points_label_batch is not None:
        center_points_label_batch += (center_points_label_batch[-1],) * (
            batch_size - len(center_points_label_batch)
        )
        assert len(center_points_label_batch) == batch_size
    if task_type == "amg":
        return mask_generator.generate_batch(image_tensors)
    elif task_type == "sps":
        mask_generator.predictor.set_image_batch(image_tensors)
        masks, scores, _ = mask_generator.predictor.predict_batch(
            point_coords_batch=center_points_batch,
            point_labels_batch=center_points_label_batch,
            multimask_output=True,
            return_logits=False,
            return_type="torch",
        )
        # TODO: This isn't exactly efficient
        masks = [
            m[0][s]
            for (m, s) in zip(
                masks, torch.stack(scores).squeeze(1).argmax(dim=1).tolist()
            )
        ]
        return masks
    elif task_type == "mps":
        mask_generator.predictor.set_image_batch(image_tensors)

        center_points_torch_batch = [
            torch.from_numpy(t).unsqueeze(1) for t in center_points_batch
        ]
        center_points_label_torch_batch = [
            torch.from_numpy(t).unsqueeze(1) for t in center_points_label_batch
        ]
        from torchao._models.sam2.map_tensor import to_map_tensor

        center_points_torch_batch = list(map(to_map_tensor, center_points_torch_batch))
        center_points_label_torch_batch = list(
            map(to_map_tensor, center_points_label_torch_batch)
        )
        masks_batch, scores_batch, _ = mask_generator.predictor.predict_batch(
            point_coords_batch=center_points_torch_batch,
            point_labels_batch=center_points_label_torch_batch,
            multimask_output=True,
            return_logits=False,
            return_type="torch",
        )
        result_masks = []
        for masks_m, scores_m in zip(masks_batch, scores_batch):
            # Unwrapping MapTensor
            masks = masks_m.elems.squeeze(1)
            scores = scores_m.elems.squeeze(1)
            # TODO: This isn't exactly efficient
            result_masks.append(
                torch.stack(
                    [
                        mask[i]
                        for (mask, i) in zip(
                            masks.unbind(), torch.argmax(scores, dim=1).tolist()
                        )
                    ]
                )
            )
        return result_masks
    raise ValueError("gen_masks_ao_batch doesn't support {task_type}")


@record_function("generate_ao")
def gen_masks_ao(
    task_type,
    image_tensor,
    mask_generator,
    center_points=None,
    center_points_label=None,
):
    if task_type == "amg":
        masks = mask_generator.generate(image_tensor)
    elif task_type == "sps":
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
        masks = torch.stack(
            [
                mask[i]
                for (mask, i) in zip(
                    masks.unbind(), torch.argmax(scores, dim=1).tolist()
                )
            ]
        )
    return masks


def gen_masks(
    task_type,
    image_tensors,
    mask_generator,
    center_points_batch,
    center_points_label_batch,
    baseline,
    verbose,
    batch_size,
):
    if verbose:
        for image_tensor in image_tensors:
            timestamped_print(f"Generating mask of size {tuple(image_tensor.shape)}.")
    if baseline:
        masks_batch = []
        for data_i in zip(
            image_tensors, center_points_batch, center_points_label_batch
        ):
            (image_tensor, center_points, center_points_label) = data_i
            masks = gen_masks_baseline(
                task_type,
                image_tensor,
                mask_generator,
                center_points,
                center_points_label,
            )
            masks_batch.append(masks)
        return masks_batch
    if batch_size > 1 and task_type in ["amg", "sps", "mps"]:
        return gen_masks_ao_batch(
            task_type,
            image_tensors,
            mask_generator,
            batch_size,
            center_points_batch,
            center_points_label_batch,
        )
    masks_batch = []
    for data_i in zip(image_tensors, center_points_batch, center_points_label_batch):
        (image_tensor, center_points, center_points_label) = data_i
        masks = gen_masks_ao(
            task_type, image_tensor, mask_generator, center_points, center_points_label
        )
        masks_batch.append(masks)
    assert len(masks_batch) == 1
    return masks_batch


def data_from_file_path(
    task_type, verbose, meta_path, input_path, gpu_preproc, baseline
):
    center_points, center_points_label = None, None
    if task_type != "amg":
        if verbose:
            timestamped_print(f"Loading meta from {meta_path}")
        with open(meta_path, "r") as file:
            amg_masks = list(json.load(file).values())
            amg_masks = sorted(amg_masks, key=(lambda x: x["area"]), reverse=True)
            # center points for biggest area first.
            center_points = [mask["center_point"] for mask in amg_masks]
            center_points = np.array(center_points)
            center_points_label = np.array(len(center_points) * [1])
            if task_type == "sps":
                center_points = center_points[:1]
                center_points_label = center_points_label[:1]

    with record_function("load image bytes from disk"):
        if gpu_preproc:
            # NOTE: We have to use numpy for the baseline
            assert not baseline
            from torchvision import io as tio

            img_bytes_tensor = tio.read_file(input_path)
        else:
            img_bytes_tensor = bytearray(open(input_path, "rb").read())

    return img_bytes_tensor, center_points, center_points_label


def decode_img_bytes(img_bytes_tensors, gpu_preproc, baseline):
    image_tensors = []
    for img_bytes_tensor in img_bytes_tensors:
        with record_function("decode image bytes"):
            if gpu_preproc:
                image_tensor = file_bytes_to_image_tensor(img_bytes_tensor)
                from torchvision.transforms import ToTensor, v2

                if not baseline:
                    image_tensor = torch.from_numpy(image_tensor)
                    image_tensor = image_tensor.permute((2, 0, 1))
                    image_tensor = image_tensor.cuda()
                    with record_function("v2.ToDtype"):
                        image_tensor = v2.ToDtype(torch.float32, scale=True)(
                            image_tensor
                        )
            else:
                image_tensor = file_bytes_to_image_tensor(img_bytes_tensor)
                from torchvision.transforms import ToTensor

                if not baseline:
                    image_tensor = ToTensor()(image_tensor)
            image_tensors.append(image_tensor)
    return image_tensors


def rle_dict_from_masks(task_type, masks, mask_to_rle_pytorch):
    with record_function("mask_to_rle_pytorch"):
        if task_type == "sps":
            masks = mask_to_rle_pytorch(masks.unsqueeze(0))[0]
            masks = [{"segmentation": masks}]
        elif task_type == "mps":
            masks = mask_to_rle_pytorch(masks)
            masks = [{"segmentation": mask} for mask in masks]

    with record_function("masks_to_rle_dict"):
        rle_dict = masks_to_rle_dict(masks)

    return rle_dict


def save_rle_dict_to_path(rle_dict, output_rle_json_path, verbose):
    with record_function("json.dumps"):
        if verbose:
            timestamped_print(f"Storing rle under {output_rle_json_path}")
        output_rle_json_path.parent.mkdir(parents=False, exist_ok=True)
        with open(output_rle_json_path, "w") as file:
            file.write(json.dumps(rle_dict, indent=4))


def batched_zip(
    input_paths, output_image_paths, output_rle_json_paths, meta_paths, batch_size
):
    i = 0
    batch = []
    for input_path, output_image_path, output_rle_json_path, meta_path in zip(
        input_paths, output_image_paths, output_rle_json_paths, meta_paths
    ):
        if i == batch_size:
            yield batch
            i = 0
            batch = []
        batch.append((input_path, output_image_path, output_rle_json_path, meta_path))
        i += 1
    if len(batch) > 0:
        yield batch


# AMG: Automatic mask generation
# SPS: Single point segmentation
# MPS: Multi point segmentation


def main_docstring():
    return f"""
    Args:
        checkpoint_path (str): Path to folder containing checkpoints from https://github.com/facebookresearch/sam2?tab=readme-ov-file#download-checkpoints
        model_type (str): Choose from one of {", ".join(MODEL_TYPES_TO_MODEL.keys())}
        input_path (str): Path to input image
        output_path (str): Path to output image
    """


TASK_TYPES = ["amg", "sps", "mps"]


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
    allow_recompiles=False,
    quiet=False,
    gpu_preproc=False,
    batch_size=1,
    seed=42,
):
    if batch_size <= 0:
        raise ValueError("Expected --batch_size to be at least 1 but got {batch_size}")
    start_time = time.time()
    if task_type not in TASK_TYPES:
        raise ValueError(
            f"Expected task_type to be one of {','.join(TASK_TYPES)}, but got {task_type}"
        )
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
        from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator
        from sam2.build_sam import build_sam2
        from sam2.utils.amg import mask_to_rle_pytorch
    else:
        from torchao._models.sam2.automatic_mask_generator import (
            SAM2AutomaticMaskGenerator,
        )
        from torchao._models.sam2.build_sam import build_sam2
        from torchao._models.sam2.utils.amg import (
            mask_to_rle_pytorch_2 as mask_to_rle_pytorch,
        )
    torch.manual_seed(seed)
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

        export_model_fn(
            mask_generator,
            export_model,
            task_type,
            furious=furious,
            batch_size=batch_size,
            points_per_batch=points_per_batch,
            overwrite=overwrite,
        )
    if load_exported_model == "":
        if furious:
            set_furious(mask_generator)
    else:
        from compile_export_utils import load_exported_model as load_exported_model_fn

        load_exported_model_fn(
            mask_generator,
            load_exported_model,
            task_type,
            furious=furious,
            batch_size=batch_size,
            points_per_batch=points_per_batch,
        )
    if fast:
        set_fast(
            mask_generator,
            task_type,
            loaded_exported_model=(load_exported_model != ""),
            allow_recompiles=allow_recompiles,
        )

    # TODO: Write out an optional unit test based on dog.jpg and rerun
    latencies = []
    num_images = len(input_paths) if num_images is None else num_images
    num_batches = (num_images + batch_size - 1) // batch_size
    input_paths = input_paths[:num_images]

    all_input_paths = input_paths
    all_output_image_paths = output_image_paths
    all_output_rle_json_paths = output_rle_json_paths
    all_meta_paths = meta_paths

    for batch in tqdm(
        batched_zip(
            all_input_paths,
            all_output_image_paths,
            all_output_rle_json_paths,
            all_meta_paths,
            batch_size,
        ),
        total=num_batches,
        disable=quiet,
    ):
        data = []
        for input_path, _, _, meta_path in batch:
            # img_bytes_tensor, center_points, center_points_label
            data.append(
                data_from_file_path(
                    task_type, verbose, meta_path, input_path, gpu_preproc, baseline
                )
            )

        # We're including decoding the image, but not
        # disk I/O in our latency calculation
        t1 = time.time()

        image_tensors = decode_img_bytes(list(zip(*data))[0], gpu_preproc, baseline)

        masks_batch = gen_masks(
            task_type,
            image_tensors,
            mask_generator,
            list(zip(*data))[1],  # center_points
            list(zip(*data))[2],  # center_points_label
            baseline,
            verbose,
            batch_size,
        )

        rle_dicts = []
        for masks in masks_batch:
            rle_dicts.append(rle_dict_from_masks(task_type, masks, mask_to_rle_pytorch))

        latencies.append((time.time() - t1) / len(batch))

        for (rle_dict), (_, _, output_rle_json_path, _) in zip(rle_dicts, batch):
            save_rle_dict_to_path(rle_dict, output_rle_json_path, verbose)

    end_time = time.time()
    total_time = end_time - start_time
    all_stats = {}
    all_stats["batch_size"] = batch_size
    all_stats["total_time"] = f"{total_time}s"
    all_stats["total_img_s"] = f"{len(input_paths) / total_time}img/s"
    if len(input_paths) > 0:
        all_stats["total_ms_per_img"] = f"{total_time / len(input_paths) * 1000}ms"

        for key, value in latencies_statistics(latencies).items():
            all_stats[key] = str(value)
            if not isinstance(value, int):
                all_stats[key] = str(int(value * 1000)) + "ms"

    mma_stats = max_memory_allocated_stats()
    mma_stats["bytes_MiB"] = mma_stats["bytes"] >> 20
    print(json.dumps(all_stats | mma_stats))


main.__doc__ = main_docstring()
if __name__ == "__main__":
    # profiler_runner("asdf.json.gz", fire.Fire, main)
    # memory_runner("asdf.pickle", fire.Fire, main)
    fire.Fire(main)
