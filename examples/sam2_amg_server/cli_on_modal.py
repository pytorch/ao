# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD 3-Clause license found in the
# LICENSE file in the root directory of this source tree.
import asyncio
import json
from pathlib import Path

import fire
import modal

TARGET = "/root/"
SAM2_GIT_SHA = "c2ec8e14a185632b0a5d8b161928ceb50197eddc"

image = (
    modal.Image.debian_slim(python_version="3.12.7")
    .pip_install("numpy<3", "tqdm")
    .pip_install(
        "torch",
        pre=True,
        index_url="https://download.pytorch.org/whl/nightly/cu126",
    )
    .pip_install(
        "torchvision",
        pre=True,
        index_url="https://download.pytorch.org/whl/nightly/cu124",
    )
    .apt_install("git")
    .apt_install("libopencv-dev")
    .apt_install("python3-opencv")
    .run_commands([f"git clone https://github.com/pytorch/ao.git {TARGET}ao_src_0"])
    .run_commands([f"cd {TARGET}ao_src_0; python setup.py develop"])
    .pip_install(
        "gitpython",
    )
    .apt_install("wget")
    .run_commands(
        [
            "wget https://raw.githubusercontent.com/pytorch/ao/refs/heads/main/examples/sam2_amg_server/requirements.txt"
        ]
    )
    .pip_install_from_requirements(
        "requirements.txt",
    )
    .pip_install(
        f"git+https://github.com/facebookresearch/sam2.git@{SAM2_GIT_SHA}",
    )
)

app = modal.App("torchao-sam-2-cli", image=image)

checkpoints = modal.Volume.from_name(
    "torchao-sam-2-cli-checkpoints", create_if_missing=True
)
data = modal.Volume.from_name("torchao-sam-2-cli-data", create_if_missing=True)
exported_models = modal.Volume.from_name(
    "torchao-sam-2-exported-models", create_if_missing=True
)
traces = modal.Volume.from_name("torchao-sam-2-traces", create_if_missing=True)


@app.cls(
    gpu="H100",
    container_idle_timeout=20 * 60,
    concurrency_limit=10,
    allow_concurrent_inputs=1,
    timeout=20 * 60,
    volumes={
        TARGET + "checkpoints": checkpoints,
        TARGET + "data": data,
        TARGET + "exported_models": exported_models,
        TARGET + "traces": traces,
    },
)
class Model:
    task_type: str = modal.parameter(default="amg")
    baseline: int = modal.parameter(default=0)

    @modal.build()
    @modal.enter()
    def build(self):
        import os

        import numpy as np
        import torch

        if self.baseline:
            from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator
            from sam2.build_sam import build_sam2
        else:
            from torchao._models.sam2.automatic_mask_generator import (
                SAM2AutomaticMaskGenerator,
            )
            from torchao._models.sam2.build_sam import build_sam2

        os.chdir(f"{TARGET}ao_src_0/examples/sam2_amg_server")
        import sys

        sys.path.append(".")

        from server import (
            file_bytes_to_image_tensor,
            masks_to_rle_dict,
            model_type_to_paths,
            profiler_runner,
            show_anns,
        )

        device = "cuda"
        checkpoint_path = Path(TARGET) / Path("checkpoints")
        sam2_checkpoint, model_cfg = model_type_to_paths(checkpoint_path, "large")
        sam2 = build_sam2(
            model_cfg, sam2_checkpoint, device=device, apply_postprocessing=False
        )
        points_per_batch = None
        if self.task_type == "amg":
            points_per_batch = 64 if self.baseline else 1024
        if self.task_type == "sps":
            points_per_batch = 1
        mask_generator = SAM2AutomaticMaskGenerator(
            sam2, points_per_batch=points_per_batch, output_mode="uncompressed_rle"
        )
        from compile_export_utils import load_exported_model

        export_model_path = Path(TARGET) / Path("exported_models")
        export_model_path = (
            export_model_path / Path("sam2") / Path(f"sam2_{self.task_type}")
        )
        if not self.baseline:
            load_exported_model(
                mask_generator,
                export_model_path,
                self.task_type,
                furious=True,
                batch_size=1,
                points_per_batch=points_per_batch,
            )
        self.mask_generator = mask_generator
        from torchvision import io as tio
        from torchvision.transforms.v2 import functional as tio_F

        if self.baseline:
            from sam2.utils.amg import mask_to_rle_pytorch as mask_to_rle_pytorch_2
            from sam2.utils.amg import rle_to_mask
        else:
            from torchao._models.sam2.utils.amg import (
                mask_to_rle_pytorch_2,
                rle_to_mask,
            )
        from torchao._models.sam2.utils.amg import area_from_rle

        self.np = np
        self.tio = tio
        self.tio_F = tio_F
        self.torch = torch
        self.masks_to_rle_dict = masks_to_rle_dict
        self.profiler_runner = profiler_runner
        self.file_bytes_to_image_tensor = file_bytes_to_image_tensor
        self.show_anns = show_anns
        self.mask_to_rle_pytorch_2 = mask_to_rle_pytorch_2
        self.area_from_rle = area_from_rle
        self.rle_to_mask = rle_to_mask

        from annotate_with_rle import _get_center_point

        self._get_center_point = _get_center_point

        # Baseline
        if self.baseline:
            from generate_data import gen_masks_baseline as gen_masks
        else:
            from generate_data import gen_masks_ao as gen_masks
        self.gen_masks = gen_masks

    def decode_img_bytes(self, img_bytes_tensor, baseline=False):
        import torch

        image_tensor = self.file_bytes_to_image_tensor(img_bytes_tensor)
        from torchvision.transforms import v2

        if not self.baseline:
            image_tensor = torch.from_numpy(image_tensor)
            image_tensor = image_tensor.permute((2, 0, 1))
            image_tensor = image_tensor.cuda()
            image_tensor = v2.ToDtype(torch.float32, scale=True)(image_tensor)
        return image_tensor

    @modal.web_endpoint(docs=True, method="POST")
    async def upload_rle(self, image):
        def upload_rle_inner(input_bytes):
            image_tensor = self.file_bytes_to_image_tensor(input_bytes)
            masks = self.mask_generator.generate(image_tensor)
            return self.masks_to_rle_dict(masks)

        return upload_rle_inner(bytearray(await image.read()))

    @modal.method()
    def inference_amg_rle(self, input_bytes) -> dict:
        image_tensor = self.decode_img_bytes(input_bytes)
        masks = self.gen_masks("amg", image_tensor, self.mask_generator)
        return self.masks_to_rle_dict(masks)

    @modal.method()
    def inference_amg_meta(self, input_bytes) -> dict:
        image_tensor = self.decode_img_bytes(input_bytes)
        masks = self.gen_masks("amg", image_tensor, self.mask_generator)
        rle_dict = self.masks_to_rle_dict(masks)
        masks = {}
        for key in rle_dict:
            masks[key] = {
                "segmentation": rle_dict[key],
                "area": self.area_from_rle(rle_dict[key]),
                "center_point": self._get_center_point(self.rle_to_mask(rle_dict[key])),
            }
        return masks

    @modal.method()
    def inference_sps_rle(self, input_bytes, prompts) -> dict:
        import numpy as np

        prompts = np.array(prompts)
        prompts_label = np.array([1] * len(prompts))
        image_tensor = self.decode_img_bytes(input_bytes)
        masks = self.gen_masks(
            "sps",
            image_tensor,
            self.mask_generator,
            center_points=prompts,
            center_points_label=prompts_label,
        )
        masks = self.mask_to_rle_pytorch_2(masks.unsqueeze(0))[0]
        masks = [{"segmentation": masks}]
        return self.masks_to_rle_dict(masks)

    @modal.method()
    def inference_mps_rle(self, input_bytes, prompts) -> dict:
        import numpy as np

        prompts = np.array(prompts)
        prompts_label = np.array([1] * len(prompts))
        image_tensor = self.decode_img_bytes(input_bytes)
        masks = self.gen_masks(
            "mps",
            image_tensor,
            self.mask_generator,
            center_points=prompts,
            center_points_label=prompts_label,
        )
        masks = self.mask_to_rle_pytorch_2(masks)
        masks = [{"segmentation": mask} for mask in masks]
        return self.masks_to_rle_dict(masks)

    def plot_image_tensor(self, image_tensor, masks, output_format, prompts=None):
        from io import BytesIO

        import matplotlib.pyplot as plt

        fig = plt.figure(
            figsize=(image_tensor.shape[1] / 100.0, image_tensor.shape[0] / 100.0),
            dpi=100,
        )
        plt.imshow(image_tensor)
        self.show_anns(masks, self.rle_to_mask, sort_by_area=False, seed=42)
        plt.axis("off")
        plt.tight_layout()
        if prompts is not None:
            ax = plt.gca()
            marker_size = 375
            ax.scatter(
                prompts[:, 0],
                prompts[:, 1],
                color="green",
                marker="*",
                s=marker_size,
                edgecolor="white",
                linewidth=1.25,
            )
        buf = BytesIO()
        plt.savefig(buf, format=output_format)
        buf.seek(0)
        result = buf.getvalue()
        plt.close(fig)
        return result

    @modal.method()
    def inference_amg(self, input_bytes, output_format="png"):
        image_tensor = self.decode_img_bytes(input_bytes)
        masks = self.gen_masks("amg", image_tensor, self.mask_generator)
        return self.plot_image_tensor(image_tensor, masks, output_format)

    @modal.method()
    def inference_sps(self, input_bytes, prompts, output_format="png"):
        import numpy as np

        prompts = np.array(prompts)
        prompts_label = np.array([1] * len(prompts))
        image_tensor = self.decode_img_bytes(input_bytes)
        masks = self.gen_masks(
            "sps",
            image_tensor,
            self.mask_generator,
            center_points=prompts,
            center_points_label=prompts_label,
        )
        masks = self.mask_to_rle_pytorch_2(masks.unsqueeze(0))[0]
        masks = [{"segmentation": masks}]
        return self.plot_image_tensor(
            image_tensor, masks, output_format, prompts=prompts
        )

    @modal.method()
    def inference_mps(self, input_bytes, prompts, output_format="png"):
        import numpy as np

        prompts = np.array(prompts)
        prompts_label = np.array([1] * len(prompts))
        image_tensor = self.decode_img_bytes(input_bytes)
        masks = self.gen_masks(
            "mps",
            image_tensor,
            self.mask_generator,
            center_points=prompts,
            center_points_label=prompts_label,
        )
        masks = self.mask_to_rle_pytorch_2(masks)
        masks = [{"segmentation": mask} for mask in masks]
        return self.plot_image_tensor(
            image_tensor, masks, output_format, prompts=prompts
        )


def get_center_points(task_type, meta_path):
    with open(meta_path, "r") as file:
        amg_masks = list(json.load(file).values())
        amg_masks = sorted(amg_masks, key=(lambda x: x["area"]), reverse=True)
        # center points for biggest area first.
        center_points = [mask["center_point"] for mask in amg_masks]
        if task_type == "sps":
            center_points = center_points[:1]
        return center_points


def timed_print(msg):
    from datetime import datetime

    current_time = datetime.now()
    timestamp_with_nanoseconds = (
        current_time.strftime("%Y-%m-%d %H:%M:%S.")
        + f"{current_time.microsecond * 1000:09d}"
    )
    print(f"{str(timestamp_with_nanoseconds)}: {msg}")


def main(
    task_type,
    input_paths,
    output_directory,
    output_rle=False,
    output_meta=False,
    meta_paths=None,
    baseline=False,
    name=None,
):
    assert task_type in ["amg", "sps", "mps"]
    if task_type in ["sps", "mps"]:
        assert meta_paths is not None
    input_paths = open(input_paths).read().split("\n")[:-1]
    for input_path in input_paths:
        assert Path(input_path).exists()

    output_directory = Path(output_directory)
    if not (output_directory.exists() and output_directory.is_dir()):
        raise ValueError(
            f"Expected output_directory {output_directory} to be a directory and exist"
        )

    if meta_paths is not None:
        meta_mapping = {}
        meta_paths = open(meta_paths).read().split("\n")[:-1]
        for meta_path in meta_paths:
            assert Path(meta_path).exists()
            key = Path(meta_path).name.split("_meta.json")[0]
            key = f"{Path(meta_path).parent.name}/{key}"
            meta_mapping[key] = meta_path

    try:
        if name is None:
            name = "torchao-sam-2-cli"
        model = modal.Cls.lookup(name, "Model")
        model = model(task_type=task_type, baseline=int(baseline))
    except modal.exception.NotFoundError:
        print(
            "Can't find running app. To deploy the app run the following",
            "command. Note that this costs money!",
            "See https://modal.com/pricing",
        )
        print("modal deploy cli_on_modal.py")
        return

    outputs = []
    output_paths = []
    timed_print(f"Queueing {len(input_paths)} tasks...")
    for input_path in input_paths:
        key = Path(input_path).name.split(".jpg")[0]
        key = f"{Path(input_path).parent.name}/{key}"
        if meta_paths is not None:
            meta_path = meta_mapping[key]
            center_points = get_center_points(task_type, meta_path)

        input_bytes = bytearray(open(input_path, "rb").read())

        output_path = output_directory / Path(key)
        output_paths.append(str(output_path))
        output_path.parent.mkdir(parents=False, exist_ok=True)
        if output_meta:
            assert task_type == "amg"
            outputs.append(model.inference_amg_meta.remote.aio(input_bytes))
        elif output_rle:
            if task_type == "amg":
                outputs.append(model.inference_amg_rle.remote.aio(input_bytes))
            if task_type == "sps":
                outputs.append(
                    model.inference_sps_rle.remote.aio(input_bytes, center_points)
                )
            if task_type == "mps":
                outputs.append(
                    model.inference_mps_rle.remote.aio(input_bytes, center_points)
                )
        else:
            if task_type == "amg":
                outputs.append(model.inference_amg.remote.aio(input_bytes))
            if task_type == "sps":
                outputs.append(
                    model.inference_sps.remote.aio(input_bytes, center_points)
                )
            if task_type == "mps":
                outputs.append(
                    model.inference_mps.remote.aio(input_bytes, center_points)
                )

    async def run_all(outputs):
        outputs = await asyncio.gather(*outputs)
        return outputs

    timed_print("Awaiting tasks...")
    outputs = asyncio.run(run_all(outputs))

    timed_print("Processing task output...")
    for output, output_path in zip(outputs, output_paths):
        if output_meta:
            with open(f"{output_path}_meta.json", "w") as file:
                file.write(json.dumps(output, indent=4))
        elif output_rle:
            with open(f"{output_path}_masks.json", "w") as file:
                file.write(json.dumps(output, indent=4))
        else:
            with open(f"{output_path}_annotated.png", "wb") as file:
                file.write(output)
    timed_print("Done.")


if __name__ == "__main__":
    fire.Fire(main)
