# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD 3-Clause license found in the
# LICENSE file in the root directory of this source tree.
import time
from pathlib import Path
from typing import Optional

import torch

from torchao._models.sam2.sam2_video_predictor import SAM2VideoPredictor

# Tools used to avoid compilation cold start and dynamo cache lookups
# We take the compiled model and export it using the largest
# inputs possible (to avoid recompilations).
# We track the largest size and fail if we size something larger
# We export every compile-able subregion after wrapping it into
# a class to make export happy.

TASK_TYPES = ["amg", "sps", "mps"]


class SAM2VideoPredictor_forward_sam_heads(torch.nn.Module):
    def __init__(
        self,
        predictor: Optional[SAM2VideoPredictor],
        batch_size=1,
        aoti_compiled_model=None,
        furious=False,
    ):
        super().__init__()
        self.predictor = predictor
        self.batch_size = batch_size
        self.aoti_compiled_model = aoti_compiled_model
        self.furious = furious

    def forward(
        self,
        backbone_features,
        point_inputs=None,
        mask_inputs=None,
        high_res_features=None,
        multimask_output=False,
    ):
        assert mask_inputs is None
        assert multimask_output
        if self.predictor is None:
            assert self.aoti_compiled_model is not None
            return self.aoti_compiled_model(
                backbone_features=backbone_features,
                point_inputs=point_inputs,
                mask_inputs=mask_inputs,
                high_res_features=high_res_features,
                multimask_output=multimask_output,
            )
        return self.predictor._forward_sam_heads(
            backbone_features=backbone_features,
            point_inputs=point_inputs,
            mask_inputs=mask_inputs,
            high_res_features=high_res_features,
            multimask_output=multimask_output,
        )


def aot_compile(
    model_directory,
    name,
    fn,
    sample_args,
    sample_kwargs=None,
    options=None,
    overwrite=False,
):
    path = Path(model_directory) / Path(f"{name}.pt2")
    if path.exists() and not overwrite:
        raise ValueError(f"{path} already exists and overwrite is {overwrite}")
    print(f"Saving at {path=}")
    if options is None:
        options = {
            "max_autotune": True,
            "triton.cudagraphs": True,
        }

    from torch.export import export_for_training

    exported = export_for_training(fn, sample_args, sample_kwargs, strict=True)
    exported.run_decompositions()
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


def export_model(
    predictor,
    model_directory,
    furious=False,
    batch_size=1,
    overwrite=False,
):
    if furious:
        set_furious(predictor)

    example_input = torch.empty(batch_size, 3, 1024, 1024)
    # example_input = example_input.to(predictor._image_dtype)
    example_input = example_input.to(torch.bfloat16)
    # example_input = (example_input.to(predictor.device),)
    example_input = (example_input.to("cuda:0"),)
    aot_compile(
        model_directory,
        "sam2_image_encoder_trunk",
        predictor.image_encoder.trunk,
        example_input,
        overwrite=overwrite,
    )

    example_input_args = ()
    example_input_kwargs = {
        "backbone_features": torch.randn(
            batch_size, 256, 64, 64, dtype=torch.float32, device="cuda"
        ),
        # "point_inputs": {
        #     "point_coords": torch.ones(batch_size, 1, 2, dtype=torch.float32, device="cuda"),
        #     "point_labels": torch.ones(batch_size, 1, dtype=torch.int32, device="cuda"),
        # },
        "point_inputs": None,
        "mask_inputs": None,
        "high_res_features": [
            torch.randn(
                batch_size,
                32,
                256,
                256,
                dtype=torch.bfloat16,
                device="cuda",
            ),
            torch.randn(
                batch_size,
                64,
                128,
                128,
                dtype=torch.bfloat16,
                device="cuda",
            ),
        ],
        "multimask_output": True,
    }
    sam2_video_forward_sam_heads = SAM2VideoPredictor_forward_sam_heads(
        predictor,
        batch_size=batch_size,
        furious=False,
    )
    aot_compile(
        model_directory,
        "sam2_video_forward_sam_heads",
        sam2_video_forward_sam_heads,
        example_input_args,
        sample_kwargs=example_input_kwargs,
        overwrite=overwrite,
    )

    return predictor


class LoadedModel(torch.nn.Module):
    def __init__(self, aoti_compiled_model):
        super().__init__()
        self.aoti_compiled_model = aoti_compiled_model

    def forward(self, *args, **kwargs):
        return self.aoti_compiled_model(*args, **kwargs)


class LoadedDecoder(torch.nn.Module):
    def __init__(self, aoti_compiled_model, other):
        super().__init__()
        self.aoti_compiled_model = aoti_compiled_model
        self.other = other

    def forward(self, *args):
        return self.aoti_compiled_model(*args)

    def get_dense_pe(self, *args, **kwargs) -> torch.Tensor:
        return self.other.get_dense_pe(*args, **kwargs)


def load_exported_model(
    predictor,
    model_directory,
    furious=False,
    batch_size=1,
):
    if furious:
        set_furious(predictor)
    t0 = time.time()
    path = Path(model_directory) / Path("sam2_image_encoder_trunk.pt2")
    assert path.exists(), f"Expected {path} to exist"
    print(f"Start load from {path}")
    pkg = torch._inductor.aoti_load_package(str(path))
    pkg_m = LoadedModel(pkg)
    predictor.image_encoder.trunk = pkg_m

    path = Path(model_directory) / Path("sam2_video_forward_sam_heads.pt2")
    assert path.exists(), f"Expected {path} to exist"
    print(f"Start load from {path}")
    pkg = torch._inductor.aoti_load_package(str(path))
    pkg_m = SAM2VideoPredictor_forward_sam_heads(
        None,
        batch_size=batch_size,
        aoti_compiled_model=pkg,
        furious=furious,
    )
    predictor._forward_sam_heads = pkg_m.forward

    print(f"End load image encoder and _forward_sam_heads. Took {time.time() - t0}s")
    return predictor


def set_fast(predictor, loaded_exported_model=False):
    if not loaded_exported_model:
        predictor.image_encoder.trunk.forward = torch.compile(
            predictor.image_encoder.trunk.forward,
            mode="max-autotune",
            fullgraph=True,
            dynamic=False,
        )
    if not loaded_exported_model:
        predictor._forward_sam_heads = torch.compile(
            predictor._forward_sam_heads,
            mode="max-autotune",
            fullgraph=True,
            dynamic=False,
        )
    predictor.memory_attention = torch.compile(
        predictor.memory_attention,
        mode="max-autotune",
        fullgraph=True,
        dynamic=True,
    )
    predictor.memory_encoder.forward = torch.compile(
        predictor.memory_encoder.forward,
        mode="max-autotune",
        fullgraph=True,
        dynamic=False,
    )


def set_furious(mask_generator):
    mask_generator.predictor.model.image_encoder = (
        mask_generator.predictor.model.image_encoder.to(torch.float16)
    )
    # NOTE: Not baseline feature
    mask_generator.predictor._image_dtype = torch.float16
    mask_generator.predictor._transforms_device = mask_generator.predictor.device
    torch.set_float32_matmul_precision("high")
    mask_generator.predictor.model.sam_mask_decoder = (
        mask_generator.predictor.model.sam_mask_decoder.to(torch.float16)
    )
    # NOTE: Not baseline feature
    mask_generator.predictor.model.sam_mask_decoder._src_dtype = torch.float16
