import time
from pathlib import Path
from typing import Optional

import torch

from torchao._models.sam2.sam2_image_predictor import SAM2ImagePredictor

# Tools used to avoid compilation cold start and dynamo cache lookups
# We take the compiled model and export it using the largest
# inputs possible (to avoid recompilations).
# We track the largest size and fail if we size something larger
# We export every compile-able subregion after wrapping it into
# a class to make export happy.

TASK_TYPES = ["amg", "sps", "mps"]


# NOTE: We have to declare a separate class, because torch.export demands it.
# We build this explicitly for the sole purpose of exporting _predict_masks
# We made sure _predict_masks is fullgraph=True compileable so it can be exported
# We must be sure to export using example args that are big enough and past
# any expected recompilations. We'll add in guards to prevent unexpectedly
# large inputs.
class SAM2ImagePredictor_predict_masks(torch.nn.Module):
    def __init__(
        self,
        predictor: Optional[SAM2ImagePredictor],
        batch_size=1,
        points_per_batch=1024,
        aoti_compiled_model=None,
        furious=False,
    ):
        super().__init__()
        self.predictor = predictor
        self.batch_size = batch_size
        self.points_per_batch = points_per_batch
        self.aoti_compiled_model = aoti_compiled_model
        self.furious = furious

    def forward(
        self,
        high_res_feats,
        image_embed,
        image_pe,
        point_coords,
        point_labels,
        boxes: Optional[torch.Tensor] = None,
        mask_input: Optional[torch.Tensor] = None,
        multimask_output: bool = True,
        img_idx: int = -1,
    ):
        assert high_res_feats[0].size() == (self.batch_size, 32, 256, 256)
        assert high_res_feats[1].size() == (self.batch_size, 64, 128, 128)
        if self.furious:
            assert high_res_feats[0].dtype == torch.float16
            assert high_res_feats[1].dtype == torch.float16
        else:
            assert high_res_feats[0].dtype == torch.float32
            assert high_res_feats[1].dtype == torch.float32

        assert image_embed.size() == (self.batch_size, 256, 64, 64)
        assert image_pe.size() == (self.batch_size, 256, 64, 64)
        assert image_embed.dtype == torch.float32
        assert image_pe.dtype == torch.float32

        assert point_coords.size() == (self.points_per_batch, 1, 2)
        assert point_labels.size() == (self.points_per_batch, 1)
        assert point_coords.dtype == torch.float32
        assert point_labels.dtype == torch.int32

        # Here we encode all the assumptions made during export
        assert boxes is None
        assert mask_input is None
        assert multimask_output
        assert img_idx == -1
        if self.predictor is None:
            assert self.aoti_compiled_model is not None
            return self.aoti_compiled_model(
                high_res_feats,
                image_embed,
                image_pe,
                point_coords,
                point_labels,
                boxes=boxes,
                mask_input=mask_input,
                multimask_output=multimask_output,
                img_idx=img_idx,
            )
        return self.predictor._predict_masks(
            high_res_feats,
            image_embed,
            image_pe,
            point_coords,
            point_labels,
            boxes=boxes,
            mask_input=mask_input,
            multimask_output=multimask_output,
            img_idx=img_idx,
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

    from torch.export import export_for_inference

    exported = export_for_inference(fn, sample_args, sample_kwargs)
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
    mask_generator,
    model_directory,
    task_type,
    furious=False,
    batch_size=1,
    points_per_batch=None,
    overwrite=False,
):
    if furious:
        set_furious(mask_generator)
    assert task_type in TASK_TYPES, f"Expected {task_type} to be one of {TASK_TYPES}"
    if task_type in ["sps", "amg"]:
        assert (
            points_per_batch is not None
        ), f"Specify points_per_batch for task {task_type}"
    if task_type == "sps":
        assert (
            points_per_batch == 1
        ), f"Expected points_per_batch set to 1 for {task_type} but got {points_per_batch}"

    example_input = torch.empty(batch_size, 3, 1024, 1024)
    example_input = example_input.to(mask_generator.predictor._image_dtype)
    example_input = (example_input.to(mask_generator.predictor.device),)
    aot_compile(
        model_directory,
        "sam2_image_encoder",
        mask_generator.predictor.model.image_encoder,
        example_input,
        overwrite=overwrite,
    )

    print(f"{task_type} cannot export _predict_masks")
    return

    if task_type in ["sps"]:
        example_input_high_res_feats = [
            torch.randn(
                batch_size,
                32,
                256,
                256,
                dtype=mask_generator.predictor._image_dtype,
                device=mask_generator.predictor.device,
            ),
            torch.randn(
                batch_size,
                64,
                128,
                128,
                dtype=mask_generator.predictor._image_dtype,
                device=mask_generator.predictor.device,
            ),
        ]
        example_input_image_embed = torch.randn(
            batch_size,
            256,
            64,
            64,
            dtype=torch.float32,
            device=mask_generator.predictor.device,
        )
        example_input_image_pe = torch.randn(
            batch_size,
            256,
            64,
            64,
            dtype=torch.float32,
            device=mask_generator.predictor.device,
        )
        example_input_point_coords = torch.randn(
            points_per_batch,
            1,
            2,
            dtype=torch.float32,
            device=mask_generator.predictor.device,
        )
        example_input_point_labels = torch.ones(
            points_per_batch,
            1,
            dtype=torch.int32,
            device=mask_generator.predictor.device,
        )
        example_input_args = (
            example_input_high_res_feats,
            example_input_image_embed,
            example_input_image_pe,
            example_input_point_coords,
            example_input_point_labels,
        )

        example_input_kwargs = {
            "boxes": None,
            "mask_input": None,
            "multimask_output": True,
            "img_idx": -1,
        }

        sam2_image_predict_masks = SAM2ImagePredictor_predict_masks(
            mask_generator.predictor,
            batch_size=batch_size,
            points_per_batch=points_per_batch,
            furious=furious,
        )
        aot_compile(
            model_directory,
            "sam2_image_predict_masks",
            sam2_image_predict_masks,
            example_input_args,
            sample_kwargs=example_input_kwargs,
            overwrite=overwrite,
        )
    else:
        print(f"{task_type} cannot export _predict_masks")


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
    mask_generator,
    model_directory,
    task_type,
    furious=False,
    batch_size=1,
    points_per_batch=1024,
):
    if furious:
        set_furious(mask_generator)
    assert task_type in TASK_TYPES, f"Expected {task_type} to be one of {TASK_TYPES}"
    t0 = time.time()
    path = Path(model_directory) / Path("sam2_image_encoder.pt2")
    assert path.exists(), f"Expected {path} to exist"
    print(f"Start load from {path}")
    pkg = torch._inductor.aoti_load_package(str(path))
    pkg_m = LoadedModel(pkg)
    mask_generator.predictor.model.image_encoder = pkg_m

    print(f"End load image encoder. Took {time.time() - t0}s")
    return mask_generator

    if task_type in ["amg", "mps"]:
        return mask_generator

    path = Path(model_directory) / Path("sam2_image_predict_masks.pt2")
    assert path.exists(), f"Expected {path} to exist"
    print(f"Start load from {path}")
    pkg = torch._inductor.aoti_load_package(str(path))
    if task_type == "amg":
        assert points_per_batch > 1
    if task_type == "sps":
        assert points_per_batch == 1
    if task_type == "mps":
        assert points_per_batch is None
    pkg_m = SAM2ImagePredictor_predict_masks(
        None,
        batch_size=batch_size,
        points_per_batch=points_per_batch,
        aoti_compiled_model=pkg,
        furious=furious,
    )
    mask_generator.predictor._predict_masks = pkg_m.forward

    print(f"End load image encoder and predict masks. Took {time.time() - t0}s")


def set_fast(
    mask_generator, task_type, loaded_exported_model=False, allow_recompiles=True
):
    if task_type == "":
        task_type = "amg"

    assert task_type in TASK_TYPES, f"Expected {task_type} to be one of {TASK_TYPES}"
    if not loaded_exported_model:
        # TODO: Using CUDA graphs can cause numerical differences?
        mask_generator.predictor.model.image_encoder = torch.compile(
            mask_generator.predictor.model.image_encoder,
            mode="max-autotune",
            fullgraph=True,
            dynamic=False,
        )

    # TODO: Only the sps task can export _predict_masks
    if task_type == "sps":
        if not loaded_exported_model:
            mask_generator.predictor._predict_masks = torch.compile(
                mask_generator.predictor._predict_masks,
                mode="max-autotune",
                fullgraph=True,
                dynamic=False,
            )
    elif task_type == "amg":
        mask_generator.predictor._predict_masks = torch.compile(
            mask_generator.predictor._predict_masks,
            mode="max-autotune",
            fullgraph=True,
            dynamic=False,
        )
    else:
        # TODO: This might need to be under "allow_recompiles"
        # mps encounters rapidly changing points per batch
        mask_generator.predictor._predict_masks = torch.compile(
            mask_generator.predictor._predict_masks,
            fullgraph=True,
            dynamic=True,
        )

    import torchao

    if allow_recompiles:
        # A bunch of extra compiles at module level
        # Note that this can cause recompilations!
        # We might want to guard on that
        torchao._models.sam2.utils.amg._mask_to_rle_pytorch_2_0_0 = torch.compile(
            fullgraph=True, dynamic=True
        )(torchao._models.sam2.utils.amg._mask_to_rle_pytorch_2_0_0)
        torchao._models.sam2.utils.amg._mask_to_rle_pytorch_2_0_1 = torch.compile(
            fullgraph=True, dynamic=True
        )(torchao._models.sam2.utils.amg._mask_to_rle_pytorch_2_0_1)
        mask_generator.calculate_stability_score = torch.compile(
            fullgraph=True, dynamic=True
        )(mask_generator.calculate_stability_score)
        mask_generator.batched_mask_to_box = torch.compile(
            fullgraph=True, dynamic=True
        )(mask_generator.batched_mask_to_box)


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
