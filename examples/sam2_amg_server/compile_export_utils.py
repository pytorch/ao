# Tools used to avoid compilation cold start and dynamo cache lookups
# We take the compiled model and export it using the largest
# inputs possible (to avoid recompilations).
# We track the largest size and fail if we size something larger
# We export every compile-able subregion after wrapping it into
# a class to make export happy.

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

    # NOTE: Good for AMG and SPS tasks
    # mask_generator.predictor._predict_masks = torch.compile(
    #     mask_generator.predictor._predict_masks,
    #     # mode="max-autotune", # NOTE: cudagraphs and aot_load don't seem to combine well
    #     mode="max-autotune-no-cudagraphs",
    #     fullgraph=True,
    #     dynamic=False,
    # )

    mask_generator.predictor._predict_masks = torch.compile(
        mask_generator.predictor._predict_masks,
        fullgraph=True,
        dynamic=True,
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
