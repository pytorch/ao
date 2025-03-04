import math
import resource
import time

import fire
import torch
import tqdm
from data import build_data, setup_coco_img_ids
from metrics import calculate_miou, create_result_entry

import torchao
from benchmarks._models.utils import (
    get_arch_name,
    write_json_result_local,
    write_json_result_ossci,
)
from torchao.dtypes import SemiSparseLayout
from torchao.prototype.quantization.autoquant_v2 import autoquant_v2
from torchao.quantization import (
    autoquant,
    int4_weight_only,
    int8_dynamic_activation_int8_weight,
    quantize_,
)
from torchao.sparsity import apply_fake_sparsity, semi_sparse_weight, sparsify_
from torchao.utils import TORCH_VERSION_AT_LEAST_2_5, unwrap_tensor_subclass

torch._dynamo.config.cache_size_limit = 50000


def unbind_jagged(device, data, sizes, offsets):
    if data is None:
        return None
    data = data.to(device=device, non_blocking=True)
    return [
        data[offsets[batch_idx] : offsets[batch_idx + 1]].view(sizes[batch_idx])
        for batch_idx in range(len(sizes))
    ]


PADDED_TENSOR = None


# Preallocate a "landing" Tensor for incoming data and reuse it across launches.
def pad_to_batch_size(batch, batch_size, device):
    assert batch.dim() == 4
    # assert batch.is_pinned()
    global PADDED_TENSOR
    if PADDED_TENSOR is None:
        batch = batch.to(device=device, non_blocking=True)
        full_batch_size = (batch_size, batch.size(1), batch.size(2), batch.size(3))
        first_entry = batch[0].unsqueeze(0)
        repeat_first_entry = first_entry.expand(full_batch_size)
        padded_batch = torch.cat(
            [batch, repeat_first_entry[batch.size(0) : batch_size]], dim=0
        )
        assert padded_batch.size() == full_batch_size
        PADDED_TENSOR = padded_batch
    PADDED_TENSOR[: batch.size(0)].copy_(batch, non_blocking=True)
    return PADDED_TENSOR


def get_features_batch(
    encoder, input_image_batch, pad_input_image_batch, batch_size, device
):
    if pad_input_image_batch:
        features_batch = encoder(
            pad_to_batch_size(input_image_batch, batch_size, device)
        )
        return features_batch[: input_image_batch.size(0)]
    return encoder(input_image_batch)


def build_results_batch(predictor, batch, batch_size, pad_input_image_batch):
    encoder = predictor.model.image_encoder
    device = predictor.device

    input_image_batch = batch[0]
    # The number of valid data points varies slightly per batch
    orig_input_image_batch_size = input_image_batch.size(0)
    if input_image_batch is None:
        return (None, None, None)

    with torch.autograd.profiler.record_function("data transfer"):
        coords_lists = unbind_jagged(*([device] + batch[1:4]))
        gt_masks_lists = unbind_jagged(*([device] + batch[4:7]))
        if coords_lists is None:
            return (None, None, None)
        datapoints = list(zip(*(batch[7:] + [coords_lists, gt_masks_lists])))
        if pad_input_image_batch:
            # Pad to a static shape to avoid recompilation
            input_image_batch = pad_to_batch_size(input_image_batch, batch_size, device)
        else:
            input_image_batch = input_image_batch.to(device=device, non_blocking=True)

    # We explicitly exclude data transfers from the timing to focus
    # only on the kernel performance.
    # Next we synchronize and set two events to start timing.
    if torch.cuda.is_available():
        torch.cuda.synchronize()
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)
        start_event.record()
    else:
        t0 = time.time()

    with torch.autograd.profiler.record_function("timed region"):
        with torch.autograd.profiler.record_function("image encoder"):
            features_batch = encoder(input_image_batch)
            features_batch = features_batch[:orig_input_image_batch_size]

        with torch.autograd.profiler.record_function("predict_torch"):
            result_batch = []
            for batch_idx, (
                anns,
                image,
                input_size,
                idx,
                coords,
                gt_masks,
            ) in enumerate(datapoints):
                features = features_batch.narrow(0, batch_idx, 1)
                predictor.reset_image()
                predictor.original_size = image.shape[:2]
                predictor.input_size = input_size
                predictor.features = features
                predictor.is_image_set = True
                coords = coords.unsqueeze(1)
                fg_labels = torch.ones(
                    (coords.size(0), 1), dtype=torch.int, device=device
                )
                masks, scores, logits = predictor.predict_torch(
                    point_coords=coords,
                    point_labels=fg_labels,
                    multimask_output=True,
                )
                entry = create_result_entry(anns, gt_masks, masks, scores, idx)
                result_batch += entry

        # After all kernels have been launched we synchronize again and measure
        # the amount of time spent on the GPU. This is a fairly tight measurement
        # around the launched GPU kernels and excludes data movement from host
        # to device.
        if torch.cuda.is_available():
            end_event.record()
            torch.cuda.synchronize()
            elapsed_time = start_event.elapsed_time(end_event)
        else:
            elapsed_time = time.time() - t0
    return result_batch, orig_input_image_batch_size, elapsed_time


def build_results(
    batched_data_iter,
    predictor,
    mask_debug_out_dir,
    batch_size,
    use_compile,
    use_compile_decoder,
    pad_input_image_batch,
    compress,
    use_fullgraph=False,
):
    # TODO: Re-enable this for datapoints
    assert not use_compile_decoder

    batch_runner = build_results_batch

    results = []
    batch_idx = 0
    num_images = 0
    num_batches = 0
    elapsed_time = 0
    partial_batch = False
    for batch in tqdm.tqdm(batched_data_iter):
        with torch.no_grad():
            if batch_idx == 0:
                with torch.autograd.profiler.record_function("compilation and warmup"):
                    if str(use_compile) != "False":
                        predictor.model.image_encoder = torch.compile(
                            predictor.model.image_encoder,
                            mode=use_compile,
                            fullgraph=use_fullgraph,
                        )
                    # Run first batch a few times for warmup and exclude it from the final timings
                    for _ in range(5):
                        _ = batch_runner(
                            predictor, batch, batch_size, pad_input_image_batch
                        )
            result_batch, num_datapoints, kernel_time = batch_runner(
                predictor, batch, batch_size, pad_input_image_batch
            )
            if result_batch is not None:
                results += result_batch
        # We expect a partial batch to only happens once at the end
        assert not partial_batch
        # Only measure timing on full batches
        if num_datapoints == batch_size:
            num_images += num_datapoints
            num_batches += 1
            # We consistently exclude the last (512 - filtered) images
            # Since batch sizes must be powers of two and less than
            # or equal 512 this ensures consistent timing across varying
            # batch sizes.
            if num_images <= 4488:
                elapsed_time += kernel_time
        else:
            partial_batch = True
        batch_idx += 1

    avg_ms_per_img = None
    if num_images > 0:
        avg_ms_per_img = elapsed_time
        avg_ms_per_img = avg_ms_per_img / num_images

    return results, avg_ms_per_img, num_batches, num_images


def identity_runner(fn, *args, **kwargs):
    return fn(*args, **kwargs)


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


def profile_top_runner(fn, *args, **kwargs):
    with torch.profiler.profile(
        activities=[
            torch.profiler.ProfilerActivity.CPU,
            torch.profiler.ProfilerActivity.CUDA,
        ],
        record_shapes=True,
    ) as prof:
        result = fn(*args, **kwargs)
    if torch.cuda.is_available():
        print(prof.key_averages().table(sort_by="self_cuda_time_total", row_limit=-1))
    else:
        print(prof.key_averages().table(sort_by="self_cpu_time_total", row_limit=-1))
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


def run(
    coco_root_dir,
    coco_slice_name,
    sam_checkpoint_base_path,
    sam_model_type,
    point_sampling_cache_dir,
    mask_debug_out_dir,
    batch_size=1,
    print_header=False,
    coco_category_names=None,
    limit=None,
    img_id=None,
    use_half=None,
    use_compile="False",
    use_compile_decoder=False,
    compress=None,
    min_sqnr=None,
    num_workers=0,
    use_rel_pos=True,
    pad_input_image_batch=True,
    profile_path=None,
    profile_top=False,
    memory_path=None,
    device="cuda",
    output_json_path=None,
    output_json_local=False,
):
    from torch._inductor import config as inductorconfig

    inductorconfig.triton.unique_kernel_names = True
    inductorconfig.epilogue_fusion = True
    inductorconfig.coordinate_descent_tuning = True
    inductorconfig.coordinate_descent_check_all_directions = True
    inductorconfig.force_fuse_int_mm_with_mul = True
    inductorconfig.use_mixed_mm = True
    from torch.sparse import SparseSemiStructuredTensor

    SparseSemiStructuredTensor._FORCE_CUTLASS = False

    if use_half is not None:
        if use_half == "float16":
            use_half = torch.float16
        elif use_half == "bfloat16":
            use_half = torch.bfloat16
        else:
            raise ValueError(
                "Expected one of float16 or bfloat for specified {use_half}"
            )

    # Batch size needs to be a multiple of two and at most 512.
    assert math.log2(batch_size).is_integer()
    assert batch_size <= 512

    # https://github.com/facebookresearch/segment-anything/tree/main#model-checkpoints
    # largest to smallest: vit_h, vit_l, vit_b
    model_type_to_checkpoint = {
        "vit_h": f"{sam_checkpoint_base_path}/sam_vit_h_4b8939.pth",
        "vit_l": f"{sam_checkpoint_base_path}/sam_vit_l_0b3195.pth",
        "vit_b": f"{sam_checkpoint_base_path}/sam_vit_b_01ec64.pth",
    }

    from segment_anything_fast import SamPredictor, sam_model_registry

    checkpoint_path = model_type_to_checkpoint[sam_model_type]
    sam = sam_model_registry[sam_model_type](checkpoint=checkpoint_path).to(
        torch.device(device)
    )
    predictor = SamPredictor(sam)

    from segment_anything_fast import tools

    tools.apply_eval_dtype_predictor(predictor, use_half)

    for block in predictor.model.image_encoder.blocks:
        block.attn.use_rel_pos = use_rel_pos

    # Helper filter functions
    def attn_only(mod, name):
        return isinstance(mod, torch.nn.Linear) and "attn" in name

    def mlp_lin1_only(mod, name):
        return isinstance(mod, torch.nn.Linear) and "lin1" in name

    def mlp_lin2_only(mod, name):
        return isinstance(mod, torch.nn.Linear) and "lin2" in name

    def mlp_only(mod, name):
        return isinstance(mod, torch.nn.Linear) and "mlp" in name

    if compress == "int8_dynamic_quant":
        quantize_(predictor.model.image_encoder, int8_dynamic_activation_int8_weight())
        if not TORCH_VERSION_AT_LEAST_2_5:
            predictor.model.image_encoder = unwrap_tensor_subclass(
                predictor.model.image_encoder
            )
    elif compress == "sparse_mlp_only":

        def mlp_only(mod, name):
            return isinstance(mod, torch.nn.Linear) and "mlp" in name

        apply_fake_sparsity(predictor.model.image_encoder, filter_fn=mlp_only)
        sparsify_(
            predictor.model.image_encoder, semi_sparse_weight(), filter_fn=mlp_only
        )
    elif compress == "sparse":
        apply_fake_sparsity(predictor.model.image_encoder)
        sparsify_(predictor.model.image_encoder, semi_sparse_weight())
    elif compress == "int8_dynamic_quant_sparse":
        # apply sparsify first to set qparams
        apply_fake_sparsity(predictor.model.image_encoder, filter_fn=mlp_only)

        quantize_(
            predictor.model.image_encoder,
            int8_dynamic_activation_int8_weight(),
            attn_only,
        )
        quantize_(
            predictor.model.image_encoder,
            int8_dynamic_activation_int8_weight(layout=SemiSparseLayout()),
            mlp_lin1_only,
        )
        sparsify_(predictor.model.image_encoder, semi_sparse_weight(), mlp_lin2_only)
        if not TORCH_VERSION_AT_LEAST_2_5:
            predictor.model.image_encoder = unwrap_tensor_subclass(
                predictor.model.image_encoder
            )
    elif compress == "int4_weight_only_sparse":
        # apply sparsify first to set qparams
        apply_fake_sparsity(predictor.model.image_encoder, filter_fn=mlp_only)
        from torchao.dtypes import MarlinSparseLayout

        quantize_(
            predictor.model.image_encoder,
            int8_dynamic_activation_int8_weight(),
            attn_only,
        )
        quantize_(
            predictor.model.image_encoder,
            int4_weight_only(layout=MarlinSparseLayout()),
            mlp_lin1_only,
        )
        sparsify_(predictor.model.image_encoder, semi_sparse_weight(), mlp_lin2_only)
        if not TORCH_VERSION_AT_LEAST_2_5:
            predictor.model.image_encoder = unwrap_tensor_subclass(
                predictor.model.image_encoder
            )

    elif compress is not None and "autoquant_v2" in compress:
        example_input = torch.randn(
            1, 3, 1024, 1024, dtype=torch.bfloat16, device=device
        )
        if "autoquant_v2-int4" == compress:
            autoquant_v2(
                predictor.model.image_encoder,
                example_input=example_input,
                manual=True,
                qtensor_class_list=torchao.prototype.quantization.autoquant_v2.DEFAULT_INT4_AUTOQUANT_CLASS_LIST,
            )
        elif "autoquant_v2-float8" == compress:
            autoquant_v2(
                predictor.model.image_encoder,
                example_input=example_input,
                manual=True,
                qtensor_class_list=torchao.prototype.quantization.autoquant_v2.OTHER_AUTOQUANT_CLASS_LIST,
            )
        elif "autoquant_v2-all" == compress:
            autoquant_v2(
                predictor.model.image_encoder,
                example_input=example_input,
                manual=True,
                qtensor_class_list=torchao.prototype.quantization.autoquant_v2.ALL_AUTOQUANT_CLASS_LIST,
            )
        else:
            autoquant_v2(
                predictor.model.image_encoder, example_input=example_input, manual=True
            )

        predictor.model.image_encoder(example_input)
        predictor.model.image_encoder.finalize_autoquant()

    elif compress is not None and "autoquant" in compress:
        example_input = torch.randn(
            1, 3, 1024, 1024, dtype=torch.bfloat16, device=device
        )
        if "autoquant-int4" == compress:
            autoquant(
                predictor.model.image_encoder,
                example_input=example_input,
                manual=True,
                qtensor_class_list=torchao.quantization.DEFAULT_INT4_AUTOQUANT_CLASS_LIST,
                min_sqnr=min_sqnr,
            )
        elif "autoquant-float8" == compress:
            autoquant(
                predictor.model.image_encoder,
                example_input=example_input,
                manual=True,
                qtensor_class_list=torchao.quantization.OTHER_AUTOQUANT_CLASS_LIST,
                min_sqnr=min_sqnr,
            )
        elif "autoquant-sparse" == compress:
            autoquant(
                predictor.model.image_encoder,
                example_input=example_input,
                manual=True,
                qtensor_class_list=torchao.quantization.DEFAULT_SPARSE_AUTOQUANT_CLASS_LIST,
                min_sqnr=min_sqnr,
            )
        elif "autoquant-all" == compress:
            autoquant(
                predictor.model.image_encoder,
                example_input=example_input,
                manual=True,
                qtensor_class_list=torchao.quantization.ALL_AUTOQUANT_CLASS_LIST,
                min_sqnr=min_sqnr,
            )
        else:
            autoquant(
                predictor.model.image_encoder,
                example_input=example_input,
                manual=True,
                min_sqnr=min_sqnr,
            )
        predictor.model.image_encoder(example_input)
        predictor.model.image_encoder.finalize_autoquant()
    else:
        assert compress is None, f"Unsupported compress mode {compress}"

    coco_img_ids_, cat_id_to_cat, catIds, coco = setup_coco_img_ids(
        coco_root_dir, coco_slice_name, coco_category_names, img_id
    )

    coco_img_ids = []
    for imgId in coco_img_ids_:
        img = coco.loadImgs(imgId)[0]
        annIds = coco.getAnnIds(imgIds=img["id"], catIds=catIds, iscrowd=None)
        anns = coco.loadAnns(annIds)
        if len(anns) != 0:
            coco_img_ids.append(imgId)

    build_batch = build_data(
        coco_img_ids,
        coco,
        catIds,
        coco_root_dir,
        coco_slice_name,
        point_sampling_cache_dir,
        predictor,
        use_half,
        pad_input_image_batch,
    )

    limit = len(coco_img_ids) if limit is None else limit
    batched_data_iter = torch.utils.data.DataLoader(
        list(range(limit)),
        batch_size=batch_size,
        collate_fn=build_batch,
        num_workers=num_workers,
        pin_memory=False,
    )
    runner = identity_runner

    if profile_path is not None:
        import functools

        runner = functools.partial(profiler_runner, profile_path)

    if profile_top:
        runner = profile_top_runner

    if memory_path is not None:
        assert (
            use_compile != "max-autotune"
        ), f"Memory path does not support {use_compile}"
        import functools

        runner = functools.partial(memory_runner, memory_path)

    results, avg_ms_per_img, num_batches, num_images = runner(
        build_results,
        batched_data_iter,
        predictor,
        mask_debug_out_dir,
        batch_size,
        use_compile,
        use_compile_decoder,
        pad_input_image_batch,
        compress,
    )

    results = [[r[0], r[1], r[2], r[3].item()] for r in results]

    img_s, batch_ms_batch_size = None, None
    if avg_ms_per_img is not None:
        img_s = 1000 / avg_ms_per_img
        batch_ms_batch_size = (avg_ms_per_img * num_images) / num_batches / batch_size

    mIoU = calculate_miou(results, mask_debug_out_dir, True, cat_id_to_cat)
    if torch.cuda.is_available():
        max_memory_allocated_bytes = torch.cuda.max_memory_allocated()
        _, total_memory = torch.cuda.mem_get_info()
        max_memory_allocated_percentage = int(
            100 * (max_memory_allocated_bytes / total_memory)
        )
        max_memory_allocated_bytes = max_memory_allocated_bytes >> 20
    else:
        import psutil

        total_memory = psutil.virtual_memory().total
        max_memory_allocated_bytes = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        max_memory_allocated_percentage = int(
            100 * (max_memory_allocated_bytes / (total_memory >> 10))
        )
        max_memory_allocated_bytes = max_memory_allocated_bytes >> 10

    with open("results.csv", "a") as f:
        if print_header:
            header = ",".join(
                [
                    "device",
                    "sam_model_type",
                    "batch_size",
                    "memory(MiB)",
                    "memory(%)",
                    "img_s(avg)",
                    "batch_ms(avg)/batch_size",
                    "mIoU",
                    "use_compile",
                    "use_half",
                    "compress",
                    "use_compile_decoder",
                    "use_rel_pos",
                    "pad_input_image_batch",
                    "num_workers",
                    "num_batches",
                    "num_images",
                    "profile_path",
                    "memory_path",
                ]
            )
            f.write(header + "\n")
        vals = ",".join(
            map(
                str,
                [
                    device,
                    sam_model_type,
                    batch_size,
                    max_memory_allocated_bytes,
                    max_memory_allocated_percentage,
                    img_s,
                    batch_ms_batch_size,
                    mIoU,
                    use_compile,
                    use_half,
                    compress,
                    use_compile_decoder,
                    use_rel_pos,
                    pad_input_image_batch,
                    num_workers,
                    num_batches,
                    num_images,
                    profile_path,
                    memory_path,
                ],
            )
        )
        f.write(vals + "\n")

    if output_json_path:
        headers = [
            "name",
            "dtype",
            "min_sqnr",
            "compile",
            "device",
            "arch",
            "metric",
            "actual",
            "target",
        ]
        name = sam_model_type
        arch = get_arch_name()
        dtype = compress or "noquant"
        # boolean flag to indicate whether compile is used
        compile = use_compile != "False"
        memory_result = [
            name,
            dtype,
            min_sqnr,
            compile,
            device,
            arch,
            "memory(MiB)",
            max_memory_allocated_bytes,
            None,
        ]
        performance_result = [
            name,
            dtype,
            min_sqnr,
            compile,
            device,
            arch,
            "img_s(avg)",
            img_s,
            None,
        ]
        write_json_result = (
            write_json_result_local if output_json_local else write_json_result_ossci
        )
        write_json_result(output_json_path, headers, memory_result)
        write_json_result(output_json_path, headers, performance_result)


if __name__ == "__main__":
    fire.Fire(run)
