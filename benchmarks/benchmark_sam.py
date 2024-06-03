import argparse
from itertools import product

import pandas as pd
from segment_anything_fast import sam_model_registry
import torch
from torch.utils.benchmark import Timer
from torch.sparse import SparseSemiStructuredTensor, SparseSemiStructuredTensorCUTLASS, SparseSemiStructuredTensorCUSPARSELT
from torchao.quantization.quant_api import (
    _replace_with_custom_fn_if_matches_filter,
    _get_subclass_inserter,
    _is_linear,
    QuantizedLinearWeightBase,
    Int8DynamicallyQuantizedLinearWeight,
)
from torchao.quantization import change_linear_weights_to_int8_dqtensors
from torchao.sparsity import (
    apply_sparse_semi_structured,
    apply_fake_sparsity,
)
from torchao.sparsity.prototype.dynamic_quant_sparse import Int8DynamicallyQuantized24CusparseltLinearFuseMulWeight, Int8DynamicallyQuantizedSemiStructuredSparseLinearWeight
from tqdm import tqdm

sam_checkpoint_base_path = "/home/jessecai/local/MODELS"
model_type = 'vit_h'
model_name = 'sam_vit_h_4b8939.pth'
checkpoint_path = f"{sam_checkpoint_base_path}/{model_name}"

torch._inductor.config.epilogue_fusion = True
torch._inductor.config.coordinate_descent_tuning = True
torch._inductor.config.coordinate_descent_check_all_directions = True
torch._inductor.config.force_fuse_int_mm_with_mul = True

@torch.no_grad()
def benchmark(f, *args, **kwargs):
    for _ in range(3):
        f(*args, **kwargs)
        torch.cuda.synchronize()

    torch.cuda.reset_peak_memory_stats()
    t0 = Timer(
        stmt="f(*args, **kwargs)", globals={"args": args, "kwargs": kwargs, "f": f}
    )
    res = t0.adaptive_autorange(.03, min_run_time=.2, max_run_time=20)
    return {'time':res.median * 1e3, 'memory': torch.cuda.max_memory_allocated()/1e9}

def get_sam_model(only_one_block=False, batchsize=1):
    sam = sam_model_registry[model_type](checkpoint=checkpoint_path).cuda()
    model = sam.image_encoder.eval()
    image = torch.randn(batchsize, 3, 1024, 1024, device='cuda')

    # code to use just a single block of the model
    if only_one_block:
        model = model.blocks[0]
        image = torch.randn(batchsize, 64, 64, 1280, device='cuda')
    return model, image

def qkv_only(mod, name):
    return isinstance(mod, torch.nn.Linear) and 'qkv' in name

def proj_only(mod, name):
    return isinstance(mod, torch.nn.Linear) and 'proj' in name

def lin1_only(mod, name):
    return isinstance(mod, torch.nn.Linear) and 'lin1' in name

def lin2_only(mod, name):
    return isinstance(mod, torch.nn.Linear) and 'lin2' in name

SUBCLASSES = {
    "quant"                              : Int8DynamicallyQuantizedLinearWeight,
    "quant+sparse (cutlass)"             : Int8DynamicallyQuantizedSemiStructuredSparseLinearWeight,
    "quant+sparse (cusparselt)"          : Int8DynamicallyQuantized24CusparseltLinearFuseMulWeight,
    "sparse (cutlass)"                   : SparseSemiStructuredTensorCUTLASS,
    "sparse (cusparselt)"                : SparseSemiStructuredTensorCUSPARSELT,
}

def run_once(block_only=False, dtype=torch.bfloat16, batchsize=32, compile=True, qkv=None, proj=None, lin1=None, lin2=None):
    res = {
        "block_only": block_only,
        "batchsize": batchsize,
        "dtype": dtype,
        "compile": compile,
        "qkv" : qkv,
        "proj": proj,
        "lin1": lin1,
        "lin2": lin2,
    }
    with torch.no_grad():
        model, image = get_sam_model(block_only, batchsize)
        model = model.to(dtype)
        image = image.to(dtype)

        # 2:4 prune model
        apply_fake_sparsity(model)
        option_and_filter_fn = zip([qkv, proj, lin1, lin2], [qkv_only, proj_only, lin1_only, lin2_only])

        for option, filter_fn in option_and_filter_fn:
            subclass = SUBCLASSES.get(option, None)
            if subclass and issubclass(subclass, SparseSemiStructuredTensor):
                # replace with to_sparse_semi_structured
                for name, mod in model.named_modules():
                    if filter_fn(mod, name):
                        mod.weight = torch.nn.Parameter(subclass.from_dense(mod.weight))
            elif subclass and issubclass(subclass, QuantizedLinearWeightBase):
                _replace_with_custom_fn_if_matches_filter(model, _get_subclass_inserter(subclass), filter_fn)

        if compile:
            model = torch.compile(model, mode='max-autotune')

        res.update(benchmark(model, image))
        res["img/s"] = 1 / (res['time'] / 1000 / res['batchsize'])
        return res

if __name__ == "__main__":
    print("BENCHMARKING")
    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('--eager', action='store_true', help='enable/disable torch.compile')
    args = parser.parse_args()
    # ALL_RUNS = [run_once(qkv="quant+sparse (cutlass)", proj="quant", lin1="quant+sparse (cutlass)", lin2="quant+sparse (cutlass)")]
    ALL_RUNS = [
        run_once(compile=not args.eager),
        run_once(compile=not args.eager, lin1="sparse (cusparselt)", lin2="sparse (cusparselt)"),
        run_once(compile=not args.eager, lin1="sparse (cutlass)", lin2="sparse (cutlass)"),
        run_once(compile=not args.eager, qkv="sparse (cusparselt)",       proj="sparse (cusparselt)",       lin1="sparse (cusparselt)",          lin2="sparse (cusparselt)"),
        run_once(compile=not args.eager, qkv="sparse (cutlass)",          proj="sparse (cutlass)",          lin1="sparse (cutlass)",             lin2="sparse (cutlass)"),
        # run_once(qkv="quant",                     proj="quant",                     lin1="quant",                        lin2="quant"),
        # run_once(qkv="quant+sparse (cusparselt)", proj="quant+sparse (cusparselt)", lin1="quant+sparse (cusparselt)",    lin2="quant+sparse (cutlass)"),
        # run_once(qkv="quant+sparse (cusparselt)", proj="quant",                     lin1="quant+sparse (cutlass)",       lin2="quant+sparse (cutlass)"),
        # run_once(qkv="quant",                     proj="quant",                     lin1="quant+sparse (cusparselt)",    lin2="quant+sparse (cusparselt)"),
        # run_once(qkv="quant+sparse (cutlass)",    proj="quant+sparse (cutlass)",    lin1="quant+sparse (cutlass)",       lin2="quant+sparse (cutlass)"),
    ]
    df = pd.DataFrame(ALL_RUNS)
    df.to_csv("sam_benchmark_results.csv")
    print(df)
