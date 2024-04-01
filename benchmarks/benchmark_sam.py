from pprint import pprint
import pandas as pd
import torch
from torchao.quantization import change_linear_weights_to_int8_dqtensors
from torchao.sparsity import change_linear_weights_to_int8_dq_semi_structured_sparsetensors
from torchao.sparsity.sparse_api import apply_sparse_semi_structured, apply_fake_sparsity
from torchao.sparsity.dynamic_quant_sparse import Int8DynamicallyQuantized24CusparseltLinearWeight, Int8DynamicallyQuantized24CutlassLinearWeight
from segment_anything import sam_model_registry
from torch.utils.benchmark import Timer
from torch.sparse import SparseSemiStructuredTensor, SparseSemiStructuredTensorCUTLASS, SparseSemiStructuredTensorCUSPARSELT
from torchao.quantization.quant_api import (
    _replace_with_custom_fn_if_matches_filter,
    _get_subclass_inserter,
    _is_linear,
    QuantizedLinearWeightBase,
    Int8DynamicallyQuantizedLinearWeight,
)
from itertools import product
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
    # "baseline": None,
    "quant": Int8DynamicallyQuantizedLinearWeight,
    "sparse (cutlass)": SparseSemiStructuredTensorCUTLASS,
    "sparse (cusparselt)": SparseSemiStructuredTensorCUSPARSELT,
    "quant+sparse (cusparselt fuse one mul)": Int8DynamicallyQuantized24CusparseltLinearWeight,
    "quant+sparse (cutlass)": Int8DynamicallyQuantized24CutlassLinearWeight,
}

def run_once(dtype=torch.bfloat16, batchsize=32, compile=True, qkv=None, proj=None, lin1=None, lin2=None):
    res = {
        "batchsize": batchsize,
        "dtype": dtype,
        "compile": compile,
        "qkv" : qkv,
        "proj": proj,
        "lin1": lin1,
        "lin2": lin2,
    }
    with torch.no_grad():
        model, image = get_sam_model(False, batchsize)
        model = model.to(dtype)
        image = image.to(dtype)

        # 2:4 prune model
        apply_fake_sparsity(model)
        option_and_filter_fn = zip([qkv, proj, lin1, lin2],
                                   [qkv_only, proj_only, lin1_only, lin2_only])

        for option, filter_fn in option_and_filter_fn:
            if option:
                subclass = SUBCLASSES[option]
                if issubclass(subclass, SparseSemiStructuredTensor):
                    for name, mod in model.named_modules():
                        if filter_fn(mod, name):
                            mod.weight = torch.nn.Parameter(subclass.from_dense(mod.weight))
                    # replace with to_sparse_semi_structured
                elif issubclass(subclass, QuantizedLinearWeightBase):
                    _replace_with_custom_fn_if_matches_filter(model, _get_subclass_inserter(subclass), filter_fn)

        # for name, mod in model.named_modules():
        #     if isinstance(mod, torch.nn.Linear):
        #         print(name, mod.weight.data.__class__.__name__)

        if compile:
            model = torch.compile(model, mode='max-autotune')

        res.update(benchmark(model, image))
        pprint(res)
        return res

if __name__ == "__main__":
    print("BENCHMARKING")
    # ALL_RUNS = [run_once(qkv=qkv, proj=proj, lin1=lin1, lin2=lin2)
    #             for (qkv, proj, lin1, lin2)
    #             in tqdm(list(product(SUBCLASSES, SUBCLASSES, SUBCLASSES, SUBCLASSES)))]
    ALL_RUNS = [
        run_once(),
        run_once(qkv="quant", proj="quant", lin1="quant+sparse (cutlass)", lin2="quant+sparse (cutlass)"),
        # run_once(qkv="sparse (cusparselt)", proj="sparse (cusparselt)", lin1="quant+sparse (cusparselt fuse one mul)", lin2="quant+sparse (cutlass)"),
        # run_once(qkv="sparse (cusparselt)", proj="sparse (cusparselt)", lin1="quant+sparse (cusparselt fuse one mul)", lin2="quant+sparse (cusparselt fuse one mul)"),
        run_once(qkv="sparse (cusparselt)", proj="sparse (cusparselt)", lin1="quant+sparse (cutlass)", lin2="quant+sparse (cutlass)"),
    ]
    df = pd.DataFrame(ALL_RUNS)
    df.to_csv("sam_benchmark_results.csv")
    print(df)


# optimal order goes
# qkv  - cusparselt
# proj - cusparselt
# lin1 - int8 cusparselt / cutlass
# lin2 - cutlass
