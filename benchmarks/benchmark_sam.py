from pprint import pprint
import pandas as pd
import torch
from torchao.quantization import change_linear_weights_to_int8_dqtensors
from torchao.sparsity import change_linear_weights_to_int8_dq_24_sparsetensors, apply_sparse
from segment_anything import sam_model_registry
from torch.utils.benchmark import Timer
from torch.sparse import SparseSemiStructuredTensor

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

def run_once(label, dtype=torch.bfloat16, batchsize=16, compile=True, quantize=False, sparsify=False):
    res = {
        "label": label,
        "batchsize": batchsize,
        "dtype": dtype,
        "compile": compile,
        "quantize": quantize,
        "sparsify": sparsify,
    }

    model, image = get_sam_model(False, batchsize)
    model = model.to(dtype)
    image = image.to(dtype)

    if sparsify and quantize:
        SparseSemiStructuredTensor._FORCE_CUTLASS = (sparsify == "cutlass")
        change_linear_weights_to_int8_dq_24_sparsetensors(model)
    elif quantize:
        change_linear_weights_to_int8_dqtensors(model)
    elif sparsify:
        SparseSemiStructuredTensor._FORCE_CUTLASS = (sparsify == "cutlass")
        apply_sparse(model)

    if compile:
        model = torch.compile(model, mode='max-autotune')

    res.update(benchmark(model, image))
    print(f"{label} finished in {res['time']} and {res['memory']} run with {res['batchsize']} batchsize, {res['dtype']} dtype, {res['compile']} compile, {res['quantize']} quantize, {res['sparsify']} sparsify")
    return res


if __name__ == "__main__":
    ALL_RUNS = []
    print("BENCHMARKING")
    ALL_RUNS.append(run_once("baseline"))
    ALL_RUNS.append(run_once("quant", quantize=True))
    ALL_RUNS.append(run_once("sparse", sparse="cusparselt"))
    ALL_RUNS.append(run_once("sparse", sparse="cutlass"))
    ALL_RUNS.append(run_once("quant+sparse (fuse one mul)", quantize=True, sparse="cusparselt"))
    ALL_RUNS.append(run_once("quant+sparse", quantize=True, sparse="cutlass"))
    df = pd.DataFrame(ALL_RUNS)
    df.to_csv("sam_benchmark_results.csv")
    print(df)
