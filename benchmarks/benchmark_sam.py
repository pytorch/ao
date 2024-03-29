from pprint import pprint

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

torch._inductor.config.epilogue_fusion = False
torch._inductor.config.coordinate_descent_tuning = False
torch._inductor.config.coordinate_descent_check_all_directions = False
torch._inductor.config.force_fuse_int_mm_with_mul = False

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

def mlp_only(mod, name):
    return isinstance(mod, torch.nn.Linear) and "mlp" in name

def attention_only(mod, name):
    return isinstance(mod, torch.nn.Linear) and "mlp" not in name

def run_once(label, dtype=torch.bfloat16, batchsize=16, compile=True, quantize=False, sparse=False):
    res = {
        "label": label,
        "batchsize": batchsize,
        "dtype": dtype,
        "compile": compile,
        "quantize": quantize,
        "sparse": sparse,
    }

    model, image = get_sam_model(False, batchsize)
    model = model.to(dtype)
    image = image.to(dtype)

    if sparse and quantize:
        SparseSemiStructuredTensor._FORCE_CUTLASS = (sparse == "cutlass")
        change_linear_weights_to_int8_dq_24_sparsetensors(model, filter_fn=mlp_only)
        change_linear_weights_to_int8_dqtensors(model, filter_fn=attention_only)
    elif quantize:
        change_linear_weights_to_int8_dqtensors(model)
    elif sparse:
        SparseSemiStructuredTensor._FORCE_CUTLASS = (sparse == "cutlass")
        apply_sparse(model)

    if compile:
        model = torch.compile(model, mode='max-autotune')

    res.update(benchmark(model, image))
    pprint(res)

    return res

print("BENCHMARKING")
# run_once("baseline")
# run_once("quant", quantize=True)
# run_once("sparse", sparse="cusparselt")
run_once("quant+sparse(mlp)", quantize=True, sparse="cusparselt")
