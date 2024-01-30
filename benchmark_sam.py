import torch
from torchao.quantization import change_linear_weights_to_int8_dqtensors
from torchao.quantization.quant_api import change_linear_weights_to_int8_dq_semi_structured_sparsetensors
from segment_anything import sam_model_registry
from torch.utils.benchmark import Timer
from torchao.sparsity import apply_fake_sparsity, apply_sparse

from torchao.quantization.dynamic_quant_sparse import apply_int4_dynamic_quant_sparse

sam_checkpoint_base_path = "/home/jessecai/local/MODELS"
model_type = 'vit_h'
model_name = 'sam_vit_h_4b8939.pth'
checkpoint_path = f"{sam_checkpoint_base_path}/{model_name}"
batchsize = 16
only_one_block = False

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

model, image = get_sam_model(only_one_block, batchsize)
model = model.to(torch.bfloat16)
image = image.to(torch.bfloat16)
torch._inductor.config.epilogue_fusion = False
torch._inductor.config.coordinate_descent_tuning = True
torch._inductor.config.coordinate_descent_check_all_directions = True
torch._inductor.config.force_fuse_int_mm_with_mul = True

change_linear_weights_to_int8_dqtensors(model)
model_c = torch.compile(model, mode='max-autotune')
quant_res = benchmark(model_c, image)

print(f"bf16 compiled runtime of the final quantized block is {quant_res['time']:0.2f}ms and peak memory {quant_res['memory']: 0.2f}GB")

del model_c, model, image
model, image = get_sam_model(only_one_block, batchsize)
model = model.to(torch.bfloat16)
image = image.to(torch.bfloat16)
apply_sparse(model)
torch._inductor.config.epilogue_fusion = False
torch._inductor.config.coordinate_descent_tuning = True
torch._inductor.config.coordinate_descent_check_all_directions = True
torch._inductor.config.force_fuse_int_mm_with_mul = True
model_c = torch.compile(model, mode='max-autotune')
quant_res = benchmark(model_c, image)

print(f"bf16 compiled runtime of the final sparsified block is {quant_res['time']:0.2f}ms and peak memory {quant_res['memory']: 0.2f}GB")

del model_c, model, image
model, image = get_sam_model(only_one_block, batchsize)
model = model.to(torch.bfloat16)
image = image.to(torch.bfloat16)
change_linear_weights_to_int8_dq_semi_structured_sparsetensors(model)
torch._inductor.config.epilogue_fusion = False
torch._inductor.config.coordinate_descent_tuning = True
torch._inductor.config.coordinate_descent_check_all_directions = True
torch._inductor.config.force_fuse_int_mm_with_mul = True
model_c = torch.compile(model, mode='max-autotune')
quant_res = benchmark(model_c, image)

print(f"bf16 compiled runtime of the final quant + sparsified block is {quant_res['time']:0.2f}ms and peak memory {quant_res['memory']: 0.2f}GB")

del model_c, model, image
model, image = get_sam_model(only_one_block, batchsize)
model = model.to(torch.bfloat16)
image = image.to(torch.bfloat16)
apply_int4_dynamic_quant_sparse(model)
torch._inductor.config.epilogue_fusion = False
torch._inductor.config.coordinate_descent_tuning = True
torch._inductor.config.coordinate_descent_check_all_directions = True
torch._inductor.config.force_fuse_int_mm_with_mul = True
model_c = torch.compile(model, mode='max-autotune')
quant_res = benchmark(model_c, image)

print(f"bf16 compiled runtime of the final quant + sparsified block is {quant_res['time']:0.2f}ms and peak memory {quant_res['memory']: 0.2f}GB")
