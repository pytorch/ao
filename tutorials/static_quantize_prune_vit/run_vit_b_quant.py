import torch
import torchao

from torchao.utils import benchmark_model, profiler_runner
from torchvision import models
from evaluate import evaluate
from utils import gpu_mem_use

torch.set_float32_matmul_precision("high")
# Load Vision Transformer model
weights = models.ViT_B_16_Weights.IMAGENET1K_V1
model = models.vit_b_16(weights=weights)
transforms = weights.transforms()

# Set the model to evaluation mode
model.eval().cuda().to(torch.bfloat16)

# Input tensor (batch_size, channels, height, width)
batch_size = 1024
input_tensor = torch.randn(batch_size, 3, 224, 224, dtype=torch.bfloat16, device='cuda')

## Quantization code - start
# int8 act, int8 weight dynamic quantization, see README for other APIs
torchao.apply_dynamic_quant(model)
## Quantization code - end

## compilation configs
torch._dynamo.config.automatic_dynamic_shapes = False
torch._inductor.config.force_fuse_int_mm_with_mul = True
torch._inductor.config.use_mixed_mm = True
# Enable this for more readable traces
torch._inductor.config.triton.unique_kernel_names = True
## compilation configs end

model = torch.compile(model, mode='max-autotune')

# Must run with no_grad when optimizing for inference
with torch.no_grad():
    print(f"batch_size: {batch_size}")
    # warmup
    benchmark_model(model, 5, input_tensor)

    # benchmark
    print(f"{benchmark_model(model, 100, input_tensor) * 1000 / batch_size:.3f} us per image ", end='')
    gpu_percent, gpu_mem_mb = gpu_mem_use()
    print(f"using {gpu_percent}% ({gpu_mem_mb}MB) memory")

    # evaluate
    acc1, acc5 = evaluate(model, transforms, "/scratch/cpuhrsch/data/imagenet_blurred/val", batch_size)
    print(f"\racc1: {acc1:5.2f} acc5: {acc5:5.2f}" + " " * 50) # Flush line since running in verbose

    # Create a trace
    profiler_runner("quant.json.gz", benchmark_model, model, 5, input_tensor)
