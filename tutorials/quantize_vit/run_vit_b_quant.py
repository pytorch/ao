import torch
import torchao
import torchvision.models.vision_transformer as models

from util import benchmark_model, profiler_runner

# Load Vision Transformer model
model = models.vit_b_16(pretrained=True)

# Set the model to evaluation mode
model.eval().cuda().to(torch.bfloat16)

# Input tensor (batch_size, channels, height, width)
input_tensor = torch.randn(1, 3, 224, 224, dtype=torch.bfloat16, device='cuda')

## Quantization code - start
torchao.apply_dynamic_quant(model)
from torch._inductor import config as inductorconfig
inductorconfig.force_fuse_int_mm_with_mul = True
## Quantization code - end

model = torch.compile(model, mode='max-autotune')

# Must run with no_grad when optimizing for inference
with torch.no_grad():
    # warmup
    benchmark_model(model, 5, input_tensor)
    # benchmark
    print("elapsed_time: ", benchmark_model(model, 100, input_tensor), " milliseconds")
    # Create a trace
    profiler_runner("quant.json.gz", benchmark_model, model, 5, input_tensor)
