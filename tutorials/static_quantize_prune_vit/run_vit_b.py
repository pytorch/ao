import torch

from torchao.utils import benchmark_model, profiler_runner
from torchvision import models
from evaluate import evaluate

torch.set_float32_matmul_precision("high")
# Load Vision Transformer model
weights = models.ViT_B_16_Weights.IMAGENET1K_V1
model = models.vit_b_16(weights=weights)
transforms = weights.transforms()

# Set the model to evaluation mode
model.eval().cuda().to(torch.bfloat16)

# Input tensor (batch_size, channels, height, width)
batch_size = 32
input_tensor = torch.randn(batch_size, 3, 224, 224, dtype=torch.bfloat16, device='cuda')

model = torch.compile(model, mode='max-autotune')

# Must run with no_grad when optimizing for inference
with torch.no_grad():
    # warmup
    benchmark_model(model, 5, input_tensor)
    # benchmark
    print("elapsed_time: ", benchmark_model(model, 100, input_tensor), " milliseconds")
    # evaluate
    acc1, acc5 = evaluate(model, transforms, "/scratch/cpuhrsch/data/imagenet_blurred/val", batch_size)
    print(f"\racc1: {acc1:5.2f} acc5: {acc5:5.2f}" + " " * 50) # Flush line since running in verbose
    # Create a trace
    profiler_runner("bfloat16.json.gz", benchmark_model, model, 5, input_tensor)
