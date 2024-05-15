import torch
import torchvision.models.vision_transformer as models

from torchao.utils import benchmark_model, profiler_runner
torch.set_float32_matmul_precision("high")
# Load Vision Transformer model
model = models.vit_b_16(pretrained=True)

# Set the model to evaluation mode
model.eval().cuda().to(torch.bfloat16)

# Input tensor (batch_size, channels, height, width)
input_tensor = torch.randn(1, 3, 224, 224, dtype=torch.bfloat16, device="cuda")

model = torch.compile(model, mode="max-autotune")

# Must run with no_grad when optimizing for inference
with torch.no_grad():
    # warmup
    benchmark_model(model, 5, input_tensor)
    # benchmark
    print("elapsed_time: ", benchmark_model(model, 100, input_tensor), " milliseconds")
    # Create a trace
    profiler_runner("bfloat16.json.gz", benchmark_model, model, 5, input_tensor)
