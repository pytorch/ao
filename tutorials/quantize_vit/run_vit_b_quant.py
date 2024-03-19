import torch
import torchvision.models.vision_transformer as models
import logging
logging.basicConfig(level=logging.INFO)

# Load Vision Transformer model
model = models.vit_b_16(pretrained=True)

# Set the model to evaluation mode
model.eval().cuda().to(torch.bfloat16)

# Input tensor (batch_size, channels, height, width)
input_tensor = torch.randn(1, 3, 224, 224, dtype=torch.bfloat16, device='cuda')

## Quantization code - start
from torchao.quantization import quant_api
quant_api.change_linear_weights_to_int8_dqtensors(model)
from torch._inductor import config as inductorconfig
inductorconfig.force_fuse_int_mm_with_mul = True
## Quantization code - end

model = torch.compile(model, mode='max-autotune')

def benchmark_model(model, num_runs, input_tensor):
    torch.cuda.synchronize()
    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)
    start_event.record()
    
    # benchmark
    for _ in range(num_runs):
        with torch.autograd.profiler.record_function("timed region"):
            model(input_tensor)
    
    end_event.record()
    torch.cuda.synchronize()
    return start_event.elapsed_time(end_event) / num_runs

def profiler_runner(path, fn, *args, **kwargs):
    with torch.profiler.profile(
            activities=[torch.profiler.ProfilerActivity.CPU,
                        torch.profiler.ProfilerActivity.CUDA],
            record_shapes=True) as prof:
        result = fn(*args, **kwargs)
    prof.export_chrome_trace(path)
    return result

# Must run with no_grad when optimizing for inference
with torch.no_grad():
    # warmup
    benchmark_model(model, 5, input_tensor)
    # benchmark
    print("elapsed_time: ", benchmark_model(model, 100, input_tensor), " milliseconds")
    # Create a trace
    profiler_runner("quant.json.gz", benchmark_model, model, 5, input_tensor)
