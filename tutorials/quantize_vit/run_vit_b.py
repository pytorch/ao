import torch
import torchvision.models.vision_transformer as models

# Load Vision Transformer model
model = models.vit_b_16(pretrained=True)

# Set the model to evaluation mode
model.eval().cuda().to(torch.bfloat16)

# Input tensor (batch_size, channels, height, width)
input_tensor = torch.randn(1, 3, 224, 224, dtype=torch.bfloat16, device='cuda')

model = torch.compile(model, mode='max-autotune')

# warmup
for _ in range(5):
    model(input_tensor)

torch.cuda.synchronize()
start_event = torch.cuda.Event(enable_timing=True)
end_event = torch.cuda.Event(enable_timing=True)
start_event.record()

# benchmark
for _ in range(100):
    model(input_tensor)

end_event.record()
torch.cuda.synchronize()
elapsed_time = start_event.elapsed_time(end_event) / 100
print("elapsed_time: ", elapsed_time, " milliseconds")
