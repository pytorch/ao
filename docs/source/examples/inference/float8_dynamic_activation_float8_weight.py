import torch.nn as nn

from torchao.quantization import (
    Float8DynamicActivationFloat8WeightConfig,
    PerRow,
    PerTensor,
    quantize_,
)

# Tensorwise scaling
model = nn.Sequential(nn.Linear(2048, 2048, device="cuda"))
quantize_(model, Float8DynamicActivationFloat8WeightConfig(granularity=PerTensor()))

# Rowwise scaling
model = nn.Sequential(nn.Linear(2048, 2048, device="cuda"))
quantize_(model, Float8DynamicActivationFloat8WeightConfig(granularity=PerRow()))
