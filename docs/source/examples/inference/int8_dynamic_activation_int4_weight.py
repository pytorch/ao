import torch.nn as nn

from torchao.prototype.quantization import Int8DynamicActivationInt4WeightConfig
from torchao.quantization import quantize_

model = nn.Sequential(nn.Linear(2048, 2048, device="cuda"))
quantize_(model, Int8DynamicActivationInt4WeightConfig())
