import torch.nn as nn

from torchao.quantization import IntxWeightOnlyConfig, quantize_

model = nn.Sequential(nn.Linear(2048, 2048, device="cuda"))
quantize_(model, IntxWeightOnlyConfig())
