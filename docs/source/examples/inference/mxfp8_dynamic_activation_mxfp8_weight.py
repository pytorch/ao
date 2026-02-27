import torch
import torch.nn as nn

from torchao.prototype.mx_formats.inference_workflow import (
    MXDynamicActivationMXWeightConfig,
)
from torchao.quantization import quantize_
from torchao.quantization.quantize_.common import KernelPreference

model = nn.Linear(32, 128, bias=False, dtype=torch.bfloat16, device="cuda")
config = MXDynamicActivationMXWeightConfig(
    activation_dtype=torch.float8_e4m3fn,
    weight_dtype=torch.float8_e4m3fn,
    kernel_preference=KernelPreference.AUTO,
)
quantize_(model, config=config)
model = torch.compile(model, fullgraph=True)
