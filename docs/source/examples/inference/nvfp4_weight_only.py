import torch
import torch.nn as nn

from torchao.prototype.mx_formats.inference_workflow import NVFP4WeightOnlyConfig
from torchao.quantization import quantize_

model = nn.Linear(32, 128, bias=False, dtype=torch.bfloat16, device="cuda")
config = NVFP4WeightOnlyConfig(
    use_dynamic_per_tensor_scale=True,
)
quantize_(model, config=config)
model = torch.compile(model, fullgraph=True)
