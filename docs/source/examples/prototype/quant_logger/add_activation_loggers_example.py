import torch
import torch.nn as nn

from torchao.prototype.quant_logger import add_activation_loggers

model = nn.Sequential(
    nn.Linear(128, 256),
    nn.ReLU(),
    nn.Linear(256, 512),
)
add_activation_loggers(model)
x = torch.randn(1, 128)
model(x)
# t=act, c=0, fqn='0.weight', op='linear', extra='MKN=1|128|256', max_abs=..., avg=..., std=...
# t=act, c=1, fqn='2.weight', op='linear', extra='MKN=1|256|512', max_abs=..., avg=..., std=...
