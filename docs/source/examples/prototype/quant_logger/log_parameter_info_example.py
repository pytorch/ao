import torch.nn as nn

from torchao.prototype.quant_logger import log_parameter_info

model = nn.Sequential(
    nn.Linear(128, 256),
    nn.ReLU(),
    nn.Linear(256, 512),
)
log_parameter_info(model)
# t=param, c=0, fqn='0.weight', op='', max_abs=..., avg=..., std=...
# t=param, c=1, fqn='0.bias', op='', max_abs=..., avg=..., std=...
# t=param, c=2, fqn='2.weight', op='', max_abs=..., avg=..., std=...
# t=param, c=3, fqn='2.bias', op='', max_abs=..., avg=..., std=...
