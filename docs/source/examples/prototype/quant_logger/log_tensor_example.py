import torch
import torch.nn as nn

from torchao.prototype.quant_logger import log_parameter_info


# Override the default log_tensor to print per-channel mean and std
@torch.library.custom_op("quant_logger::log_tensor", mutates_args=("x",))
def log_tensor(
    x: torch.Tensor, fqn: str, op: str, tag: str, extra: str | None = None
) -> None:
    if x.ndim >= 2:
        channel_mean = x.mean(dim=1)
        channel_std = x.std(dim=1, correction=0)
        print(f"{fqn}: channel_mean={channel_mean}, channel_std={channel_std}")
    else:
        print(f"{fqn}: mean={x.mean().item():.4f}, std={x.std().item():.4f}")


model = nn.Sequential(
    nn.Linear(128, 256),
    nn.ReLU(),
    nn.Linear(256, 512),
)
log_parameter_info(model)
# 0.weight: channel_mean=tensor([...]), channel_std=tensor([...])
# 0.bias: mean=..., std=...
# 2.weight: channel_mean=tensor([...]), channel_std=tensor([...])
# 2.bias: mean=..., std=...
