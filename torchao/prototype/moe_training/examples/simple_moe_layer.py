import torch
from torch import nn
from torch.nn import functional as F

# this feature requires CUDA and SM89+
assert torch.cuda.is_available() and torch.cuda.get_device_capability() >= (8, 9)

from torchao.prototype.moe_training.conversion_utils import MoETrainingConfig
from torchao.quantization.quant_api import quantize_

# this example uses torchtitan llama4 MoE, see
try:
    from torchtitan.experiments.llama4.model.args import TransformerModelArgs
    from torchtitan.experiments.llama4.model.moe import MoE
except ImportError as e:
    raise ImportError(
        "torchtitan not installed, see installation instructions at https://github.com/pytorch/torchtitan"
    ) from e


# initialize model
device = torch.device("cuda")
model_args = TransformerModelArgs(
    moe_enabled=True,
    num_experts=8,
    dim=256,
)
model = MoE(model_args).to(torch.bfloat16).to(device)
init_std = 0.02
model.init_weights(init_std, device)

# module filter function to define which modules to quantize
target_fqns = ["experts"]


def moe_module_filter_fn(mod: nn.Module, cur_fqn: str) -> bool:
    for target_fqn in target_fqns:
        if target_fqn in cur_fqn:
            return True
    return False


# quantize the model
config = MoETrainingConfig()
quantize_(model, config=config, filter_fn=moe_module_filter_fn)

# training loop
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
for step in range(10):
    batch, seq, dim = 8, 2048, 256
    x = torch.randn(
        batch, seq, dim, dtype=torch.bfloat16, requires_grad=True, device=device
    )

    # forward pass
    out = model(x)

    # compute loss
    labels = torch.ones_like(out)
    out_loss = F.mse_loss(out, labels)
    print(f"step {step} loss: {out_loss.item()}")

    # backward pass
    out_loss.backward()
    optimizer.step()
