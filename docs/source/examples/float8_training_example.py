import torch
import torch.nn as nn

from torchao.float8 import Float8LinearConfig, convert_to_float8_training

# create model and sample input
m = (
    nn.Sequential(
        nn.Linear(8192, 4096, bias=False),
        nn.Linear(4096, 128, bias=False),
    )
    .bfloat16()
    .cuda()
)
optimizer = torch.optim.SGD(m.parameters(), lr=0.1)


# optional: filter modules from being eligible for float8 conversion
def module_filter_fn(mod: torch.nn.Module, fqn: str):
    # don't convert the last module
    if fqn == "1":
        return False
    # don't convert linear modules with weight dimensions not divisible by 16
    if isinstance(mod, torch.nn.Linear):
        if mod.in_features % 16 != 0 or mod.out_features % 16 != 0:
            return False
    return True


# configure float8 recipe
# valid recipe names: "tensorwise", "rowwise", "rowwise_with_gw_hp"
config = Float8LinearConfig.from_recipe_name("tensorwise")

# convert specified `torch.nn.Linear` modules to `Float8Linear`
convert_to_float8_training(m, config=config, module_filter_fn=module_filter_fn)

# enable torch.compile for competitive performance
m = torch.compile(m)

# training loop
x = torch.randn(4096, 8192, device="cuda", dtype=torch.bfloat16)
for _ in range(10):
    optimizer.zero_grad()
    y = m(x)
    y.sum().backward()
    optimizer.step()
