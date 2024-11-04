import os
import torch
import torch.distributed as dist
from typing import Sequence
from torch.distributed import DeviceMesh
from torch.distributed.tensor import DTensor, Replicate, Shard, Placement
from torch.utils._python_dispatch import return_and_correct_aliasing
from torchao.dtypes.uintx.uintx import UintxTensor, to_uintx
from torchao.quantization.quant_api import quantize_, uintx_weight_only
from torchao.utils import fill_defaults

class M(torch.nn.Module):
    def __init__(self, in_features, out_features, **kwargs):
        super().__init__(**kwargs)
        self.linear = torch.nn.Linear(in_features=in_features, out_features=out_features, bias=False)

    def forward(self, x):
        return self.linear(x)

def quantize(m: torch.nn.Module, dtype, group_size=32)-> torch.nn.Module:
    """
    Quantize the model
    """
    quantize_(m, uintx_weight_only(dtype, group_size=group_size))
    return m

def shard(
    full_tensor: torch.tensor,
    device_mesh: DeviceMesh,
    placements: Sequence[Placement],
)-> DTensor:
    from torch.distributed.tensor._utils import compute_local_shape_and_global_offset

    shape, offset = compute_local_shape_and_global_offset(
        full_tensor.shape, device_mesh, placements
    )
    slices = [
        slice(cur_offset, cur_offset + cur_shape)
        for cur_shape, cur_offset in zip(shape, offset)
    ]
    local_tensor = full_tensor[slices]
    return DTensor.from_local(
        local_tensor, device_mesh, placements
    )
    

def colwise_shard(m: torch.nn.Module, mesh: DeviceMesh) -> torch.nn.Module:
    """
    Shard linear layer of the model in column-wise fashion
    """
    # Column-wise is wrt to A^T, so for A it is row-wise.
    orig_weight = m.linear.weight
    # Construct DTensor from local shard
    dtensor = shard(orig_weight, mesh, [Shard(0)])
    # Replace parameter in module
    m.linear.weight = torch.nn.Parameter(
        dtensor, requires_grad=False
    )
    return m

def rowwise_shard(m: torch.nn.Module, mesh: DeviceMesh) -> torch.nn.Module:
    """
    Shard linear layer of the model in row-wise fashion
    """
    # Row-wise is wrt to A^T, so for A it is column-wise.
    orig_weight = m.linear.weight
    # Construct DTensor from local shard
    dtensor = shard(orig_weight, mesh, [Shard(1)])
    # Replace parameter in module
    m.linear.weight = torch.nn.Parameter(
        dtensor, requires_grad=False
    )
    return m

class Linear16(torch.nn.Module):
    def __init__(self, scale, device):
        super().__init__()
        self.net = torch.nn.Sequential(
            torch.nn.Linear(scale * 2, scale, bias=False, dtype=torch.float16, device=device),
            torch.nn.Linear(scale, scale, bias=False, dtype=torch.float16, device=device),
            torch.nn.Linear(scale, scale//2, bias=False, dtype=torch.float16, device=device),
        )

    def forward(self, x):
        return self.net(x)
########
# Test #
########
def main():
    #run on cpu
    device = torch.device("cpu")
    proj_up = M(1024, 2048).to(device)
    proj_dn = M(2048, 1024).to(device)
    example_input = 100 * torch.randn(128, 1024, device=device)
    y = proj_dn(proj_up(example_input))

    # Quantize the model
    up_quant = quantize(proj_up, torch.uint6)
    dn_quant = quantize(proj_dn, torch.uint6)
    y_q = dn_quant(up_quant(example_input))
    print("Quantization works!")

    # To make sure different ranks create the same module
    torch.manual_seed(5)

    # Get rank and device
    world_size = int(os.environ["WORLD_SIZE"])
    rank = int(os.environ["RANK"])
    device = torch.device(f"cuda:{rank % torch.cuda.device_count()}")

    # Original model
    proj_up = M(1024, 2048).to(device)
    proj_dn = M(2048, 1024).to(device)
    example_input = 100 * torch.randn(128, 1024, device=device)
    y = proj_dn(proj_up(example_input))

    # Quantize the model
    up_quant = quantize(proj_up)
    dn_quant = quantize(proj_dn)
    y_q = dn_quant(up_quant(example_input))
    print("Quantization works!")

    # Create a device mesh
    dist.init_process_group(backend="nccl")
    mesh = dist.init_device_mesh("cuda", (world_size,))

    # Shard the models
    up_dist = colwise_shard(up_quant, mesh)
    dn_dist = rowwise_shard(dn_quant, mesh)

    # We need to turn inputs into DTensor form as well -- just a format change
    input_dtensor = DTensor.from_local(
        example_input, mesh, [Replicate()]
    )

    y_d = dn_dist(up_dist(input_dtensor))
    print("Distributed result:", y_d)
    print("Distributed works!")

    up_compiled = torch.compile(up_dist)
    y_up = up_compiled(input_dtensor)
    dn_compiled = torch.compile(dn_dist)
    y_dn = dn_compiled(y_up)
    print("compiled result:", y_dn)
    print("torch.compile works!")

    dist.destroy_process_group()

if __name__ == "__main__":
    main()
