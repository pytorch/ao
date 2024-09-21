import os
import torch
import torch.distributed as dist
from torch.distributed import DeviceMesh
from torch.distributed._tensor import DTensor, Replicate, Shard
from torch.utils._python_dispatch import return_and_correct_aliasing
from my_dtype_tensor_subclass import MyDTypeTensor, fill_defaults

# a tensor subclass that supports tensor parallelism with DTensor
class MyDTypeTensorTP(MyDTypeTensor):
    pass

implements = MyDTypeTensorTP.implements

aten = torch.ops.aten

@implements([aten._to_copy.default, aten.clone.default])
def _(func, types, args, kwargs):
    return return_and_correct_aliasing(
        func, args, kwargs, args[0]._apply_fn_to_data(torch.clone)
    )

@implements([aten.split.Tensor])
def _(func, types, args, kwargs):
    layout_tensor_list = func(args[0].layout_tensor, *args[1:], **kwargs)
    out = [MyDTypeTensorTP(layout_tensor, layout_tensor.shape) for layout_tensor in layout_tensor_list]
    return out

@implements([aten.empty_like.default])
def _(func, types, args, kwargs):
    empty_like_layout_tensor = func(args[0].layout_tensor, *args[1:], **kwargs)
    return MyDTypeTensorTP(empty_like_layout_tensor, empty_like_layout_tensor.shape)

@implements(aten.slice.Tensor)
def _(func, types, args, kwargs):
    self, dim, start, end, step = fill_defaults(args, 5, [0, None, None, 1])
    assert step == 1
    if end >= self.shape[dim]:
        end = self.shape[dim]
    shape = list(self.shape)
    shape[dim] = end - start
    return self.__class__(aten.slice.Tensor(self.layout_tensor, dim, start, end, step), shape, self.dtype)

# this is needed for DTensor.from_local() and for flattening tensor
@implements(aten.view.default)
def _(func, types, args, kwargs):
    x, shape = args

    if tuple(x.shape) == tuple(shape):
        return x.__class__(x.layout_tensor, x.shape, x.dtype)

    if len(shape) == 1 and shape[0] == -1:
        return x.__class__(x.layout_tensor, (x.numel(),), x.dtype)

    raise ValueError(f"{x.__class__.__name__} only supports .view() with same shape or shape=[-1]")

@implements(aten.t.default)
def _(func, types, args, kwargs):
    tensor = args[0]
    shape = tensor.shape[::-1]
    new = tensor.__class__(tensor.layout_tensor.t(), shape, tensor.dtype)
    return return_and_correct_aliasing(func, args, kwargs, new)

@implements(aten.addmm.default)
def _(func, types, args, kwargs):
    input_tensor, weight_tensor, bias = (
        args[1],
        args[2],
        args[0],
    )
    weight_tensor = weight_tensor.dequantize()
    return aten.addmm(input_tensor, weight_tensor, bias)

@implements(aten.mm.default)
def _(func, types, args, kwargs):
    input_tensor, weight_tensor, bias = (
        args[0],
        args[1],
        None
    )
    print("mm weight transposed:", weight_tensor.layout_tensor.transposed)
    weight_tensor = weight_tensor.dequantize()
    return aten.mm(input_tensor, weight_tensor)


class M(torch.nn.Module):
    def __init__(self, in_features, out_features, **kwargs) -> None:
        super().__init__(**kwargs)
        self.linear = torch.nn.Linear(in_features, out_features, bias=False, device="cuda")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear(x)

to_my_dtype_tp = MyDTypeTensorTP.from_float

def quantize(m: torch.nn.Module) -> torch.nn.Module:
    """
    Quantize the model
    """
    m.linear.weight = torch.nn.Parameter(
        to_my_dtype_tp(m.linear.weight), requires_grad=False
    )
    return m

def colwise_shard(m: torch.nn.Module, mesh: DeviceMesh) -> torch.nn.Module:
    """
    Shard linear layer of the model in column-wise fashion
    """
    # Column-wise is wrt to A^T, so for A it is row-wise.
    # Number of rows per rank
    orig_weight = m.linear.weight
    n_local_rows = orig_weight.size(0) // mesh.size()
    rank = mesh.get_local_rank()
    local_shard = orig_weight[rank * n_local_rows : (rank + 1) * n_local_rows, :]
    # Construct DTensor from local shard
    dtensor = DTensor.from_local(local_shard, mesh, [Shard(0)])
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
    # Number of rows per rank
    orig_weight = m.linear.weight
    print("rowwise original:", orig_weight.shape)
    n_local_cols = orig_weight.size(1) // mesh.size()
    rank = mesh.get_local_rank()
    print("rowwise n_local_cols:", n_local_cols)
    local_shard = orig_weight[:, rank * n_local_cols : (rank + 1) * n_local_cols]
    # BUG: `local_shard` has the same shape as the original tensor
    print("rowwise local shard:", local_shard.shape)
    # Construct DTensor from local shard
    dtensor = DTensor.from_local(local_shard, mesh, [Shard(1)])
    # Replace parameter in module
    m.linear.weight = torch.nn.Parameter(
        dtensor, requires_grad=False
    )
    return m

########
# Test #
########
def main():
    # To make sure different ranks create the same module
    torch.manual_seed(5)

    # Original model
    proj_up = M(1024, 2048)
    proj_dn = M(2048, 1024)
    example_input = 100 * torch.randn(128, 1024, device="cuda")
    y = proj_dn(proj_up(example_input))

    # Quantize the model
    q_up = quantize(proj_up)
    q_dn = quantize(proj_dn)
    y_q = q_dn(q_up(example_input))
    print("Quantization works!")

    # Create a device mesh
    world_size = int(os.environ["WORLD_SIZE"])
    rank = int(os.environ["RANK"])
    dist.init_process_group(backend="nccl")
    mesh = dist.init_device_mesh("cuda", (world_size,))

    # Shard the models
    d_up = colwise_shard(q_up, mesh)
    print("d_up weight shape:", d_up.linear.weight.shape)
    d_dn = rowwise_shard(q_dn, mesh)

    # We need to turn inputs into DTensor form as well -- just a format change
    input_dtensor = DTensor.from_local(
        example_input, mesh, [Replicate()]
    )

    y_colwise = d_up(input_dtensor)
    print("y_colwise:", y_colwise.shape)
    print("result:", d_dn(y_colwise))
    print("Distributed works!")

    c_up = torch.compile(d_up)
    y_up = c_up(input_dtensor)
    print("y_up:", y_up.shape)
    c_dn = torch.compile(d_dn)
    y_dn = c_dn(y_up)
    print("y_dn:", y_dn.shape)
    print("compiled result:", y_dn)
    print("torch.compile works!")

    dist.destroy_process_group()

if __name__ == "__main__":
    main()
