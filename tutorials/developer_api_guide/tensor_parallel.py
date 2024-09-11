import torch
from my_dtype_tensor_subclass import MyDTypeTensor, fill_defaults
from torch.utils._python_dispatch import return_and_correct_aliasing

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
    print("dim:", dim, "start:", start, " end:", end, " shape:", end - start)
    print("manual shape:", (end - start,) + self.shape[1:])
    return self.__class__(aten.slice.Tensor(self.layout_tensor, dim, start, end, step), (end - start,) + self.shape[1:], self.dtype)

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
    weight_tensor = weight_tensor.dequantize()
    return aten.mm(input_tensor, weight_tensor)


class M(torch.nn.Module):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.linear = torch.nn.Linear(1024, 512, bias=False, device="cuda")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear(x)

to_my_dtype_tp = MyDTypeTensorTP.from_float

########
# Test #
########
if __name__ == "__main__":
    # To make sure different ranks create the same module
    torch.manual_seed(5)

    m = M()
    example_input = 100 * torch.randn(128, 1024, device="cuda")
    m(example_input)


    import os
    from torch.distributed._tensor import DTensor, Replicate, Shard
    import torch.distributed as dist

    # initialize a fake process group
    world_size = int(os.environ["WORLD_SIZE"])
    rank = int(os.environ["RANK"])
    dist.init_process_group(backend="nccl")
    mesh = dist.init_device_mesh("cuda", (world_size,))

    # Shard this tensor over the mesh by sharding `big_tensor`'s 0th dimension over the 0th dimension of `mesh`.
    orig_weight = m.linear.weight
    quantized_weight = to_my_dtype_tp(orig_weight)
    print("quantized weight:", quantized_weight)
    # Number of rows per rank
    n_local_rows = orig_weight.size(0) // world_size
    # TODO: add support for aten.slice.Tensor
    quantized_shard = quantized_weight[rank * n_local_rows : (rank + 1) * n_local_rows, :]
    print("quantized shard:", quantized_shard)
    # Construct DTensor from local shard
    quantized_dtensor = DTensor.from_local(quantized_shard, mesh, [Shard(0)])
    print("quantized dtensor:", quantized_dtensor)

    # Replace parameter in module
    m.linear.weight = torch.nn.Parameter(
        quantized_dtensor, requires_grad=False
    )

    # We need to turn inputs into DTensor form as well -- just a format change
    input_dtensor = DTensor.from_local(
        example_input, mesh, [Replicate()]
    )
    print("input dtensor:", input_dtensor)

    print("result:", m(input_dtensor))

    # doesn't work
    # [rank0]: torch._dynamo.exc.TorchRuntimeError: Failed running call_function <built-in function linear>(*(DTensor(local_tensor=FakeTensor(..., device='cuda:0', size=(128, 1024)), device_mesh=DeviceMesh('cuda', [0, 1,
    # 2, 3]), placements=(Replicate(),)), DTensor(local_tensor=MyDTypeTensorTP(data=FakeTensor(..., device='cuda:0', size=(128, 1024)), shape=torch.Size([1024, 1024]), device=cuda:0, dtype=torch.float32, requires_grad=False), device_mesh=DeviceMesh('cuda', [0, 1, 2, 3]), placements=(Shard(dim=0),)), None), **{}):
    # [rank0]: a and b must have same reduction dim, but got [128, 1024] X [128, 1024].
    # m = torch.compile(m)
    # print("compiled result:", m(input_dtensor))
