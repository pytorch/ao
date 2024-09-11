import torch
from my_dtype_tensor_subclass import MyDTypeTensor
from torch.utils._python_dispatch import return_and_correct_aliasing

# a tensor subclass that supports tensor parallelism with DTensor
class MyDTypeTensorTP(MyDTypeTensor):
    pass

implements = MyDTypeTensorTP.implements

aten = torch.ops.aten

def fill_defaults(args, n, defaults_tail):
    """
    __torch_dispatch__ doesn't guarantee the number of arguments you are
    passed (e.g., defaulted arguments are not passed); but usually it is
    convenient to pad out the arguments list with defaults.  This function
    helps you do that.
    Args:
        args: the list of positional arguments passed to __torch_dispatch__
        n: the number of arguments you are expecting to get
        defaults_tail: default values for the arguments, starting from the
            end of the list
    Example:
        >>> fill_defaults([1, 2, 3], 5, [3, 4, 5])
        [1, 2, 3, 4, 5]
        >>> fill_defaults([1, 2, 3], 5, [None, None, None])
        [1, 2, 3, None, None]]
    """
    if n - len(defaults_tail) > len(args):
        raise RuntimeError("not enough defaults to fill arguments")
    r = list(args)
    for i in range(len(args), n):
        r.append(defaults_tail[i - n + len(defaults_tail)])
    return r

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

@implements([aten.slice.Tensor])
def _(func, types, args, kwargs):
    self, dim, start, end, step = fill_defaults(args, 5, [0, None, None, 1])
    print("slice:", dim, start, end, step)
    if dim == 0:
        assert step == 1
        return self.__class__(aten.slice.Tensor(self.layout_tensor), (end - start + 1,) + self.shape[1:], self.dtype)
    return


class M(torch.nn.Module):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.linear = torch.nn.Linear(1024, 1024)

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
    example_input = 100 * torch.randn(128, 1024)
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
    quantized_dtensor = DTensor.from_local(quantized_shard, device_mesh, [Shard(0)])
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

    m(input_dtensor)
