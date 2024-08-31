import torch
from my_dtype_tensor_subclass import MyDTypeTensor
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
    m = M()
    example_inputs = (100 * torch.randn(128, 1024),)
    m(*example_inputs)


    import os
    import torch
    from torch.distributed._tensor import init_device_mesh, Shard, distribute_tensor
    import torch.distributed as dist
    # initialize a fake process group
    store = torch.testing._internal.distributed.fake_pg.FakeStore()
    dist.init_process_group(
        backend="fake",
        world_size=2,
        rank=0,
        store=store,
    )
    mesh = init_device_mesh("cuda", (int(os.environ["WORLD_SIZE"]),))
    # Shard this tensor over the mesh by sharding `big_tensor`'s 0th dimension over the 0th dimension of `mesh`.
    quantized_weight = to_my_dtype_tp(m.linear.weight)
    print("quantized weight:", quantized_weight)
    quantized_weight_dtensor = distribute_tensor(quantized_weight, mesh, [Shard(dim=0)])
    print("quantized weight dtensor:", quantized_weight_dtensor)

    m.linear.weight = torch.nn.Parameter(
        quantized_weight_dtensor, requires_grad=False
    )

    m(*example_inputs)
