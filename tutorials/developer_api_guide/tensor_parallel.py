# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD 3-Clause license found in the
# LICENSE file in the root directory of this source tree.
import os
from typing import Sequence

import torch
import torch.distributed as dist
from my_dtype_tensor_subclass import MyDTypeTensor
from torch.distributed import DeviceMesh
from torch.distributed.tensor import DTensor, Placement, Replicate, Shard
from torch.utils._python_dispatch import return_and_correct_aliasing

from torchao.utils import fill_defaults


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
    tensor_impl_list = func(args[0].tensor_impl, *args[1:], **kwargs)
    out = [
        MyDTypeTensorTP(tensor_impl, tensor_impl.shape)
        for tensor_impl in tensor_impl_list
    ]
    return out


@implements([aten.empty_like.default])
def _(func, types, args, kwargs):
    empty_like_tensor_impl = func(args[0].tensor_impl, *args[1:], **kwargs)
    return MyDTypeTensorTP(empty_like_tensor_impl, empty_like_tensor_impl.shape)


@implements(aten.slice.Tensor)
def _(func, types, args, kwargs):
    self, dim, start, end, step = fill_defaults(args, 5, [0, None, None, 1])
    assert step == 1
    if end >= self.shape[dim]:
        end = self.shape[dim]
    shape = list(self.shape)
    shape[dim] = end - start
    return self.__class__(
        aten.slice.Tensor(self.tensor_impl, dim, start, end, step), shape, self.dtype
    )


# this is needed for DTensor.from_local() and for flattening tensor
@implements(aten.view.default)
def _(func, types, args, kwargs):
    x, shape = args

    if tuple(x.shape) == tuple(shape):
        return x.__class__(x.tensor_impl, x.shape, x.dtype)

    if len(shape) == 1 and shape[0] == -1:
        return x.__class__(x.tensor_impl, (x.numel(),), x.dtype)

    raise ValueError(
        f"{x.__class__.__name__} only supports .view() with same shape or shape=[-1]"
    )


@implements(aten.t.default)
def _(func, types, args, kwargs):
    tensor = args[0]
    shape = tensor.shape[::-1]
    new = tensor.__class__(tensor.tensor_impl.t(), shape, tensor.dtype)
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
    input_tensor, weight_tensor, _ = (args[0], args[1], None)
    weight_tensor = weight_tensor.dequantize()
    return aten.mm(input_tensor, weight_tensor)


class M(torch.nn.Module):
    def __init__(self, in_features, out_features, **kwargs) -> None:
        super().__init__(**kwargs)
        self.linear = torch.nn.Linear(in_features, out_features, bias=False)

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


def shard(
    full_tensor: torch.Tensor,
    device_mesh: DeviceMesh,
    placements: Sequence[Placement],
) -> DTensor:
    """
    Add a shard function to simplify both colwise_shard and rowwise_shard.  The
    shard function accepts a full tensor, and returns a DTensor based on
    indicated placements.  Goal is to move the shard function as a static method
    of DTensor, e.g.
        dtensor = DTensor.shard(full_tensor, device_mesh, placement)
    """
    from torch.distributed.tensor._utils import compute_local_shape_and_global_offset

    shape, offset = compute_local_shape_and_global_offset(
        full_tensor.shape, device_mesh, placements
    )
    slices = [
        slice(cur_offset, cur_offset + cur_shape)
        for cur_shape, cur_offset in zip(shape, offset)
    ]
    local_tensor = full_tensor[slices]
    return DTensor.from_local(local_tensor, device_mesh, placements)


def colwise_shard(m: torch.nn.Module, mesh: DeviceMesh) -> torch.nn.Module:
    """
    Shard linear layer of the model in column-wise fashion
    """
    # Column-wise is wrt to A^T, so for A it is row-wise.
    orig_weight = m.linear.weight
    # Construct DTensor from local shard
    dtensor = shard(orig_weight, mesh, [Shard(0)])
    # Replace parameter in module
    m.linear.weight = torch.nn.Parameter(dtensor, requires_grad=False)
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
    m.linear.weight = torch.nn.Parameter(dtensor, requires_grad=False)
    return m


########
# Test #
########
def main():
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
    proj_dn(proj_up(example_input))

    # Quantize the model
    up_quant = quantize(proj_up)
    dn_quant = quantize(proj_dn)
    dn_quant(up_quant(example_input))
    print("Quantization works!")

    # Create a device mesh
    dist.init_process_group(backend="nccl")
    mesh = dist.init_device_mesh("cuda", (world_size,))

    # Shard the models
    up_dist = colwise_shard(up_quant, mesh)
    dn_dist = rowwise_shard(dn_quant, mesh)

    # We need to turn inputs into DTensor form as well -- just a format change
    input_dtensor = DTensor.from_local(example_input, mesh, [Replicate()])

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
