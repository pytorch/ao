import contextlib
import functools
from typing import Dict

import torch
from torch.nested._internal.nested_tensor import nested_view_from_values_offsets
from torch.utils._pytree import tree_map

MAP_TENSOR_ATEN_OP_TABLE = {}


def implements(aten_ops_or_torch_fns):
    if not isinstance(aten_ops_or_torch_fns, (list, tuple)):
        aten_ops_or_torch_fns = [aten_ops_or_torch_fns]

    def decorator(func):
        for op in aten_ops_or_torch_fns:

            @functools.wraps(op)
            def wrapper(f, types, args, kwargs):
                return func(f, types, args, kwargs)

            MAP_TENSOR_ATEN_OP_TABLE[op] = wrapper
        return func

    return decorator


@contextlib.contextmanager
def no_dispatch():
    guard = torch._C._DisableTorchDispatch()
    try:
        yield
    finally:
        del guard


def wrap_dim(i, dim):
    if i < 0:
        return dim + i
    return i


def unwrap(t):
    if isinstance(t, MapTensor):
        with no_dispatch():
            return t.elems
    else:
        return t


def unwrap_i(t, i):
    if isinstance(t, MapTensor):
        with no_dispatch():
            return t.elems[i]
    else:
        return t


def unwrap_fn(t, fn):
    if isinstance(t, MapTensor):
        with no_dispatch():
            return fn(t.elems)
    else:
        return None


def wrap(t):
    if isinstance(t, torch.Tensor):
        return MapTensor(t)
    else:
        return t


@implements(torch.ops.aten.native_layer_norm.default)
def layer_norm_impl(func, types, args, kwargs=None):
    unwrapped_args = tree_map(unwrap, args)
    unwrapped_kwargs = tree_map(unwrap, kwargs)
    assert len(unwrapped_kwargs) == 0
    assert len(unwrapped_args) == 5, f"args: {unwrapped_args}"
    norm_res = func(*unwrapped_args)
    assert len(norm_res) == 3
    return tuple(wrap(a) for a in norm_res)


@implements(torch.ops.aten.add.Tensor)
def add_impl(func, types, args, kwargs):
    unwrapped_args = tree_map(unwrap, args)
    unwrapped_kwargs = tree_map(unwrap, kwargs)
    assert len(unwrapped_kwargs) == 0
    assert len(unwrapped_args) == 2, f"args: {unwrapped_args}"
    if not isinstance(args[0], MapTensor) and isinstance(args[1], MapTensor):
        if args[0].dim() == (args[1].dim() + 1):
            return NotImplemented
        return NotImplemented
    return wrap(func(*unwrapped_args, **unwrapped_kwargs))


@implements([torch.ops.aten.cat.default, torch.ops.aten.stack.default])
def cat_ops_impl(func, types, args, kwargs):
    unwrapped_args = tree_map(unwrap, args)
    unwrapped_kwargs = tree_map(unwrap, kwargs)
    assert len(unwrapped_kwargs) == 0
    assert len(unwrapped_args) <= 2, f"args: {unwrapped_args}"
    # TODO: Use MapTensor type for filter
    # First argument's dim
    dim = unwrapped_args[0][0].dim()
    size = unwrapped_args[0][0].size()
    for a in unwrapped_args[0]:
        if a.dim() > dim:
            dim = a.dim()
            size = a.size()
    new_args = []
    for a in unwrapped_args[0]:
        if a.dim() == dim:
            new_args.append(a)
        else:
            assert a.dim() + 1 == dim
            new_args.append(a.unsqueeze(0).expand((size[0],) + a.size()))
    orig_dim = unwrapped_args[1] if len(unwrapped_args) == 2 else 0
    return wrap(func(new_args, wrap_dim(orig_dim, dim - 1) + 1))


@implements(torch.ops.aten.select.int)
def select_impl(func, types, args, kwargs):
    unwrapped_args = tree_map(unwrap, args)
    unwrapped_kwargs = tree_map(unwrap, kwargs)
    assert len(unwrapped_kwargs) == 0
    assert len(unwrapped_args) == 3, f"args: {unwrapped_args}"
    return wrap(func(unwrapped_args[0], unwrapped_args[1] + 1, unwrapped_args[2]))


@implements(torch.ops.aten.slice.Tensor)
def slice_impl(func, types, args, kwargs):
    unwrapped_args = tree_map(unwrap, args)
    unwrapped_kwargs = tree_map(unwrap, kwargs)
    assert len(unwrapped_kwargs) == 0
    assert len(unwrapped_args) == 4, f"args: {unwrapped_args}"
    dim = unwrapped_args[0].dim()
    return wrap(
        func(
            unwrapped_args[0],
            wrap_dim(unwrapped_args[1], dim - 1) + 1,
            unwrapped_args[2],
            unwrapped_args[3],
        )
    )


@implements(
    [
        torch.ops.aten.mean.dim,
        torch.ops.aten.max.dim,
        torch.ops.aten.argmax.default,
        torch.ops.aten.min.dim,
        torch.ops.aten.any.dim,
        torch.ops.aten.amax.default,
        torch.ops.aten.amin.default,
        torch.ops.aten.all.default,
        torch.ops.aten.sum.dim_IntList,
    ]
)
def reductions_impl(func, types, args, kwargs):
    unwrapped_args = tree_map(unwrap, args)
    unwrapped_kwargs = tree_map(unwrap, kwargs)
    # TODO: THIS MIGHT BE WRONG
    if len(unwrapped_args) == 3 and len(unwrapped_kwargs) == 0:
        assert len(unwrapped_args[1]) == 1
        dim = unwrapped_args[0].dim()
        return wrap(
            func(
                unwrapped_args[0],
                [wrap_dim(u, dim - 1) + 1 for u in unwrapped_args[1]],
                unwrapped_args[2],
            )
        )
    if len(unwrapped_args) == 2 and len(unwrapped_kwargs) == 1:
        assert len(unwrapped_args[1]) == 1
        dim = unwrapped_args[0].dim()
        return wrap(
            func(
                unwrapped_args[0],
                [wrap_dim(u, dim - 1) + 1 for u in unwrapped_args[1]],
                **unwrapped_kwargs,
            )
        )
    if (
        len(unwrapped_args) == 2
        and len(unwrapped_kwargs) == 0
        and type(unwrapped_args[1]) == list
    ):
        assert len(unwrapped_args[1]) == 1
        dim = unwrapped_args[0].dim()
        return wrap(
            func(
                unwrapped_args[0], [wrap_dim(u, dim - 1) + 1 for u in unwrapped_args[1]]
            )
        )
    if (
        len(unwrapped_args) == 2
        and len(unwrapped_kwargs) == 0
        and type(unwrapped_args[1]) == int
    ):
        dim = unwrapped_args[0].dim()
        return wrap(func(unwrapped_args[0], wrap_dim(unwrapped_args[1], dim - 1) + 1))
    if len(args) == 1 and len(kwargs) == 0:
        return wrap(func(unwrapped_args[0]))
    return NotImplemented


@implements([torch.ops.aten._unsafe_view.default, torch.ops.aten.expand.default])
def view_ops_impl(func, types, args, kwargs):
    unwrapped_args = tree_map(unwrap, args)
    unwrapped_kwargs = tree_map(unwrap, kwargs)
    assert len(unwrapped_kwargs) == 0
    assert len(unwrapped_args) == 2, f"args: {unwrapped_args}"
    input_size = unwrapped_args[0].size()
    bigger_size = list(input_size[:1]) + unwrapped_args[1]
    return wrap(func(unwrapped_args[0], bigger_size))


@implements(torch.ops.aten.view.default)
def view_impl(func, types, args, kwargs):
    unwrapped_args = tree_map(unwrap, args)
    unwrapped_kwargs = tree_map(unwrap, kwargs)
    assert len(unwrapped_kwargs) == 0
    assert len(unwrapped_args) == 2, f"args: {unwrapped_args}"
    input_size = unwrapped_args[0].size()
    bigger_size = list(input_size[:1]) + unwrapped_args[1]
    if unwrapped_args[0].size() == tuple(bigger_size):
        return wrap(args[0].elems)
    return wrap(unwrapped_args[0].reshape(bigger_size))


@implements([torch.ops.aten.mm.default, torch.ops.aten.bmm.default])
def mm_ops_impl(func, types, args, kwargs):
    unwrapped_args = tree_map(unwrap, args)
    unwrapped_kwargs = tree_map(unwrap, kwargs)
    assert len(unwrapped_kwargs) == 0
    assert len(unwrapped_args) == 2, f"args: {unwrapped_args}"
    return wrap(torch.matmul(*unwrapped_args))


@implements(torch.ops.aten.unsqueeze.default)
def unsqueeze_impl(func, types, args, kwargs):
    unwrapped_args = tree_map(unwrap, args)
    unwrapped_kwargs = tree_map(unwrap, kwargs)
    assert len(unwrapped_kwargs) == 0
    assert len(unwrapped_args) == 2, f"args: {unwrapped_args}"
    new_i = unwrapped_args[1]
    if new_i >= 0:
        new_i += 1
    return wrap(func(unwrapped_args[0], new_i))


@implements(torch.ops.aten.squeeze.dim)
def squeeze_impl(func, types, args, kwargs):
    unwrapped_args = tree_map(unwrap, args)
    unwrapped_kwargs = tree_map(unwrap, kwargs)
    assert len(unwrapped_kwargs) == 0
    assert len(unwrapped_args) == 2, f"args: {unwrapped_args}"
    new_i = unwrapped_args[1]
    if new_i >= 0:
        new_i += 1
    return wrap(func(unwrapped_args[0], new_i))


@implements(torch.ops.aten.addmm.default)
def addmm_impl(func, types, args, kwargs):
    unwrapped_args = tree_map(unwrap, args)
    unwrapped_kwargs = tree_map(unwrap, kwargs)
    assert len(unwrapped_kwargs) == 0
    assert len(unwrapped_args) == 3, f"args: {unwrapped_args}"
    return wrap(torch.matmul(unwrapped_args[1], unwrapped_args[2]) + unwrapped_args[0])


@implements(torch.ops.aten.convolution.default)
def convolution_impl(func, types, args, kwargs):
    unwrapped_args = tree_map(unwrap, args)
    unwrapped_kwargs = tree_map(unwrap, kwargs)
    assert len(unwrapped_kwargs) == 0
    assert len(unwrapped_args) == 9, f"args: {unwrapped_args}"
    a = unwrapped_args[0]
    a = unwrapped_args[0].flatten(0, 1)
    # TODO: It's scary that this .contiguous seems necessary, but I we're below composite conv
    # which might expected contiguous output
    resa = func(*((a,) + unwrapped_args[1:])).contiguous()
    resb = resa.view(
        (unwrapped_args[0].size(0), unwrapped_args[0].size(1)) + resa.size()[1:]
    )
    return wrap(resb)


@implements(torch.ops.aten.upsample_bilinear2d.default)
def upsample_bilinear2d_impl(func, types, args, kwargs):
    unwrapped_args = tree_map(unwrap, args)
    unwrapped_kwargs = tree_map(unwrap, kwargs)
    assert len(unwrapped_kwargs) == 0
    assert len(unwrapped_args) == 3, f"args: {unwrapped_args}"
    a = unwrapped_args[0]
    a = unwrapped_args[0].flatten(0, 1)
    # NOTE: It's scary that this .contiguous seems necessary, but we're below composite upsample
    # which might expected contiguous output
    resa = func(*((a,) + unwrapped_args[1:])).contiguous()
    resb = resa.view(
        (unwrapped_args[0].size(0), unwrapped_args[0].size(1)) + resa.size()[1:]
    )
    return wrap(resb)


@implements(torch.ops.aten.transpose.int)
def transpose_impl(func, types, args, kwargs):
    unwrapped_args = tree_map(unwrap, args)
    unwrapped_kwargs = tree_map(unwrap, kwargs)
    assert len(unwrapped_kwargs) == 0
    assert len(unwrapped_args) == 3, f"args: {unwrapped_args}"
    dim = unwrapped_args[0].dim()
    return wrap(
        func(
            unwrapped_args[0],
            wrap_dim(unwrapped_args[1], dim - 1) + 1,
            wrap_dim(unwrapped_args[2], dim - 1) + 1,
        )
    )


@implements(torch.ops.aten.unbind.int)
def unbind_impl(func, types, args, kwargs):
    unwrapped_args = tree_map(unwrap, args)
    unwrapped_kwargs = tree_map(unwrap, kwargs)
    assert len(unwrapped_kwargs) == 0
    assert len(unwrapped_args) == 2, f"args: {unwrapped_args}"
    dim = unwrapped_args[0].dim()
    return wrap(func(unwrapped_args[0], wrap_dim(unwrapped_args[1], dim - 1) + 1))


@implements(torch.ops.aten.permute.default)
def permute_impl(func, types, args, kwargs):
    unwrapped_args = tree_map(unwrap, args)
    unwrapped_kwargs = tree_map(unwrap, kwargs)
    assert len(unwrapped_kwargs) == 0
    assert len(unwrapped_args) == 2, f"args: {unwrapped_args}"
    dim = unwrapped_args[0].dim()
    return wrap(
        func(
            unwrapped_args[0],
            ([0] + [wrap_dim(u, dim - 1) + 1 for u in unwrapped_args[1]]),
        )
    )


@implements(torch.ops.aten._scaled_dot_product_efficient_attention.default)
def _scaled_dot_product_efficient_attention_impl(func, types, args, kwargs):
    unwrapped_args = tree_map(unwrap, args)
    unwrapped_kwargs = tree_map(unwrap, kwargs)
    assert len(args) == 5
    if all(isinstance(a, MapTensor) for a in args[:3]):
        # assert len(unwrapped_kwargs) == 0
        assert len(unwrapped_args) == 5, f"args: {unwrapped_args}"
        assert unwrapped_args[0].dim() == 5
        assert unwrapped_args[1].dim() == 5
        assert unwrapped_args[2].dim() == 5
        sdpa_res = wrap(
            func(
                unwrapped_args[0].flatten(0, 1),
                unwrapped_args[1].flatten(0, 1),
                unwrapped_args[2].flatten(0, 1),
                unwrapped_args[3],
                unwrapped_args[4],
                **unwrapped_kwargs,
            )
        )
        return (wrap(sdpa_res[0].view(unwrapped_args[0].size())),) + sdpa_res[1:]
    if isinstance(args[0], MapTensor) and not any(
        isinstance(a, MapTensor) for a in args[1:]
    ):
        # assert len(unwrapped_kwargs) == 0
        assert len(unwrapped_args) == 5, f"args: {unwrapped_args}"
        assert unwrapped_args[0].dim() == 5
        assert unwrapped_args[1].dim() == 4
        assert unwrapped_args[2].dim() == 4
        a0 = unwrapped_args[0]
        a1_size = unwrapped_args[1].size()
        a1 = unwrapped_args[1].unsqueeze(0).expand((a0.size(0),) + a1_size)
        a2 = unwrapped_args[2].unsqueeze(0).expand((a0.size(0),) + a1_size)
        sdpa_res = wrap(
            func(
                a0.flatten(0, 1),
                a1.flatten(0, 1),
                a2.flatten(0, 1),
                unwrapped_args[3],
                unwrapped_args[4],
                **unwrapped_kwargs,
            )
        )
        return (wrap(sdpa_res[0].view(unwrapped_args[0].size())),) + sdpa_res[1:]
    if (
        (not isinstance(args[0], MapTensor))
        and isinstance(args[1], MapTensor)
        and (not isinstance(args[2], MapTensor))
    ):
        assert len(unwrapped_kwargs) == 0
        assert len(unwrapped_args) == 5, f"args: {unwrapped_args}"
        assert unwrapped_args[0].dim() == 4
        assert unwrapped_args[1].dim() == 5
        assert unwrapped_args[2].dim() == 4
        a1_size = unwrapped_args[1].size()
        a0 = (
            unwrapped_args[0]
            .unsqueeze(0)
            .expand((a1_size[0],) + unwrapped_args[0].size()[1:])
        )
        a2 = (
            unwrapped_args[2]
            .unsqueeze(0)
            .expand((a1_size[0],) + unwrapped_args[2].size()[1:])
        )
        sdpa_res = wrap(
            func(
                a0.flatten(0, 1),
                a1.flatten(0, 1),
                a2.flatten(0, 1),
                unwrapped_args[3],
                unwrapped_args[4],
            )
        )
        return (wrap(sdpa_res[0].view(unwrapped_args[0].size())),) + sdpa_res[1:]
    if (
        (not isinstance(args[0], MapTensor))
        and isinstance(args[1], MapTensor)
        and isinstance(args[2], MapTensor)
    ):
        # assert len(unwrapped_kwargs) == 0
        assert len(unwrapped_args) == 5, f"args: {unwrapped_args}"
        assert unwrapped_args[0].dim() == 4
        assert unwrapped_args[1].dim() == 5
        assert unwrapped_args[2].dim() == 5
        a0_size = unwrapped_args[0].size()
        a1_size = unwrapped_args[1].size()
        a0 = unwrapped_args[0].unsqueeze(0).expand((a1_size[0],) + a0_size)
        a1 = unwrapped_args[1]
        a2 = unwrapped_args[2]
        sdpa_res = wrap(
            func(
                a0.flatten(0, 1),
                a1.flatten(0, 1),
                a2.flatten(0, 1),
                unwrapped_args[3],
                unwrapped_args[4],
                **unwrapped_kwargs,
            )
        )
        return (wrap(sdpa_res[0].view((a1_size[0],) + a0_size)),) + sdpa_res[1:]
    return NotImplemented


@implements(torch.ops.aten._scaled_dot_product_flash_attention.default)
def _scaled_dot_product_flash_attention_impl(func, types, args, kwargs):
    unwrapped_args = tree_map(unwrap, args)
    unwrapped_kwargs = tree_map(unwrap, kwargs)
    assert len(args) == 3
    assert len(unwrapped_kwargs) == 1
    assert len(unwrapped_args) == 3, f"args: {unwrapped_args}"
    if all(isinstance(a, MapTensor) for a in args[:3]):
        assert unwrapped_args[0].dim() == 5
        assert unwrapped_args[1].dim() == 5
        assert unwrapped_args[2].dim() == 5
        sdpa_res = wrap(
            func(
                unwrapped_args[0].flatten(0, 1),
                unwrapped_args[1].flatten(0, 1),
                unwrapped_args[2].flatten(0, 1),
                **unwrapped_kwargs,
            )
        )
        return (wrap(sdpa_res[0].view(unwrapped_args[0].size())),) + sdpa_res[1:]
    if isinstance(args[0], MapTensor) and not any(
        isinstance(a, MapTensor) for a in args[1:]
    ):
        assert unwrapped_args[0].dim() == 5
        assert unwrapped_args[1].dim() == 4
        assert unwrapped_args[2].dim() == 4
        a0 = unwrapped_args[0]
        a1_size = unwrapped_args[1].size()
        a1 = unwrapped_args[1].unsqueeze(0).expand((a0.size(0),) + a1_size)
        a2 = unwrapped_args[2].unsqueeze(0).expand((a0.size(0),) + a1_size)
        sdpa_res = wrap(
            func(
                a0.flatten(0, 1), a1.flatten(0, 1), a2.flatten(0, 1), **unwrapped_kwargs
            )
        )
        return (wrap(sdpa_res[0].view(unwrapped_args[0].size())),) + sdpa_res[1:]
    if (
        (not isinstance(args[0], MapTensor))
        and isinstance(args[1], MapTensor)
        and (not isinstance(args[2], MapTensor))
    ):
        assert unwrapped_args[0].dim() == 4
        assert unwrapped_args[1].dim() == 5
        assert unwrapped_args[2].dim() == 4
        a1_size = unwrapped_args[1].size()
        a0 = (
            unwrapped_args[0]
            .unsqueeze(0)
            .expand((a1_size[0],) + unwrapped_args[0].size()[1:])
        )
        a2 = (
            unwrapped_args[2]
            .unsqueeze(0)
            .expand((a1_size[0],) + unwrapped_args[2].size()[1:])
        )
        sdpa_res = wrap(
            func(
                a0.flatten(0, 1), a1.flatten(0, 1), a2.flatten(0, 1), **unwrapped_kwargs
            )
        )
        return (wrap(sdpa_res[0].view(unwrapped_args[0].size())),) + sdpa_res[1:]
    if (
        (not isinstance(args[0], MapTensor))
        and isinstance(args[1], MapTensor)
        and isinstance(args[2], MapTensor)
    ):
        assert unwrapped_args[0].dim() == 4
        assert unwrapped_args[1].dim() == 5
        assert unwrapped_args[2].dim() == 5
        a0_size = unwrapped_args[0].size()
        a1_size = unwrapped_args[1].size()
        a0 = unwrapped_args[0].unsqueeze(0).expand((a1_size[0],) + a0_size)
        a1 = unwrapped_args[1]
        a2 = unwrapped_args[2]
        sdpa_res = wrap(
            func(
                a0.flatten(0, 1), a1.flatten(0, 1), a2.flatten(0, 1), **unwrapped_kwargs
            )
        )
        return (wrap(sdpa_res[0].view((a1_size[0],) + a0_size)),) + sdpa_res[1:]
    return NotImplemented


# torch.ops.aten._unsafe_index.Tensor is only needed by inductor for compile
@implements([torch.ops.aten._unsafe_index.Tensor, torch.ops.aten.index.Tensor])
def index_ops_impl(func, types, args, kwargs):
    unwrapped_args = tree_map(unwrap, args)
    unwrapped_kwargs = tree_map(unwrap, kwargs)
    assert len(unwrapped_kwargs) == 0
    assert len(unwrapped_args) == 2, f"args: {unwrapped_args}"
    # if len(args[1]) == 1 and isinstance(args[1][0], MapTensor) and isinstance(args[0], MapTensor):
    #     return wrap(func(*unwrapped_args))
    if (
        len(args[1]) == 1
        and isinstance(args[1][0], MapTensor)
        and not isinstance(args[0], MapTensor)
    ):
        tensors = [
            func(args[0], [args[1][0].elems[i]]) for i in range(len(args[1][0].elems))
        ]
        values = torch.cat(tensors)
        lengths = torch.tensor([0] + [t.size(0) for t in tensors], pin_memory=True).to(
            values.device, non_blocking=True
        )
        offsets = torch.cumsum(lengths, dim=0)
        nt = nested_view_from_values_offsets(values, offsets)
        assert nt.layout == torch.jagged
        return wrap(nt)
    if (
        isinstance(args[0], MapTensor)
        and not isinstance(args[1][0], MapTensor)
        and len(args[1]) == 1
    ):
        return wrap(func(args[0].elems, [args[1][0].unsqueeze(0)]))
    if (
        isinstance(args[0], MapTensor)
        and not isinstance(args[1][0], MapTensor)
        and isinstance(args[1][1], MapTensor)
        and len(args[1]) == 2
    ):
        res = []
        for a0, a11 in zip(args[0].elems.unbind(), args[1][1].elems.unbind()):
            res.append(func(a0, [args[1][0], a11]))
        return wrap(torch.stack(res))
    if (
        isinstance(args[0], MapTensor)
        and isinstance(args[1][0], MapTensor)
        and len(args[1]) == 1
    ):
        tensors = [
            func(args[0].elems[i], [args[1][0].elems[i]])
            for i in range(len(args[0].elems))
        ]
        values = torch.cat(tensors)
        lengths = torch.tensor([0] + [t.size(0) for t in tensors], pin_memory=True).to(
            values.device, non_blocking=True
        )
        offsets = torch.cumsum(lengths, dim=0)
        nt = nested_view_from_values_offsets(values, offsets)
        assert nt.layout == torch.jagged
        return wrap(nt)
    a = unwrapped_args[0]
    a = unwrapped_args[0].flatten(0, 1)
    resa = func(a, args[1])
    resb = resa.view(
        (unwrapped_args[0].size(0), unwrapped_args[0].size(1)) + resa.size()[1:]
    )
    return wrap(resb)


# Prims
@implements(torch.ops.aten.dim.default)
def dim_impl(func, types, args, kwargs):
    assert len(args) == 1
    assert len(kwargs) == 0
    ret_dim = func(args[0].elems) - 1
    assert ret_dim >= 0
    return ret_dim


@implements(torch.ops.aten.sym_size.default)
def sym_impl(func, types, args, kwargs):
    assert len(args) == 1
    assert len(kwargs) == 0
    elems_size = func(args[0].elems)
    assert len(elems_size) > 0
    return elems_size[1:]


@implements(torch.ops.aten.is_contiguous.default)
def is_contiguous_impl(func, types, args, kwargs):
    assert len(args) == 1
    assert len(kwargs) == 0
    return func(args[0].elems)


@implements(
    [
        torch.ops.aten.clamp.default,
        torch.ops.aten.clone.default,
        torch.ops.aten.cos.default,
        torch.ops.aten.div.Tensor,
        torch.ops.aten.eq.Scalar,
        torch.ops.aten.gelu.default,
        torch.ops.aten.mul.Tensor,
        torch.ops.aten.pow.Tensor_Scalar,
        torch.ops.aten.relu.default,
        torch.ops.aten.sigmoid.default,
        torch.ops.aten.sin.default,
        torch.ops.aten.sqrt.default,
        torch.ops.aten.sub.Tensor,
        torch.ops.aten.unbind.int,
        torch.ops.aten.where.self,
        torch.ops.aten.zeros_like.default,
        torch.ops.aten._to_copy.default,
        torch.ops.aten.gt.Scalar,
        torch.ops.aten.ge.Scalar,
        torch.ops.aten.bitwise_not.default,
        torch.ops.aten.lt.Tensor,
        torch.ops.aten.bitwise_or.Tensor,
        torch.ops.aten.eq.Tensor,
        torch.ops.aten.abs.default,
        torch.ops.aten.ne.Scalar,
        torch.ops.aten.le.Tensor,
        torch.ops.aten.view_as_complex.default,
        torch.ops.aten.view_as_real.default,
        torch.ops.aten.neg.default,
        torch.ops.aten.le.Scalar,
        torch.ops.aten.rsub.Scalar,
        # Sketchy new in place ops
        torch.ops.aten.bitwise_and_.Tensor,
        torch.ops.aten.bitwise_or_.Tensor,
        torch.ops.aten.le.Tensor,
        torch.ops.aten.logical_and.default,
        # in place ops
        torch.ops.aten.add_.Tensor,
        torch.ops.aten.copy_.default,
        # Prims
        torch.ops.prim.layout.default,
    ]
)
def forwardables_impl(func, types, args, kwargs):
    unwrapped_args = tree_map(unwrap, args)
    unwrapped_kwargs = tree_map(unwrap, kwargs)
    return wrap(func(*unwrapped_args, **unwrapped_kwargs))


def run_invariant_test(res, func, args, kwargs):
    # Compares 0th element of list of results with
    # func applied to 0th arg and kwarg.
    # Rough test to maintain per-op accuracy.
    if isinstance(res, torch.Tensor):
        unwrapped_args_0 = tree_map(lambda x: unwrap_i(x, 0), args)
        unwrapped_kwargs_0 = tree_map(lambda x: unwrap_i(x, 0), kwargs)
        if func == torch.ops.aten.view.default:
            res_0 = torch.ops.aten.reshape.default(
                *unwrapped_args_0, **unwrapped_kwargs_0
            )
        else:
            res_0 = func(*unwrapped_args_0, **unwrapped_kwargs_0)
        # TODO: Extend this all elems not just elems[0]
        if res.elems[0].size() != res_0.size():
            import pdb

            pdb.set_trace()
        if not torch.allclose(res.elems[0], res_0, atol=1e-3, rtol=1e-3):
            import pdb

            pdb.set_trace()
    else:
        pass
        # print("res got type: ", type(res))
        # import pdb; pdb.set_trace()
    return res


class MapTensor(torch.Tensor):
    @staticmethod
    def __new__(cls, elems):
        # print("elems.layout: ", elems.layout)
        return torch.Tensor._make_wrapper_subclass(
            cls,
            elems.shape[1:],
            dtype=elems.dtype,
            device=elems.device,
            layout=elems.layout,
            dispatch_layout=True,
            dispatch_sizes_strides_policy=(
                "sizes" if elems.layout == torch.jagged else None
            ),
            storage_size=(
                elems._values.untyped_storage().size()
                if elems.layout == torch.jagged
                else None
            ),
        )

    def __init__(self, elems):
        self.elems = elems

    @classmethod
    def __torch_dispatch__(cls, func, types, args, kwargs=None):
        if func in MAP_TENSOR_ATEN_OP_TABLE:
            res = MAP_TENSOR_ATEN_OP_TABLE[func](func, types, args, kwargs)
            # run_invariant_test(res, func, args, kwargs)
            return res
        return NotImplemented

    @classmethod
    def __torch_function__(cls, func, types, args=(), kwargs=None):
        if kwargs is None:
            kwargs = {}
        with torch._C.DisableTorchFunctionSubclass():
            return func(*args, **kwargs)

    # flatten/unflatten is needed for compile
    def __tensor_flatten__(self):
        ctx = {}
        inner_tensors = ["elems"]
        return inner_tensors, ctx

    @staticmethod
    def __tensor_unflatten__(inner_tensors: Dict, meta, outer_size, outer_stride):
        # inner tensors: _values, _offsets, [_lengths], [_min_seqlen], [_max_seqlen]
        assert len(inner_tensors) == 1, f"{inner_tensors}"
        elems = inner_tensors["elems"]

        return MapTensor(elems)

    def __repr__(self):
        return f"MapTensor({self.elems.size()})"

    def pin_memory(self):
        elems = self.elems.pin_memory()
        return wrap(elems)


# ts is a higher dim Tensor
def to_map_tensor(ts: torch.Tensor):
    return MapTensor(ts)
