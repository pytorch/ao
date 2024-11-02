import contextlib
import torch
from torch.utils._pytree import tree_map
from typing import Dict
from torch.nested._internal.nested_tensor import nested_view_from_values_offsets

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

def ops_impl(cls, func, types, args, kwargs=None):

    unwrapped_args = tree_map(unwrap, args)
    unwrapped_kwargs = tree_map(unwrap, kwargs)

    if func == torch.ops.aten.native_layer_norm.default:
        assert len(unwrapped_kwargs) == 0
        assert len(unwrapped_args) == 5, f"args: {unwrapped_args}"
        norm_res = func(*unwrapped_args)
        assert len(norm_res) == 3
        return tuple(wrap(a) for a in norm_res)

    if func == torch.ops.aten.div.Tensor:
        assert len(unwrapped_kwargs) == 0
        assert len(unwrapped_args) == 2, f"args: {unwrapped_args}"
        if isinstance(args[0], MapTensor) and isinstance(args[1], MapTensor):
            if args[0].dim() == 1 and args[1].dim() == 0:
                res = func(unwrapped_args[0], unwrapped_args[1].unsqueeze(-1).expand_as(unwrapped_args[0]))
                return wrap(res)

    # TODO: I guess if being added against something higher dim
    # we should increase dim overall?
    if func == torch.ops.aten.add.Tensor:
        assert len(unwrapped_kwargs) == 0
        assert len(unwrapped_args) == 2, f"args: {unwrapped_args}"
        # print("unwrapped_args")
        # print([type(a) for a in unwrapped_args])
        if not isinstance(args[0], MapTensor) and isinstance(args[1], MapTensor):
            if args[0].dim() == (args[1].dim() + 1):
                return NotImplemented
                # return wrap(func(unwrapped_args[0], unwrapped_args[1].unsqueeze(1)))
            # print("args[0].dim(): ", args[0].dim())
            # print("args[1].dim(): ", args[1].dim())
            # print("type(args[0]): ", type(args[0]))
            # print("type(args[1]): ", type(args[1]))
            # TODO: THIS GETS CALLED???
            return NotImplemented
        if isinstance(args[0], MapTensor) and isinstance(args[1], torch.Tensor):
            if args[0].dim() == 3 and args[1].dim() == 4:
                return wrap(func(args[0].elems.unsqueeze(1), args[1]))
        if isinstance(args[0], MapTensor) and isinstance(args[1], MapTensor):
            if args[0].dim() == 4 and args[1].dim() == 3:
                return wrap(func(unwrapped_args[0], unwrapped_args[1].unsqueeze(1)))
        pass

    if func in [torch.ops.aten.cat.default, torch.ops.aten.stack.default]:
        assert len(unwrapped_kwargs) == 0
        assert len(unwrapped_args) == 2, f"args: {unwrapped_args}"
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
        return wrap(func(new_args, wrap_dim(unwrapped_args[1], dim - 1) + 1))

    if func == torch.ops.aten.select.int:
        assert len(unwrapped_kwargs) == 0
        assert len(unwrapped_args) == 3, f"args: {unwrapped_args}"
        return wrap(func(unwrapped_args[0], unwrapped_args[1] + 1, unwrapped_args[2]))

    if func == torch.ops.aten.slice.Tensor:
        assert len(unwrapped_kwargs) == 0
        assert len(unwrapped_args) == 4, f"args: {unwrapped_args}"
        dim = unwrapped_args[0].dim()
        return wrap(func(unwrapped_args[0],
                         wrap_dim(unwrapped_args[1], dim - 1) + 1,
                         unwrapped_args[2],
                         unwrapped_args[3]))

    if func in [torch.ops.aten.mean.dim,
                torch.ops.aten.max.dim,
                torch.ops.aten.min.dim,
                torch.ops.aten.any.dim,
                torch.ops.aten.amax.default,
                torch.ops.aten.amin.default,
                torch.ops.aten.all.default,
                torch.ops.aten.sum.dim_IntList]:
        # TODO: THIS MIGHT BE WRONG
        if len(unwrapped_args) == 3 and len(unwrapped_kwargs) == 0:
            assert len(unwrapped_args[1]) == 1
            dim = unwrapped_args[0].dim()
            return wrap(func(unwrapped_args[0],
                             [wrap_dim(u, dim - 1) + 1 for u in unwrapped_args[1]],
                             unwrapped_args[2]))
        if len(unwrapped_args) == 2 and len(unwrapped_kwargs) == 1:
            assert len(unwrapped_args[1]) == 1
            dim = unwrapped_args[0].dim()
            return wrap(func(unwrapped_args[0],
                             [wrap_dim(u, dim - 1) + 1 for u in unwrapped_args[1]],
                             **unwrapped_kwargs))
        if len(unwrapped_args) == 2 and len(unwrapped_kwargs) == 0 and type(unwrapped_args[1]) == list:
            assert len(unwrapped_args[1]) == 1
            dim = unwrapped_args[0].dim()
            return wrap(func(unwrapped_args[0],
                             [wrap_dim(u, dim - 1) + 1 for u in unwrapped_args[1]]))
        if len(unwrapped_args) == 2 and len(unwrapped_kwargs) == 0 and type(unwrapped_args[1]) == int:
            dim = unwrapped_args[0].dim()
            return wrap(func(unwrapped_args[0], wrap_dim(unwrapped_args[1], dim - 1) + 1))
        if len(args) == 1 and len(kwargs) == 0:
            return wrap(func(unwrapped_args[0]))
        return NotImplemented

    view_ops = [torch.ops.aten._unsafe_view.default,
                torch.ops.aten.expand.default]
    if func in view_ops:
        assert len(unwrapped_kwargs) == 0
        assert len(unwrapped_args) == 2, f"args: {unwrapped_args}"
        input_size = unwrapped_args[0].size()
        bigger_size = list(input_size[:1]) + unwrapped_args[1]
        return wrap(func(unwrapped_args[0], bigger_size))

    if func is torch.ops.aten.view.default:
        assert len(unwrapped_kwargs) == 0
        assert len(unwrapped_args) == 2, f"args: {unwrapped_args}"
        input_size = unwrapped_args[0].size()
        bigger_size = list(input_size[:1]) + unwrapped_args[1]
        if unwrapped_args[0].size() == tuple(bigger_size):
            return wrap(args[0].elems)
        return wrap(unwrapped_args[0].reshape(bigger_size))

    if func in [torch.ops.aten.mm.default, torch.ops.aten.bmm.default]:
        assert len(unwrapped_kwargs) == 0
        assert len(unwrapped_args) == 2, f"args: {unwrapped_args}"
        return wrap(torch.matmul(*unwrapped_args))

    if func in [torch.ops.aten.unsqueeze.default]:
        assert len(unwrapped_kwargs) == 0
        assert len(unwrapped_args) == 2, f"args: {unwrapped_args}"
        dim = unwrapped_args[0].dim()
        new_i = unwrapped_args[1]
        if new_i >= 0:
            new_i += 1
        return wrap(func(unwrapped_args[0], new_i))

    if func == torch.ops.aten.addmm.default:
        assert len(unwrapped_kwargs) == 0
        assert len(unwrapped_args) == 3, f"args: {unwrapped_args}"
        return wrap(torch.matmul(unwrapped_args[1], unwrapped_args[2]) + unwrapped_args[0])

    if func == torch.ops.aten.convolution.default:
        assert len(unwrapped_kwargs) == 0
        assert len(unwrapped_args) == 9, f"args: {unwrapped_args}"
        a = unwrapped_args[0]
        # print("0 a.size(): ", a.size())
        a = unwrapped_args[0].flatten(0, 1)
        # print("1 a.size(): ", a.size())
        # TODO: It's scary that this .contiguous seems necessary, but I guess we're below composite conv
        # which might expected contiguous output
        resa = func(*((a,) + unwrapped_args[1:])).contiguous()
        # print("0 resa.size(): ", resa.size())
        resb = resa.view((unwrapped_args[0].size(0), unwrapped_args[0].size(1)) + resa.size()[1:])
        # print("1 resb.size(): ", resb.size())
        # res_0 = func(*((unwrapped_args[0][0],) + unwrapped_args[1:]))
        # if not torch.allclose(resb[0], res_0):
        #     print("139203")
        #     import pdb; pdb.set_trace()
        #     pass
        return wrap(resb)

    if func == torch.ops.aten.upsample_bilinear2d.default:
        assert len(unwrapped_kwargs) == 0
        assert len(unwrapped_args) == 3, f"args: {unwrapped_args}"
        a = unwrapped_args[0]
        # print("0 a.size(): ", a.size())
        a = unwrapped_args[0].flatten(0, 1)
        # print("1 a.size(): ", a.size())
        # TODO: It's scary that this .contiguous seems necessary, but I guess we're below composite conv
        # which might expected contiguous output
        resa = func(*((a,) + unwrapped_args[1:])).contiguous()
        # print("0 resa.size(): ", resa.size())
        resb = resa.view((unwrapped_args[0].size(0), unwrapped_args[0].size(1)) + resa.size()[1:])
        # print("1 resb.size(): ", resb.size())
        # res_0 = func(*((unwrapped_args[0][0],) + unwrapped_args[1:]))
        # if not torch.allclose(resb[0], res_0):
        #     print("139203")
        #     import pdb; pdb.set_trace()
        #     pass
        return wrap(resb)

    if func == torch.ops.aten.transpose.int:
        assert len(unwrapped_kwargs) == 0
        assert len(unwrapped_args) == 3, f"args: {unwrapped_args}"
        dim = unwrapped_args[0].dim()
        return wrap(func(unwrapped_args[0],
                         wrap_dim(unwrapped_args[1], dim - 1) + 1,
                         wrap_dim(unwrapped_args[2], dim - 1) + 1))

    if func == torch.ops.aten.permute.default:
        assert len(unwrapped_kwargs) == 0
        assert len(unwrapped_args) == 2, f"args: {unwrapped_args}"
        dim = unwrapped_args[0].dim()
        return wrap(func(unwrapped_args[0],
                         ([0] + [wrap_dim(u, dim - 1) + 1 for u in unwrapped_args[1]])))

    if func == torch.ops.aten._scaled_dot_product_efficient_attention.default:
        assert len(args) == 5
        if all(isinstance(a, MapTensor) for a in args[:3]):
            assert len(unwrapped_kwargs) == 0
            assert len(unwrapped_args) == 5, f"args: {unwrapped_args}"
            assert unwrapped_args[0].dim() == 5
            assert unwrapped_args[1].dim() == 5
            assert unwrapped_args[2].dim() == 5
            sdpa_res = wrap(func(unwrapped_args[0].flatten(0, 1),
                                 unwrapped_args[1].flatten(0, 1),
                                 unwrapped_args[2].flatten(0, 1),
                                 unwrapped_args[3],
                                 unwrapped_args[4]))
            return (wrap(sdpa_res[0].view(unwrapped_args[0].size())),) + sdpa_res[1:]
        if isinstance(args[0], MapTensor) and not any(isinstance(a, MapTensor) for a in args[1:]):
            assert len(unwrapped_kwargs) == 0
            assert len(unwrapped_args) == 5, f"args: {unwrapped_args}"
            assert unwrapped_args[0].dim() == 5
            assert unwrapped_args[1].dim() == 4
            assert unwrapped_args[2].dim() == 4
            a0 = unwrapped_args[0]
            a1_size = unwrapped_args[1].size()
            a1 = unwrapped_args[1].unsqueeze(0).expand((a0.size(0),) + a1_size)
            a2 = unwrapped_args[2].unsqueeze(0).expand((a0.size(0),) + a1_size)
            sdpa_res = wrap(func(a0.flatten(0, 1),
                                 a1.flatten(0, 1),
                                 a2.flatten(0, 1),
                                 unwrapped_args[3],
                                 unwrapped_args[4]))
            return (wrap(sdpa_res[0].view(unwrapped_args[0].size())),) + sdpa_res[1:]
        if ((not isinstance(args[0], MapTensor)) and isinstance(args[1], MapTensor) and (not isinstance(args[2], MapTensor))):
            assert len(unwrapped_kwargs) == 0
            assert len(unwrapped_args) == 5, f"args: {unwrapped_args}"
            assert unwrapped_args[0].dim() == 4
            assert unwrapped_args[1].dim() == 5
            assert unwrapped_args[2].dim() == 4
            a1_size = unwrapped_args[1].size()
            a0 = unwrapped_args[0].unsqueeze(0).expand((a1_size[0],) + unwrapped_args[0].size()[1:])
            a2 = unwrapped_args[2].unsqueeze(0).expand((a1_size[0],) + unwrapped_args[2].size()[1:])
            sdpa_res = wrap(func(a0.flatten(0, 1),
                                 a1.flatten(0, 1),
                                 a2.flatten(0, 1),
                                 unwrapped_args[3],
                                 unwrapped_args[4]))
            return (wrap(sdpa_res[0].view(unwrapped_args[0].size())),) + sdpa_res[1:]
        if ((not isinstance(args[0], MapTensor)) and isinstance(args[1], MapTensor) and isinstance(args[2], MapTensor)):
            assert len(unwrapped_kwargs) == 0
            assert len(unwrapped_args) == 5, f"args: {unwrapped_args}"
            assert unwrapped_args[0].dim() == 4
            assert unwrapped_args[1].dim() == 5
            assert unwrapped_args[2].dim() == 5
            a0_size = unwrapped_args[0].size()
            a1_size = unwrapped_args[1].size()
            a0 = unwrapped_args[0].unsqueeze(0).expand((a1_size[0],) + a0_size)
            a1 = unwrapped_args[1]
            a2 = unwrapped_args[2]
            sdpa_res = wrap(func(a0.flatten(0, 1),
                                 a1.flatten(0, 1),
                                 a2.flatten(0, 1),
                                 unwrapped_args[3],
                                 unwrapped_args[4]))
            return (wrap(sdpa_res[0].view((a1_size[0],) + a0_size)),) + sdpa_res[1:]
        return NotImplemented

    if func == torch.ops.aten._scaled_dot_product_flash_attention.default:
        assert len(args) == 3
        assert len(unwrapped_kwargs) == 1
        assert len(unwrapped_args) == 3, f"args: {unwrapped_args}"
        if all(isinstance(a, MapTensor) for a in args[:3]):
            assert unwrapped_args[0].dim() == 5
            assert unwrapped_args[1].dim() == 5
            assert unwrapped_args[2].dim() == 5
            sdpa_res = wrap(func(unwrapped_args[0].flatten(0, 1),
                                 unwrapped_args[1].flatten(0, 1),
                                 unwrapped_args[2].flatten(0, 1),
                                 **unwrapped_kwargs))
            return (wrap(sdpa_res[0].view(unwrapped_args[0].size())),) + sdpa_res[1:]
        if isinstance(args[0], MapTensor) and not any(isinstance(a, MapTensor) for a in args[1:]):
            assert unwrapped_args[0].dim() == 5
            assert unwrapped_args[1].dim() == 4
            assert unwrapped_args[2].dim() == 4
            a0 = unwrapped_args[0]
            a1_size = unwrapped_args[1].size()
            a1 = unwrapped_args[1].unsqueeze(0).expand((a0.size(0),) + a1_size)
            a2 = unwrapped_args[2].unsqueeze(0).expand((a0.size(0),) + a1_size)
            sdpa_res = wrap(func(a0.flatten(0, 1),
                                 a1.flatten(0, 1),
                                 a2.flatten(0, 1),
                                 **unwrapped_kwargs))
            return (wrap(sdpa_res[0].view(unwrapped_args[0].size())),) + sdpa_res[1:]
        if ((not isinstance(args[0], MapTensor)) and isinstance(args[1], MapTensor) and (not isinstance(args[2], MapTensor))):
            assert unwrapped_args[0].dim() == 4
            assert unwrapped_args[1].dim() == 5
            assert unwrapped_args[2].dim() == 4
            a1_size = unwrapped_args[1].size()
            a0 = unwrapped_args[0].unsqueeze(0).expand((a1_size[0],) + unwrapped_args[0].size()[1:])
            a2 = unwrapped_args[2].unsqueeze(0).expand((a1_size[0],) + unwrapped_args[2].size()[1:])
            sdpa_res = wrap(func(a0.flatten(0, 1),
                                 a1.flatten(0, 1),
                                 a2.flatten(0, 1),
                                 **unwrapped_kwargs))
            return (wrap(sdpa_res[0].view(unwrapped_args[0].size())),) + sdpa_res[1:]
        if ((not isinstance(args[0], MapTensor)) and isinstance(args[1], MapTensor) and isinstance(args[2], MapTensor)):
            assert unwrapped_args[0].dim() == 4
            assert unwrapped_args[1].dim() == 5
            assert unwrapped_args[2].dim() == 5
            a0_size = unwrapped_args[0].size()
            a1_size = unwrapped_args[1].size()
            a0 = unwrapped_args[0].unsqueeze(0).expand((a1_size[0],) + a0_size)
            a1 = unwrapped_args[1]
            a2 = unwrapped_args[2]
            sdpa_res = wrap(func(a0.flatten(0, 1),
                                 a1.flatten(0, 1),
                                 a2.flatten(0, 1),
                                 **unwrapped_kwargs))
            return (wrap(sdpa_res[0].view((a1_size[0],) + a0_size)),) + sdpa_res[1:]
        return NotImplemented

    # torch.ops.aten._unsafe_index.Tensor is only needed by inductor for compile
    if func in [torch.ops.aten._unsafe_index.Tensor, torch.ops.aten.index.Tensor]:
        assert len(unwrapped_kwargs) == 0
        assert len(unwrapped_args) == 2, f"args: {unwrapped_args}"
        # if len(args[1]) == 1 and isinstance(args[1][0], MapTensor) and isinstance(args[0], MapTensor):
        #     return wrap(func(*unwrapped_args))
        if len(args[1]) == 1 and isinstance(args[1][0], MapTensor) and not isinstance(args[0], MapTensor):
            tensors = [func(args[0], [args[1][0].elems[i]]) for i in range(len(args[1][0].elems))]
            values = torch.cat(tensors)
            lengths = torch.tensor([0] + [t.size(0) for t in tensors], pin_memory=True).to(values.device, non_blocking=True)
            offsets = torch.cumsum(lengths, dim=0)
            nt = nested_view_from_values_offsets(values, offsets)
            assert nt.layout == torch.jagged
            return wrap(nt)
        if isinstance(args[0], MapTensor) and not isinstance(args[1][0], MapTensor) and len(args[1]) == 1:
            return wrap(func(args[0].elems, [args[1][0].unsqueeze(0)]))
        if isinstance(args[0], MapTensor) and isinstance(args[1][0], MapTensor) and len(args[1]) == 1:
            tensors = [func(args[0].elems[i], [args[1][0].elems[i]]) for i in range(len(args[0].elems))]
            values = torch.cat(tensors)
            lengths = torch.tensor([0] + [t.size(0) for t in tensors], pin_memory=True).to(values.device, non_blocking=True)
            offsets = torch.cumsum(lengths, dim=0)
            nt = nested_view_from_values_offsets(values, offsets)
            assert nt.layout == torch.jagged
            return wrap(nt)
        a = unwrapped_args[0]
        a = unwrapped_args[0].flatten(0, 1)
        resa = func(a, args[1])
        resb = resa.view((unwrapped_args[0].size(0), unwrapped_args[0].size(1)) + resa.size()[1:])
        return wrap(resb)

    # Prims
    if func == torch.ops.aten.dim.default:
        assert len(args) == 1
        assert len(kwargs) == 0
        ret_dim = func(args[0].elems) - 1
        assert ret_dim >= 0
        return ret_dim

    if func == torch.ops.aten.sym_size.default:
        assert len(args) == 1
        assert len(kwargs) == 0
        elems_size = func(args[0].elems)
        assert len(elems_size) > 0
        return elems_size[1:]

    if func == torch.ops.aten.is_contiguous.default:
        assert len(args) == 1
        assert len(kwargs) == 0
        return func(args[0].elems)


    forwardables = [
                       torch.ops.aten.add.Tensor,
                       torch.ops.aten.clamp.default,
                       torch.ops.aten.clone.default,
                       torch.ops.aten.copy_.default,
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
                       # Sketchy new in place ops
                       torch.ops.aten.bitwise_and_.Tensor,
                       torch.ops.aten.bitwise_or_.Tensor,
                       torch.ops.aten.le.Tensor,
                       torch.ops.aten.logical_and.default,
                       # Prims
                       torch.ops.prim.layout.default,
                   ]
    if func in forwardables:
        return wrap(func(*unwrapped_args, **unwrapped_kwargs))
    print(f"WARNING! Not officially marked as forwardable: torch.ops.{func}")
    import pdb; pdb.set_trace()
    return wrap(func(*unwrapped_args, **unwrapped_kwargs))

class MapTensor(torch.Tensor):
    @staticmethod
    def __new__(cls, elems):
        # print("elems.layout: ", elems.layout)
        if elems.is_nested:
            new_shape = (2,) * (elems.dim() - 1)
        else:
            new_shape = elems.shape[1:]
        return torch.Tensor._make_wrapper_subclass(cls,
                                                   new_shape,
                                                   dtype=elems.dtype,
                                                   device=elems.device,
                                                   layout=elems.layout,
                                                   dispatch_layout=True,
                                                   dispatch_sizes_strides_policy=("sizes" if elems.layout == torch.jagged else None),
                                                   storage_size=(elems._values.untyped_storage().size() if elems.layout == torch.jagged else None))

    def __init__(self, elems):
        self.elems = elems

    @classmethod
    def __torch_dispatch__(cls, func, types, args, kwargs=None):
        # print("func: ", func)
        # print("func: ", func, "args: ", [type(a.elems) if isinstance(a, MapTensor) else None for a in args])
        # if func == torch.ops.aten.gt.Scalar:
        #     import pdb; pdb.set_trace()
        # print("func: ", func, "args: ", [a.size() if isinstance(a, torch.Tensor) else a for a in args])
        return ops_impl(cls, func, types, args, kwargs)
        res = ops_impl(cls, func, types, args, kwargs)
        if isinstance(res, torch.Tensor):
            unwrapped_args_0 = tree_map(lambda x: unwrap_i(x, 0), args)
            unwrapped_kwargs_0 = tree_map(lambda x: unwrap_i(x, 0), kwargs)
            if func == torch.ops.aten.view.default:
                res_0 = torch.ops.aten.reshape.default(*unwrapped_args_0, **unwrapped_kwargs_0)
            else:
                res_0 = func(*unwrapped_args_0, **unwrapped_kwargs_0)
            if res.elems[0].size() != res_0.size():
                import pdb; pdb.set_trace()
                print("02390")
            if not torch.allclose(res.elems[0], res_0, atol=1e-3, rtol=1e-3):
                import pdb; pdb.set_trace()
                print("SDJFKL")
        else:
            pass
            print("res got type: ", type(res))
        return res

    # __torch_function__ = torch._C._disabled_torch_function_impl

    @classmethod
    def __torch_function__(cls, func, types, args=(), kwargs=None):
        if kwargs is None:
            kwargs = {}
        # print("TF func: ", func)
        if torch._C.TensorBase.flatten == func:
            # import pdb; pdb.set_trace()
            pass
        if 'interpolate' in str(func):
            ret = []
            for i, _ in enumerate(kwargs['size']):
                new_kwargs = dict(kwargs)
                new_kwargs['size'] = new_kwargs['size'][i]
                ret.append(func(args[0].elems[i], **new_kwargs))
            # Requires jagged 2d because images
            return wrap(torch.nested.nested_tensor(ret))
        with torch._C.DisableTorchFunctionSubclass():
            return func(*args, **kwargs)

    # flatten/unflatten is needed for compile
    def __tensor_flatten__(self):
        ctx = {}
        inner_tensors = ["elems"]
        return inner_tensors, ctx

    @staticmethod
    def __tensor_unflatten__(inner_tensors: Dict, meta, outer_size, outer_stride):
        from torch._subclasses.fake_tensor import FakeTensor

        # inner tensors: _values, _offsets, [_lengths], [_min_seqlen], [_max_seqlen]
        assert len(inner_tensors) == 1, f"{inner_tensors}"
        elems = inner_tensors["elems"]

        return MapTensor(elems)

    def __repr__(self):
        return f"MapTensor({self.elems.size()})"

# ts is a higher dim Tensor
def to_map_tensor(ts: torch.Tensor):
    return MapTensor(ts)
