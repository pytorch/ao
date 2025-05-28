import torch
import torch.utils._pytree as pytree

from torchao.prototype.scaled_grouped_mm import _scaled_grouped_mm

aten = torch.ops.aten

OP_OVERRIDES_TABLE = {}


def implements(aten_ops):
    """Register aten ops to the op override table"""
    def decorator(func):
        for op in aten_ops:
            OP_OVERRIDES_TABLE[op] = func
        return func
    return decorator


@implements([aten.detach.default, aten.empty_like.default])
def _desugar(aten_op, args, kwargs=None):
    new_data = aten_op(args[0]._data, *args[1:], **kwargs)
    return ScaledGroupedMMTensor(new_data)


@implements([aten._grouped_mm.default])
def _grouped_mm(aten_op, args, kwargs=None):
    a, b, offs = args
    assert not isinstance(a, ScaledGroupedMMTensor), (
        "expected activations tensor 'a' to be a regular tensor"
    )
    assert isinstance(b, ScaledGroupedMMTensor), (
        "expected weights tensor 'b' to be a ScaledGroupedMMTensor"
    )
    return _scaled_grouped_mm(a, b._data, offs=offs)


class ScaledGroupedMMTensor(torch.Tensor):
    """
    ScaledGroupedMMTensor is a simple tensor subclass that wraps a regular tensor
    and overrides the torch._grouped_mm op by dispatching to the
    differentiable _scaled_grouped_mm autograd function.
    """

    _data: torch.Tensor

    def __new__(
        cls,
        data: torch.Tensor,
    ):
        self = torch.Tensor._make_wrapper_subclass(
            cls,
            data.size(),
            strides=data.stride(),
            storage_offset=data.storage_offset(),
            dtype=data.dtype,
            layout=data.layout,
            requires_grad=data.requires_grad,
            device=data.device,
        )
        self._data = data
        return self

    def __repr__(self):
        return f"ScaledGroupedMMTensor(data={self._data})"

    def __tensor_flatten__(self):
        ctx = {}
        return ["_data"], ctx

    @staticmethod
    def __tensor_unflatten__(inner_tensors: dict, metadata, outer_size, outer_stride):
        assert len(inner_tensors) == 1
        return ScaledGroupedMMTensor(
            inner_tensors["_data"],
        )

    @classmethod
    def __torch_dispatch__(cls, func, types, args, kwargs={}):
        # Don't support mixed tensor subclasses. This will trigger the handler for
        # the next type in the dispatch list
        def allowed_subclasses(type):
            return (
                issubclass(cls, type)
                or issubclass(torch._subclasses.fake_tensor.FakeTensor, type)
                or issubclass(
                    torch._subclasses.functional_tensor.FunctionalTensor, type
                )
            )

        if not all(allowed_subclasses(t) for t in types):
            return NotImplemented

        if func in OP_OVERRIDES_TABLE:
            return OP_OVERRIDES_TABLE[func](func, args, kwargs)

        # fallback to regular tensor behavior for all other ops, returning a regular tensor as well.
        args_a = pytree.tree_map_only(ScaledGroupedMMTensor, lambda x: x._data, args)
        kwargs_a = pytree.tree_map_only(
            ScaledGroupedMMTensor, lambda x: x._data, kwargs
        )
        out_a = func(*args_a, **kwargs_a)
        out_a_flat, spec = pytree.tree_flatten(out_a)
        return pytree.tree_unflatten(out_a_flat, spec)

    # Do not force the ScaledGroupedMMTensor type on the returned tensor
    __torch_function__ = torch._C._disabled_torch_function_impl
