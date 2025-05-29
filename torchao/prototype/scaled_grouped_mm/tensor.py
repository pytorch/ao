import torch
import torch.utils._pytree as pytree

from torchao.prototype.scaled_grouped_mm import _scaled_grouped_mm

aten = torch.ops.aten

c10d_functional = torch.ops.c10d_functional
_c10d_functional = torch.ops._c10d_functional
OP_OVERRIDES_TABLE = {}

# FSDP pads its local tensor on dim-0. The subclass should be preserved such
# that the padded local tensor (and any transformations like copying to GPU)
# is of the subclass as well.
_ops_to_preserve_subclass = {
    # fsdp ops
    torch.ops.aten.empty_like.default,
    torch.ops.aten.new_zeros.default,
    torch.ops.aten.slice.Tensor,
    torch.ops.aten.copy_.default,
    torch.ops.aten.view.default,
    torch.ops.aten.as_strided.default,
    torch.ops.aten._to_copy.default,
    torch.ops.aten._pin_memory.default,
    torch.ops.aten.split.Tensor,
    torch.ops.aten.clone.default,

    # other ops
    aten.detach.default,
    aten.t.default,
    aten.transpose.int,
}

def implements(aten_ops):
    """Register aten ops to the op override table"""
    def decorator(func):
        for op in aten_ops:
            OP_OVERRIDES_TABLE[op] = func
        return func
    return decorator


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



@implements([aten.split.Tensor])
def _split(aten_op, args, kwargs=None):
    new_data_tensors = aten_op(args[0]._data, *args[1:], **kwargs)

    def make_subclass(data):
        return ScaledGroupedMMTensor(
            data,
        )

    out = map(make_subclass, new_data_tensors)
    return list(out)


@implements([aten.cat.default])
def _cat(aten_op, args, kwargs=None):
    chunked_tensors: tuple[ScaledGroupedMMTensor] = args[0]
    chunk_data = []
    for chunk in chunked_tensors:
        assert isinstance(chunk, ScaledGroupedMMTensor), (
            "Expecting all chunks to be of type ScaledGroupedMMTensor"
        )
        chunk_data.append(chunk._data)
    new_data = aten_op(chunk_data, *args[1:], **kwargs)
    return ScaledGroupedMMTensor(new_data)


@implements(
    [
        c10d_functional.all_gather_into_tensor.default,
        _c10d_functional.all_gather_into_tensor.default,
    ]
)
def _allgather(aten_op, args, kwargs=None):
    input = args[0]
    assert isinstance(input, ScaledGroupedMMTensor), (
        f"expecting a ScaledGroupedMMTensor for allgather but found {type(input)}"
    )

    data = input._data
    data = data.contiguous()
    out = aten_op(data, *args[1:], **kwargs)
    return ScaledGroupedMMTensor(out)


@implements([c10d_functional.wait_tensor.default, _c10d_functional.wait_tensor.default])
def _wait_tensor(aten_op, args, kwargs=None):
    input = args[0]
    assert isinstance(input, ScaledGroupedMMTensor)
    out = aten_op(input._data, *args[1:], **kwargs)
    return ScaledGroupedMMTensor(out)

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
            print(f"{func}: calling override")
            return OP_OVERRIDES_TABLE[func](func, args, kwargs)

        # return pytree.tree_unflatten(out_a_flat, spec)
        unwrap = lambda x: x._data if isinstance(x, ScaledGroupedMMTensor) else x
        args, kwargs = pytree.tree_map_only(
            ScaledGroupedMMTensor, unwrap, (args, kwargs or {})
        )
        out = func(*args, **kwargs)
        if func not in _ops_to_preserve_subclass:
            print(f"{func}: not preserving subclass")
            return out
        print(f"{func}: preserving subclass")
        return pytree.tree_map_only(
            torch.Tensor,
            lambda x: hp_to_scaled_grouped_mm_tensor(x),
            out,
        )

    # Do not force the ScaledGroupedMMTensor type on the returned tensor
    __torch_function__ = torch._C._disabled_torch_function_impl


@torch._dynamo.allow_in_graph
class _ConvertFunc(torch.autograd.Function):
    """
    A differentiable conversion to ScaledGroupedMMTensor.
    """

    @staticmethod
    def forward(ctx, x: torch.Tensor) -> torch.Tensor:
        return ScaledGroupedMMTensor(x)

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor) -> torch.Tensor:
        return grad_output
        
def hp_to_scaled_grouped_mm_tensor(hp_tensor: torch.Tensor) -> ScaledGroupedMMTensor:
    return _ConvertFunc.apply(hp_tensor)
