import torch
from torch.utils._pytree import tree_map

from torchao.prototype.moe_training import _scaled_grouped_mm


class ScaledGroupedMMTensor(torch.Tensor):
    """
    ScaledGroupedMMTensor is a simple tensor subclass that wraps a regular tensor
    and overrides the torch._grouped_mm op by dispatching to the
    differentiable _scaled_grouped_mm autograd function.
    """

    grouped_mm_func_name = "_grouped_mm"
    offs_arg_name = "offs"

    def __init__(self, data: torch.Tensor):
        self._data = data

    def __repr__(self):
        return f"ScaledGroupedMMTensor(data={self._data})"

    @classmethod
    def __torch_function__(cls, func, types, args, kwargs={}):
        print(func.__name__)
        if func.__name__ == cls.grouped_mm_func_name:
            # Use torchao scaled grouped mm with dynamic quant for
            # "2d x 3d with offsets" case (used for routed experts).
            # Otherwise, fall back to regular grouped mm.
            #
            # TODO: support "3d x 3d without offsets" case, which is
            # used for shared experts. This is basically the grouped_mm
            # kernel handling a bmm.
            A, B = args[0], args[1]
            A_is_2d = A.dim() == 2
            B_is_3d = B.dim() == 3
            has_offs = kwargs.get(cls.offs_arg_name) is not None
            if A_is_2d and B_is_3d and has_offs:
                return _scaled_grouped_mm(*args, **kwargs)

        return super().__torch_function__(func, types, args, kwargs)

    @classmethod
    def __torch_dispatch__(cls, func, types, args, kwargs={}):
        unwrap = lambda x: x._data if isinstance(x, cls) else x
        wrap = lambda x: cls(x) if isinstance(x, torch.Tensor) else x
        unwrapped_args, unwrapped_kwargs = tree_map(unwrap, (args, kwargs))
        output = super().__torch_dispatch__(func, types, unwrapped_args, unwrapped_kwargs)
        wrapped_output = tree_map(wrap, output)
        return wrapped_output
