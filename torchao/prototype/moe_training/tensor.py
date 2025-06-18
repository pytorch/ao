import torch

from torchao.prototype.moe_training import _scaled_grouped_mm


class ScaledGroupedMMTensor(torch.Tensor):
    """
    ScaledGroupedMMTensor is a simple tensor subclass that wraps a regular tensor
    and overrides the torch._grouped_mm op by dispatching to the
    differentiable _scaled_grouped_mm autograd function.
    """

    grouped_mm_func_name = "_grouped_mm"
    offs_arg_name = "offs"
    use_triton_for_per_group_scales = True

    def __init__(
        self, data: torch.Tensor, use_triton_for_per_group_scales: bool = True
    ):
        self._data = data
        self._use_triton_for_per_group_scales = use_triton_for_per_group_scales

    def __repr__(self):
        return f"ScaledGroupedMMTensor(use_triton_for_per_group_scales={self._use_triton_for_per_group_scales}, {self._data})"

    @classmethod
    def __torch_function__(cls, func, types, args, kwargs={}):
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
                use_triton = (
                    A._use_triton_for_per_group_scales
                    if isinstance(A, cls)
                    else B._use_triton_for_per_group_scales
                )
                return _scaled_grouped_mm(
                    *args,
                    use_triton_for_per_group_scales=use_triton,
                    **kwargs,
                )
        return super().__torch_function__(func, types, args, kwargs)
