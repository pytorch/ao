"""
Following is a example for a simple dtype implemented with tensor subclass
it shows
    * the basic structure of a new dtype tensor subclass (__new__, __init__, __tensor_flatten__, __tensor_unflatten__)
    * two types of dispatch that people can overwrite (__torch_function__, __torch_dispatch__)
    * how to abstract away packing format with layout
    * how the tensor subclass composes with torch.compile to get speedup
"""


import functools
from collections import defaultdict
from typing import Any, Callable, Dict, Optional, Tuple

import torch

from torch.utils._python_dispatch import return_and_correct_aliasing
from torchao.quantization.quant_primitives import (
    choose_qparams_affine,
    MappingType,
    quantize_affine,
    dequantize_affine,
)
from torchao.dtypes.utils import (
    Layout,
    PlainLayout,
)
from torchao.utils import (
    TorchAOBaseTensor,
    fill_defaults,
)

aten = torch.ops.aten

###############################
# Base Tensor Impl Subclass #
###############################
class MyDTypeTensorImpl(torch.Tensor):
    """
    Base class for the tensor impl for `MyDTypeTensor`
    """
    # get the original unpacked Tensors
    def get_plain(self) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.int_data, self.scale

    def get_layout(self) -> Layout:
        return self._layout

    @classmethod
    def from_plain(
        cls,
        int_data: torch.Tensor,
        scale: torch.Tensor,
        _layout: Layout,
    ):
        """Construct a tensor impl from plain tensors and a _layout, which main contain
        extra metadata for packing etc.
        """
        pass

    def __repr__(self):
        int_data, scale = self.get_plain()
        _layout = self.get_layout()
        return f"{self.__class__.__name__}(int_data={int_data}, scale={scale}, _layout={_layout})"

    __torch_function__ = torch._C._disabled_torch_function_impl

##############################
# Tensor Subclass Definition #
##############################

class MyDTypeTensor(TorchAOBaseTensor):
    """Inheriting from `TorchAOBaseTensor` gives us some helper functions, please see docs
    for :class:`~torchao.utils.TorchAOBaseTensor` for more details
    """

    """We need to define __new__ for constructing a new tensor subclass instance and __init__ for initialize
    the instance. There is no requirement on what the argument list should look like here, only requirement is
    that `__new__` must return a Tensor instance with `torch.Tensor._make_wrapper_subclass(cls, shape, ...)` call
    """

    @staticmethod
    def __new__(
        cls,
        tensor_impl: MyDTypeTensorImpl,
        shape: torch.Size,
        dtype: Optional[torch.dtype] = None,
        requires_grad: bool = False,
    ):
        kwargs = {}
        kwargs["device"] = tensor_impl.device
        kwargs["layout"] = (
            kwargs.get("layout")
            if kwargs.get("layout", False)
            else tensor_impl.layout
        )
        kwargs["dtype"] = dtype
        kwargs["requires_grad"] = requires_grad
        return torch.Tensor._make_wrapper_subclass(cls, shape, **kwargs)  # type: ignore[attr-defined]

    def __init__(
        self,
        tensor_impl: MyDTypeTensorImpl,
        shape: torch.Size,
        dtype: Optional[torch.dtype] = None,
        requires_grad: bool = False,
    ):
        self.tensor_impl = tensor_impl

    """__tensor_flatten__ and __tensor_unflatten__ are used to desugar the tensor into native Tensors/attributes and
    reconstruct the tensor subclass instance from the desugared tensor and attributes, these are required to define
    a Tensor subclass for torch.compile support
    """

    def __tensor_flatten__(self):
        """
        Given the class, returns the fields of the class as two lists
        The first one contains any tensor fields such as int_data and scale as keys to a dictionary
        The second one contains all other non tensor type fields as values of a list
        """
        return ["tensor_impl"], [self.shape, self.dtype, self.requires_grad]

    @classmethod
    def __tensor_unflatten__(
        cls, tensor_data_dict, tensor_attributes, outer_size, outer_stride
    ):
        """
        Given the flattened data from above, returns a class instance
        tensor_data_dict contains the tensor fields of the class as a dictionary
        tensor_attributes contains all other non tensor type fields
        """
        tensor_impl = tensor_data_dict["tensor_impl"]
        shape, dtype, requires_grad = tensor_attributes
        return cls(
            tensor_impl,
            shape if outer_size is None else outer_size,
            dtype=dtype,
            requires_grad=requires_grad,
        )

    """classmethod that converts from a floating point Tensor (fp32/fp16/bf16) to the current dtype
    """

    @classmethod
    def from_float(
        cls,
        input_float: torch.Tensor,
        _layout: Layout = PlainLayout(),
    ):
        mapping_type = MappingType.SYMMETRIC
        block_size = (1, input_float.shape[-1])
        dtype = torch.int16
        scale, zero_point = choose_qparams_affine(input_float, mapping_type, block_size, dtype)
        int_data = quantize_affine(input_float, block_size, scale, zero_point, dtype)
        tensor_impl_ctr = get_tensor_impl_constructor(type(_layout))
        tensor_impl = tensor_impl_ctr(int_data, scale, _layout)
        return cls(tensor_impl, input_float.shape)

    """[Optional] We can overwrite layout property of the Tensor to represent different packing formats
    """

    @property
    def _layout(self) -> Layout:
        return self.tensor_impl._layout

    def dequantize(self, output_dtype=None):
        """We can define a dequantize method to convert the quantized tensor to a floating point tensor"""
        if output_dtype is None:
            output_dtype = torch.get_default_dtype()
        int_data, scale = self.tensor_impl.get_plain()
        transposed = False
        block_size = (1, int_data.shape[-1])
        if hasattr(self.tensor_impl, "transposed") and self.tensor_impl.transposed:
            transposed = True
        res = dequantize_affine(int_data, block_size, scale, None, int_data.dtype, output_dtype=output_dtype)
        if transposed:
            res = res.t()
        return res

    def __repr__(self):
        return (
            f"{self.__class__.__name__}(data={self.dequantize()}, shape={self.shape}, "
            f"device={self.device}, dtype={self.dtype}, requires_grad={self.requires_grad})"
        )

    def _apply_fn_to_data(self, fn):
        """
        Used for implementing aten ops by applying them only to the relevant tensor atributes
        In this case we only want to call things like to() or view() on the tensor impl
        """
        return self.__class__(
            fn(self.tensor_impl),
            self.shape,
            self.dtype,
        )

    """There are two entry points that we can modify the behavior of a pytorch op: torch_function and torch_dispatch:

    __torch_function__: will be called whenever a torch level function is called on the Tensor object, for example: torch.nn.functional.linear,
    tensor.detach, tensor.reshape, tensor.t etc.

    __torch_dispatch__: will be called in the C++ dispatcher, when an aten operator is called on the Tensor object, for example:
    aten.mm, aten.addmm, aten.detach.default, aten.t.default etc.

    We have some helper functions that can dispatch to the functions registered with MyDTypeTensor.implements, but if the default implementation does not work for your use case, please feel free to customize it
    """

######################################################
# Layout and TensorImpl Subclass Registration #
######################################################

register_layout = MyDTypeTensor.register_layout
get_tensor_impl_constructor = MyDTypeTensor.get_tensor_impl_constructor

@register_layout(PlainLayout)
class PlainMyDTypeTensorImpl(MyDTypeTensorImpl):
    def __new__(
        cls,
        int_data: torch.Tensor,
        scale: torch.Tensor,
        transposed: bool,
        _layout: Layout,
    ):
        kwargs = {}
        kwargs["device"] = int_data.device
        kwargs["layout"] = (
            kwargs.get("layout") if kwargs.get("layout", False) else int_data.layout
        )
        kwargs["dtype"] = int_data.dtype
        kwargs["requires_grad"] = False
        shape = int_data.shape
        return torch.Tensor._make_wrapper_subclass(cls, shape, **kwargs)  # type: ignore[attr-defined]

    def __init__(
        self,
        int_data: torch.Tensor,
        scale: torch.Tensor,
        transposed: bool,
        _layout: Layout,
    ):
        self.int_data = int_data
        self.scale = scale
        self.transposed = transposed
        self._layout = _layout

    def __tensor_flatten__(self):
        return ["int_data", "scale"], [self.transposed, self._layout]

    @classmethod
    def __tensor_unflatten__(
        cls, tensor_data_dict, tensor_attributes, outer_size, outer_stride
    ):
        int_data, scale = tensor_data_dict["int_data"], tensor_data_dict["scale"]
        transposed, _layout, = tensor_attributes
        return cls(int_data, scale, transposed, _layout)

    @classmethod
    def from_plain(
        cls,
        int_data: torch.Tensor,
        scale: torch.Tensor,
        _layout: Layout,
    ):
        """Construct a tensor impl from plain tensors and a _layout, which main contain
        extra metadata for packing etc.
        """
        assert isinstance(_layout, PlainLayout)
        return cls(int_data, scale, False, _layout)

    def _apply_fn_to_data(self, fn):
        return self.__class__(
            fn(self.int_data),
            fn(self.scale),
            self.transposed,
            self._layout,
        )

    @classmethod
    def __torch_dispatch__(cls, func, types, args, kwargs):
        kwargs = {} if kwargs is None else kwargs

        if func is aten.detach.default:
            return return_and_correct_aliasing(
                func, args, kwargs, args[0]._apply_fn_to_data(torch.detach)
            )

        # Tensor parallel support START
        elif func in [aten._to_copy.default, aten.clone.default]:
            return return_and_correct_aliasing(
                func, args, kwargs, args[0]._apply_fn_to_data(torch.clone)
            )
        elif func is aten.split.Tensor:
            int_data_list = func(args[0].int_data, *args[1:], **kwargs)
            scale_list = func(args[0].scale, *args[1:], **kwargs)
            out = [PlainMyDTypeTensorImpl(int_data, scale, args[0].transposed, args[0]._layout) for int_data, scale in zip(int_data_list, scale_list)]
            return out
        elif func is aten.empty_like.default:
            int_data_empty_like = func(args[0].int_data, *args[1:], **kwargs)
            return PlainMyDTypeTensorImpl(int_data_empty_like, args[0].scale, args[0].transposed, args[0]._layout)
        elif func is aten.slice.Tensor:
            self, dim, start, end, step = fill_defaults(args, 5, [0, None, None, 1])
            if dim == 0:
                return return_and_correct_aliasing(
                    func, args, kwargs, args[0]._apply_fn_to_data(lambda x: aten.slice.Tensor(x, dim, start, end, step))
                )
            elif dim == 1:
                return PlainMyDTypeTensorImpl(aten.slice.Tensor(self.int_data, dim, start, end, step), self.scale.view(-1), self.transposed, self._layout)
            else:
                raise NotImplementedError(f"PlainMyDTypeTensorImpl dispatch: attempting to run {func}, with dim={dim}, that is not supported")
        elif func is aten.t.default:
            return return_and_correct_aliasing(func, args, kwargs, PlainMyDTypeTensorImpl(args[0].int_data, args[0].scale, not args[0].transposed, args[0]._layout))

        # Tensor parallel support END

        raise NotImplementedError(
            f"PlainMyDTypeTensorImpl dispatch: attempting to run {func}, this is not supported"
        )

#####################################################
# torch functional and aten operator implementation #
#####################################################

implements = MyDTypeTensor.implements

def _quantized_linear_op(input_tensor, weight_tensor, bias):
    if isinstance(input_tensor, MyDTypeTensor):
        input_tensor = input_tensor.dequantize()
    if isinstance(weight_tensor, MyDTypeTensor):
        weight_tensor = weight_tensor.dequantize()
    return torch.nn.functional.linear(input_tensor, weight_tensor, bias)


@implements(torch.nn.functional.linear)
def _(func, types, args, kwargs):
    input_tensor, weight_tensor, bias = (
        args[0],
        args[1],
        args[2] if len(args) > 2 else None,
    )
    # using try/except here so that we can have a general fallback when input_tensor/weight_tensor
    # is not picked up by any of the dispatch paths in `_quantized_linear_op`, this allows us to
    # make the branches easier to understand in `_quantized_linear_op`
    try:
        return _quantized_linear_op(input_tensor, weight_tensor, bias)
    except NotImplementedError:
        if isinstance(input_tensor, MyDTypeTensor):
            input_tensor = input_tensor.dequantize()
        if isinstance(weight_tensor, MyDTypeTensor):
            weight_tensor = weight_tensor.dequantize()
        return torch.nn.functional.linear(input_tensor, weight_tensor, bias)


@implements(aten.detach.default)
def _(func, types, args, kwargs):
    # `return_and_correct_aliasing` should be used by wrapper tensor ``__torch_dispatch__`` subclasses that would like to
    # work with torch.compile. It ensures that the subclass properly implements the aliasing behavior of every op,
    # which is needed for correctness in AOTAutograd.

    # `_apply_fn_to_data` just applies the function to the tensor data in `args[0]`, `args[0]` is a tensor subclass
    # of `my_dtype`
    return return_and_correct_aliasing(
        func, args, kwargs, args[0]._apply_fn_to_data(torch.detach)
    )

#####################
# Factory functions #
#####################
to_my_dtype = MyDTypeTensor.from_float


########
# Test #
########
def main():
    from torchao.utils import benchmark_model

    class M(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.linear = torch.nn.Linear(1024, 128)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return self.linear(x)

    m = M()
    example_inputs = (100 * torch.randn(512, 1024),)
    NUM_WARMUPS = 10
    NUM_RUNS = 100

    for _ in range(NUM_WARMUPS):
        m(*example_inputs)
    print("before quantization:", benchmark_model(m, NUM_RUNS, example_inputs))

    compiled = torch.compile(m, mode="max-autotune")
    for _ in range(NUM_WARMUPS):
        compiled(*example_inputs)
    print("after compile:", benchmark_model(compiled, NUM_RUNS, example_inputs))

    # convert weights to quantized weights
    m.linear.weight = torch.nn.Parameter(
        to_my_dtype(m.linear.weight), requires_grad=False
    )

    for _ in range(NUM_WARMUPS):
        m(*example_inputs)

    print("after quantization:", benchmark_model(m, NUM_RUNS, example_inputs))

    m = torch.compile(m, mode="max-autotune")

    for _ in range(NUM_WARMUPS):
        m(*example_inputs)

    # NOTE: currently there is no speedup because we just dequantize the weight in the _quantized_linear op
    # we plan to add custom op example in the future and that will help us to get speedup
    print("after quantization and compile:", benchmark_model(m, NUM_RUNS, example_inputs))

if __name__ == "__main__":
    main()
