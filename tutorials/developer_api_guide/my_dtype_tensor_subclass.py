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
from torchao.quantization.quant_primitives import choose_qparams_affine, MappingType
from torchao.dtypes.utils import (
    _implements,
    _dispatch__torch_function__,
    _dispatch__torch_dispatch__,
    _register_layout_cls,
    _get_layout_tensor_constructor,
    LayoutType,
    PlainLayoutType,
)

aten = torch.ops.aten

###############################
# Base Layout Tensor Subclass #
###############################
class MyDTypeLayout(torch.Tensor):
    """
    Base class for the layout tensor for `MyDTypeTensor`
    """
    # get the original unpacked Tensors
    def get_plain(self) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.int_data, self.scale

    def get_layout_type(self) -> LayoutType:
        return self.layout_type

    @classmethod
    def from_plain(
        cls,
        int_data: torch.Tensor,
        scale: torch.Tensor,
        layout_type: LayoutType,
    ):
        """Construct a layout tensor from plain tensors and a layout_type, which main contain
        extra metadata for packing etc.
        """
        pass

    def __repr__(self):
        int_data, scale = self.get_plain()
        layout_type = self.get_layout_type()
        return f"{self.__class__.__name__}(int_data={int_data}, scale={scale}, layout_type={layout_type})"

    __torch_function__ = torch._C._disabled_torch_function_impl

##############################
# Tensor Subclass Definition #
##############################

class MyDTypeTensor(torch.Tensor):
    """We need to define __new__ for constructing a new tensor subclass instance and __init__ for initialize
    the instance. There is no requirement on what the argument list should look like here, only requirement is
    that `__new__` must return a Tensor instance with `torch.Tensor._make_wrapper_subclass(cls, shape, ...)` call
    """

    @staticmethod
    def __new__(
        cls,
        layout_tensor: MyDTypeLayout,
        shape: torch.Size,
        dtype: Optional[torch.dtype] = None,
    ):
        kwargs = {}
        kwargs["device"] = layout_tensor.device
        kwargs["layout"] = (
            kwargs.get("layout")
            if kwargs.get("layout", False)
            else layout_tensor.layout
        )
        kwargs["dtype"] = dtype
        kwargs["requires_grad"] = False
        return torch.Tensor._make_wrapper_subclass(cls, shape, **kwargs)  # type: ignore[attr-defined]

    def __init__(
        self,
        layout_tensor: MyDTypeLayout,
        shape: torch.Size,
        dtype: Optional[torch.dtype] = None,
    ):
        self.layout_tensor = layout_tensor

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
        return ["layout_tensor"], [self.shape, self.dtype]

    @classmethod
    def __tensor_unflatten__(
        cls, tensor_data_dict, tensor_attributes, outer_size, outer_stride
    ):
        """
        Given the flattened data from above, returns a class instance
        tensor_data_dict contains the tensor fields of the class as a dictionary
        tensor_attributes contains all other non tensor type fields
        """
        layout_tensor = tensor_data_dict["layout_tensor"]
        shape, dtype = tensor_attributes
        return cls(
            layout_tensor,
            shape if outer_size is None else outer_size,
            dtype=dtype,
        )

    """classmethod that converts from a floating point Tensor (fp32/fp16/bf16) to the current dtype
    """

    @classmethod
    def from_float(
        cls,
        input_float: torch.Tensor,
        layout_type: LayoutType = PlainLayoutType(),
    ):
        mapping_type = MappingType.SYMMETRIC
        block_size = input_float.shape
        dtype = torch.int16
        scale, _ = choose_qparams_affine(input_float, mapping_type, block_size, dtype)
        int_data = (input_float / scale).to(torch.int8)
        layout_tensor_ctr = get_layout_tensor_constructor(type(layout_type))
        layout_tensor = layout_tensor_ctr(int_data, scale, layout_type)
        return cls(layout_tensor, input_float.shape)

    """[Optional] We can overwrite layout property of the Tensor to represent different packing formats
    """

    @property
    def layout_type(self) -> LayoutType:
        return self.layout_tensor.layout_type

    def dequantize(self, output_dtype=None):
        """We can define a dequantize method to convert the quantized tensor to a floating point tensor"""
        if output_dtype is None:
            output_dtype = torch.get_default_dtype()
        int_data, scale = self.layout_tensor.get_plain()
        return int_data.to(output_dtype) * scale

    def __repr__(self):
        return (
            f"{self.__class__.__name__}(data={self.dequantize()}, shape={self.shape}, "
            f"device={self.device}, dtype={self.dtype}, requires_grad={self.requires_grad})"
        )

    def _apply_fn_to_data(self, fn):
        """
        Used for implementing aten ops by applying them only to the relevant tensor atributes
        In this case we only want to call things like to() or view() on the layout tensor
        """
        return self.__class__(
            fn(self.layout_tensor),
            self.shape,
            self.dtype,
        )

    implements = classmethod(_implements)

    """There are two entry points that we can modify the behavior of a pytorch op: torch_function and torch_dispatch:

    __torch_function__: will be called whenever a torch level function is called on the Tensor object, for example: torch.nn.functional.linear,
    tensor.detach, tensor.reshape, tensor.t etc.

    __torch_dispatch__: will be called in the C++ dispatcher, when an aten operator is called on the Tensor object, for example:
    aten.mm, aten.addmm, aten.detach.default, aten.t.default etc.

    We have some helper functions that can dispatch to the functions registered with MyDTypeTensor.implements, but if the default implementation does not work for your use case, please feel free to customize it
    """
    __torch_dispatch__ = classmethod(_dispatch__torch_dispatch__)
    __torch_function__ = classmethod(_dispatch__torch_function__)

######################################################
# LayoutType and Layout Tensor Subclass Registration #
######################################################

def register_layout_cls(layout_type_class: type(LayoutType)):
    return _register_layout_cls(MyDTypeTensor, layout_type_class)

def get_layout_tensor_constructor(layout_type_class: type(LayoutType)):
    return _get_layout_tensor_constructor(MyDTypeTensor, layout_type_class)


@register_layout_cls(PlainLayoutType)
class PlainMyDTypeLayout(MyDTypeLayout):
    def __new__(
        cls,
        int_data: torch.Tensor,
        scale: torch.Tensor,
        layout_type: LayoutType,
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
        layout_type: LayoutType,
    ):
        self.int_data = int_data
        self.scale = scale
        self.layout_type = layout_type

    def __tensor_flatten__(self):
        return ["int_data", "scale"], [self.layout_type]

    @classmethod
    def __tensor_unflatten__(
        cls, tensor_data_dict, tensor_attributes, outer_size, outer_stride
    ):
        int_data, scale = tensor_data_dict["int_data"], tensor_data_dict["scale"]
        layout_type, = tensor_attributes
        return cls(int_data, scale, layout_type)

    @classmethod
    def from_plain(
        cls,
        int_data: torch.Tensor,
        scale: torch.Tensor,
        layout_type: LayoutType,
    ):
        """Construct a layout tensor from plain tensors and a layout_type, which main contain
        extra metadata for packing etc.
        """
        assert isinstance(layout_type, PlainLayoutType)
        return cls(int_data, scale, layout_type)

    def _apply_fn_to_data(self, fn):
        return self.__class__(
            fn(self.int_data),
            fn(self.scale),
            self.layout_type,
        )

    @classmethod
    def __torch_dispatch__(cls, func, types, args, kwargs):
        kwargs = {} if kwargs is None else kwargs

        if func is aten.detach.default:
            return return_and_correct_aliasing(
                func, args, kwargs, args[0]._apply_fn_to_data(torch.detach)
            )

        raise NotImplementedError(
            f"MyDTypeLayout dispatch: attempting to run {func}, this is not supported"
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


class M(torch.nn.Module):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.linear = torch.nn.Linear(1024, 1024)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear(x)

#####################
# Factory functions #
#####################
to_my_dtype = MyDTypeTensor.from_float


########
# Test #
########
from torchao.utils import benchmark_model

m = M()
example_inputs = (100 * torch.randn(1024, 1024),)
NUM_WARMUPS = 10
NUM_RUNS = 100

for _ in range(NUM_WARMUPS):
    m(*example_inputs)
print("before quantization:", benchmark_model(m, NUM_RUNS, example_inputs[0]))

compiled = torch.compile(m, mode="max-autotune")
for _ in range(NUM_WARMUPS):
    compiled(*example_inputs)
print("after compile:", benchmark_model(compiled, NUM_RUNS, example_inputs[0]))

# convert weights to quantized weights
m.linear.weight = torch.nn.Parameter(
    to_my_dtype(m.linear.weight), requires_grad=False
)

for _ in range(NUM_WARMUPS):
    m(*example_inputs)

print("after quantization:", benchmark_model(m, NUM_RUNS, example_inputs[0]))

m = torch.compile(m, mode="max-autotune")

for _ in range(NUM_WARMUPS):
    m(*example_inputs)

# NOTE: currently there is no speedup because we just dequantize the weight in the _quantized_linear op
# we plan to add custom op example in the future and that will help us to get speedup
print("after quantization and compile:", benchmark_model(m, NUM_RUNS, example_inputs[0]))
