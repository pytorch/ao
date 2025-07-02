# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD 3-Clause license found in the
# LICENSE file in the root directory of this source tree.

import torch
from torch.utils._python_dispatch import (
    return_and_correct_aliasing,
)

aten = torch.ops.aten

from enum import Enum, auto
from typing import List, Optional, Tuple, Union

from torchao.quantization.quant_api import (
    _QUANTIZE_CONFIG_HANDLER,
    AOBaseConfig,
    dataclass,
    register_quantize_module_handler,
)
from torchao.utils import DummyModule, fill_defaults


class FakeExtraDimTensor(torch.Tensor):
    """This is a subclass of torch.Tensor that simulates a tensor of n+1 dimensions, akin to concatenating several tensors along the 0th dimension.
    It takes a list of tensors with the same dtype, device and shape and creates a representation of shape (num_tensors, orig_shape). It can handle a
    variety of ops like detach and clone but most importantly, supports any slicing and indexing along the extra dimension.
    This is most useful when you have another tensor subclass that you'd like to concatenate together but don't want to support all the necessary
    pieces of 3D scaffolding required to make it work.

    The structure of this tensor subclass is a linked_list of tensors with each instance of FakeExtraDimTensor containing a head tensor and a tail consisting of
    either another intance of FakeExtraDimTensor or None if we've reached the end of the linked list. This implementation structure is necessary to
    support compilation of this tensor subclass since compile requires each tensor component of the tensor subclass to have its own attribute.
    """

    def __new__(
        cls,
        tensors: Union[Tuple[torch.Tensor], List[torch.Tensor]],
        tensor_tail: Optional["FakeExtraDimTensor"] = None,
    ):
        assert len(tensors) > 0 or tensor_tail is not None
        num_tensors = len(tensors)
        if tensor_tail is not None:
            num_tensors += tensor_tail.num_tensors
            test_tensor = tensor_tail.head_tensor
        else:
            test_tensor = tensors[0]

        dtype = test_tensor.dtype
        shape = test_tensor.shape
        device = test_tensor.device
        layout = test_tensor.layout
        for tensor in tensors:
            assert tensor.dtype == dtype, (
                f"all tensors in FakeExtraDimTensor must have same dtype but got {tensor.dtype} and {dtype}"
            )
            assert tensor.shape == shape, (
                f"all tensors in FakeExtraDimTensor must have same shape but got {tensor.shape} and {shape}"
            )
            assert tensor.device == device, (
                f"all tensors in FakeExtraDimTensor must have same device but got {tensor.device} and {device}"
            )
            assert tensor.layout == layout, (
                f"all tensors in FakeExtraDimTensor must have same layout but got {tensor.layout} and {layout}"
            )
        kwargs = {}
        kwargs["dtype"] = dtype
        kwargs["layout"] = layout
        kwargs["device"] = device
        kwargs["requires_grad"] = False
        new_shape = (num_tensors, *shape)
        return torch.Tensor._make_wrapper_subclass(cls, new_shape, **kwargs)

    def __repr__(
        self,
    ):
        return f"{self.__class__.__name__}(shape={self.shape}, containing {self.num_tensors}: {self.head_tensor})"

    def __init__(
        self,
        tensors: Union[Tuple[torch.Tensor], List[torch.Tensor]],
        tensor_tail: Optional["FakeExtraDimTensor"] = None,
    ):
        tensors = list(tensors)
        assert len(tensors) > 0 or tensor_tail is not None

        # count num_tensors and make tensor_list
        self.num_tensors = len(tensors)
        if tensor_tail is not None:
            self.num_tensors += tensor_tail.num_tensors
            tail_list = tensor_tail.tensor_list
        else:
            tail_list = []
        self.tensor_list = tensors + tail_list

        # 3 cases
        # 0) tensors has 0 elements -> take element from tail then do case 1 instead
        # 1) tensors has 1 element,  -> pop element and tail is None
        # 2) tensors has >1 elements, -> pop element and recurse

        # convert case 0 to case 1 by taking 1 element from tail
        if len(tensors) == 0 and tensor_tail is not None:
            tensors = [
                tensor_tail.head_tensor,
            ]
            tensor_tail = tensor_tail.tensor_tail

        if len(tensors) > 1:
            # case (1): remove first element from tensors, then recurse
            self.head_tensor = tensors[0]  # remove one
            self.tensor_tail = self.__class__(tensors[1:], tensor_tail)  # recurse
        elif len(tensors) == 1:
            # case (2) take final element from tensors, attach tensor_tail then stop recursion
            self.head_tensor = tensors[0]
            self.tensor_tail = tensor_tail

    def _apply_fn_to_data(self, fn):
        self.head_tensor = fn(self.head_tensor)
        if self.tensor_tail is not None:
            self.tensor_tail = self.tensor_tail._apply_fn_to_data(fn)
        return self.__class__([self.head_tensor], self.tensor_tail)

    def __tensor_flatten__(self):
        if self.tensor_tail is None:
            return [
                "head_tensor",
            ], [self.num_tensors]
        else:
            return [
                "head_tensor",
                "tensor_tail",
            ], [self.num_tensors]

    @classmethod
    def __tensor_unflatten__(
        cls,
        tensor_data_dict,
        tensor_attributes,
        outer_size,
        outer_stride,
    ):
        head_tensor = tensor_data_dict["head_tensor"]
        tensor_tail = tensor_data_dict.get("tensor_tail", None)
        return cls([head_tensor], tensor_tail)

    @classmethod
    def __torch_function__(cls, func, types, args, kwargs=None):
        kwargs = {} if kwargs is None else kwargs
        if func is torch.nn.functional.linear:
            x, w, bias = (
                args[0],
                args[1],
                args[2] if len(args) > 2 else None,
            )
            assert w.num_tensors == 1, (
                "FakeExtraDimTensor used in a linear op when it had multiple tensors"
            )
            return func(x, w.head_tensor, bias)
        try:
            with torch._C.DisableTorchFunctionSubclass():
                return func(*args, **kwargs)
        except Exception as e:
            print(f"ERR: subclass {cls} doesn't implement {func}, got error: {e}")

    @classmethod
    def __torch_dispatch__(cls, func, types, args, kwargs):
        kwargs = {} if kwargs is None else kwargs

        if func == aten.slice.Tensor:
            self, dim, start, end, step = fill_defaults(args, 5, [0, None, None, 1])
            if dim == 0:
                return return_and_correct_aliasing(
                    func, args, kwargs, cls(self.tensor_list[start:end:step])
                )

        elif func == aten.select.int:
            self, dim, index = fill_defaults(args, 3, [0, 0])
            if dim == 0:
                return return_and_correct_aliasing(
                    func, args, kwargs, cls([self.tensor_list[index]])
                )
        elif func == aten.index.Tensor:
            self, indices, dim = fill_defaults(args, 3, [0])
            if dim == 0:
                # this handles a weird bug where indices gets turned into a list
                # between the function dispatch and torch dispatch but just for this function
                if isinstance(indices, list) and len(indices) == 1:
                    indices = indices[0]
                return return_and_correct_aliasing(
                    func,
                    args,
                    kwargs,
                    cls([self.tensor_list[index] for index in indices]),
                )
        try:
            return return_and_correct_aliasing(
                func,
                args,
                kwargs,
                args[0]._apply_fn_to_data(lambda x: func(x, *args[1:], **kwargs)),
            )
        except Exception as e:
            print(
                f"function {func} failed for FakeExtraDimTensor, following error occured when trying to"
                "run function on its elements: "
            )
            raise e


class UseFakeExtraDimTensor(Enum):
    """Enum that indicate whether to use FakeExtraDimTensor"""

    TRUE = auto()
    FALSE = auto()
    AS_FALLBACK = auto()


@dataclass
class MoEQuantConfig(AOBaseConfig):
    """Configuration for applying quantization to MoE
    Args:
        `base_config`: normal AO Config
    """

    base_config: AOBaseConfig
    use_fake_extra_dim_tensor: UseFakeExtraDimTensor = UseFakeExtraDimTensor.AS_FALLBACK
    set_inductor_config: bool = True


# Module-level flag to track if we've already printed the error
_moe_quant_tensor_has_printed_error = False


def _moe_quant_tensor(weight, config):
    def _moe_quant_tensor_base(weight, config):
        base_config_handler = _QUANTIZE_CONFIG_HANDLER[type(config.base_config)]
        dummy_mod = DummyModule(weight)
        quant_mod = base_config_handler(dummy_mod, config.base_config)
        return quant_mod.weight

    def _moe_quant_tensor_fake_extra_dim_tensor(weight, config):
        base_config_handler = _QUANTIZE_CONFIG_HANDLER[type(config.base_config)]
        # break 3D tensor
        tensors = [weight[i] for i in range(weight.shape[0])]
        # put tensors into modules since the handlers target modules not tensors
        dummy_modules = [DummyModule(tensor) for tensor in tensors]
        # apply handler to each module
        quant_mods = list(
            map(lambda x: base_config_handler(x, config.base_config), dummy_modules)
        )
        # pack quantized subclasses into FakeExtraDimTensor
        quant_weight = FakeExtraDimTensor([mod.weight for mod in quant_mods])
        return quant_weight

    global _moe_quant_tensor_has_printed_error

    use_fake = config.use_fake_extra_dim_tensor
    if use_fake == UseFakeExtraDimTensor.FALSE:
        return _moe_quant_tensor_base(weight, config)
    elif use_fake == UseFakeExtraDimTensor.AS_FALLBACK:
        try:
            return _moe_quant_tensor_base(weight, config)
        except Exception as e:
            if not _moe_quant_tensor_has_printed_error:
                print(f"tried to do moe_quant but got error: {e}")
                _moe_quant_tensor_has_printed_error = True
            return _moe_quant_tensor_fake_extra_dim_tensor(weight, config)
    else:  # This handles UseFakeExtraDimTensor.TRUE
        return _moe_quant_tensor_fake_extra_dim_tensor(weight, config)


@register_quantize_module_handler(MoEQuantConfig)
def moe_quant_fn(module, config: MoEQuantConfig):
    import warnings

    warnings.simplefilter("ignore", lineno=84)
    warnings.simplefilter("ignore", lineno=105)

    for weight_attr in ["w1", "w2", "w3"]:
        param = getattr(module, weight_attr)
        assert param.dim() == 3, (
            f"when applying moe_quant to {module} expected 3D tensor for {weight_attr} but got {param.dim()}"
        )
        assert isinstance(config.base_config, AOBaseConfig), (
            f"MoEQuantConfig expected to be initialized with an AOBaseConfig but got {type(config.base_config)}"
            + "this can happen if you initiaze with MoEQuantConfig(AOConfig) rather than MoEQuantConfig(AOConfig())"
        )
        new_param = _moe_quant_tensor(param, config)
        new_param = torch.nn.Parameter(new_param, requires_grad=False)
        setattr(module, weight_attr, new_param)
        del param
    return module


def moe_filter(module, fqn):
    return "MOEFeedForwardAOQuantizable" in str(type(module))


def cond_ffn_filter(module, fqn):
    return "ConditionalFeedForwardAOQuantizable" in str(type(module))
