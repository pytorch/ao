# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD 3-Clause license found in the
# LICENSE file in the root directory of this source tree.

import torch
import torch.nn.functional as F
from torch.utils._python_dispatch import (
    return_and_correct_aliasing,
)

import torchao

aten = torch.ops.aten

import warnings
from enum import Enum, auto
from typing import Callable, List, Optional, Tuple, Union

from torch.ao.quantization.utils import getattr_from_fqn

from torchao.quantization.quant_api import (
    _QUANTIZE_CONFIG_HANDLER,
    AOBaseConfig,
    dataclass,
    register_quantize_module_handler,
)
from torchao.utils import DummyModule, fill_defaults

from .quantizable_moe_modules import MoEFeedForwardAOQuantizable

warnings.simplefilter("ignore", lineno=84)
warnings.simplefilter("ignore", lineno=105)

__all__ = [
    "MoEQuantConfig",
    "MoEMapping",
    "FakeExtraDimTensor",
    "UseFakeExtraDimTensor",
]


class UseFakeExtraDimTensor(Enum):
    """Enum that indicate whether to use FakeExtraDimTensor"""

    TRUE = auto()
    FALSE = auto()
    AS_FALLBACK = auto()


@dataclass
class MoEQuantConfig(AOBaseConfig):
    """Configuration for applying quantization to MoE
    Args:
        `Optional[base_config]`: normal AO Config to be applied to a MoEFeedforwardAOQuantizable module,
            if None, then will only do the conversion to MoEFeedforwardAOQuantizable using the mapping
        `Optional[mapping]`: MoEMapping, if None, then this will do no conversion, note: only
            MoEFeedforwardAOQuantizable modules can be quantized.
    """

    base_config: Optional[AOBaseConfig] = None
    mapping: Optional["MoEMapping"] = None

    use_fake_extra_dim_tensor: UseFakeExtraDimTensor = UseFakeExtraDimTensor.AS_FALLBACK
    set_inductor_config: bool = True


@register_quantize_module_handler(MoEQuantConfig)
def moe_convert_and_quant_fn(module: torch.nn.Module, config: MoEQuantConfig):
    mapping = config.mapping
    base_config = config.base_config
    assert mapping is not None or base_config is not None, (
        "need one of mapping or base_config to use MoEQuantConfig"
    )

    # maybe convert module to quantizable
    if mapping is not None and isinstance(module, mapping.target_module_type):
        module = _convert_module_to_ao_quantizable(module, mapping)

    # maybe quantize module
    if base_config is not None and isinstance(module, MoEFeedForwardAOQuantizable):
        module = _quantize_moe_module(module, config)

    return module


def _quantize_moe_module(module: torch.nn.Module, config: MoEQuantConfig):
    assert isinstance(module, MoEFeedForwardAOQuantizable), (
        f"can only apply quantization to MoEFeedForwardAOQuantizable modules but got {type(module)}"
    )

    experts = module.experts

    for weight_attr in experts.weight_attrs:
        param = getattr(experts, weight_attr)
        assert param.dim() == 3, (
            f"when applying moe_quant to {module} expected 3D tensor for {weight_attr} but got {param.dim()}"
        )
        assert isinstance(config.base_config, AOBaseConfig), (
            f"MoEQuantConfig expected to be initialized with an AOBaseConfig but got {type(config.base_config)}"
            + "this can happen if you initiaze with MoEQuantConfig(AOConfig) rather than MoEQuantConfig(AOConfig())"
        )
        new_param = _quantize_moe_tensor(param, config)
        new_param = torch.nn.Parameter(new_param, requires_grad=False)
        setattr(experts, weight_attr, new_param)
        del param
    return module


# Module-level flag to track if we've already printed the error
_quantize_moe_tensor_has_printed_error = False


def _quantize_moe_tensor(weight: torch.Tensor, config: MoEQuantConfig):
    def _quantize_moe_tensor_base(weight, config):
        base_config_handler = _QUANTIZE_CONFIG_HANDLER[type(config.base_config)]
        dummy_mod = DummyModule(weight)
        quant_mod = base_config_handler(dummy_mod, config.base_config)
        return quant_mod.weight

    def _quantize_moe_tensor_fake_extra_dim_tensor(
        weight: torch.Tensor, config: MoEQuantConfig
    ):
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

    global _quantize_moe_tensor_has_printed_error

    use_fake = config.use_fake_extra_dim_tensor
    if use_fake == UseFakeExtraDimTensor.FALSE:
        return _quantize_moe_tensor_base(weight, config)
    elif use_fake == UseFakeExtraDimTensor.AS_FALLBACK:
        try:
            return _quantize_moe_tensor_base(weight, config)
        except Exception as e:
            if not _quantize_moe_tensor_has_printed_error:
                print(f"tried to do moe_quant but got error: {e}")
                _quantize_moe_tensor_has_printed_error = True
            return _quantize_moe_tensor_fake_extra_dim_tensor(weight, config)
    else:  # This handles UseFakeExtraDimTensor.TRUE
        return _quantize_moe_tensor_fake_extra_dim_tensor(weight, config)


@dataclass
class MoEMapping:
    """This mapping dataclass is used to map an existing MoE module to the AOQuantizable one
    and is used with the convert_moe_with_mapping fn to convert a model to use the AO moe modules
    """

    target_module_type: type

    router_fqn: str = "gate"
    top_k_fqn: Optional[str] = "num_activated_experts"

    # if up_proj is a single tensor, leave up_proj_part2_fqn as None, otherwise list the fqn
    # for w1 and up_proj_fqn and w3 as up_proj_part2_fqn
    up_proj_fqn: str = "cond_ffn.w1"
    up_proj_part2_fqn: Optional[str] = "cond_ffn.w3"
    down_proj_fqn: str = "cond_ffn.w2"  # also known as down_proj

    # what is the order of indices of the weights,
    # specifically which order are the experts, out_features, in_features indices in?
    # for up_proj this would be experts, expert_dim*2, hidden_dim,
    # for down_proj this would be experts, hidden_dim, expert_dim,
    order_of_weight_indices: Union[Tuple[int], Tuple[int]] = (
        0,
        1,
        2,
    )

    # can't both be None
    act_fn: Optional[Callable[[torch.Tensor], torch.Tensor]] = F.silu
    act_fn_fqn: Optional[str] = None

    # Options
    shared_expert_fqn: Optional[str] = None
    return_scores: bool = False
    decompose_grouped_mm: bool = False


def _convert_module_to_ao_quantizable(module: torch.nn.Module, mapping: MoEMapping):
    assert isinstance(module, mapping.target_module_type), (
        f"_convert_module_to_ao_quantizable only works on modules of type {mapping.target_module_type} but got {type(module)}"
    )

    # get router and top_k
    router = getattr_from_fqn(module, mapping.router_fqn)
    top_k = getattr_from_fqn(module, mapping.top_k_fqn)

    # get up and down_proj
    order_of_indices = mapping.order_of_weight_indices
    if mapping.up_proj_part2_fqn is None:
        up_proj = (
            getattr_from_fqn(module, mapping.up_proj_fqn)
            .permute(*order_of_indices)
            .contiguous()
        )
    else:
        w1 = getattr_from_fqn(module, mapping.up_proj_fqn).permute(*order_of_indices)
        w3 = getattr_from_fqn(module, mapping.up_proj_part2_fqn).permute(
            *order_of_indices
        )
        up_proj = torch.cat((w1, w3), dim=1).contiguous()

    down_proj = (
        getattr_from_fqn(module, mapping.down_proj_fqn)
        .permute(*order_of_indices)
        .contiguous()
    )

    # get sizes
    num_experts, hidden_dim, expert_dim = down_proj.shape

    # get act_fn
    act_fn = mapping.act_fn
    if act_fn is None:
        act_fn = getattr_from_fqn(module, mapping.act_fn_fqn)
    assert act_fn is not None, (
        "both act_fn and act_fn_fqn can't be None in the MoEMapping"
    )

    # get final options
    shared_expert = None
    if isinstance(mapping.shared_expert_fqn, str):
        shared_expert = getattr_from_fqn(module, mapping.shared_expert_fqn)
    return_scores = mapping.return_scores
    decompose_grouped_mm = mapping.decompose_grouped_mm

    # make new module
    new_module = torchao.prototype.moe_quant.MoEFeedForwardAOQuantizable(
        num_experts=num_experts,
        hidden_dim=hidden_dim,
        expert_dim=expert_dim,
        top_k=top_k,
        act_fn=act_fn,
        shared_expert=shared_expert,
        return_scores=return_scores,
        decompose_grouped_mm=decompose_grouped_mm,
    )

    new_module.router = router
    new_module.experts.up_proj = torch.nn.Parameter(up_proj)
    new_module.experts.down_proj = torch.nn.Parameter(down_proj)

    return new_module


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
