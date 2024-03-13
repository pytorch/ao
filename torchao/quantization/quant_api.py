# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""
Quantization APIs

Generally these APIs can be applied directly to any model
with Linear modules to obtain quantized linear ops. The intended
usage involves applying torch.compile to the model afterwards
both because primitives were designed based on the fusions that
come along with it and because that is how we access the intended quantized
and mixed GEMM kernels
"""

import torch
import torch.nn.functional as F
import torch.nn as nn
from .dynamic_quant import (
    DynamicallyPerAxisQuantizedLinear,
)
from .subclass import (
    QuantizedLinearWeightBase,
    Int8DynamicallyQuantizedLinearWeight,
    Int8WeightOnlyQuantizedLinearWeight,
    Int4WeightOnlyQuantizedLinearWeight,
)
from .weight_only import (
    WeightOnlyInt8QuantLinear,
)
from .quant_primitives import (
    get_group_qparams_symmetric,
    per_token_dynamic_quant,
)
from typing import Dict, Tuple

__all__ = [
    "apply_weight_only_int8_quant",
    "apply_dynamic_quant",
    "change_linear_weights_to_int8_dqtensors",
    "change_linear_weights_to_int8_woqtensors",
    "change_linear_weights_to_int4_woqtensors",
    "swap_conv2d_1x1_to_linear",
    "Quantizer",
    "TwoStepQuantizer",
]

############################# Unified Quantization APIs ##############################
# API 1, single quantize call to create a quantized model with quantized state_dict
class Quantizer:
    def quantize(self, model: torch.nn.Module, *args, **kwargs) -> torch.nn.Module:
        pass


# API 2, flow that needs calibration or training
class TwoStepQuantizer:
    def prepare(self, model: torch.nn.Module, *args, **kwargs) -> torch.nn.Module:
        pass

    def convert(self, model: torch.nn.Module, *args, **kwargs) -> torch.nn.Module:
        pass

############################# Unified Quantization APIs ##############################

def _replace_with_custom_fn_if_matches_filter(
    model, replacement_fn, filter_fn, cur_fqn=""
) -> None:
    """
    For each `child` in `model`, replaces it with `replacement_fn(child)`
    if `filter_fn(child)` is `True`
    """
    if filter_fn(model, cur_fqn[:-1]):
        model = replacement_fn(model)
        return model
    else:
        for name, child in model.named_children():
            new_child = _replace_with_custom_fn_if_matches_filter(
                child, replacement_fn, filter_fn, f"{cur_fqn}{name}."
            )
            if new_child is not child:
                setattr(model, name, new_child)
        return model


def _is_linear(mod, *args):
    return (
        isinstance(mod, torch.nn.Linear) and
        hasattr(mod, "weight") and
        not isinstance(mod.weight, QuantizedLinearWeightBase)
    )

def _in_features_greater_than_16(mod, *args):
    return hasattr(mod, "in_features") and mod.in_features > 16

def apply_weight_only_int8_quant(model, filter_fn=None):
    """
    Applies weight-only symmetric per-channel int8 quantization to all linear layers
    in the given model using module swaps.
    """
    _replace_with_custom_fn_if_matches_filter(
        model,
        WeightOnlyInt8QuantLinear.from_float,
        _is_linear if filter_fn is None else filter_fn,
    )


def apply_dynamic_quant(model, filter_fn=None):
    """
    Applies dynamic symmetric per-token activation and per-channel weight
    quantization to all linear layers in the given model using
    module swaps.
    """
    _replace_with_custom_fn_if_matches_filter(
        model,
        lambda mod: DynamicallyPerAxisQuantizedLinear.from_float(mod),
        _is_linear if filter_fn is None else filter_fn,
    )


def _get_subclass_inserter(cls, **kwargs):
    def insert_subclass(lin):
        lin.weight = torch.nn.Parameter(
            cls.from_float(lin.weight, **kwargs), requires_grad=False
        )
        return lin

    return insert_subclass


def change_linear_weights_to_int8_dqtensors(model, filter_fn=None):
    """
    Converts all linear weight tensors to the `Int8DynamicallyQuantizedLinearWeight`
    Tensor subclass, effectively applying the same form of quantization
    as apply_dynamic_quant while not modifying the linear modules.
    """
    if filter_fn is None:
        filter_fn = (
            lambda *args:
            _is_linear(*args) and
            _in_features_greater_than_16(*args)
        )

    _replace_with_custom_fn_if_matches_filter(
        model,
        _get_subclass_inserter(Int8DynamicallyQuantizedLinearWeight),
        filter_fn
    )


def change_linear_weights_to_int8_woqtensors(model, filter_fn=None):
    """
    Converts all linear weight tensors to the
    `Int8WeightOnlyQuantizedLinearWeight` tensor subclass,
    effectively applying the same form of quantization
    as apply_dynamic_quant while not modifying the linear modules.
    """
    _replace_with_custom_fn_if_matches_filter(
        model,
        _get_subclass_inserter(Int8WeightOnlyQuantizedLinearWeight),
        _is_linear if filter_fn is None else filter_fn,
    )


def change_linear_weights_to_int4_woqtensors(model, **kwargs):
    """
    Converts all linear weight tensors to the
    `Int4WeightOnlyQuantizedLinearWeight` tensor subclass,
    effectively applying the same form of quantization
    as apply_dynamic_quant while not modifying the linear modules.
    """
    filter_fn = kwargs.pop("filter_fn", _is_linear)

    _replace_with_custom_fn_if_matches_filter(
        model,
        _get_subclass_inserter(Int4WeightOnlyQuantizedLinearWeight, **kwargs),
        filter_fn,
    )

def swap_conv2d_1x1_to_linear(model, filter_fn=None):
    """
    Changes all conv2d 1x1 modules to equivalent linear modules so that they can then be quantized.
    """
    class PermuteSandwich(torch.nn.Module):
        def __init__(self, mod):
            super().__init__()
            self.mod = mod

        def forward(self, *args):
            return self.mod(args[0].permute(0, 2, 3, 1)).permute(-0,3,1,2)


    def replace_conv2d_1x1(conv):
        assert conv.kernel_size == (1, 1)
        lin = torch.nn.Linear(conv.in_channels, conv.out_channels, bias=(conv.bias is None))
        lin.weight=torch.nn.Parameter(conv.weight.squeeze(-1,-2))
        lin.bias = conv.bias
        return PermuteSandwich(lin)

    if filter_fn is None:
        filter_fn=lambda mod, *args: isinstance(mod, torch.nn.Conv2d) and mod.kernel_size==(1,1)

    _replace_with_custom_fn_if_matches_filter(
        model,
        replace_conv2d_1x1,
        filter_fn=filter_fn
    )


from .GPTQ import lm_eval_available

if lm_eval_available:
    from .GPTQ import (  # pyre-ignore[21]
        evaluate,
        GenericGPTQRunner,
        get_task_dict,
        InputRecorder,
        lm_eval,
        MultiInput,
    )
else:
    print("lm_eval not available, skip defining GPTQQuantizer")


class GPTQQuantizer(Quantizer):
    """
    This class implements a GPTQ Quantizer that can be used to apply GPTQ to a model in concert with the GenericGPTQRunner class.
    Unlike the base Quantizer class, the user does not need to implement the create_quantized_state_dict, instead they have to reimplement
    __init__ such that it defines the functions for the quantization mode. User is expected to reimplement convert_for_runtime.

    The following functions (which must be defined in __init__) are used to define the quantization mode for both GPTQ and
    create_quantized_state_dict. Here is a description of each function.

    get_qparams_func:
        A function that calculates the quantization qparams for an input tensor.
        Args:
            weight: A 2d weight tensor with non-integer dtype.
        Returns:
            qparams: it can have any format but will need to be handled by the other defined functions below.

    quantize_func:
        A function that applies quantization to an input tensor. It should be noted
        that this function needs to be able to handle quantizing the entire weight tensor, a single group,
        or a single column.
        Args:
            weight: A 2d weight tensor with non-integer dtype.
            qparams: the output from get_qparams_func
        Returns:
            quantized_weight: A 2d quantized weight tensor (generally with an integer dtype)


    dequantize_func:
        A function that dequantizes an input quantized weight tensor. It should be noted
        that this function needs to be able to handle dequantizing the entire weight tensor, a single group,
        or a single column.
        Args:
            quantized_weight: A 2d quantized weight tensor (generally with an integer dtype)
            qparams: the output from get_qparams_func
        Returns:
            weight: A 2d weight tensor with non-integer dtype.

    combine_qparams_list_func:
        A function that combines several qparams into one qparam.
        Args:
            qparams_list: a list of qparams objects, each obtained by calling get_qparams_func
            on a single group from a weight tensor
        Returns:
            qparams: an object of the same format as the qparams above.

    skip_layer_func:
        A function that determines which linear layers should be skipped during GPTQ
        Args:
            weight: A 2d weight tensor with non-integer dtype.
        Returns:
            skip: boolean indicating whether layer should be skipped

    make_names_and_values_dict_func:
        A function that prepares the qparams and quantized_weight and creates a dictionary indicating how they
        should be inserted into the state_dict. Generally any packing of the weight and qparams should be done here.
        Args:
            quantized_weight: A 2d quantized weight tensor (generally with an integer dtype)
            qparams: the output from get_qparams_func
        Returns:
            names_and_values_dict: a dictionary mapping the name of the parameters of the quantized module to the
            corresponding quantized weights and qparams.
    """

    def __init__(self):
        assert self.get_qparams_func is not None
        assert self.quantize_func is not None
        assert self.dequantize_func is not None
        assert self.combine_qparams_list_func is not None
        assert self.make_names_and_values_dict_func is not None

    @staticmethod
    def get_inputs(
        model,
        tokenizer,
        calibration_tasks,
        calibration_limit,
        calibration_seq_length,
        pad_calibration_inputs,
    ) -> "MultiInput":  # pyre-ignore[11]
        input_recorder = InputRecorder(
            model,
            tokenizer,
            calibration_seq_length,
            pad_calibration_inputs,
        )

        try:
            lm_eval.tasks.initialize_tasks()
        except:
            pass
        task_dict = get_task_dict(calibration_tasks)
        print("Obtaining GPTQ calibration inputs on: ", calibration_tasks)

        evaluate(
            input_recorder,
            task_dict,
            limit=calibration_limit,
        )
        inputs = input_recorder.get_recorded_inputs()
        assert inputs is not None, (
            f"No inputs were collected, use a task other than {calibration_tasks}, "
            + "use option pad_calibration_inputs, or decrease calibration_sequence_length (currently "
            + f"{calibration_seq_length})"
        )
        print(f"Obtained {len(inputs[0].values)} calibration samples")
        return inputs

    @torch.no_grad()
    def _create_quantized_state_dict(
        self,
        model,
        tokenizer,
        blocksize,
        percdamp,
        groupsize,
        calibration_tasks,
        calibration_limit,
        calibration_seq_length,
        pad_calibration_inputs,
    ) -> Dict:
        inputs = GPTQQuantizer.get_inputs(
            model,
            tokenizer,
            calibration_tasks,
            calibration_limit,
            calibration_seq_length,
            pad_calibration_inputs,
        )
        print("Tracing model for GPTQ")
        GPTQ_runner = GenericGPTQRunner(
            model,
            inputs,
            blocksize,
            percdamp,
            groupsize,
        ).configure_quantization_mode(
            self.get_qparams_func,  # pyre-ignore[16]
            self.quantize_func,  # pyre-ignore[16]
            self.dequantize_func,  # pyre-ignore[16]
            self.combine_qparams_list_func,  # pyre-ignore[16]
            self.make_names_and_values_dict_func,  # pyre-ignore[16]
            self.skip_layer_func,  # pyre-ignore[16]
        )
        print("Applying GPTQ to weights")
        GPTQ_runner.run()
        return GPTQ_runner.get_quantized_state_dict()

    def _convert_for_runtime(self, model: torch.nn.Module) -> "nn.Module":
        raise NotImplementedError("_convert_for_runtime not implemented")

    @torch.no_grad()
    def quantize(
        self,
        model,
    ) -> torch.nn.Module:
        state_dict = self._create_quantized_state_dict(
            model,
            self.tokenizer,
            self.blocksize,
            self.percdamp,
            self.groupsize,
            self.calibration_tasks,
            self.calibration_limit,
            self.calibration_seq_length,
            self.pad_calibration_inputs,
        )
        model = self._convert_for_runtime(model)
        model.load_state_dict(state_dict, strict=False)
        return model


class Int8DynActInt4WeightLinear(torch.nn.Module):
    __constants__ = ["in_features", "out_features"]

    in_features: int
    out_features: int
    weight: torch.Tensor

    """
    This module implements a dynamic quantized linear layer with int4 weight.
    Weights are per channel groupwise quantized. Parameters of importance
    group_size: the number of elements in each quantized group
    precision: precision of input and output. e.g. torch.float32 means input
    activation is float32 and output is float32.
    scales_precision: precision of per group scale.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias=True,
        device=None,
        dtype=None,
        group_size: int = 256,
        precision: torch.dtype = torch.float32,
        scales_precision: torch.dtype = torch.float32,
    ) -> None:
        super().__init__()
        # always pad if needed since it becomes a noop at runtime if not needed
        # self.origin_in_features = in_features
        assert (
            in_features % group_size == 0
        ), f"require in_features:{in_features} % group_size:{group_size} == 0"
        # in_features = _calc_padded_size_linear_int4(
        #    in_features, group_size
        # )
        self.in_features = in_features
        self.out_features = out_features
        assert not bias, "require bias=False"
        self.group_size = group_size
        # Precision of the activation which also indicates
        # output precision of the dynamically quantized linear layer
        # that his module represents.
        self.precision = precision

        # currently storing unpacked int8 weights
        self.register_buffer(
            "weight",
            torch.empty((out_features, in_features), dtype=torch.int8),
        )
        self.register_buffer(
            "scales",
            torch.empty(
                (out_features, in_features // group_size),
                dtype=scales_precision,
            ),
        )
        self.register_buffer(
            "zeros",
            torch.empty(
                (out_features, in_features // group_size),
                dtype=scales_precision,
            ),
        )

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        input = input.to(self.precision)
        # padding is removed for perf
        # input = F.pad(input, pad=(0, self.in_features - self.origin_in_features))
        return linear_forward_8da4w(
            input,
            self.weight,
            self.scales,
            self.zeros,
            self.out_features,
            self.groupsize,
            self.precision,
        )

from math import gcd
from functools import reduce


def find_multiple(n: int, *args: Tuple[int]) -> int:
    # TODO: this change is reverted right now in gpt-fast
    k: int = reduce(lambda x, y: x * y // gcd(x, y), args + (1,))  # type: ignore[9]
    if n % k == 0:
        return n
    return n + k - (n % k)


def _check_linear_int4_k(k, group_size=1):
    return k % group_size == 0


def _calc_padded_size_linear_int4(k, groupsize=1):
    return find_multiple(k, groupsize)


def pack_scales_and_zeros(scales, zeros, precision=torch.float32):
    assert scales.shape == zeros.shape
    assert scales.dtype == precision
    assert zeros.dtype == precision
    return (
        torch.cat(
            [
                scales.reshape(scales.size(0), scales.size(1), 1),
                zeros.reshape(zeros.size(0), zeros.size(1), 1),
            ],
            2,
        )
        .transpose(0, 1)
        .contiguous()
    )


def unpack_scales_and_zeros(scales_and_zeros):
    assert len(scales_and_zeros.shape) == 3 and scales_and_zeros.shape[2] == 2
    assert scales_and_zeros.dtype == torch.float
    return torch.split(scales_and_zeros.transpose(0, 1), 1, 2)


def replace_linear_8da4w(
    module,
    group_size,
    padding_allowed,
    precision,
    scales_precision,
):
    for name, child in module.named_children():
        if isinstance(child, nn.Linear):
            if _check_linear_int4_k(child.in_features, group_size) or padding_allowed:
                setattr(
                    module,
                    name,
                    Int8DynActInt4WeightLinear(
                        child.in_features,
                        child.out_features,
                        bias=False,
                        group_size=group_size,
                        precision=precision,
                        scales_precision=scales_precision,
                    ),
                )
        else:
            replace_linear_8da4w(
                child,
                group_size,
                padding_allowed,
                precision,
                scales_precision,
            )


class Int8DynActInt4WeightGPTQQuantizer(GPTQQuantizer):
    def __init__(
        self,
        tokenizer,
        blocksize,
        percdamp,
        groupsize,
        calibration_tasks,
        calibration_limit,
        calibration_seq_length,
        pad_calibration_inputs,
        inner_k_tiles=8,
        padding_allowed=True,
        precision=torch.float32,
    ):

        self.tokenizer = tokenizer
        self.blocksize = blocksize
        self.percdamp = percdamp
        self.groupsize = groupsize
        self.calibration_tasks = calibration_tasks
        self.calibration_limit = calibration_limit
        self.calibration_seq_length = calibration_seq_length
        self.pad_calibration_inputs = pad_calibration_inputs
        self.inner_k_tiles = inner_k_tiles
        self.padding_allowed = padding_allowed
        self.precision = precision
        self.dyn_quant_func = lambda x: per_token_dynamic_quant(x)
        n_bit = 4
        self.get_qparams_func = lambda w: get_group_qparams_symmetric(
            w, n_bit, groupsize, self.precision
        )
        quant_min = -(2 ** (n_bit - 1))
        quant_max = 2 ** (n_bit - 1) - 1
        self.quantize_func = lambda w, qparams: torch.ops.quantized_decomposed.quantize_per_channel_group(
            w, qparams[0], qparams[1], quant_min, quant_max, torch.int8, groupsize
        )
        self.dequantize_func = lambda q, qparams: torch.ops.quantized_decomposed.dequantize_per_channel_group(
            q,
            qparams[0],
            qparams[1],
            quant_min,
            quant_max,
            torch.int8,
            groupsize,
            self.precision,
        )
        self.combine_qparams_list_func = lambda qparams_list: [
            torch.cat(x, dim=1) for x in zip(*qparams_list)
        ]
        # skip unless padding_allowed=True or its correctly sized
        self.skip_layer_func = lambda linear_weight: not (
            _check_linear_int4_k(linear_weight.shape[-1], groupsize)
            or padding_allowed
        )

        # we need to do the padding here, both for q and the qparams if necessary
        def make_names_and_values_dict_func(q, qparams):
            k = q.shape[1]
            new_k = _calc_padded_size_linear_int4(k, groupsize)
            # how much we need to pad the weight
            delta_k = new_k - q.shape[1]
            final_q = F.pad(q, pad=(0, delta_k))
            scales = qparams[0].to(self.precision)
            zeros = qparams[1].to(self.precision)
            # scales_and_zeros = pack_scales_and_zeros(*qparams, precision=self.precision)
            # how many new groups we need for padded weight
            # delta_groups = new_k // groupsize - scales_and_zeros.shape[0]
            # TODO: split scales and zero_points
            # final_s_and_z = F.pad(
            #     scales_and_zeros, pad=(0, 0, 0, 0, 0, delta_groups), value=1
            # )
            return {"weight": final_q, "scales": scales, "zeros": zeros}

        self.make_names_and_values_dict_func = make_names_and_values_dict_func
        super().__init__()

    def _convert_for_runtime(self, model):
        replace_linear_8da4w(
            model,
            self.groupsize,
            self.padding_allowed,
            torch.int8,
            self.precision,
        )
        return model
