# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD 3-Clause license found in the
# LICENSE file in the root directory of this source tree.
"""
Traditional calibration flow has the following flow (see static_quant.py for code examples):

(1). insert input/output observers to the modules
(2). run the model with calibration data so the observers in the model can record the statistics of the data flowing through them, observation does not change the output of a layer
(3). convert the observed module to quantized module (or quantize the weights with the quantization parameters based on the observer statistics)

By GPTQ like flow we mean a flow that does not fit into the above flow very well because
(1) optimize (quantize) one layer (module) at a time and the the output of each layer is calculated based on the optimized (quantized) module, and then pass down to the next layer, this means layers are not independent
(2) with each optimization step, we need to use all the input data for that layer instead of just some derived statistics like min_val/max_val

To use the traditional flow, we'd need to
(1). insert observers for the layer we want to optimize, that will record all the inputs
(2). each time, run the entire model upto layer N, then optimize layer N, and then
continue the process for layer N+1, this means we'll need to run O(N^2) layers in total.

So we'd like to use a flow that only runs each layer a constant time so we get O(N) time complexity.

In this tutorial we mainly use two things:
(1) MultiTensor subclass https://gist.github.com/HDCharles/a1b575bbf8875f994af8a01b225e1227

It stores a list of Tensors (calibration data). This is used to pass around all the calibration data to a layer, we can optimize the layer, and then output another MultiTensor object for future layers.

(2) Module forward pre hooks (https://pytorch.org/docs/stable/generated/torch.nn.Module.html#torch.nn.Module.register_forward_pre_hook)

This is used for modifying the behavior of the forward function of `module`, it allows [modification](https://discuss.pytorch.org/t/use-forward-pre-hook-to-modify-nn-module-parameters/108498/2) of the module itself, and also allows modifying the input of the module.

This can be used when we try to optimize (quantize) the layer, and then want the next layer to consume the output of the optimized layer directly.
"""

from typing import Any, Dict, Tuple

import torch
from torch.utils._pytree import tree_flatten, tree_unflatten

from torchao.core.config import AOBaseConfig
from torchao.dtypes import (
    to_affine_quantized_intx,
    to_affine_quantized_intx_static,
)
from torchao.quantization import (
    AffineQuantizedMinMaxObserver,
    LinearActivationQuantizedTensor,
    MappingType,
    PerTensor,
    fake_quantize_affine,
    quantize_,
    to_linear_activation_quantized,
)
from torchao.quantization.quant_api import _replace_with_custom_fn_if_matches_filter
from torchao.quantization.transform_module import (
    register_quantize_module_handler,
)
from torchao.quantization.utils import compute_error

torch.manual_seed(0)


class MultiTensor(torch.Tensor):
    @staticmethod
    def __new__(cls, input, **kwargs):
        if isinstance(input, (list, tuple)):
            input = input[0]
        kwargs["dtype"] = kwargs.get("dtype", input.dtype)
        shape = kwargs.pop("shape", input.shape)
        return torch.Tensor._make_wrapper_subclass(cls, shape, **kwargs)

    def __init__(self, input, **kwargs):
        self.values = []
        self.count = 0
        self.add_tensors(input)
        self.debug = True

    def __repr__(self):
        return f"{self.__class__.__name__}(data={self.values})"

    def __iter__(self):
        for v in self.values:
            yield v

    def add_tensors(self, input):
        if isinstance(input, (tuple, list)):
            for inp in input:
                self.add_tensors(inp)
        else:
            assert isinstance(input, torch.Tensor), (
                f"MultiTensor can only use add_tensors for Tensors or lists of tensors but got {type(input)}"
            )
            self.count += 1
            self.values.append(input)
        return self

    def pad_to_length(self, length):
        if self.count > length:
            return self
        self.add_tensors([self.values[-1]] * (length - self.count))
        return self

    @classmethod
    def __torch_function__(cls, func, types, args=(), kwargs=None, skip_gptq=False):
        def flat_to_grouped(flat):
            # size of biggest MultiTensor
            multi_tensor_size = max(
                [x.count if isinstance(x, MultiTensor) else 1 for x in flat]
            )
            # convert [A, MultiTensor(b1,b2,b3), MultiTensor(c1,c2,c3)] => [[A,b1,c1], [A,b2,c2] [A,b3,c3]]
            grouped = list(
                zip(
                    *[
                        x.pad_to_length(multi_tensor_size).values
                        if isinstance(x, MultiTensor)
                        else [x] * multi_tensor_size
                        for x in flat
                    ]
                )
            )
            return grouped

        # convert [[A,b1,c1], [A,b2,c2] [A,b3,c3]] => [A, MultiTensor(b1,b2,b3), MultiTensor(c1,c2,c3)]
        # where A is nontensor, b's,c's are tensors
        def grouped_to_flat(grouped):
            # convert [[A,b1,c1], [A,b2,c2] [A,b3,c3]] => [(A,A,A), (b1,b2,b3), (c1,c2,c3)]
            flat_tups = list(zip(*grouped))
            # convert [(A,A,A), (b1,b2,b3), (c1,c2,c3)] => [A, MultiTensor(b1,b2,b3), MultiTensor(c1,c2,c3)]
            flattened = [
                cls(tup).cpu() if isinstance(tup[0], torch.Tensor) else tup[0]
                for tup in flat_tups
            ]
            # need to check that getting rid of all but one from each nonTensor tuple is OK
            non_tensors_equal = min(
                [True]
                + [
                    min(
                        [True]
                        + [  # handle situation where tuples have size 0
                            tup[0] == x
                            for x in tup  # check all elements match
                        ]
                    )
                    for tup in flat_tups
                    if not isinstance(
                        tup[0], torch.Tensor
                    )  # look at tuples of nonTensors
                ]
            )
            return flattened, non_tensors_equal

        kwargs = {} if kwargs is None else kwargs
        # combine args and kwargs and remove lists and tuples
        flat_args, spec = tree_flatten((args, kwargs))
        # convert [A, MultiTensor(b1,b2,b3), MultiTensor(c1,c2,c3)] => [[A,b1,c1], [A,b2,c2] [A,b3,c3]]
        grouped_args = flat_to_grouped(flat_args)
        # run function for each of the multitensors and return a multitensor
        outputs = []
        with torch._C.DisableTorchFunctionSubclass():
            for inp in grouped_args:
                # inp = tensors_to_cuda(inp)
                cur_args, cur_kwargs = tree_unflatten(inp, spec)
                out = func(*cur_args, **cur_kwargs)
                # outputs.append(out.cpu() if isinstance(out, torch.Tensor) else out)
                outputs.append(out)
            grouped_outputs = [tree_flatten(x)[0] for x in outputs]
            out_spec = tree_flatten(outputs[0])[1]
            # convert [[A,b1,c1], [A,b2,c2] [A,b3,c3]] => [A, MultiTensor(b1,b2,b3), MultiTensor(c1,c2,c3)]
            flat_outputs, non_tensors_equal = grouped_to_flat(grouped_outputs)
            assert non_tensors_equal, (
                f"ERR: found a function in model: {func} which "
                + "caused an error in MultiInput, the function dispatch only works for functions"
                + " with Tensor outputs or that have the same non-Tensor output value for all across all inputs"
            )
            return tree_unflatten(flat_outputs, out_spec)

    @classmethod
    def __torch_dispatch__(cls, func, types, args=(), kwargs={}, skip_gptq=False):
        pass

    def __tensor_flatten__(self):
        return ["values"], None

    @classmethod
    def __tensor_unflatten__(
        cls, tensor_data_dict, tensor_attributes, outer_size, outer_stride
    ):
        return cls(tensor_data_dict["values"])


class M(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(64, 128)

    def forward(self, x):
        x = self.linear(x)
        return x


def _is_linear(mod, fqn):
    return isinstance(mod, torch.nn.Linear)


# Adapted from https://github.com/pytorch/ao/pull/581
def prepare_model_for_optimization_(model):
    def forward_pre_hook(
        module,
        args: Tuple[MultiTensor],
        kwargs: Dict[str, Any],
    ):
        # remove the hook to avoid recursive calls
        module._forward_pre_hook_handle.remove()
        # we'll have a single MultiTensor as argument, that contains a list of activation Tensors
        # from previous layer

        # we can use the MultiTensor to calculate the quantization parameters for each input Tensor
        act_obs = AffineQuantizedMinMaxObserver(
            MappingType.ASYMMETRIC,
            torch.uint8,
            granularity=PerTensor(),
            eps=torch.finfo(torch.float32).eps,
            scale_dtype=torch.float32,
            zero_point_dtype=torch.int32,
        )
        for inp in args[0]:
            act_obs(inp)

        input_scale, input_zp = act_obs.calculate_qparams()

        # we can optimize/modify the module here
        module.input_scale = input_scale
        module.input_zp = input_zp

        # rerun the module with quantized and dequantized inputs
        new_input = []
        for inp in args[0]:
            new_input.append(
                fake_quantize_affine(inp, inp.shape, input_scale, input_zp, torch.uint8)
            )

        mt = MultiTensor(new_input)

        # tuple of modified args and kwargs
        return ((mt,), {})

    def _register_forward_pre_hook(module: torch.nn.Module):
        """Adds a forward pre hook for the module, that runs before module.forward is run that can
        modify the module and the input of the module
        docs for `module.register_forward_pre_hook` can be found in https://pytorch.org/docs/stable/generated/torch.nn.Module.html#torch.nn.Module.register_forward_pre_hook
        """
        forward_pre_hook_handle = module.register_forward_pre_hook(
            forward_pre_hook, with_kwargs=True
        )
        module._forward_pre_hook_handle = forward_pre_hook_handle
        return module

    _replace_with_custom_fn_if_matches_filter(
        model, _register_forward_pre_hook, _is_linear
    )


class ApplyActivationStaticWeightQuantConfig(AOBaseConfig):
    pass


# using a function to align with the API in quant_api
@register_quantize_module_handler(ApplyActivationStaticWeightQuantConfig)
def _apply_activation_static_weight_quant_transform(
    module: torch.nn.Module,
    config: ApplyActivationStaticWeightQuantConfig,
):
    observed_linear = module
    target_dtype = torch.uint8

    # we can quantize the weight here as well

    # activation quantization
    act_scale, act_zero_point = (
        observed_linear.input_scale,
        observed_linear.input_zp,
    )
    input_quant_func = lambda x: to_affine_quantized_intx_static(
        x, act_scale, act_zero_point, x.shape, target_dtype
    )
    # for demo purpose only, we quantize the weight here
    weight = observed_linear.weight
    weight = to_affine_quantized_intx(
        weight, MappingType.SYMMETRIC, (1, weight.shape[-1]), torch.int8
    )
    observed_linear.weight = torch.nn.Parameter(
        to_linear_activation_quantized(weight, input_quant_func),
        requires_grad=False,
    )

    del observed_linear.input_scale
    del observed_linear.input_zp
    return observed_linear


example_inputs = (torch.randn(32, 64),)
m = M().eval()
before_quant = m(*example_inputs)
prepare_model_for_optimization_(m)
inputs = []
for _ in range(10):
    inputs.append(torch.randn(32, 64))

mt_input = MultiTensor(inputs)

out = m(mt_input)

# just quantizing activation since we only observed quantization, this could be extended to support
# quantizing weight as well
quantize_(m, ApplyActivationStaticWeightQuantConfig(), _is_linear)
for l in m.modules():
    if isinstance(l, torch.nn.Linear):
        assert isinstance(l.weight, LinearActivationQuantizedTensor)

after_quant = m(*example_inputs)
print("sqnr:", compute_error(before_quant, after_quant))
assert compute_error(before_quant, after_quant) > 35
