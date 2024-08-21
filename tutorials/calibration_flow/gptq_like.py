"""
This is a example flow for GPTQ like calibration flows, where we:
(1) optimize (quantize) one module at a time
(2) with each optimization step, we need to get a set of all calibration data
(3) the output of each module is calculated based on the optimized (quantized) module, and then pass down to the next module

In this tutorial we mainly use two things:
(1) MultiTensor subclass https://gist.github.com/HDCharles/a1b575bbf8875f994af8a01b225e1227
(2) Module forward hooks
"""
import torch
import torch.nn as nn
from torch.utils._pytree import tree_flatten, tree_unflatten
import gc
from typing import Tuple, Dict
from torchao.quantization.utils import compute_error
from torchao.dtypes import to_affine_quantized_static
from torchao.quantization import quantize_
from torchao.quantization import to_linear_activation_quantized
from torchao.quantization import LinearActivationQuantizedTensor
from torchao.quantization.quant_api import _replace_with_custom_fn_if_matches_filter
from torchao.quantization.observer import (
    AffineQuantizedMinMaxObserver,
    PerTensor,
)
from torchao.quantization.quant_primitives import (
    MappingType,
    fake_quantize_affine,
)

class MultiTensor(torch.Tensor):
    @staticmethod
    def __new__(cls, input, **kwargs):
        if isinstance(input, (list, tuple)):
            input = input[0]
        kwargs["dtype"]=kwargs.get("dtype", input.dtype)
        shape = kwargs.pop("shape", input.shape)
        return torch.Tensor._make_wrapper_subclass(cls, shape, **kwargs)

    def __init__(self, input, **kwargs):
        self.values = []
        self.count = 0
        self.add_tensors(input)
        self.debug = True

    def __repr__(self):
        return (
            f"{self.__class__.__name__}(data={self.values})"
        )

    def __iter__(self):
        for v in self.values:
            yield v

    def add_tensors(self, input):
        if isinstance(input, (tuple, list)):
            for inp in input:
                self.add_tensors(inp)
        else:
            assert isinstance(input, torch.Tensor), f"MultiTensor can only use add_tensors for Tensors or lists of tensors but got {type(input)}"
            self.count += 1
            self.values.append(input)
        return self

    def pad_to_length(self, length):
        if self.count > length:
            return self
        self.add_tensors([self.values[-1]]*(length-self.count))
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
                        x.pad_to_length(multi_tensor_size).values if isinstance(x, MultiTensor) else [x] * multi_tensor_size for x in flat]
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
                cls(tup).cpu() if isinstance(tup[0], torch.Tensor) else tup[0] for tup in flat_tups
            ]
            # need to check that getting rid of all but one from each nonTensor tuple is OK
            non_tensors_equal=min([True]+[
                min([True]+[ # handle situation where tuples have size 0
                    tup[0]==x for x in tup # check all elements match
                ]) for tup in flat_tups if not isinstance(tup[0], torch.Tensor) # look at tuples of nonTensors
            ])
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
                +"caused an error in MultiInput, the function dispatch only works for functions"
                +" with Tensor outputs or that have the same non-Tensor output value for all across all inputs"
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

class TwoLinear(torch.nn.Module):
    def __init__(self, in_features=64, out_features=128):
        super().__init__()
        self.linear1 = torch.nn.Linear(in_features, out_features)
        self.linear2 = torch.nn.Linear(in_features, out_features)

    def forward(self, x, y):
        x = self.linear1(x)
        y = self.linear2(y)
        return x + y

class M(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.two_linear1 = TwoLinear()
        self.two_linear2 = TwoLinear(128, 256)

    def forward(self, x, y):
        x1 = self.two_linear1(x, y)
        x2 = self.two_linear2(x1, x1)
        return x2

def _is_two_linear(mod, fqn):
    return isinstance(mod, TwoLinear)

# Adapted from https://github.com/pytorch/ao/pull/581
def prepare_model_for_optimization_(model):
    def forward_hook(
        module,
        args: Tuple[MultiTensor],
        kwargs: Dict[str, MultiTensor],
        output: Tuple[MultiTensor],
    ):
        # remove the hook to avoid recursive calls
        module._forward_hook_handle.remove()
        # in this case args will be a tuple of 2 MultiTensors since
        # the TwoLinear module takes 2 inputs, and each input will collect a list of normal Tensors

        # we can use these two MultiTensors to calculate the quantization parameters for each input Tensor
        act_obs1 = AffineQuantizedMinMaxObserver(MappingType.ASYMMETRIC, torch.uint8, granularity_type=PerTensor(), eps=torch.finfo(torch.float32).eps, scale_dtype=torch.float32, zero_point_dtype=torch.int32)
        for inp in args[0]:
            act_obs1(inp)

        act_obs2 = AffineQuantizedMinMaxObserver(MappingType.ASYMMETRIC, torch.uint8, granularity_type=PerTensor(), eps=torch.finfo(torch.float32).eps, scale_dtype=torch.float32, zero_point_dtype=torch.int32)
        for inp in args[1]:
            act_obs2(inp)

        input_scale1, input_zp1 = act_obs1.calculate_qparams()
        input_scale2, input_zp2 = act_obs2.calculate_qparams()

        # we can optimize/modify the module here
        module.input_scale1 = input_scale1
        module.input_zp1 = input_zp1
        module.input_scale2 = input_scale2
        module.input_zp2 = input_zp2

        # rerun the module with quantized and dequantized inputs
        new_input1 = []
        for inp in args[0]:
            new_input1.append(fake_quantize_affine(inp, inp.shape, input_scale1, input_zp1, torch.uint8))

        new_input2 = []
        for inp in args[1]:
            new_input2.append(fake_quantize_affine(inp, inp.shape, input_scale2, input_zp2, torch.uint8))

        mt1 = MultiTensor(new_input1)
        mt2 = MultiTensor(new_input2)

        output_with_fake_quantized_inputs = module(mt1, mt2)
        # we can return the modified output so it can be consumed by future modules
        return output_with_fake_quantized_inputs

    def _register_forward_hook(module: torch.nn.Module):
        forward_hook_handle = module.register_forward_hook(
            forward_hook, with_kwargs=True
        )
        module._forward_hook_handle = forward_hook_handle
        return module

    _replace_with_custom_fn_if_matches_filter(
        model, _register_forward_hook, _is_two_linear
    )

# using a function to align with the API in quant_api
def apply_activation_static_quant():
    def _apply_activation_static_quant(observed_two_linear):
        target_dtype = torch.uint8

        linear1 = observed_two_linear.linear1
        # activation quantization
        act_scale, act_zero_point = observed_two_linear.input_scale1, observed_two_linear.input_zp1
        input_quant_func1 = lambda x: to_affine_quantized_static(x, act_scale, act_zero_point, x.shape, target_dtype)
        linear1.weight = torch.nn.Parameter(to_linear_activation_quantized(linear1.weight, input_quant_func1), requires_grad=False)

        linear2 = observed_two_linear.linear2
        # activation quantization
        act_scale, act_zero_point = observed_two_linear.input_scale2, observed_two_linear.input_zp2
        input_quant_func2 = lambda x: to_affine_quantized_static(x, act_scale, act_zero_point, x.shape, target_dtype)
        linear2.weight = torch.nn.Parameter(to_linear_activation_quantized(linear2.weight, input_quant_func2), requires_grad=False)
        del observed_two_linear.input_scale1
        del observed_two_linear.input_zp1
        del observed_two_linear.input_scale2
        del observed_two_linear.input_zp2
        return observed_two_linear

    return _apply_activation_static_quant


example_inputs = (torch.randn(32, 64), torch.randn(32, 64),)
m = M().eval()
before_quant = m(*example_inputs)
prepare_model_for_optimization_(m)
input1 = []
input2 = []
for _ in range(10):
    input1.append(torch.randn(32, 64))
    input2.append(torch.randn(32, 64))

mt_input1 = MultiTensor(input1)
mt_input2 = MultiTensor(input2)

out = m(mt_input1, mt_input2)

# just quantizing activation since we only observed quantization, this could be extended to support
# quantizing weight as well
quantize_(m, apply_activation_static_quant(), _is_two_linear)
for l in m.modules():
    if isinstance(l, torch.nn.Linear):
        assert isinstance(l.weight, LinearActivationQuantizedTensor)

after_quant = m(*example_inputs)
assert compute_error(before_quant, after_quant) > 35
