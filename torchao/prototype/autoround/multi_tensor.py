# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD 3-Clause license found in the
# LICENSE file in the root directory of this source tree.
import dataclasses
from typing import List

import torch
from torch.utils._pytree import tree_flatten, tree_unflatten


@dataclasses.dataclass
class _MultiTensorConfig:
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    ops_to_accelerate: List[str] = dataclasses.field(
        default_factory=lambda: [
            torch.nn.functional.linear,
            torch.matmul,
            torch.bmm,
            torch.nn.functional.scaled_dot_product_attention,
        ]
    )
    offload: bool = False


# Note: As the `MultiTensor` includes a list of tensors, during the calibration stage,
# placing all output tensors on the GPU would consume a significant amount of GPU memory.
# This is especially true for models with a large `lm-head`, such as Llama-3.1.
# In these cases, we load the model onto the DRAM and only transfer tensors to the GPU for compute-intensive operations.
_multi_tensor_config = _MultiTensorConfig()


class MultiTensor(torch.Tensor):
    # Modified from https://gist.github.com/HDCharles/a1b575bbf8875f994af8a01b225e1227
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

    def add_tensors(self, input):
        if isinstance(input, (tuple, list)):
            for inp in input:
                self.add_tensors(inp)
        else:
            assert isinstance(
                input, torch.Tensor
            ), f"MultiTensor can only use add_tensors for Tensors or lists of tensors but got {type(input)}"
            self.count += 1
            self.values.append(input)
        return self

    def get_value(self, i):
        # Instead of copy the last tensor to pad the MultiTensor, we use this function to do fake padding
        # to avoid introducing extra memory usage.
        if i + 1 <= self.count:
            return self.values[i]
        else:
            return self.values[-1]

    @classmethod
    def flat_to_grouped(cls, flat):
        # size of biggest MultiTensor
        multi_tensor_size = max(
            [x.count if isinstance(x, MultiTensor) else 1 for x in flat]
        )

        grouped = []
        for i in range(multi_tensor_size):
            sub_group = []
            for x in flat:
                if isinstance(x, MultiTensor):
                    sub_group.append(x.get_value(i))
                else:
                    sub_group.append(x)
            grouped.append(sub_group)
        return grouped

    @classmethod
    def grouped_to_flat(cls, grouped):
        # convert [[A,b1,c1], [A,b2,c2] [A,b3,c3]] => [A, MultiTensor(b1,b2,b3), MultiTensor(c1,c2,c3)]
        # where A is nontensor, b's,c's are tensors
        # convert [[A,b1,c1], [A,b2,c2] [A,b3,c3]] => [(A,A,A), (b1,b2,b3), (c1,c2,c3)]
        flat_tups = list(zip(*grouped))
        # convert [(A,A,A), (b1,b2,b3), (c1,c2,c3)] => [A, MultiTensor(b1,b2,b3), MultiTensor(c1,c2,c3)]
        flattened = [
            cls(tup) if isinstance(tup[0], torch.Tensor) else tup[0]
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
                if not isinstance(tup[0], torch.Tensor)  # look at tuples of nonTensors
            ]
        )
        return flattened, non_tensors_equal

    @classmethod
    def revert_to_tensor_pairs(cls, args, kwargs=None):
        if kwargs is None:
            kwargs = {}
        # combine args and kwargs and remove lists and tuples
        flat_args, spec = tree_flatten((args, kwargs))
        # convert [A, MultiTensor(b1,b2,b3), MultiTensor(c1,c2,c3)] => [[A,b1,c1], [A,b2,c2] [A,b3,c3]]
        grouped_args = cls.flat_to_grouped(flat_args)
        args_kwargs_pairs = []
        for i, inp in enumerate(grouped_args):
            cur_args, cur_kwargs = tree_unflatten(inp, spec)
            args_kwargs_pairs.append((cur_args, cur_kwargs))
        return args_kwargs_pairs

    @classmethod
    def __torch_function__(cls, func, types, args=(), kwargs=None):
        args_kwargs_pairs = cls.revert_to_tensor_pairs(args, kwargs)
        # run function for each of the multitensors and return a multitensor
        outputs = []
        with torch._C.DisableTorchFunctionSubclass():
            for cur_args, cur_kwargs in args_kwargs_pairs:
                if func in _multi_tensor_config.ops_to_accelerate:
                    device = _multi_tensor_config.device
                    cur_args = [
                        (arg.to(device) if isinstance(arg, torch.Tensor) else arg)
                        for arg in cur_args
                    ]
                    cur_kwargs = {
                        k: (v.to(device) if isinstance(v, torch.Tensor) else v)
                        for k, v in cur_kwargs.items()
                    }
                out = func(*cur_args, **cur_kwargs)
                offload = _multi_tensor_config.offload
                outputs.append(
                    out.to("cpu") if isinstance(out, torch.Tensor) and offload else out
                )
            grouped_outputs = [tree_flatten(x)[0] for x in outputs]
            out_spec = tree_flatten(outputs[0])[1]
            # convert [[A,b1,c1], [A,b2,c2] [A,b3,c3]] => [A, MultiTensor(b1,b2,b3), MultiTensor(c1,c2,c3)]
            flat_outputs, non_tensors_equal = cls.grouped_to_flat(grouped_outputs)
            assert non_tensors_equal, (
                f"ERR: found a function in model: {func} which "
                "caused an error in MultiTensor, the function dispatch only works for functions"
                " with Tensor outputs or that have the same non-Tensor output value for all across all inputs"
            )
            return tree_unflatten(flat_outputs, out_spec)

    @classmethod
    def __torch_dispatch__(cls, func, types, args=(), kwargs={}):
        pass

    def __tensor_flatten__(self):
        return ["values"], None

    @classmethod
    def __tensor_unflatten__(
        cls, tensor_data_dict, tensor_attributes, outer_size, outer_stride
    ):
        return cls(tensor_data_dict["values"])
