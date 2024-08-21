import dataclasses
from typing import List

import torch
from torch.utils._pytree import tree_flatten, tree_unflatten
import os

def _get_accelerator_name():
    if torch.xpu.is_available():
        return "xpu"
    elif torch.cuda.is_available():
        return "cuda"
    else:
        return "cpu"


@dataclasses.dataclass
class MultiTensorConfig:
    accelerator_device: str = _get_accelerator_name()
    offload_device: str = "cpu"
    ops_to_accelerate: List[str] = dataclasses.field(
        default_factory=lambda: [
            "linear",
            "matmul",
            "bmm",
            "scaled_dot_product_attention",
        ]
    )

# TODO(Yi): refine this logic to expose a simple API(one parameter) for modeling users
# If we have enough gpu memory:
# multi_tensor_config_for_large_vram = MultiTensorConfig(accelerator_device="cuda", offload_device="cuda")
# If we have limited gpu memory:
# multi_tensor_config_for_small_vram = MultiTensorConfig(accelerator_device="cuda", offload_device="cpu")
# If we don't have gpu:
# multi_tensor_config_cpu_only = MultiTensorConfig(accelerator_device="cpu", offload_device="cpu")

multi_tensor_config = MultiTensorConfig(accelerator_device="cuda", offload_device="cuda")


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

    def pad_to_length(self, length):
        if self.count > length:
            return self
        self.add_tensors([self.values[-1]] * (length - self.count))
        return self

    @classmethod
    def flat_to_grouped(cls, flat):
        # size of biggest MultiTensor
        multi_tensor_size = max(
            [x.count if isinstance(x, MultiTensor) else 1 for x in flat]
        )
        # convert [A, MultiTensor(b1,b2,b3), MultiTensor(c1,c2,c3)] => [[A,b1,c1], [A,b2,c2] [A,b3,c3]]
        grouped = list(
            zip(
                *[
                    (
                        x.pad_to_length(multi_tensor_size).values
                        if isinstance(x, MultiTensor)
                        else [x] * multi_tensor_size
                    )
                    for x in flat
                ]
            )
        )
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
                        tup[0] == x for x in tup  # check all elements match
                    ]
                )
                for tup in flat_tups
                if not isinstance(tup[0], torch.Tensor)  # look at tuples of nonTensors
            ]
        )
        return flattened, non_tensors_equal

    @classmethod
    def __torch_function__(cls, func, types, args=(), kwargs=None):
        kwargs = {} if kwargs is None else kwargs
        # combine args and kwargs and remove lists and tuples
        flat_args, spec = tree_flatten((args, kwargs))
        # convert [A, MultiTensor(b1,b2,b3), MultiTensor(c1,c2,c3)] => [[A,b1,c1], [A,b2,c2] [A,b3,c3]]
        grouped_args = cls.flat_to_grouped(flat_args)
        # run function for each of the multitensors and return a multitensor
        outputs = []
        with torch._C.DisableTorchFunctionSubclass():
            for i, inp in enumerate(grouped_args):
                cur_args, cur_kwargs = tree_unflatten(inp, spec)
                if any(
                    [op in str(func) for op in multi_tensor_config.ops_to_accelerate]
                ):
                    cur_args = [
                        (
                            arg.to(multi_tensor_config.accelerator_device)
                            if isinstance(arg, torch.Tensor)
                            else arg
                        )
                        for arg in cur_args
                    ]
                    cur_kwargs = {
                        k: (
                            v.to(multi_tensor_config.accelerator_device)
                            if isinstance(v, torch.Tensor)
                            else v
                        )
                        for k, v in cur_kwargs.items()
                    }
                out = func(*cur_args, **cur_kwargs)
                outputs.append(
                    out.to(multi_tensor_config.offload_device)
                    if isinstance(out, torch.Tensor)
                    else out
                )
            grouped_outputs = [tree_flatten(x)[0] for x in outputs]
            out_spec = tree_flatten(outputs[0])[1]
            # convert [[A,b1,c1], [A,b2,c2] [A,b3,c3]] => [A, MultiTensor(b1,b2,b3), MultiTensor(c1,c2,c3)]
            flat_outputs, non_tensors_equal = cls.grouped_to_flat(grouped_outputs)
            assert non_tensors_equal, (
                f"ERR: found a function in model: {func} which "
                + "caused an error in MultiTensor, the function dispatch only works for functions"
                + " with Tensor outputs or that have the same non-Tensor output value for all across all inputs"
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
