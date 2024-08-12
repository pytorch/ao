from dataclasses import dataclass
from typing import Dict, Tuple

import torch
from torch.utils._pytree import tree_flatten, tree_unflatten

import torchao.prototype.autoround.utils as ar_utils
import torchao.quantization as ao_quant
from torchao.dtypes import to_affine_quantized_static

# TODO: remove it before merge
ar_utils.freeze_random()


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
                # TODO: Cann't release the cuda memory, fix it later
                # cur_args = ar_utils.move_data_to_device(cur_args, "cuda")
                # cur_kwargs = ar_utils.move_data_to_device(cur_kwargs, "cuda")
                out = func(*cur_args, **cur_kwargs)
                outputs.append(out.cpu() if isinstance(out, torch.Tensor) else out)
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
        cls(tensor_data_dict["values"])


@dataclass
class AutoRoundConfig:
    bits: int = 4
    sym: bool = False
    iters: int = 200
    group_size: int = 128
    train_bs: int = 8
    eval_bs: int = 4
    seed: int = 42
    amp: bool = False
    nsamples: int = 128
    seqlen: int = 2048


auto_round_config = AutoRoundConfig()


@torch.no_grad()
def create_qmodel_from_qdq_model(qdq_model: torch.nn.Module):
    """Create a quantized model from a qdq model.

    The qdq_model includes Linear quantized by auto-round, which includes qdq weight, scale, zp.
    """

    @torch.no_grad()
    def apply_static_quant(observed_linear):
        device = observed_linear.weight.device
        weight_scale = observed_linear.scale.to(device)
        weight_zero_point = observed_linear.zp.to(device)

        def weight_quant_func(weight):
            block_size = (1, observed_linear.group_size)
            # TODO: shift the zero and prepack the weight to use tinygemm?
            return to_affine_quantized_static(
                input_float=weight,
                scale=weight_scale,
                zero_point=weight_zero_point,
                block_size=block_size,
                target_dtype=torch.uint8,
                quant_min=0,
                quant_max=15,
                zero_point_domain=ao_quant.quant_primitives.ZeroPointDomain.INT,
            )

        observed_linear.weight = torch.nn.Parameter(
            weight_quant_func(observed_linear.weight), requires_grad=False
        )
        del observed_linear.scale
        del observed_linear.zp
        return observed_linear

    def _is_observed_linear(mod: torch.nn.Module, fqn: str):
        return hasattr(mod, "scale")

    qmodel = ao_quant.quant_api._replace_with_custom_fn_if_matches_filter(
        qdq_model, apply_static_quant, _is_observed_linear
    )
    return qmodel


@torch.no_grad()
def apply_auto_round(observed_block, grouped_args, spec, block_outputs):
    # Call the auto-round to execute the optimization process
    import auto_round

    ar_utils.see_memory_usage("Before apply auto-round")

    global auto_round_config

    block = observed_block

    # Start the training process to update the v, alpha and betta.
    rounder = auto_round.AutoRound(
        model=block,
        tokenizer=None,
        sym=auto_round_config.sym,  # Both True and False are OK
        bits=auto_round_config.bits,
        iters=auto_round_config.iters,
        use_quant_input=False,  # disable it for now
        amp=False,
        low_gpu_mem_usage=False,
        model_dtype=next(block.parameters()).dtype,
    )

    @torch.no_grad()
    def _unflatten_grouped_args(grouped_args, spec):
        inputs = []
        for i, inp in enumerate(grouped_args):
            cur_args, cur_kwargs = tree_unflatten(inp, spec)
            inputs.append((cur_args, cur_kwargs))
        return inputs

    block_inputs = _unflatten_grouped_args(grouped_args, spec)
    with torch.enable_grad():
        rounder.quant_block_v2_(
            block, inputs=block_inputs, outputs=block_outputs, device="cuda"
        )
    block.to("cpu")
    qmodel = create_qmodel_from_qdq_model(block)
    ar_utils.see_memory_usage("After apply auto-round.")
    return qmodel


@torch.no_grad()
def optimize_module(
    module: torch.nn.Module,
    args: Tuple[MultiTensor],
    kwargs: Dict[str, MultiTensor],
    output: tuple[MultiTensor],
):
    # Remove the hook before otpimization process to avoid the recursive call
    module._forward_hook_handle_for_autoround.remove()
    flat_args, spec = tree_flatten((args, kwargs))
    grouped_args = MultiTensor.flat_to_grouped(flat_args)
    output_flat_args, output_spec = tree_flatten((output, {}))
    output_grouped_args = MultiTensor.flat_to_grouped(output_flat_args)
    apply_auto_round(module, grouped_args, spec, output_grouped_args)
    torch.cuda.empty_cache()


@torch.no_grad()
def prepare_model_for_applying_auto_round_(model, is_decoder):
    def forward_hook(
        module,
        args: Tuple[MultiTensor],
        kwargs: Dict[str, MultiTensor],
        output: tuple[MultiTensor],
    ):
        optimize_module(module, args, kwargs, output)
        return output

    def _register_forward_hook(module: torch.nn.Module):
        forward_hook_handle = module.register_forward_hook(
            forward_hook, with_kwargs=True
        )
        module._forward_hook_handle_for_autoround = forward_hook_handle
        return module

    ao_quant.quant_api._replace_with_custom_fn_if_matches_filter(
        model, _register_forward_hook, is_decoder
    )
