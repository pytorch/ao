# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.


# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import logging
from typing import Any, Callable, Dict, List, Optional, Type

import torch
import torch.fx as fx
import torch.nn as nn
import torch.nn.functional as F
from torch.utils._pytree import tree_flatten, tree_unflatten

from torchao.dtypes.utils import is_device
from torchao.utils import (
    TORCH_VERSION_AT_LEAST_2_3,
    TORCH_VERSION_AT_LEAST_2_6,
    find_multiple,
)

from .quant_primitives import (
    MappingType,
    dequantize_affine,
)
from .unified import Quantizer
from .utils import (
    _MultiInput,
    get_group_qparams_symmetric,
    get_groupwise_affine_qparams,
    group_quantize_tensor_symmetric,
    groupwise_affine_dequantize_tensor_from_qparams,
    groupwise_affine_quantize_tensor,
    groupwise_affine_quantize_tensor_from_qparams,
    pack_tinygemm_scales_and_zeros,
    per_token_dynamic_quant,
)

aten = torch.ops.aten

add_ons = []

if TORCH_VERSION_AT_LEAST_2_3:
    add_ons += ["Int8DynActInt4WeightQuantizer", "Int8DynActInt4WeightGPTQQuantizer"]


__all__ = [
    "Int4WeightOnlyGPTQQuantizer",
    "Int4WeightOnlyQuantizer",
] + add_ons


class GenericGPTQRunner(fx.Interpreter):
    """
    This is a generic GPTQ runner that takes an existing model and applies GPTQ.
    It uses torch._dynamo.export to obtain a graph of the model and then hooks
    into function calls and when it detects a linear, it applies GPTQ to the weight
    given the calibration of inputs passed in at initialization. It puts the results
    into the state_dict so that the quantized model weights/qparams can be loaded
    directly into the model.

    intended to be used in concert with a GPTQQuantizer class to define the quantization mode.
    """

    def __init__(
        self,
        model,
        inputs: _MultiInput,
        blocksize=128,
        percdamp=0.01,
        groupsize=128,
    ):
        self.id_to_name = {
            id(value): name for name, value in dict(model.named_parameters()).items()
        }

        # trace model for one input
        one_input = [multi.values[0].cpu() for multi in inputs]  # pyre-ignore[16]
        # needed for GPTQ on the torchao llama model
        import torchao

        torchao._models.llama.model.use_index_put_for_kv_cache = True
        exported_model = torch._dynamo.export(
            model.cpu(), aten_graph=True, pre_dispatch=True, tracing_mode="fake"
        )(*one_input)
        super().__init__(exported_model.graph_module)

        self.new_state_dict = model.state_dict()

        self.blocksize = blocksize

        self.percdamp = percdamp

        self.groupsize = groupsize
        self.inputs = inputs
        self.gptq_done = False
        self.debug = False

    def configure_quantization_mode(
        self,
        get_qparams_func,
        quantize_func,
        dequantize_func,
        combine_qparams_list_func,
        make_names_and_values_dict_func,
        skip_layer_func,
        act_fake_quant_func=None,
    ):
        # these functions need to already be curried with all inputs other than weight, qparams

        self.get_qparams_func = (
            get_qparams_func  # accepts [2d weight tensor], outputs qparams.
        )

        self.quantize_func = quantize_func  # accepts [2d weight tensor], [qparams], outputs a 2d quantized tensor of desired dtype

        self.dequantize_func = dequantize_func
        # accepts [quantized] tensor and [qparams], outputs a 2d dequantized tensor of type float,
        # assumes this output .to(w_orig_dtype) is ~eventual desired dequant behavior

        #  `combine_qparams_list_func`.
        self.combine_qparams_list_func = combine_qparams_list_func
        # accepts [`list` of qparams] from quantizing one group at a time,
        # outputs a qparams object that could be passed into quant/dequantize_func

        self.skip_layer_func = skip_layer_func  # accepts [weight tensor], outputs a bool on whether or not to apply gptq to this layer

        #  `make_names_and_values_dict_func`.
        self.make_names_and_values_dict_func = make_names_and_values_dict_func  # accepts [2d quantized tensor], [qparams], returns a dict of names, values to put in state_dict
        # note any final packing for storage should happen here

        # `act_fake_quant_func`
        if act_fake_quant_func is None:
            self.act_fake_quant_func = lambda x: x
        else:
            self.act_fake_quant_func = act_fake_quant_func  # accepts [activation tensor], returns a fake-quantized activation tensor
        return self

    def run(self):
        assert (
            self.get_qparams_func is not None
        ), "need to configure quantization mode before running"
        self.gptq_done = True
        super().run(*self.inputs)

    def get_quantized_state_dict(self):
        assert (
            self.gptq_done
        ), "need to run GPTQRunner before you can get_quantized_state_dict"
        quantized_state_dict = self.new_state_dict
        # Don't want to store/load the kv_cache so remove it from the state_dict
        del_list = []
        for param_fqn in quantized_state_dict:
            if "kv_cache" in param_fqn:
                del_list.append(param_fqn)
        for param_fqn in del_list:
            quantized_state_dict.pop(param_fqn)
        return quantized_state_dict

    def call_function(self, target, args, kwargs, already_quantized=False):  # noqa: C901
        def tensors_to_cuda(args):
            new_args = []
            for x in args:
                new_args.append(x.cuda() if isinstance(x, torch.Tensor) else x)
            return new_args

        # flatten args and kwargs together
        flat_args, spec = tree_flatten((args, kwargs))
        # move all single tensors to cuda, will move _MultiInputs to cuda one at a time
        flat_args = tensors_to_cuda(flat_args)

        has_multi_input = _MultiInput in [type(x) for x in flat_args]
        if has_multi_input:
            # Just some trickery to convert
            # [_MultiInput[a, a, a], _MultiInput(b, b, b)] => [a, b], [a, b], [a, b]
            multi_input_count = max(
                [len(x.values) if isinstance(x, _MultiInput) else 1 for x in flat_args]
            )
            transposed_args = list(
                zip(
                    *[
                        (
                            x.values
                            if isinstance(x, _MultiInput)
                            else [x] * multi_input_count
                        )
                        for x in flat_args
                    ]
                )
            )
        else:
            transposed_args = [flat_args]
        outputs = []

        # check whether we apply GPTQ to this module
        quantize_linear = (
            (target == aten.linear.default)  # if its a linear
            and id(args[1]) in self.id_to_name  # and if we know the layer name
            # and we haven't already quantized this layer
            and not already_quantized
            # and if the skip_layer_func doesn't say we should skip
            and not (self.skip_layer_func is not None and self.skip_layer_func(args[1]))
        )  # then we will quantize this linear layer/weight

        if quantize_linear:  # instantiate variables for GPTQ
            H = 0
            total_batches = 0

        for inp in transposed_args:
            inp = tensors_to_cuda(inp)
            cur_args, cur_kwargs = tree_unflatten(inp, spec)

            if quantize_linear:  # calculate H instead of output (will run the linear eventually with updated weight)
                x = cur_args[0].float()
                x = self.act_fake_quant_func(x)
                shape = x.shape
                n = 1 if len(shape) == 2 else shape[0]
                H *= total_batches / (total_batches + n)
                total_batches += n
                x = ((2 / total_batches) ** (1 / 2)) * x.reshape(
                    -1, shape[-1]
                ).t().float()
                H += x.matmul(x.t())
            else:
                # weight has already been quantized but still need to apply
                # activation quant for final calculation
                if already_quantized:
                    cur_args = (self.act_fake_quant_func(cur_args[0]), *cur_args[1:])

                # get output if its not a linear
                out = super().call_function(target, cur_args, cur_kwargs)
                if isinstance(out, torch.Tensor):
                    outputs.append(out.cpu())
                else:
                    outputs.append(out)

        if quantize_linear:
            mod_fqn = ".".join(self.id_to_name[id(args[1])].split(".")[:-1])

            W = args[1].to(H.device)

            Q, DQ, qparams = self.faster_quant(H, W.detach())
            print(mod_fqn)

            #  `make_names_and_values_dict_func`.
            names_and_values_dict = self.make_names_and_values_dict_func(Q, qparams)

            # delete old weight
            if mod_fqn + ".weight" in self.new_state_dict:
                self.new_state_dict.pop(mod_fqn + ".weight")
            if len(args) > 2:
                self.new_state_dict[mod_fqn + ".bias"] = args[2]
            for name, value in names_and_values_dict.items():
                self.new_state_dict[mod_fqn + "." + name] = value

            # run linear with new weight to get corrected output
            new_out = self.call_function(
                target, (args[0], DQ, *args[2:]), kwargs, already_quantized=True
            )

            if self.debug:
                old_out = self.call_function(
                    target,
                    (args[0][:2], args[1], *args[2:]),
                    kwargs,
                    already_quantized=True,
                )

                def SQNR(x, y):
                    # TODO: Use of deprecated function torch.norm
                    return 20 * torch.log10(
                        torch.linalg.norm(x) / torch.linalg.norm(x - y)
                    )

                #  `dequantize_func`.
                DQ_after = self.dequantize_func(Q, qparams).to(W.dtype)
                print(
                    "SQNR for QDQ (this should be inf)", SQNR(DQ, DQ_after)
                )  # matches
                print(
                    "SQNR for weight (can be low)", SQNR(W, DQ.cuda())
                )  # fine to not match
                print(
                    "SQNR for output with GPTQ (hopefully 35+)",
                    torch.cat(
                        [
                            SQNR(old.cpu(), new.cpu()).unsqueeze(0)
                            for (old, new) in zip(old_out.values, new_out.values[:2])
                        ]
                    ).mean(),
                )

                #  `get_qparams_func`.
                qparams2 = self.get_qparams_func(W)

                Q2 = self.quantize_func(W, qparams2)
                DQ2 = self.dequantize_func(Q2, qparams2).to(W.dtype)
                old_q_out = self.call_function(
                    target,
                    (args[0][:2], DQ2, *args[2:]),
                    kwargs,
                    already_quantized=True,
                )

                print(
                    "SQNR for output without GPTQ (should be less than above)",
                    torch.cat(
                        [
                            SQNR(old.cpu(), old_q.cpu()).unsqueeze(0)
                            for (old, old_q) in zip(old_out.values, old_q_out.values)
                        ]
                    ).mean(),
                )
            return new_out

        return _MultiInput(outputs) if has_multi_input else outputs[0]

    def faster_quant(self, H, W):
        percdamp = self.percdamp
        blocksize = self.blocksize
        groupsize = self.groupsize
        orig_dtype = W.dtype
        W = W.detach().float()
        _, columns = W.shape[0], W.shape[1]
        device = W.device

        if groupsize == -1:
            cur_qparams = self.get_qparams_func(W)
        dead = torch.diag(H) == 0
        H[dead, dead] = 1
        W[:, dead] = 0

        Losses = torch.zeros_like(W)
        DQ = torch.zeros_like(W)

        damp = percdamp * torch.mean(torch.diag(H))
        diag = torch.arange(columns, device=device)
        H[diag, diag] += damp
        H = torch.linalg.cholesky(H)
        H = torch.cholesky_inverse(H)
        H = torch.linalg.cholesky(H, upper=True)
        Hinv = H

        all_qparams = []
        for i1 in range(0, columns, blocksize):
            i2 = min(i1 + blocksize, columns)
            count = i2 - i1
            W1 = W[:, i1:i2].clone()
            DQ1 = torch.zeros_like(W1)
            Err1 = torch.zeros_like(W1)
            Losses1 = torch.zeros_like(W1)
            Hinv1 = Hinv[i1:i2, i1:i2]
            for i in range(count):
                w = W1[:, i]
                d = Hinv1[i, i]

                if groupsize != -1 and (i1 + i) % groupsize == 0:  # start of new group
                    cur_qparams = self.get_qparams_func(
                        W[:, (i1 + i) : (i1 + i + groupsize)]
                    )
                    all_qparams.append(cur_qparams)

                q = self.quantize_func(w.unsqueeze(1), cur_qparams).flatten()

                #  `dequantize_func`.

                dq = self.dequantize_func(q.unsqueeze(1), cur_qparams).flatten()

                DQ1[:, i] = dq
                Losses1[:, i] = (w - dq) ** 2 / d**2

                err1 = (w - dq) / d
                W1[:, i:] -= (
                    err1.to(Hinv1.dtype).unsqueeze(1).matmul(Hinv1[i, i:].unsqueeze(0))
                )
                Err1[:, i] = err1

            DQ[:, i1:i2] = DQ1
            Losses[:, i1:i2] = Losses1 / 2

            W[:, i2:] -= Err1.to(Hinv.dtype).matmul(Hinv[i1:i2, i2:])

        torch.cuda.synchronize()

        if all_qparams == []:
            all_qparams.append(cur_qparams)

        # convert a list of qparams objects into a single one. enerally by
        # concatenating a bunch of n,1 scale/zeros tensors into a n,num_groups tensor

        #  `combine_qparams_list_func`.
        all_qparams = self.combine_qparams_list_func(all_qparams)
        Q = self.quantize_func(DQ, all_qparams)
        return Q, DQ.to(orig_dtype), all_qparams


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

    act_fake_quant_func (optional):
            A function that (dynamically) quantizes activation to input
            Args:
                input: input Tensor in f32/bf16/f16
            Returns:
                output: dynamically quantized and dequantized Tensor (with the same dtype as input)

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

        #  `make_names_and_values_dict_func`.
        assert self.make_names_and_values_dict_func is not None

    @torch.no_grad()
    def _create_quantized_state_dict(
        self,
        model,
        inputs,
        blocksize,
        percdamp,
        groupsize,
        #  `typing.Dict[<key type>, <value type>]` to avoid runtime subscripting errors.
    ) -> Dict:
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
            self.act_fake_quant_func
            if hasattr(self, "act_fake_quant_func")
            else None,  # pyre-ignore[16]
        )
        print("Applying GPTQ to weights")
        GPTQ_runner.run()
        return GPTQ_runner.get_quantized_state_dict()

    def _convert_for_runtime(self, model: torch.nn.Module) -> "nn.Module":
        raise NotImplementedError("_convert_for_runtime not implemented")

    @torch.no_grad()
    def quantize(
        self, model: torch.nn.Module, inputs: List[_MultiInput], **kwargs: Any
    ) -> torch.nn.Module:
        pass


def _check_linear_int4_k(k, groupsize=1, inner_k_tiles=None):
    k_divisible_by_groupsize = k % groupsize == 0
    if inner_k_tiles is not None:
        k_divisible_by_16_times_inner_k_tiles = k % (inner_k_tiles * 16) == 0
        return k_divisible_by_groupsize and k_divisible_by_16_times_inner_k_tiles
    return k_divisible_by_groupsize


def linear_forward_int4(
    x: torch.Tensor,
    weight_int4pack: torch.Tensor,
    scales_and_zeros: torch.Tensor,
    out_features: int,
    groupsize: int,
    precision: torch.dtype = torch.bfloat16,
    scales_precision: torch.dtype = torch.bfloat16,
):
    origin_x_size = x.size()
    x = x.reshape(-1, origin_x_size[-1])
    if is_device(x.device.type, "cpu") and TORCH_VERSION_AT_LEAST_2_6:
        c = torch.ops.aten._weight_int4pack_mm_for_cpu(
            x.to(precision),
            weight_int4pack,
            groupsize,
            scales_and_zeros.to(scales_precision),
        ).to(dtype=x.dtype)
    else:
        c = torch.ops.aten._weight_int4pack_mm(
            x.to(precision),
            weight_int4pack,
            groupsize,
            scales_and_zeros.to(scales_precision),
        ).to(dtype=x.dtype)
    new_shape = origin_x_size[:-1] + (out_features,)
    c = c.reshape(new_shape)
    return c


class WeightOnlyInt4Linear(torch.nn.Module):
    __constants__ = ["in_features", "out_features"]
    in_features: int
    out_features: int
    weight: torch.Tensor

    def __init__(
        self,
        in_features: int,
        out_features: int,
        # TODO: remove dtype field, not used
        bias=False,
        device=None,
        dtype=None,
        groupsize: int = 128,
        inner_k_tiles: int = 8,
        precision: torch.dtype = torch.bfloat16,
        scales_precision: torch.dtype = torch.bfloat16,
    ) -> None:
        super().__init__()
        self.padding = not _check_linear_int4_k(in_features, groupsize, inner_k_tiles)
        if self.padding:
            self.origin_in_features = in_features
            in_features = find_multiple(in_features, 1024)

        self.in_features = in_features
        self.out_features = out_features
        assert not bias, "require bias=False"
        self.device = device
        self.groupsize = groupsize
        self.inner_k_tiles = inner_k_tiles
        self.precision = precision
        self.scales_precision = scales_precision

        if dtype is not None:
            raise ValueError("Please specify 'precision' instead of 'dtype'")

        assert out_features % 8 == 0, "require out_features % 8 == 0"
        assert (
            in_features % (inner_k_tiles * 16) == 0
        ), "require in_features % (innerKTiles * 16) == 0"
        if is_device(device.type, "cpu"):
            self.register_buffer(
                "weight",
                torch.zeros(
                    (
                        out_features,
                        in_features // 2,
                    ),
                    dtype=torch.uint8,
                    device=device,
                ),
            )
        else:
            self.register_buffer(
                "weight",
                torch.zeros(
                    (
                        out_features // 8,
                        in_features // (inner_k_tiles * 16),
                        32,
                        inner_k_tiles // 2,
                    ),
                    dtype=torch.int32,
                    device=device,
                ),
            )
        self.dtype = dtype
        self.register_buffer(
            "scales_and_zeros",
            torch.zeros(
                (in_features // groupsize, out_features, 2),
                dtype=self.scales_precision,
                device=device,
            ),
        )

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        if self.padding:
            input = F.pad(input, pad=(0, self.in_features - self.origin_in_features))
        return linear_forward_int4(
            input,
            self.weight,
            self.scales_and_zeros,
            self.out_features,
            self.groupsize,
            self.precision,
            self.scales_precision,
        )


def _replace_linear_int4(
    module: torch.nn.Module,
    groupsize: int,
    inner_k_tiles: Optional[int],
    padding_allowed: bool,
    skip_layer_func: Optional[Callable] = None,
    precision: torch.dtype = torch.bfloat16,
    scales_precision: torch.dtype = torch.bfloat16,
    linear_class: Type[torch.nn.Module] = WeightOnlyInt4Linear,
    copy_weights: bool = False,
):
    for name, child in module.named_children():
        # TODO: support linear bias
        if (
            isinstance(child, nn.Linear)
            and child.bias is None
            and (skip_layer_func is None or not skip_layer_func(child.weight))
        ):
            if (
                _check_linear_int4_k(child.in_features, groupsize, inner_k_tiles)
                or padding_allowed
            ):
                new_linear = linear_class(
                    child.in_features,
                    child.out_features,
                    bias=False,
                    device=child.weight.device,
                    groupsize=groupsize,
                    inner_k_tiles=inner_k_tiles,
                    precision=precision,
                    scales_precision=scales_precision,
                )
                # TODO: merge with 8da4w?
                # In distributed training, the model may be instantiated
                # on the meta device, in which case there is no need to
                # copy the weights, and doing so will result in an error
                if copy_weights and child.weight.device != torch.device("meta"):
                    new_linear.weight = child.weight
                setattr(module, name, new_linear)
        else:
            _replace_linear_int4(
                child,
                groupsize,
                inner_k_tiles,
                padding_allowed,
                skip_layer_func,
                precision,
                scales_precision,
                linear_class,
                copy_weights,
            )


def replace_linear_int4(
    module, groupsize, inner_k_tiles, padding_allowed, skip_layer_func=None
):
    _replace_linear_int4(
        module,
        groupsize,
        inner_k_tiles,
        padding_allowed,
        skip_layer_func,
        linear_class=WeightOnlyInt4Linear,
    )


class Int4WeightOnlyQuantizer(Quantizer):
    def __init__(
        self,
        groupsize: int = 256,
        padding_allowed: bool = True,
        inner_k_tiles: Optional[int] = 8,
        device: torch.device = torch.device("cuda"),
        precision: torch.dtype = torch.bfloat16,
    ) -> None:
        super().__init__()
        assert inner_k_tiles in [2, 4, 8]
        assert groupsize in [32, 64, 128, 256]

        self.inner_k_tiles = inner_k_tiles
        self.groupsize: int = groupsize
        self.padding_allowed: bool = padding_allowed
        self.device: torch.device = device
        # precision and dtype are being used interchangeably here
        self.precision: torch.dtype = precision

    @torch.no_grad()
    def _create_quantized_state_dict(
        self, model: torch.nn.Module
    ) -> Dict[str, torch.Tensor]:
        cur_state_dict = model.state_dict()
        for fqn, mod in model.named_modules():
            if isinstance(mod, torch.nn.Linear) and mod.bias is None:
                out_features = mod.out_features
                in_features = mod.in_features
                # assert out_features % 8 == 0, "require out_features % 8 == 0"
                logging.info(f"linear: {fqn}, in={in_features}, out={out_features}")

                assert (
                    in_features % self.groupsize == 0
                ), f"require in_features:{in_features} % self.groupsize:{self.groupsize} == 0"

                weight = mod.weight.data
                if not _check_linear_int4_k(
                    in_features, self.groupsize, self.inner_k_tiles
                ):
                    if self.padding_allowed:
                        import torch.nn.functional as F

                        logging.warning(
                            f"warning: {fqn} is padded to satisfy in_features % 1024 == 0"
                        )
                        padded_in_features = find_multiple(in_features, 1024)
                        weight = F.pad(
                            weight, pad=(0, padded_in_features - in_features)
                        )
                    else:
                        logging.warning(
                            f"warning: {fqn} is skipped, int4 requires that in_features is 32, 64, or is divisible by 1024, "
                            + "and that groupsize and inner_k_tiles*16 evenly divide into it"
                        )
                        continue
                (w_int4x8, scales_and_zeros) = groupwise_affine_quantize_tensor(
                    weight,
                    4,  # n_bit
                    self.groupsize,
                    self.precision,  # dtype for scales_and_zeros
                )
                # TODO: just get the device from mod.weight.device?
                if (
                    is_device(w_int4x8.device.type, "cpu")
                    and TORCH_VERSION_AT_LEAST_2_6
                ):
                    weight_int4pack = (
                        torch.ops.aten._convert_weight_to_int4pack_for_cpu(
                            w_int4x8.to(self.device), self.inner_k_tiles
                        )
                    )
                else:
                    weight_int4pack = torch.ops.aten._convert_weight_to_int4pack(
                        w_int4x8.to(self.device), self.inner_k_tiles
                    )
                cur_state_dict[f"{fqn}.weight"] = weight_int4pack.to(self.device)
                cur_state_dict[f"{fqn}.scales_and_zeros"] = scales_and_zeros.to(
                    self.device
                )
        return cur_state_dict

    def _convert_for_runtime(self, model: torch.nn.Module) -> torch.nn.Module:
        _replace_linear_int4(
            model,
            self.groupsize,
            self.inner_k_tiles,
            self.padding_allowed,
            skip_layer_func=None,
            precision=self.precision,
            scales_precision=self.precision,
        )
        return model

    def quantize(
        self, model: torch.nn.Module, *args: Any, **kwargs: Any
    ) -> torch.nn.Module:
        state_dict = self._create_quantized_state_dict(model)
        model = self._convert_for_runtime(model)
        # TODO: make it strict
        model.load_state_dict(state_dict, strict=False)
        return model


class Int4WeightOnlyGPTQQuantizer(GPTQQuantizer):
    def __init__(
        self,
        blocksize=128,
        percdamp=0.01,
        groupsize=64,
        inner_k_tiles=8,
        padding_allowed=True,
        device: torch.device = torch.device("cuda"),
    ):
        self.blocksize = blocksize
        self.percdamp = percdamp
        self.groupsize = groupsize
        self.inner_k_tiles = inner_k_tiles
        self.padding_allowed = padding_allowed
        self.device = device
        self.act_fake_quant_func = None
        n_bit = 4
        self.get_qparams_func = lambda w: get_groupwise_affine_qparams(
            w, n_bit, groupsize
        )
        self.quantize_func = (
            lambda w, qparams: groupwise_affine_quantize_tensor_from_qparams(
                w, qparams[0], qparams[1], n_bit, groupsize
            )
        )
        self.dequantize_func = (
            lambda q, qparams: groupwise_affine_dequantize_tensor_from_qparams(
                q,
                qparams[0],
                qparams[1],
                n_bit,
                groupsize,
            )
        )
        self.combine_qparams_list_func = lambda qparams_list: [
            torch.cat(x, dim=1) for x in zip(*qparams_list)
        ]
        # skip unless padding_allowed=True or its correctly sized
        self.skip_layer_func = lambda linear_weight: not (
            _check_linear_int4_k(linear_weight.shape[-1], groupsize) or padding_allowed
        )

        # we need to do the padding here, both for q and the qparams if necessary

        # TODO: this is the gpt-fast version, merge with the main version later
        def make_names_and_values_dict_func(q, qparams):
            k = q.shape[1] * 2
            if not _check_linear_int4_k(k, groupsize):
                new_k = find_multiple(k, 1024)
            else:
                new_k = k
            # how much we need to pad the weight
            delta_k = int((new_k - k) / 2)
            q = q.to(self.device)
            if is_device(self.device.type, "cpu") and TORCH_VERSION_AT_LEAST_2_6:
                final_q = torch.ops.aten._convert_weight_to_int4pack_for_cpu(
                    F.pad(q, pad=(0, delta_k)), inner_k_tiles
                )
            else:
                final_q = torch.ops.aten._convert_weight_to_int4pack(
                    F.pad(q, pad=(0, delta_k)), inner_k_tiles
                )
            scales = qparams[0].to(torch.bfloat16).to(self.device)
            zeros = qparams[1].to(torch.bfloat16).to(self.device)
            scales_and_zeros = pack_tinygemm_scales_and_zeros(scales, zeros)
            # how many new groups we need for padded weight
            delta_groups = new_k // groupsize - scales_and_zeros.shape[0]
            final_s_and_z = F.pad(
                scales_and_zeros, pad=(0, 0, 0, 0, 0, delta_groups), value=1
            )
            return {"weight": final_q, "scales_and_zeros": final_s_and_z}

        self.make_names_and_values_dict_func = make_names_and_values_dict_func
        super().__init__()

    def _convert_for_runtime(self, model):
        replace_linear_int4(
            model,
            self.groupsize,
            self.inner_k_tiles,
            self.padding_allowed,
            skip_layer_func=self.skip_layer_func,
        )
        return model

    def quantize(
        self, model: torch.nn.Module, inputs: List[_MultiInput], **kwargs: Any
    ) -> torch.nn.Module:
        state_dict = self._create_quantized_state_dict(
            model,
            inputs,
            self.blocksize,
            self.percdamp,
            self.groupsize,
        )
        model = self._convert_for_runtime(model)
        model.load_state_dict(state_dict, strict=False)
        return model


def linear_forward_8da4w(
    x,
    weight_int8,
    bias,
    scales,
    zeros,
    out_features,
    groupsize,
    precision,
):
    x = per_token_dynamic_quant(x)
    # TODO: verify and remove following reshape code
    # origin_x_size = x.size()
    # x = x.reshape(-1, origin_x_size[-1])

    # TODO: better API
    # weight_int8 = torch.ops.quantized_decomposed.unpack_int4_to_int8(weight_int4packed)
    n_bit = 4
    quant_min = -(2 ** (n_bit - 1))
    quant_max = 2 ** (n_bit - 1) - 1
    block_size = (1, groupsize)

    w_dq = dequantize_affine(
        weight_int8,
        block_size,
        scales,
        zeros,
        torch.int8,
        quant_min,
        quant_max,
        output_dtype=precision,
    )

    # x = x.to(torch.float16)
    # w_dq = w_dq.to(torch.float16)
    c = torch.nn.functional.linear(x, w_dq, bias)

    # new_shape = origin_x_size[:-1] + (out_features,)
    # c = c.reshape(new_shape)

    return c


class Int8DynActInt4WeightLinear(torch.nn.Module):
    __constants__ = ["in_features", "out_features"]

    in_features: int
    out_features: int
    weight: torch.Tensor
    bias: torch.Tensor

    """
    This module implements a dynamic quantized linear layer with int4 weight.
    Weights are per channel groupwise quantized. Parameters of importance
    groupsize: the number of elements in each quantized group
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
        # TODO: remove this field, not used
        dtype=None,
        groupsize: int = 256,
        precision: torch.dtype = torch.float32,
        scales_precision: torch.dtype = torch.float32,
    ) -> None:
        super().__init__()
        # always pad if needed since it becomes a noop at runtime if not needed
        # self.origin_in_features = in_features
        assert (
            in_features % groupsize == 0
        ), f"require in_features:{in_features} % groupsize:{groupsize} == 0"
        # in_features = _calc_padded_size_linear_int4(
        #    in_features, groupsize
        # )
        self.in_features = in_features
        self.out_features = out_features
        # TODO: align groupsize naming
        self.groupsize = groupsize
        # Precision of the activation which also indicates
        # output precision of the dynamically quantized linear layer
        # that his module represents.
        self.precision = precision

        if dtype is not None:
            raise ValueError("Please specify 'precision' instead of 'dtype'")

        # currently storing unpacked int8 weights
        self.register_buffer(
            "weight",
            torch.zeros((out_features, in_features), dtype=torch.int8),
        )
        self.register_buffer(
            "scales",
            torch.zeros(
                (out_features, in_features // groupsize),
                dtype=scales_precision,
            ),
        )
        self.register_buffer(
            "zeros",
            torch.zeros(
                (out_features, in_features // groupsize),
                dtype=scales_precision,
            ),
        )

        if bias:
            self.register_buffer("bias", torch.zeros(out_features, dtype=precision))
        else:
            self.bias = None

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        input = input.to(self.precision)
        # padding is removed for perf
        # input = F.pad(input, pad=(0, self.in_features - self.origin_in_features))
        return linear_forward_8da4w(
            input,
            self.weight,
            self.bias,
            self.scales,
            self.zeros,
            self.out_features,
            self.groupsize,
            self.precision,
        )


def _replace_linear_8da4w(
    module: torch.nn.Module,
    groupsize: int,
    padding_allowed: bool,
    precision: torch.dtype,
    scales_precision: torch.dtype,
    linear_class: Type[torch.nn.Module],
    copy_weights: bool = False,
):
    # import the util function here to avoid circular dependency
    from torchao.quantization.quant_api import _replace_with_custom_fn_if_matches_filter

    def filter_fn(child: torch.nn.Module, cur_fqn: str) -> bool:
        return isinstance(child, nn.Linear) and (
            _check_linear_int4_k(child.in_features, groupsize) or padding_allowed
        )

    def replacement_fn(child: torch.nn.Module) -> torch.nn.Module:
        new_linear = linear_class(
            child.in_features,
            child.out_features,
            bias=child.bias is not None,
            device=child.weight.device,
            groupsize=groupsize,
            precision=precision,
            scales_precision=scales_precision,
        )
        # In distributed training, the model may be instantiated
        # on the meta device, in which case there is no need to
        # copy the weights, and doing so will result in an error
        if copy_weights and child.weight.device != torch.device("meta"):
            new_linear.weight = child.weight
            new_linear.bias = child.bias
        return new_linear

    _replace_with_custom_fn_if_matches_filter(module, replacement_fn, filter_fn)


def replace_linear_8da4w(
    module: torch.nn.Module,
    groupsize: int,
    padding_allowed: bool,
    precision: torch.dtype,
    scales_precision: torch.dtype,
):
    _replace_linear_8da4w(
        module,
        groupsize,
        padding_allowed,
        precision,
        scales_precision,
        Int8DynActInt4WeightLinear,
    )


class Int8DynActInt4WeightQuantizer(Quantizer):
    def __init__(
        self,
        groupsize: int = 256,
        padding_allowed: bool = False,
        precision: torch.dtype = torch.float32,
        scales_precision: torch.dtype = torch.float32,
        device: torch.device = torch.device("cpu"),
        mapping_type: MappingType = MappingType.SYMMETRIC,
    ) -> None:
        super().__init__()
        self.groupsize: int = groupsize
        self.padding_allowed: bool = padding_allowed
        self.precision: torch.dtype = precision
        self.scales_precision: torch.dtype = scales_precision
        self.device: torch.device = device
        self.mapping_type: MappingType = mapping_type

    @torch.no_grad()
    def _create_quantized_state_dict(
        self, model: torch.nn.Module
    ) -> Dict[str, torch.Tensor]:
        cur_state_dict = model.state_dict()
        for fqn, mod in model.named_modules():
            if isinstance(mod, torch.nn.Linear):
                out_features = mod.out_features
                in_features = mod.in_features
                # assert out_features % 8 == 0, "require out_features % 8 == 0"
                logging.info(f"linear: {fqn}, in={in_features}, out={out_features}")

                assert (
                    in_features % self.groupsize == 0
                ), f"require in_features:{in_features} % self.groupsize:{self.groupsize} == 0"

                weight = mod.weight.data
                if not _check_linear_int4_k(in_features, self.groupsize):
                    if self.padding_allowed:
                        import torch.nn.functional as F

                        logging.warning(
                            f"warning: {fqn} is padded to satisfy in_features % 1024 == 0"
                        )
                        padded_in_features = find_multiple(in_features, 1024)
                        weight = F.pad(
                            weight, pad=(0, padded_in_features - in_features)
                        )
                    else:
                        logging.warning(
                            f"warning: {fqn} is skipped, int4 requires that in_features is 32, 64, or is divisible by 1024, "
                            + "and that groupsize and inner_k_tiles*16 evenly divide into it"
                        )
                        continue
                (
                    weight_int8,
                    scales,
                    zeros,
                ) = group_quantize_tensor_symmetric(
                    weight.to(self.precision),
                    4,  # n_bit
                    self.groupsize,
                    self.scales_precision,
                    mapping_type=self.mapping_type,
                )
                cur_state_dict[f"{fqn}.weight"] = weight_int8.to(self.device)
                cur_state_dict[f"{fqn}.scales"] = scales.to(self.device)
                cur_state_dict[f"{fqn}.zeros"] = zeros.to(self.device)

        return cur_state_dict

    def _convert_for_runtime(self, model: torch.nn.Module) -> torch.nn.Module:
        replace_linear_8da4w(
            model,
            self.groupsize,
            self.padding_allowed,
            self.precision,
            # TODO: this should be self.scales_precision?
            self.precision,
        )
        return model

    def quantize(
        self, model: torch.nn.Module, *args: Any, **kwargs: Any
    ) -> torch.nn.Module:
        state_dict = self._create_quantized_state_dict(model)
        model = self._convert_for_runtime(model)
        # TODO: make it strict
        model.load_state_dict(state_dict, strict=False)
        return model


class Int8DynActInt4WeightGPTQQuantizer(GPTQQuantizer):
    def __init__(
        self,
        blocksize=128,
        percdamp=0.01,
        groupsize=64,
        inner_k_tiles=8,
        padding_allowed=True,
        precision=torch.float32,
    ):
        self.blocksize = blocksize
        self.percdamp = percdamp
        self.groupsize = groupsize
        self.inner_k_tiles = inner_k_tiles
        self.padding_allowed = padding_allowed
        self.precision = precision

        self.act_fake_quant_func = per_token_dynamic_quant
        n_bit = 4
        self.get_qparams_func = lambda w: get_group_qparams_symmetric(
            w, n_bit, groupsize, self.precision
        )
        quant_min = -(2 ** (n_bit - 1))
        quant_max = 2 ** (n_bit - 1) - 1

        from torchao._executorch_ops import (
            _quantized_decomposed_quantize_per_channel_group_wrapper,
        )

        self.quantize_func = (
            lambda w, qparams: _quantized_decomposed_quantize_per_channel_group_wrapper(
                w, qparams[0], qparams[1], quant_min, quant_max, torch.int8, groupsize
            )
        )

        from torchao._executorch_ops import (
            _quantized_decomposed_dequantize_per_channel_group_wrapper,
        )

        self.dequantize_func = (
            lambda q,
            qparams: _quantized_decomposed_dequantize_per_channel_group_wrapper(
                q,
                qparams[0],
                qparams[1],
                quant_min,
                quant_max,
                torch.int8,
                groupsize,
                self.precision,
            )
        )

        self.combine_qparams_list_func = lambda qparams_list: [
            torch.cat(x, dim=1) for x in zip(*qparams_list)
        ]
        # skip unless padding_allowed=True or its correctly sized

        self.skip_layer_func = lambda linear_weight: not (
            _check_linear_int4_k(linear_weight.shape[-1], groupsize) or padding_allowed
        )

        # we need to do the padding here, both for q and the qparams if necessary
        def make_names_and_values_dict_func(q, qparams):
            k = q.shape[1]
            new_k = find_multiple(k, 1 if groupsize is None else groupsize)
            # how much we need to pad the weight
            delta_k = new_k - q.shape[1]
            final_q = F.pad(q, pad=(0, delta_k))
            scales = qparams[0].to(self.precision)
            zeros = qparams[1].to(self.precision)
            return {"weight": final_q, "scales": scales, "zeros": zeros}

        self.make_names_and_values_dict_func = make_names_and_values_dict_func
        super().__init__()

    def _convert_for_runtime(self, model):
        replace_linear_8da4w(
            model,
            self.groupsize,
            self.padding_allowed,
            self.precision,
            # TODO: this should be self.scales_precision?
            self.precision,
        )
        return model

    def quantize(
        self, model: torch.nn.Module, inputs: List[_MultiInput], **kwargs: Any
    ) -> torch.nn.Module:
        state_dict = self._create_quantized_state_dict(
            model,
            inputs,
            self.blocksize,
            self.percdamp,
            self.groupsize,
        )
        model = self._convert_for_runtime(model)
        model.load_state_dict(state_dict, strict=False)
        return model
