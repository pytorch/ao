from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD 3-Clause license found in the
# LICENSE file in the root directory of this source tree.
from torchao.core.config import AOBaseConfig
from torchao.quantization.transform_module import register_quantize_module_handler
from torchao.utils import TorchAOBaseTensor
from functools import partial
from torchao.quantization.quant_api import _module_extra_repr
import types

from torchao.quantization.utils import (
    get_groupwise_affine_qparams,
    groupwise_affine_dequantize_tensor_from_qparams,
    groupwise_affine_quantize_tensor_from_qparams,
)

import math
from torchao.quantization.quant_primitives import (
    ZeroPointDomain,
)


@dataclass
class ObserverConfig(AOBaseConfig):
    step: str = "observe"


@register_quantize_module_handler(ObserverConfig)
def _observer_config_transform(
    module: torch.nn.Module, config: ObserverConfig, *, parameter_name="weight"
) -> torch.nn.Module:
    if config.step == "observe":
        tensor = getattr(module, parameter_name)
        new_tensor = ObserverTensor.from_hp(tensor)
        setattr(module, parameter_name, nn.Parameter(new_tensor, requires_grad=False))
        module.extra_repr = types.MethodType(
            partial(
                _module_extra_repr,
                original_extra_repr=module.extra_repr,
                parameter_name=parameter_name,
            ),
            module,
        )
        return module
    elif config.step == "convert":
        tensor = getattr(module, parameter_name)
        assert isinstance(tensor, ObserverTensor)
        new_tensor = tensor.to_accelerated("gptq_int4")
        setattr(module, parameter_name, nn.Parameter(new_tensor, requires_grad=False))
        module.extra_repr = types.MethodType(
            partial(
                _module_extra_repr,
                original_extra_repr=module.extra_repr,
                parameter_name=parameter_name,
            ),
            module,
        )
        return module


class ObserverTensor(TorchAOBaseTensor):
    """
    We create ObserverTensor with two modes, OBSERVE and REPLAY.

    if in OBSERVE mode, when it comes across a mm it will add the input to saved activations, and return a meta tensor.

    if in REPLAY mode, when it comes across a meta input to mm, it will pop an input from the saved activations, and return the quantized mm output.



    Then to sequentially quantize we can do the following:


    quantize_(layer N, ObserverTensor.OBSERVE)

    for batch in calibration_dataset:
        model(batch)

    # repeat below for all layers
    layer1.calculate_qparams_gptq()
    quantize(layer N, ObserverTensor.REPLAY)
    quantize(layer N+1, ObserverTensor.OBSERVE)

    for batch in calibration_dataset:
        model(batch.to(meta))

    quantize(layer N, Int4Tensor)
    move layer N to meta
    """

    tensor_data_names = ["hp_data"]
    tensor_attribute_names = ["inputs"]
    optional_tensor_attribute_names = []

    def __new__(cls, hp_data: torch.Tensor, inputs: List[torch.Tensor] = []):
        shape = hp_data.shape
        kwargs = {}
        kwargs["device"] = hp_data.device
        kwargs["dtype"] = hp_data.dtype
        kwargs["requires_grad"] = False
        return torch.Tensor._make_wrapper_subclass(cls, shape, **kwargs)  # type: ignore[attr-defined]

    def __init__(self, hp_data: torch.Tensor, inputs: List[torch.Tensor] = []):
        super().__init__()
        self.hp_data = hp_data
        self.inputs = inputs

    @classmethod
    def from_hp(cls, hp_tensor):
        return ObserverTensor(hp_tensor, [])

    def to_accelerated(self, torchao_base_tensor_type):
        if torchao_base_tensor_type == "gptq_int4":
            H = _calculate_hessian(self.inputs, None, self.device)

            W = self.hp_data
            Q, DQ, all_qparams = self.faster_quant(
                H, W.view(-1, W.shape[-1]).detach(), self.device
            )

            return Int4Tensor(
                qdata=Q, 
                scale=all_qparams[0], 
                zero_point=all_qparams[1], 
                block_size=None,
                shape=W.shape,
                act_pre_scale=None,
            )

    def faster_quant(self, H, W, device):
        """
        GPTQ quantization implementation.

        Args:
            H: Hessian matrix approximation
            W: Weight matrix to quantize
            device: accelerator device
        """

        zero_point_domain = ZeroPointDomain.FLOAT
        n_bit = 4
        group_size = -1
        # self.scale = hp_data
        self.get_qparams_func = lambda w, precision: get_groupwise_affine_qparams(
            w,
            n_bit,
            group_size,
            dtype=precision,
            zero_point_domain=self.zero_point_domain,
        )
        self.quantize_func = (
            lambda w, qparams: groupwise_affine_quantize_tensor_from_qparams(
                w,
                qparams[0],
                qparams[1],
                n_bit,
                group_size,
                zero_point_domain=self.zero_point_domain,
            )
        )
        self.dequantize_func = (
            lambda q, qparams: groupwise_affine_dequantize_tensor_from_qparams(
                q,
                qparams[0],
                qparams[1],
                n_bit,
                group_size,
                zero_point_domain=self.zero_point_domain,
            )
        )
        self.combine_qparams_list_func = lambda qparams_list: [
            torch.cat(x, dim=1) for x in zip(*qparams_list)
        ]

        percdamp = 0.01
        blocksize = 128
        group_size = -1
        orig_dtype = W.dtype
        W = W.detach().float()
        _, columns = W.shape[0], W.shape[1]
        device = W.device

        if group_size == -1:
            group_size = columns
        else:
            blocksize = math.ceil(blocksize / group_size) * group_size

        dead = torch.diag(H) == 0
        H[dead, dead] = 1
        W[:, dead] = 0

        DQ = torch.zeros_like(W)

        damp = percdamp * torch.mean(torch.diag(H))
        diag = torch.arange(columns, device=device)
        H[diag, diag] += damp
        H = torch.linalg.cholesky(H)
        H = torch.cholesky_inverse(H)
        H = torch.linalg.cholesky(H, upper=True)
        Hinv = H

        cur_qparams = None
        all_qparams = []

        for block_start in range(
            0, columns, blocksize
        ):  # go through all columns block by block
            block_end = min(block_start + blocksize, columns)
            W1 = W[:, block_start:block_end].clone()
            DQ1 = torch.zeros_like(W1)
            Err1 = torch.zeros_like(W1)
            Hinv1 = Hinv[block_start:block_end, block_start:block_end]
            for group_start in range(
                block_start, block_end, group_size
            ):  # break up blocks by groupsize
                group_end = min(group_start + group_size, columns)
                if group_start % group_size == 0:
                    # needed for when group_size == columns so only calculate qparams once
                    cur_qparams = self.get_qparams_func(
                        W[:, group_start:group_end], orig_dtype
                    )
                    all_qparams.append(cur_qparams)

                for index in range(group_start, group_end):  # within each group
                    i = index - block_start
                    if i >= W1.shape[1]:
                        break
                    w = W1[:, i]
                    d = Hinv1[i, i]

                    q = self.quantize_func(w.unsqueeze(1), cur_qparams).flatten()
                    dq = self.dequantize_func(q.unsqueeze(1), cur_qparams).flatten()

                    DQ1[:, i] = dq

                    err1 = (w - dq) / d
                    W1[:, i:] -= (
                        err1.to(Hinv1.dtype)
                        .unsqueeze(1)
                        .matmul(Hinv1[i, i:].unsqueeze(0))
                    )
                    Err1[:, i] = err1

            DQ[:, block_start:block_end] = DQ1
            W[:, block_end:] -= Err1.to(Hinv.dtype).matmul(
                Hinv[block_start:block_end, block_end:]
            )

        if "xpu" in device.type:
            torch.xpu.synchronize()
        elif "cuda" in device.type:
            torch.cuda.synchronize()
        else:
            pass

        if all_qparams == []:
            all_qparams.append(cur_qparams)

        all_qparams = self.combine_qparams_list_func(all_qparams)
        Q = self.quantize_func(DQ, all_qparams)
        return Q, DQ.to(orig_dtype), all_qparams


implements = ObserverTensor.implements
implements_torch_function = ObserverTensor.implements_torch_function
aten = torch.ops.aten


@implements(aten.linear.default)
@implements_torch_function(torch.nn.functional.linear)
def _(func, types, args, kwargs):
    input_tensor, weight_tensor, bias = (
        args[0],
        args[1],
        args[2] if len(args) > 2 else None,
    )
    weight_tensor.inputs.append(input_tensor.detach())
    return F.linear(input_tensor, weight_tensor.hp_data, bias)


@implements(aten.bmm.default)
def _(func, types, args, kwargs):
    input_tensor, weight_tensor = (
        args[0],
        args[1],
    )
    weight_tensor.inputs.append(input_tensor.view(-1, input_tensor.shape[-1]).detach())
    return func(input_tensor, weight_tensor.hp_data)


def _calculate_hessian(grouped_args, spec, device=torch.device("cuda")):
    """
    Calculate the Hessian matrix for GPTQ.

    Args:
        grouped_args: Grouped arguments
        spec: Original structure specification
        device: accelerator device

    Returns:
        torch.Tensor: Hessian matrix
    """
    H = 0
    total_batches = 0
    for inp in grouped_args:
        # Move all remaining CPU tensors to CUDA
        device_inp = [x.to(device) if isinstance(x, torch.Tensor) else x for x in inp]

        # Setup x (activation tensor)
        x = inp.float()
        shape = x.shape
        n = 1 if len(shape) == 2 else shape[0]
        x = x.reshape(-1, shape[-1])

        # Update Hessian with running average
        H *= total_batches / (total_batches + n)
        total_batches += n

        x = ((2 / total_batches) ** (1 / 2)) * x.t()
        H += x.matmul(x.t())

    return H
