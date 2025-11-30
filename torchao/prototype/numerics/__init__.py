from torchao.quantization.quantize_.workflows.int4.int4_tensor import int4_row_quantize_zp
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
from torchao.quantization import Int4Tensor
from fbgemm_gpu.experimental.gen_ai.quantize import int4_row_quantize_zp, pack_int4
from torchao.quantization import Int4WeightOnlyConfig
import time


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
            acceleration_config = Int4WeightOnlyConfig()
            block_size = [1, acceleration_config.group_size]
            # calculate hessian
            H = _calculate_hessian(self.inputs, None, self.device)
            W = self.hp_data.view(-1, self.shape[-1])
            gptq_block_size = 256
            percdamp = .01
            group_size = acceleration_config.group_size

            W = W.detach()
            _, columns = W.shape[0], W.shape[1]
            device = W.device

            if acceleration_config.group_size == -1:
                group_size = columns
            else:
                blocksize = math.ceil(gptq_block_size/ group_size) * group_size

            dead = torch.diag(H) == 0
            H[dead, dead] = 1
            W[:, dead] = 0

            Q = torch.zeros_like(W, dtype=torch.int8)

            damp = percdamp * torch.mean(torch.diag(H))
            diag = torch.arange(columns, device=device)
            H[diag, diag] += damp
            H = torch.linalg.cholesky(H)
            H = torch.cholesky_inverse(H)
            H = torch.linalg.cholesky(H, upper=True)
            Hinv = H

            all_qparams = []

            for block_start in range(
                0, columns, blocksize
            ):  # go through all columns block by block
                block_end = min(block_start + blocksize, columns)
                W1 = W[:, block_start:block_end].clone()
                Q1 = torch.zeros_like(W1, dtype=torch.int8)
                Err1 = torch.zeros_like(W1)
                Hinv1 = Hinv[block_start:block_end, block_start:block_end]
                for group_start in range(
                    block_start, block_end, group_size
                ):  # break up blocks by groupsize
                    group_end = min(group_start + group_size, columns)
                    if group_start % group_size == 0:
                        # needed for when group_size == columns so only calculate qparams once
                        _, scale, zero = int4_row_quantize_zp(W[:, group_start:group_end], group_size) 
                        all_qparams.append((scale, zero))

                    for index in range(group_start, group_end):  # within each group
                        i = index - block_start
                        w = W1[:, i]
                        d = Hinv1[i, i]

                        q = Int4Tensor.int4_row_quantize_zp_precomputed_qparams(w.unsqueeze(1), scale, zero, group_size=group_size)
                        Q1[:, i] = q.flatten()

                        dq = Int4Tensor(
                            qdata=q,
                            scale=scale,
                            zero_point=zero,
                            block_size=block_size,
                            shape=q.shape,
                            act_pre_scale=None
                        ).dequantize().flatten()

                        err1 = (w - dq) / d
                        W1[:, i:] -= (
                            err1.to(Hinv1.dtype)
                            .unsqueeze(1)
                            .matmul(Hinv1[i, i:].unsqueeze(0))
                        )
                        Err1[:, i] = err1

                Q[:, block_start:block_end] = Q1
                W[:, block_end:] -= Err1.to(Hinv.dtype).matmul(
                    Hinv[block_start:block_end, block_end:]
                )

            if "cuda" in device.type:
                torch.cuda.synchronize()

            final_qparams = [torch.cat(x, dim=0) for x in zip(*all_qparams)]
            return Int4Tensor(
                qdata=pack_int4(Q),
                scale=final_qparams[0].to(self.dtype),
                zero_point=final_qparams[1].to(self.dtype), 
                block_size=block_size, 
                shape=W.shape,
                act_pre_scale=None,
            )



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
