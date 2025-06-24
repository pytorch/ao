# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD 3-Clause license found in the
# LICENSE file in the root directory of this source tree.
import copy
from typing import Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributed._tensor import Replicate, Shard, distribute_tensor
from torch.distributed.device_mesh import DeviceMesh
from torch.distributed.tensor.parallel import (
    ColwiseParallel,
    PrepareModuleInput,
    RowwiseParallel,
    parallelize_module,
)

from torchao.float8 import Float8LinearConfig
from torchao.float8.float8_linear_utils import convert_to_float8_training
from torchao.float8.float8_tensor_parallel import (
    Float8ColwiseParallel,
    Float8RowwiseParallel,
    PrepareFloat8ModuleInput,
)
from torchao.prototype.mx_formats.config import MXLinearConfig
from torchao.quantization import quantize_


class FeedForward(nn.Module):
    """MLP based model"""

    def __init__(self):
        super(FeedForward, self).__init__()
        self.w1 = nn.Linear(16, 32, bias=False)
        self.w2 = nn.Linear(16, 32, bias=False)
        self.out_proj = nn.Linear(32, 16, bias=False)

    def forward(self, x):
        x = F.silu(self.w1(x)) * self.w2(x)
        x = self.out_proj(x)
        return x


class ToyModel(nn.Module):
    def __init__(self):
        super(ToyModel, self).__init__()
        self.ffn = FeedForward()

    def forward(self, x):
        return self.ffn(x)


def _test_lowp_mlp_tensor_parallelism_base(
    mesh: DeviceMesh,
    config: Union[Float8LinearConfig, MXLinearConfig],
    size=16,
    compile: bool = False,
    allgather_in_lowp: bool = False,
):
    device = mesh.device_type

    # TODO(future): remove this once float8 training works with `quantize_` API
    convert_model_func = convert_to_float8_training
    if isinstance(config, MXLinearConfig):
        convert_model_func = quantize_

    toy_model = ToyModel().to(device)
    toy_model_fp8 = copy.deepcopy(toy_model)
    convert_model_func(toy_model_fp8, config=config)

    tp_model = copy.deepcopy(toy_model)
    convert_model_func(tp_model, config=config)
    sp_model = copy.deepcopy(toy_model)
    convert_model_func(sp_model, config=config)

    # For tensorwise scaling, enable float8 all_gather.
    # For rowwise scaling, keep high precision all_gather. Motivation for
    # not doing float8 all-gather for rowwise: tensors need to be scaled both ways,
    # so for float8 all-gather we'd need to send two float8 copies per tensor,
    # which is similar # bytes over the wire than just doing bfloat16 all-gather.
    if not allgather_in_lowp:
        colwise_parallel_cls = ColwiseParallel
        rowwise_parallel_cls = RowwiseParallel
        prepare_input_cls = PrepareModuleInput
    else:
        colwise_parallel_cls = Float8ColwiseParallel
        rowwise_parallel_cls = Float8RowwiseParallel
        prepare_input_cls = PrepareFloat8ModuleInput

    # vanilla TP
    tp_model = parallelize_module(
        tp_model,
        mesh,
        {
            "ffn.w1": colwise_parallel_cls(),
            "ffn.w2": colwise_parallel_cls(),
            "ffn.out_proj": rowwise_parallel_cls(),
        },
    )

    # "sequence parallel" mlp computation
    sp_model = parallelize_module(
        sp_model,
        mesh,
        {
            "ffn": prepare_input_cls(
                input_layouts=Shard(1), desired_input_layouts=Replicate()
            ),
            "ffn.w1": colwise_parallel_cls(),
            "ffn.w2": colwise_parallel_cls(),
            "ffn.out_proj": rowwise_parallel_cls(
                output_layouts=Shard(1), use_local_output=False
            ),
        },
    )

    # prepare_input_cls with specific submodule fqn
    sp_model2 = copy.deepcopy(toy_model)
    convert_model_func(sp_model2, config=config)

    if not allgather_in_lowp:
        prepare_input = prepare_input_cls(
            input_layouts=Shard(1),
            desired_input_layouts=Replicate(),
        )
    else:
        prepare_input = prepare_input_cls(
            input_layouts=Shard(1),
            desired_input_layouts=Replicate(),
            fwd_config_submodule_fqn="w2",
        )

    sp_model2 = parallelize_module(
        sp_model2,
        mesh,
        {
            "ffn": prepare_input,
            "ffn.w1": colwise_parallel_cls(),
            "ffn.w2": colwise_parallel_cls(),
            "ffn.out_proj": rowwise_parallel_cls(
                output_layouts=Shard(1), use_local_output=False
            ),
        },
    )

    if compile:
        tp_model = torch.compile(tp_model)
        sp_model = torch.compile(sp_model)
        sp_model2 = torch.compile(sp_model2)

    x_fp32 = torch.rand(size, size * 2, size, device=device, requires_grad=False)
    go_fp32 = torch.rand(size, size * 2, size, device=device, requires_grad=False)
    x_fp32_tp_input = x_fp32.clone()
    go_fp32_tp = go_fp32.clone()
    x_fp32_sp_input = distribute_tensor(x_fp32.clone(), mesh, [Shard(0)])
    go_fp32_sp = distribute_tensor(go_fp32.clone(), mesh, [Shard(0)])

    tp_out = tp_model(x_fp32_tp_input)
    tp_out.backward(go_fp32_tp)
    sp_out = sp_model(x_fp32_sp_input)
    sp_out.backward(go_fp32_sp)
    global_out = toy_model_fp8(x_fp32)
    global_out.backward(go_fp32)
    torch.testing.assert_close(tp_out, global_out)
    torch.testing.assert_close(sp_out.full_tensor(), global_out)
    torch.testing.assert_close(tp_model.ffn.w1.weight.grad, sp_model.ffn.w1.weight.grad)
    torch.testing.assert_close(
        tp_model.ffn.out_proj.weight.grad, sp_model.ffn.out_proj.weight.grad
    )

    sp_out2 = sp_model2(x_fp32_sp_input)
    sp_out2.backward(go_fp32_sp)
    torch.testing.assert_close(sp_out2.full_tensor(), global_out)
    torch.testing.assert_close(
        tp_model.ffn.w1.weight.grad, sp_model2.ffn.w1.weight.grad
    )
    torch.testing.assert_close(
        tp_model.ffn.out_proj.weight.grad, sp_model2.ffn.out_proj.weight.grad
    )
