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
from torchao.prototype.moe_training.config import MXFP8TrainingOpConfig
from torchao.quantization import quantize_
from torchao.quantization.utils import compute_error


class FeedForward(nn.Module):
    """MLP based model"""

    def __init__(self, size):
        super(FeedForward, self).__init__()
        self.w1 = nn.Linear(size, size * 2, bias=False)
        self.w2 = nn.Linear(size, size * 2, bias=False)
        self.out_proj = nn.Linear(size * 2, size, bias=False)

    def forward(self, x):
        x = F.silu(self.w1(x)) * self.w2(x)
        x = self.out_proj(x)
        return x


class ToyModel(nn.Module):
    def __init__(self, size):
        super(ToyModel, self).__init__()
        self.ffn = FeedForward(size)

    def forward(self, x):
        return self.ffn(x)


def _test_lowp_mlp_tensor_parallelism_base(
    mesh: DeviceMesh,
    config: Union[Float8LinearConfig, MXFP8TrainingOpConfig],
    size=32,
    compile: bool = False,
    allgather_in_lowp: bool = False,
):
    device = mesh.device_type

    # TODO(future): remove this once float8 training works with `quantize_` API
    convert_model_func = convert_to_float8_training
    if isinstance(config, MXFP8TrainingOpConfig):
        convert_model_func = quantize_

    toy_model = ToyModel(size).to(device).to(torch.bfloat16)
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

    x_bf16 = torch.rand(
        2, size * 2, size, device=device, requires_grad=False, dtype=torch.bfloat16
    )
    go_bf16 = torch.rand(
        2, size * 2, size, device=device, requires_grad=False, dtype=torch.bfloat16
    )
    x_bf16_tp_input = x_bf16.clone()
    go_bf16_tp = go_bf16.clone()
    x_bf16_sp_input = distribute_tensor(x_bf16.clone(), mesh, [Shard(0)])
    go_bf16_sp = distribute_tensor(go_bf16.clone(), mesh, [Shard(0)])

    tp_out = tp_model(x_bf16_tp_input)
    tp_out.backward(go_bf16_tp)
    sp_out = sp_model(x_bf16_sp_input)
    sp_out.backward(go_bf16_sp)
    global_out = toy_model_fp8(x_bf16)
    global_out.backward(go_bf16)

    MIN_SQNR = 23.0

    if not torch.allclose(tp_out, global_out):
        print(
            f"tp out comparison not close, shapes: tp={tp_out.shape}, global={global_out.shape}"
        )
        print(
            f"tp_out stats: min={tp_out.min()}, max={tp_out.max()}, mean={tp_out.mean()}"
        )
        print(
            f"global_out stats: min={global_out.min()}, max={global_out.max()}, mean={global_out.mean()}"
        )
        diff = (tp_out - global_out).abs()
        print(f"diff stats: min={diff.min()}, max={diff.max()}, mean={diff.mean()}")

    tp_out_sqnr = compute_error(tp_out, global_out)
    print(f"tp_out SQNR: {tp_out_sqnr}")
    assert tp_out_sqnr >= MIN_SQNR, f"tp_out SQNR {tp_out_sqnr} < {MIN_SQNR}"

    sp_out_sqnr = compute_error(sp_out.full_tensor(), global_out)
    assert sp_out_sqnr >= MIN_SQNR, f"sp_out SQNR {sp_out_sqnr} < {MIN_SQNR}"

    w1_grad_sqnr = compute_error(
        tp_model.ffn.w1.weight.grad, sp_model.ffn.w1.weight.grad
    )
    assert w1_grad_sqnr >= MIN_SQNR, f"w1.weight.grad SQNR {w1_grad_sqnr} < {MIN_SQNR}"

    out_proj_grad_sqnr = compute_error(
        tp_model.ffn.out_proj.weight.grad, sp_model.ffn.out_proj.weight.grad
    )
    assert out_proj_grad_sqnr >= MIN_SQNR, (
        f"out_proj.weight.grad SQNR {out_proj_grad_sqnr} < {MIN_SQNR}"
    )

    sp_out2 = sp_model2(x_bf16_sp_input)
    sp_out2.backward(go_bf16_sp)

    sp_out2_sqnr = compute_error(sp_out2.full_tensor(), global_out)
    assert sp_out2_sqnr >= MIN_SQNR, f"sp_out2 SQNR {sp_out2_sqnr} < {MIN_SQNR}"

    w1_grad2_sqnr = compute_error(
        tp_model.ffn.w1.weight.grad, sp_model2.ffn.w1.weight.grad
    )
    assert w1_grad2_sqnr >= MIN_SQNR, (
        f"w1.weight.grad (sp_model2) SQNR {w1_grad2_sqnr} < {MIN_SQNR}"
    )

    out_proj_grad2_sqnr = compute_error(
        tp_model.ffn.out_proj.weight.grad, sp_model2.ffn.out_proj.weight.grad
    )
    assert out_proj_grad2_sqnr >= MIN_SQNR, (
        f"out_proj.weight.grad (sp_model2) SQNR {out_proj_grad2_sqnr} < {MIN_SQNR}"
    )
