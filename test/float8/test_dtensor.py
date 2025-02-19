# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD 3-Clause license found in the
# LICENSE file in the root directory of this source tree.
"""
Test numerics of manually defined float16 TP vs float8 TP of toy models

Note: for now, this does not run in CI.
TODO(future): make this run in CI
"""

import copy
import os

import pytest
import torch

from torchao.utils import TORCH_VERSION_AT_LEAST_2_5

if not TORCH_VERSION_AT_LEAST_2_5:
    pytest.skip("Unsupported PyTorch version", allow_module_level=True)

from torch.distributed._tensor import DTensor, Replicate, Shard, distribute_tensor
from torch.distributed.device_mesh import DeviceMesh, init_device_mesh
from torch.distributed.tensor.parallel import (
    ColwiseParallel,
    PrepareModuleInput,
    RowwiseParallel,
    parallelize_module,
)
from torch.testing._internal.distributed._tensor.common_dtensor import (
    ModelArgs,
    Transformer,
)
from tqdm import tqdm

from torchao.float8 import Float8LinearConfig
from torchao.float8.config import (
    CastConfig,
    Float8LinearRecipeName,
    ScalingType,
    e4m3_dtype,
)
from torchao.float8.float8_linear_utils import convert_to_float8_training
from torchao.float8.float8_scaling_utils import NoopFwToFloat8BwDynamic
from torchao.float8.float8_tensor import (
    Float8Tensor,
    GemmInputRole,
    LinearMMConfig,
    hp_tensor_and_scale_to_float8,
)
from torchao.float8.float8_tensor_parallel import (
    Float8ColwiseParallel,
    Float8RowwiseParallel,
    PrepareFloat8ModuleInput,
)
from torchao.float8.float8_utils import tensor_to_scale
from torchao.float8.fsdp_utils import WeightWithDynamicFloat8CastTensor
from torchao.testing.float8.dtensor_utils import ToyModel

torch.set_float32_matmul_precision("high")


def setup_distributed():
    world_size = int(os.environ.get("WORLD_SIZE", -1))
    device_mesh = init_device_mesh("cuda", (world_size,))
    # seed must be the same in all processes
    torch.manual_seed(1)
    return device_mesh


def _test_scaled_mm(mesh: DeviceMesh, size=16):
    device = mesh.device_type
    fp8_dtype = e4m3_dtype
    world_size = mesh.size()

    x_fp32 = torch.rand(size, size, device=device)
    y_fp32 = torch.eye(size, device=device).t()

    placement_combs = (
        (Shard(0), Replicate()),
        (Replicate(), Shard(1)),
        (Shard(1), Shard(0)),
    )
    expected_dt_out_shape = (
        (size * world_size, size),
        (size, size * world_size),
        (size, size),
    )
    for idx, (lhs_placement, rhs_placement) in enumerate(placement_combs):
        x_scale = tensor_to_scale(x_fp32, fp8_dtype).float()
        y_scale = tensor_to_scale(y_fp32, fp8_dtype).float()

        x_fp8 = hp_tensor_and_scale_to_float8(
            x_fp32, x_scale, fp8_dtype, None, GemmInputRole.INPUT
        )
        y_fp8 = hp_tensor_and_scale_to_float8(
            y_fp32, y_scale, fp8_dtype, None, GemmInputRole.WEIGHT
        )

        dist_x_fp8 = DTensor.from_local(x_fp8, mesh, [lhs_placement], run_check=False)
        dist_y_fp8 = DTensor.from_local(y_fp8, mesh, [rhs_placement], run_check=False)

        assert isinstance(dist_x_fp8.to_local(), Float8Tensor)
        assert isinstance(dist_y_fp8.to_local(), Float8Tensor)
        assert dist_x_fp8.to_local()._orig_dtype == torch.float32
        out_fp8 = torch.mm(dist_x_fp8, dist_y_fp8)
        local_fp8_out = out_fp8.to_local()
        assert out_fp8.shape == expected_dt_out_shape[idx], (idx, local_fp8_out.shape)

        # after mm the out dtype should be fp32
        assert local_fp8_out.dtype == torch.float32


def _test_fp8_redistribute(mesh: DeviceMesh, size=16):
    device = mesh.device_type
    fp8_dtype = e4m3_dtype
    world_size = mesh.size()

    x_fp32 = torch.rand(size, size, device=device)

    x_scale = tensor_to_scale(x_fp32, fp8_dtype).float()

    x_fp8 = hp_tensor_and_scale_to_float8(x_fp32, x_scale, fp8_dtype)

    dist_x_fp8 = DTensor.from_local(x_fp8, mesh, [Shard(0)], run_check=False)
    out_dist = dist_x_fp8.redistribute(placements=[Replicate()])
    assert out_dist.shape == (size * world_size, size)
    assert out_dist.placements == (Replicate(),)
    out_local = out_dist.to_local()
    # after allgather the out shape should be replicate
    assert out_local.shape == (size * world_size, size)
    from torch.distributed._functional_collectives import AsyncCollectiveTensor

    if isinstance(out_local, AsyncCollectiveTensor):
        out_local = out_local.wait()

    assert isinstance(out_local, Float8Tensor)
    assert out_local._data.dtype == fp8_dtype


def _test_dtensor_cast_to_fp8(mesh: DeviceMesh, size=16):
    device = mesh.device_type
    fp8_dtype = e4m3_dtype

    x_fp32 = torch.rand(size, size, device=device)
    dist_x_fp32 = distribute_tensor(x_fp32, mesh, [Shard(0)])

    dist_x_scale = tensor_to_scale(dist_x_fp32, fp8_dtype).float()
    assert isinstance(dist_x_scale, DTensor)

    dist_x_fp8 = hp_tensor_and_scale_to_float8(dist_x_fp32, dist_x_scale, fp8_dtype)
    assert isinstance(dist_x_fp8, DTensor)


def _test_dtensor_fp8_autograd(mesh: DeviceMesh, size=16):
    device = mesh.device_type
    fp8_dtype = e4m3_dtype

    x_fp32 = torch.rand(size, size, device=device, requires_grad=True)
    local_weight = torch.rand(2 * size, size, device=device, requires_grad=True)
    target = torch.rand(size, 2 * size, device=device)

    dist_x_fp32 = distribute_tensor(x_fp32, mesh, [Shard(0)])
    dist_x_scale = tensor_to_scale(dist_x_fp32, fp8_dtype).float()

    dist_wight_fp32 = distribute_tensor(local_weight, mesh, [Shard(0)])
    dist_weight_scale = tensor_to_scale(dist_wight_fp32, fp8_dtype).float()
    dist_target = distribute_tensor(target, mesh, [Shard(0)])

    dist_x_fp8 = hp_tensor_and_scale_to_float8(
        dist_x_fp32,
        dist_x_scale,
        fp8_dtype,
        None,
        GemmInputRole.INPUT,
    )
    dist_weight_fp8 = hp_tensor_and_scale_to_float8(
        dist_wight_fp32,
        dist_weight_scale,
        fp8_dtype,
        None,
        GemmInputRole.WEIGHT,
    )

    out = torch.nn.functional.linear(dist_x_fp8, dist_weight_fp8)
    out = NoopFwToFloat8BwDynamic.apply(out, LinearMMConfig(), fp8_dtype)
    assert isinstance(out, DTensor), f"Expected DTensor, got {type(out)}"
    loss = torch.sum(torch.abs(out - dist_target))
    loss.backward()


def _test_fp8_mlp_tensor_parallelism_base(
    mesh: DeviceMesh, size=16, compile: bool = False, rowwise: bool = False
):
    device = mesh.device_type

    if rowwise:
        config = Float8LinearConfig.from_recipe_name(Float8LinearRecipeName.ROWWISE)
        # hack around config being frozen
        # TODO(future PR): we should make this nicer at the config level
        object.__setattr__(config, "emulate", True)
    else:
        config = Float8LinearConfig(emulate=True)

    toy_model = ToyModel().to(device)
    toy_model_fp8 = convert_to_float8_training(toy_model, config=config)

    tp_model = copy.deepcopy(toy_model)
    tp_model = convert_to_float8_training(tp_model, config=config)
    sp_model = copy.deepcopy(toy_model)
    sp_model = convert_to_float8_training(sp_model, config=config)

    # For tensorwise scaling, enable float8 all_gather.
    # For rowwise scaling, keep high precision all_gather. Motivation for
    # not doing float8 all-gather for rowwise: tensors need to be scaled both ways,
    # so for float8 all-gather we'd need to send two float8 copies per tensor,
    # which is similar # bytes over the wire than just doing bfloat16 all-gather.
    if rowwise:
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
    sp_model2 = convert_to_float8_training(sp_model2, config=config)

    if rowwise:
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
    x_fp32_tp_input = x_fp32.clone()
    x_fp32_sp_input = distribute_tensor(x_fp32.clone(), mesh, [Shard(0)])

    tp_out = tp_model(x_fp32_tp_input)
    tp_out.sum().backward()
    sp_out = sp_model(x_fp32_sp_input)
    sp_out.sum().backward()
    global_out = toy_model_fp8(x_fp32)
    global_out.sum().backward()
    torch.testing.assert_close(tp_out, global_out)
    torch.testing.assert_close(sp_out.full_tensor(), global_out)
    torch.testing.assert_close(tp_model.ffn.w1.weight.grad, sp_model.ffn.w1.weight.grad)
    torch.testing.assert_close(
        tp_model.ffn.out_proj.weight.grad, sp_model.ffn.out_proj.weight.grad
    )

    sp_out2 = sp_model2(x_fp32_sp_input)
    sp_out2.sum().backward()
    torch.testing.assert_close(sp_out2.full_tensor(), global_out)
    torch.testing.assert_close(
        tp_model.ffn.w1.weight.grad, sp_model2.ffn.w1.weight.grad
    )
    torch.testing.assert_close(
        tp_model.ffn.out_proj.weight.grad, sp_model2.ffn.out_proj.weight.grad
    )


def _test_fp8_mlp_tensor_parallelism_eager(mesh: DeviceMesh, size=16):
    _test_fp8_mlp_tensor_parallelism_base(mesh, size, compile=False, rowwise=False)
    _test_fp8_mlp_tensor_parallelism_base(mesh, size, compile=False, rowwise=True)


def _test_fp8_mlp_tensor_parallelism_compile(mesh: DeviceMesh, size=16):
    _test_fp8_mlp_tensor_parallelism_base(mesh, size, compile=True, rowwise=False)
    _test_fp8_mlp_tensor_parallelism_base(mesh, size, compile=True, rowwise=True)


def _test_distribute_fsdp_tensor_subclass(tp_mesh: DeviceMesh):
    torch.manual_seed(42)
    model = Transformer(ModelArgs(dropout_p=0.0, weight_tying=False)).cuda()
    convert_to_float8_training(
        model,
        config=Float8LinearConfig(
            enable_fsdp_float8_all_gather=True,
            cast_config_weight=CastConfig(scaling_type=ScalingType.DYNAMIC),
        ),
    )
    # test Float8ColwiseParallel
    colwise_param = distribute_tensor(
        model.layers[0].attention.wq.weight, tp_mesh, [Shard(0)]
    )
    assert (
        isinstance(colwise_param, DTensor)
        and isinstance(colwise_param._local_tensor, WeightWithDynamicFloat8CastTensor)
    ), f"expect DTensor(local_tensor={WeightWithDynamicFloat8CastTensor}) but got {colwise_param}"
    # test Float8RowwiseParallel
    rowwise_param = distribute_tensor(
        model.layers[0].attention.wo.weight, tp_mesh, [Shard(1)]
    )
    assert (
        isinstance(rowwise_param, DTensor)
        and isinstance(rowwise_param._local_tensor, WeightWithDynamicFloat8CastTensor)
    ), f"expect DTensor(local_tensor={WeightWithDynamicFloat8CastTensor}) but got {colwise_param}"


if __name__ == "__main__":
    # float8 only works on CUDA H100 so we only test cuda and we follow
    # other test files to not use TestCase but instead just add the test
    # cases in the main func.
    device_mesh = setup_distributed()
    tests = [
        _test_scaled_mm,
        _test_fp8_redistribute,
        _test_dtensor_cast_to_fp8,
        _test_dtensor_fp8_autograd,
        _test_fp8_mlp_tensor_parallelism_eager,
        _test_fp8_mlp_tensor_parallelism_compile,
        _test_distribute_fsdp_tensor_subclass,
    ]

    for test in tqdm(tests, desc="Running tests"):
        try:
            test(device_mesh)
        except Exception as e:
            print(f"Test {test.__name__} failed with error: {e}")
            raise e

    torch.distributed.destroy_process_group()
