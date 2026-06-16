"""Distributed tests for MoE QAT operator semantics. Run with: torchrun --nproc_per_node=4 -m pytest test/prototype/moe_qat/test_distributed_ops.py"""

import pytest
import torch
from torch.distributed._tensor import Shard, distribute_tensor

from torchao.prototype.moe_qat.wrapper_tensor import (
    FakeQuantizedWeightWrapperBaseTensor,
    Float8FakeQuantizedWeightWrapperTensor,
)
from torchao.quantization.qat.fake_quantize_config import Float8FakeQuantizeConfig

from .testing_utils import distributed_env


@pytest.mark.parametrize("wrapper_cls,weight_config,activation_config", [
    (FakeQuantizedWeightWrapperBaseTensor, Float8FakeQuantizeConfig(), None),
    (FakeQuantizedWeightWrapperBaseTensor, Float8FakeQuantizeConfig(), Float8FakeQuantizeConfig()),
    (Float8FakeQuantizedWeightWrapperTensor, Float8FakeQuantizeConfig(), None),
    (Float8FakeQuantizedWeightWrapperTensor, Float8FakeQuantizeConfig(), Float8FakeQuantizeConfig()),
])
def test_scatter_preserves_subclass(wrapper_cls, weight_config, activation_config, distributed_env):
    """c10d.scatter_.default preserves the wrapper subclass on the local shard."""
    rank = torch.distributed.get_rank()
    device = distributed_env["device_type"]
    shape = (4, 64, 128)
    if rank == 0:
        weight = torch.randn(*shape, device=device)
        wrapper = wrapper_cls(weight.clone(), activation_config=activation_config, weight_config=weight_config)
    else:
        weight = torch.zeros(*shape, device=device)
        wrapper = wrapper_cls(weight.clone(), activation_config=activation_config, weight_config=weight_config)

    mesh = distributed_env["tp_mesh"]
    dt = distribute_tensor(wrapper, mesh, [Shard(0)])

    assert isinstance(dt._local_tensor, wrapper_cls), "Wrapper subclass should be preserved"
    torch.distributed.broadcast(weight, src=0)
    assert torch.equal(dt._local_tensor.to_tensor(), weight.chunk(mesh.size(), dim=0)[rank])
