"""Distributed tests for FSDP2 compatibility. Run with: torchrun --nproc_per_node=2 -m pytest test_fsdp2.py"""

import os

import pytest
import torch
from torch import nn
from torch.distributed._composable.fsdp import fully_shard
from torch.distributed.device_mesh import init_device_mesh

from torchao.prototype.moe_qat import MoEQATConfig
from torchao.prototype.moe_qat.tensor import Float8FakeQuantizedWeightWrapperTensor
from torchao.quantization.qat.fake_quantize_config import Float8FakeQuantizeConfig
from torchao.quantization.granularity import PerRow
from torchao.quantization.quant_api import quantize_


def _init_dist():
    if os.environ.get("RANK") is None:
        raise RuntimeError(
            "Distributed tests require a launcher that sets RANK, WORLD_SIZE, MASTER_ADDR, and MASTER_PORT, "
            "e.g.: torchrun --nproc_per_node=2 -m pytest test_fsdp2.py"
        )
    if not torch.distributed.is_initialized():
        torch.distributed.init_process_group(backend="gloo")
    torch.cuda.set_device(int(os.environ.get("LOCAL_RANK", "0")))


class ExpertLayer(nn.Module):
    """A minimal module with a 3D expert weight for FSDP2 testing."""

    def __init__(self, num_experts=4, hidden_dim=16, intermediate_dim=32):
        super().__init__()
        self.w1 = nn.Parameter(torch.randn(num_experts, intermediate_dim, hidden_dim))

    def forward(self, x):
        return torch._grouped_mm(
            x, self.w1.transpose(-2, -1), offs=torch.tensor([x.shape[0]], dtype=torch.int32)
        )


@pytest.mark.skipif(not torch.distributed.is_available(), reason="distributed not available")
class TestFSDP2:
    @staticmethod
    def _expert_weight_filter(param, fqn):
        return param.ndim == 3

    def test_fsdp2_prepare_forward(self):
        _init_dist()
        device = torch.device(f"cuda:{torch.cuda.current_device()}" if torch.cuda.is_available() else "cpu")

        model = ExpertLayer().to(device)
        weight_config = Float8FakeQuantizeConfig(dtype=torch.float8_e4m3fn, granularity=PerRow())

        qat_config = MoEQATConfig(
            weight_config=weight_config,
            step="prepare",
            params_filter_fn=TestFSDP2._expert_weight_filter,
        )
        quantize_(model, qat_config, filter_fn=lambda m, fqn: isinstance(m, ExpertLayer))

        mesh = init_device_mesh(device.type, [torch.distributed.get_world_size()])
        fully_shard(model, mesh=mesh)

        x = torch.randn(4, 16, device=device)
        with torch.no_grad():
            out = model(x)
        # Verify wrapper preserved through FSDP2 unshard
        assert isinstance(model.w1.data, Float8FakeQuantizedWeightWrapperTensor)

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required for _grouped_mm backward")
    def test_fsdp2_prepare_backward(self):
        _init_dist()
        device = torch.device(f"cuda:{torch.cuda.current_device()}")

        model = ExpertLayer().to(device)
        weight_config = Float8FakeQuantizeConfig(dtype=torch.float8_e4m3fn, granularity=PerRow())

        qat_config = MoEQATConfig(
            weight_config=weight_config,
            step="prepare",
            params_filter_fn=TestFSDP2._expert_weight_filter,
        )
        quantize_(model, qat_config, filter_fn=lambda m, fqn: isinstance(m, ExpertLayer))

        mesh = init_device_mesh(device.type, [torch.distributed.get_world_size()])
        fully_shard(model, mesh=mesh)

        x = torch.randn(4, 16, device=device)
        out = model(x)
        loss = out.sum()
        loss.backward()
        assert model.w1.grad is not None

    class _MixedPrecisionModel(nn.Module):
        """Minimal model using torch.mm for CPU-safe backward."""

        def __init__(self):
            super().__init__()
            self.w = nn.Parameter(torch.randn(32, 16))

        def forward(self, x):
            return torch.mm(x, self.w)

    def test_fsdp2_mixed_precision_no_cast(self):
        """FSDP2 with same dtype (no-cast path, storage-sharing)."""
        from torch.distributed.fsdp import MixedPrecisionPolicy

        _init_dist()
        device = torch.device(f"cuda:{torch.cuda.current_device()}" if torch.cuda.is_available() else "cpu")

        model = self._MixedPrecisionModel().to(device)
        weight_config = Float8FakeQuantizeConfig(dtype=torch.float8_e4m3fn, granularity=PerRow())

        qat_config = MoEQATConfig(weight_config=weight_config, step="prepare")
        quantize_(model, qat_config, filter_fn=lambda m, fqn: isinstance(m, self._MixedPrecisionModel))

        mesh = init_device_mesh(device.type, [torch.distributed.get_world_size()])
        mp_policy = MixedPrecisionPolicy(param_dtype=torch.float32)
        fully_shard(model, mesh=mesh, mp_policy=mp_policy)

        x = torch.randn(4, 32, device=device)
        out = model(x)
        loss = out.sum()
        loss.backward()
        assert model.w.grad is not None

    def test_fsdp2_mixed_precision_cast(self):
        """FSDP2 with different dtype (cast + copy_ path)."""
        from torch.distributed.fsdp import MixedPrecisionPolicy

        _init_dist()
        device = torch.device(f"cuda:{torch.cuda.current_device()}" if torch.cuda.is_available() else "cpu")

        model = self._MixedPrecisionModel().to(device)
        weight_config = Float8FakeQuantizeConfig(dtype=torch.float8_e4m3fn, granularity=PerRow())

        qat_config = MoEQATConfig(weight_config=weight_config, step="prepare")
        quantize_(model, qat_config, filter_fn=lambda m, fqn: isinstance(m, self._MixedPrecisionModel))

        mesh = init_device_mesh(device.type, [torch.distributed.get_world_size()])
        mp_policy = MixedPrecisionPolicy(param_dtype=torch.bfloat16)
        fully_shard(model, mesh=mesh, mp_policy=mp_policy)

        x = torch.randn(4, 32, device=device)
        out = model(x)
        loss = out.sum()
        loss.backward()
        assert model.w.grad is not None

    def test_fsdp2_convert_unwraps(self):
        _init_dist()
        device = torch.device(f"cuda:{torch.cuda.current_device()}" if torch.cuda.is_available() else "cpu")

        model = ExpertLayer().to(device)
        weight_config = Float8FakeQuantizeConfig(dtype=torch.float8_e4m3fn, granularity=PerRow())

        qat_config = MoEQATConfig(
            weight_config=weight_config,
            step="prepare",
            params_filter_fn=TestFSDP2._expert_weight_filter,
        )
        quantize_(model, qat_config, filter_fn=lambda m, fqn: isinstance(m, ExpertLayer))

        mesh = init_device_mesh(device.type, [torch.distributed.get_world_size()])
        fully_shard(model, mesh=mesh)

        convert_config = MoEQATConfig(step="convert")
        quantize_(model, convert_config, filter_fn=lambda m, fqn: isinstance(m, ExpertLayer))

        from torchao.prototype.moe_qat.tensor import FakeQuantizedWeightWrapperBaseTensor
        assert not isinstance(model.w1.data, FakeQuantizedWeightWrapperBaseTensor)
