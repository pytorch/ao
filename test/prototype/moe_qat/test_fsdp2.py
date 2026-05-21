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

from .reference_moe import MoE, MoEArgs


def _get_device():
    if torch.cuda.is_available():
        return torch.device(f"cuda:{torch.cuda.current_device()}")
    return torch.device("cpu")


def _init_dist():
    if os.environ.get("RANK") is None:
        raise RuntimeError(
            "Distributed tests require a launcher that sets RANK, WORLD_SIZE, MASTER_ADDR, and MASTER_PORT, "
            "e.g.: torchrun --nproc_per_node=2 -m pytest test_fsdp2.py"
        )
    if not torch.distributed.is_initialized():
        torch.distributed.init_process_group(backend="gloo")
    if torch.cuda.is_available():
        torch.cuda.set_device(int(os.environ.get("LOCAL_RANK", "0")))


def _make_moe(dim, hidden_dim, use_grouped_mm, device):
    args = MoEArgs(
        num_experts=4,
        num_shared_experts=0,
        use_grouped_mm=use_grouped_mm,
        load_balance_coeff=None,
    )
    model = MoE(args, dim=dim, hidden_dim=hidden_dim)
    with torch.no_grad():
        for param in model.parameters():
            nn.init.trunc_normal_(param, std=0.5)
    return model.to(device)


@pytest.mark.skipif(not torch.distributed.is_available(), reason="distributed not available")
class TestFSDP2:
    @staticmethod
    def _expert_weight_filter(param, fqn):
        return param.ndim == 3

    def test_fsdp2_prepare_forward(self):
        _init_dist()
        device = _get_device()

        use_grouped_mm = torch.cuda.is_available()
        model = _make_moe(dim=16, hidden_dim=32, use_grouped_mm=use_grouped_mm, device=device)
        weight_config = Float8FakeQuantizeConfig(dtype=torch.float8_e4m3fn, granularity=PerRow())

        qat_config = MoEQATConfig(
            weight_config=weight_config,
            step="prepare",
            params_filter_fn=TestFSDP2._expert_weight_filter,
        )
        quantize_(model, qat_config, filter_fn=lambda m, fqn: isinstance(m, MoE))

        mesh = init_device_mesh(device.type, [torch.distributed.get_world_size()])
        fully_shard(model, mesh=mesh)

        x = torch.randn(2, 4, 16, device=device)
        with torch.no_grad():
            model(x)
        # Verify wrapper preserved through FSDP2 unshard
        assert isinstance(model.experts.w1.data, Float8FakeQuantizedWeightWrapperTensor)
        assert isinstance(model.experts.w2.data, Float8FakeQuantizedWeightWrapperTensor)
        assert isinstance(model.experts.w3.data, Float8FakeQuantizedWeightWrapperTensor)

    def test_fsdp2_prepare_backward(self):
        _init_dist()
        device = _get_device()

        use_grouped_mm = torch.cuda.is_available()
        model = _make_moe(dim=16, hidden_dim=32, use_grouped_mm=use_grouped_mm, device=device)
        weight_config = Float8FakeQuantizeConfig(dtype=torch.float8_e4m3fn, granularity=PerRow())

        qat_config = MoEQATConfig(
            weight_config=weight_config,
            step="prepare",
            params_filter_fn=TestFSDP2._expert_weight_filter,
        )
        quantize_(model, qat_config, filter_fn=lambda m, fqn: isinstance(m, MoE))

        mesh = init_device_mesh(device.type, [torch.distributed.get_world_size()])
        fully_shard(model, mesh=mesh)

        x = torch.randn(2, 4, 16, device=device)
        out = model(x)
        loss = out.sum()
        loss.backward()
        assert model.experts.w1.grad is not None
        assert model.experts.w1.grad.abs().sum() > 0, "w1 gradient is zero"
        assert model.experts.w2.grad is not None
        assert model.experts.w2.grad.abs().sum() > 0, "w2 gradient is zero"
        assert model.experts.w3.grad is not None
        assert model.experts.w3.grad.abs().sum() > 0, "w3 gradient is zero"

    def test_fsdp2_mixed_precision_no_cast(self):
        """FSDP2 with same dtype (no-cast path, storage-sharing)."""
        from torch.distributed.fsdp import MixedPrecisionPolicy

        _init_dist()
        device = _get_device()

        use_grouped_mm = torch.cuda.is_available()
        model = _make_moe(dim=16, hidden_dim=32, use_grouped_mm=use_grouped_mm, device=device)
        weight_config = Float8FakeQuantizeConfig(dtype=torch.float8_e4m3fn, granularity=PerRow())

        qat_config = MoEQATConfig(weight_config=weight_config, step="prepare")
        quantize_(model, qat_config, filter_fn=lambda m, fqn: isinstance(m, MoE))

        mesh = init_device_mesh(device.type, [torch.distributed.get_world_size()])
        mp_policy = MixedPrecisionPolicy(param_dtype=torch.float32)
        fully_shard(model, mesh=mesh, mp_policy=mp_policy)

        x = torch.randn(2, 4, 16, device=device)
        out = model(x)
        loss = out.sum()
        loss.backward()
        for name in ("w1", "w2", "w3"):
            p = getattr(model.experts, name)
            assert p.grad is not None, f"{name} grad is None"
            assert p.grad.abs().sum() > 0, f"{name} gradient is zero"

    def test_fsdp2_mixed_precision_cast(self):
        """FSDP2 with different dtype (cast + copy_ path)."""
        from torch.distributed.fsdp import MixedPrecisionPolicy

        _init_dist()
        device = _get_device()

        use_grouped_mm = torch.cuda.is_available()
        model = _make_moe(dim=16, hidden_dim=32, use_grouped_mm=use_grouped_mm, device=device)
        weight_config = Float8FakeQuantizeConfig(dtype=torch.float8_e4m3fn, granularity=PerRow())

        qat_config = MoEQATConfig(weight_config=weight_config, step="prepare")
        quantize_(model, qat_config, filter_fn=lambda m, fqn: isinstance(m, MoE))

        mesh = init_device_mesh(device.type, [torch.distributed.get_world_size()])
        mp_policy = MixedPrecisionPolicy(param_dtype=torch.bfloat16)
        fully_shard(model, mesh=mesh, mp_policy=mp_policy)

        x = torch.randn(2, 4, 16, device=device)
        out = model(x)
        loss = out.sum()
        loss.backward()
        for name in ("w1", "w2", "w3"):
            p = getattr(model.experts, name)
            assert p.grad is not None, f"{name} grad is None"
            assert p.grad.abs().sum() > 0, f"{name} gradient is zero"

    def test_fsdp2_convert_unwraps(self):
        _init_dist()
        device = _get_device()

        use_grouped_mm = torch.cuda.is_available()
        model = _make_moe(dim=16, hidden_dim=32, use_grouped_mm=use_grouped_mm, device=device)
        weight_config = Float8FakeQuantizeConfig(dtype=torch.float8_e4m3fn, granularity=PerRow())

        qat_config = MoEQATConfig(
            weight_config=weight_config,
            step="prepare",
            params_filter_fn=TestFSDP2._expert_weight_filter,
        )
        quantize_(model, qat_config, filter_fn=lambda m, fqn: isinstance(m, MoE))

        mesh = init_device_mesh(device.type, [torch.distributed.get_world_size()])
        fully_shard(model, mesh=mesh)

        convert_config = MoEQATConfig(step="convert")
        quantize_(model, convert_config, filter_fn=lambda m, fqn: isinstance(m, MoE))

        from torchao.prototype.moe_qat.tensor import FakeQuantizedWeightWrapperBaseTensor
        assert not isinstance(model.experts.w1.data, FakeQuantizedWeightWrapperBaseTensor)
        assert not isinstance(model.experts.w2.data, FakeQuantizedWeightWrapperBaseTensor)
        assert not isinstance(model.experts.w3.data, FakeQuantizedWeightWrapperBaseTensor)

    # =========================================================================
    # Expert Parallel (EP)
    # =========================================================================

    def test_ep_prepare_forward(self):
        """EP distributes expert params with Shard(0); wrapper preserved on local shards."""
        from .reference_parallel_styles import ExpertParallel

        _init_dist()
        device = _get_device()

        use_grouped_mm = torch.cuda.is_available()
        model = _make_moe(dim=16, hidden_dim=32, use_grouped_mm=use_grouped_mm, device=device)
        weight_config = Float8FakeQuantizeConfig(dtype=torch.float8_e4m3fn, granularity=PerRow())

        qat_config = MoEQATConfig(
            weight_config=weight_config,
            step="prepare",
            params_filter_fn=TestFSDP2._expert_weight_filter,
        )
        quantize_(model, qat_config, filter_fn=lambda m, fqn: isinstance(m, MoE))

        # Apply EP to the experts submodule (matching moe_training pattern)
        ep_mesh = init_device_mesh(device.type, (torch.distributed.get_world_size(),), mesh_dim_names=("ep",))
        ep_style = ExpertParallel()
        model.experts = ep_style._apply(model.experts, ep_mesh)

        # Run forward through the full MoE pipeline.
        # MoE.forward routes on the full input, then EP dispatch/combine wraps experts.forward.
        x = torch.randn(2, 4, 16, device=device)
        with torch.no_grad():
            out = model(x)

        # After EP distribution, expert params are DTensors. to_local() should return
        # Float8FakeQuantizedWeightWrapperTensor on each rank's local shard.
        for name in ("w1", "w2", "w3"):
            param = getattr(model.experts, name)
            local = param.to_local()
            assert isinstance(local, Float8FakeQuantizedWeightWrapperTensor), (
                f"EP local shard for {name} should preserve wrapper"
            )

    def test_ep_prepare_backward(self):
        """EP backward flows gradients through the wrapper on local shards."""
        from .reference_parallel_styles import ExpertParallel

        _init_dist()
        device = _get_device()

        use_grouped_mm = torch.cuda.is_available()
        model = _make_moe(dim=16, hidden_dim=32, use_grouped_mm=use_grouped_mm, device=device)
        weight_config = Float8FakeQuantizeConfig(dtype=torch.float8_e4m3fn, granularity=PerRow())

        qat_config = MoEQATConfig(
            weight_config=weight_config,
            step="prepare",
            params_filter_fn=TestFSDP2._expert_weight_filter,
        )
        quantize_(model, qat_config, filter_fn=lambda m, fqn: isinstance(m, MoE))

        ep_mesh = init_device_mesh(device.type, (torch.distributed.get_world_size(),), mesh_dim_names=("ep",))
        model.experts = ExpertParallel()._apply(model.experts, ep_mesh)

        x = torch.randn(2, 4, 16, device=device)
        out = model(x)
        loss = out.sum()
        loss.backward()

        for name in ("w1", "w2", "w3"):
            param = getattr(model.experts, name)
            assert param.grad is not None, f"{name} grad is None"
            assert param.grad.abs().sum() > 0, f"{name} gradient is zero"

    # =========================================================================
    # Tensor Parallel (TP)
    # =========================================================================

    def test_tp_prepare_forward(self):
        """TP distributes expert params with Shard(1)/Shard(2); wrapper preserved."""
        from .reference_parallel_styles import TensorParallel

        _init_dist()
        device = _get_device()

        use_grouped_mm = torch.cuda.is_available()
        model = _make_moe(dim=16, hidden_dim=32, use_grouped_mm=use_grouped_mm, device=device)
        weight_config = Float8FakeQuantizeConfig(dtype=torch.float8_e4m3fn, granularity=PerRow())

        qat_config = MoEQATConfig(
            weight_config=weight_config,
            step="prepare",
            params_filter_fn=TestFSDP2._expert_weight_filter,
        )
        quantize_(model, qat_config, filter_fn=lambda m, fqn: isinstance(m, MoE))

        tp_mesh = init_device_mesh(device.type, (torch.distributed.get_world_size(),), mesh_dim_names=("tp",))
        model.experts = TensorParallel()._apply(model.experts, tp_mesh)

        x = torch.randn(2, 4, 16, device=device)
        with torch.no_grad():
            out = model(x)

        for name in ("w1", "w2", "w3"):
            param = getattr(model.experts, name)
            local = param.to_local()
            assert isinstance(local, Float8FakeQuantizedWeightWrapperTensor), (
                f"TP local shard for {name} should preserve wrapper"
            )

    # =========================================================================
    # FSDP2 + Tensor Parallel (2D parallelism)
    # =========================================================================

    def test_fsdp2_tp_prepare_forward(self):
        """FSDP2+TP 2D parallelism: wrapper preserved through both sharding strategies."""
        _init_dist()
        world_size = torch.distributed.get_world_size()
        if world_size < 4:
            pytest.skip(f"FSDP2+TP requires world_size >= 4 (2×2 mesh), got {world_size}")

        from torch.distributed.fsdp import MixedPrecisionPolicy
        from .reference_parallel_styles import TensorParallel

        device = _get_device()

        model = _make_moe(dim=16, hidden_dim=32, use_grouped_mm=torch.cuda.is_available(), device=device)
        weight_config = Float8FakeQuantizeConfig(dtype=torch.float8_e4m3fn, granularity=PerRow())

        qat_config = MoEQATConfig(
            weight_config=weight_config,
            step="prepare",
            params_filter_fn=TestFSDP2._expert_weight_filter,
        )
        quantize_(model, qat_config, filter_fn=lambda m, fqn: isinstance(m, MoE))

        world_size = torch.distributed.get_world_size()
        # 2D mesh: fsdp_dim × tp_dim = world_size (e.g., 2×2 for 4 processes)
        tp_size = 2
        fsdp_size = world_size // tp_size
        mesh_2d = init_device_mesh(
            device.type, (fsdp_size, tp_size), mesh_dim_names=("fsdp", "tp")
        )

        # Apply TP to experts submodule on the tp submesh
        model.experts = TensorParallel()._apply(model.experts, mesh_2d["tp"])

        # Apply FSDP2 on the fsdp submesh
        mp_policy = MixedPrecisionPolicy(param_dtype=torch.float32)
        fully_shard(model, mesh=mesh_2d["fsdp"], mp_policy=mp_policy)

        x = torch.randn(2, 4, 16, device=device)
        with torch.no_grad():
            model(x)

        for name in ("w1", "w2", "w3"):
            param = getattr(model.experts, name)
            # After TP+FSDP2, param is a DTensor with both shardings.
            # to_local() should return a Float8FakeQuantizedWeightWrapperTensor.
            local = param.to_local()
            assert isinstance(local, Float8FakeQuantizedWeightWrapperTensor), (
                f"FSDP2+TP local shard for {name} should preserve wrapper"
            )

    # =========================================================================
    # EP + Tensor Parallel parameter distribution (all_to_all skipped for CPU)
    # =========================================================================

    def test_ep_tp_param_distribution(self):
        """EP+TP 2D param sharding [Shard(0), Shard(1)] preserves wrapper on local."""
        _init_dist()
        device = _get_device()

        weight_config = Float8FakeQuantizeConfig(dtype=torch.float8_e4m3fn, granularity=PerRow())
        w = torch.randn(4, 64, 128, device=device)
        wrapper = Float8FakeQuantizedWeightWrapperTensor(w, weight_config=weight_config)

        world_size = torch.distributed.get_world_size()
        # 2D mesh: ep_dim × tp_dim
        tp_size = 2 if world_size >= 4 else 1
        ep_size = world_size // tp_size
        if ep_size * tp_size != world_size:
            pytest.skip(f"world_size {world_size} cannot form ep×tp mesh")
        mesh_2d = init_device_mesh(
            device.type, (ep_size, tp_size), mesh_dim_names=("ep", "tp")
        )

        from torch.distributed.tensor import distribute_tensor, Shard

        # EP Shard(0) on ep dim, TP Shard(1) on tp dim
        dt = distribute_tensor(wrapper, mesh_2d, [Shard(0), Shard(1)])

        local = dt.to_local()
        assert isinstance(local, Float8FakeQuantizedWeightWrapperTensor), (
            "EP+TP local shard should preserve Float8FakeQuantizedWeightWrapperTensor"
        )
        assert local.shape[0] == 4 // ep_size, "EP shard should reduce expert count"
        assert local.shape[1] == 64 // tp_size, "TP shard should reduce hidden dim"
