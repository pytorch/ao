# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
"""
Tests for NVFP4 tensor-parallel linear (sequence-parallel TP).

Run with:
    torchrun --nproc_per_node=2 -m pytest test/prototype/mx_formats/test_nvfp4_parallel.py -s

Requires SM100 (Blackwell) hardware and 2 GPUs.
"""

import os

import pytest
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
from torch.distributed.device_mesh import DeviceMesh, init_device_mesh
from torch.distributed.tensor import DTensor, Replicate, Shard
from torch.distributed.tensor.parallel import parallelize_module
from torch.utils._triton import has_triton

from torchao.prototype.mx_formats.hadamard_utils import prepare_for_cuda_graph
from torchao.prototype.mx_formats.nvfp4_tensor_parallel import (
    _TP_RHT_SIGN_VECTOR,
    NVFP4ColwiseParallel,
    NVFP4RowwiseParallel,
    nvfp4_col_parallel_mm,
    nvfp4_row_parallel_mm,
    swap_first_dims,
)
from torchao.prototype.mx_formats.nvfp4_training import NVFP4Linear
from torchao.quantization.quantize_.common.kernel_preference import KernelPreference
from torchao.quantization.utils import compute_error
from torchao.utils import is_sm_at_least_100, torch_version_at_least

if not torch.cuda.is_available():
    pytest.skip("Requires CUDA", allow_module_level=True)

if not is_sm_at_least_100():
    pytest.skip("Requires SM100+ hardware", allow_module_level=True)


class NVFP4MLP(nn.Module):
    """Small gated MLP using NVFP4Linear layers."""

    def __init__(
        self,
        size: int,
        hidden_size: int,
        *,
        device: str | torch.device,
        dtype: torch.dtype,
    ):
        super().__init__()
        self.w1 = NVFP4Linear(
            size,
            hidden_size,
            bias=True,
            kernel_preference=KernelPreference.TRITON,
            device=device,
            dtype=dtype,
        )
        self.w2 = NVFP4Linear(
            size,
            hidden_size,
            bias=True,
            kernel_preference=KernelPreference.TRITON,
            device=device,
            dtype=dtype,
        )
        self.out_proj = NVFP4Linear(
            hidden_size,
            size,
            bias=True,
            kernel_preference=KernelPreference.TRITON,
            device=device,
            dtype=dtype,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        hidden = F.silu(self.w1(x)) * self.w2(x)
        return self.out_proj(hidden)


def _fp32_mlp_reference(
    x: torch.Tensor,
    weights: dict[str, torch.Tensor],
) -> torch.Tensor:
    w1 = F.linear(x.float(), weights["w1.weight"].float(), weights["w1.bias"].float())
    w2 = F.linear(x.float(), weights["w2.weight"].float(), weights["w2.bias"].float())
    return F.linear(
        F.silu(w1) * w2,
        weights["out_proj.weight"].float(),
        weights["out_proj.bias"].float(),
    )


def _parallelize_nvfp4_mlp(model: NVFP4MLP, mesh: DeviceMesh) -> NVFP4MLP:
    return parallelize_module(
        model,
        mesh,
        {
            "w1": NVFP4ColwiseParallel(use_local_output=False),
            "w2": NVFP4ColwiseParallel(use_local_output=False),
            "out_proj": NVFP4RowwiseParallel(),
        },
    )


@pytest.fixture(scope="module")
def distributed_env() -> DeviceMesh:
    """Set up the 2-rank CUDA device mesh shared by all tests in this module."""
    if "RANK" not in os.environ or "WORLD_SIZE" not in os.environ:
        pytest.skip(
            "Run with: torchrun --nproc_per_node=2 -m pytest "
            "test/prototype/mx_formats/test_nvfp4_parallel.py"
        )

    rank = int(os.environ["RANK"])
    local_rank = int(os.environ.get("LOCAL_RANK", rank))
    world_size = int(os.environ["WORLD_SIZE"])
    assert world_size == 2, (
        f"This test requires world_size=2, got world_size={world_size}. "
        "Run with: torchrun --nproc_per_node=2 -m pytest "
        "test/prototype/mx_formats/test_nvfp4_parallel.py"
    )

    torch.manual_seed(1)
    torch.cuda.set_device(local_rank)
    device_mesh = init_device_mesh("cuda", (world_size,), mesh_dim_names=("tp",))
    yield device_mesh
    torch.cuda.synchronize()
    torch._dynamo.reset()
    dist.destroy_process_group()


@pytest.mark.skipif(not has_triton(), reason="unsupported without triton")
@pytest.mark.skipif(not is_sm_at_least_100(), reason="Requires SM100+")
@pytest.mark.skipif(
    not torch_version_at_least("2.10.0"), reason="torch.compile requires PyTorch 2.10+"
)
def test_swap_first_dims(distributed_env: DeviceMesh):
    """Verify swap_first_dims correctly de-interleaves gathered colwise tensor."""
    mesh = distributed_env
    device = mesh.device_type
    rank = dist.get_rank()
    world_size = dist.get_world_size()

    K, M_per_rank = 128, 64
    M = M_per_rank * world_size

    # Build ground-truth [K, M//2] tensor and slice into per-rank shards [K, M_per_rank//2]
    torch.manual_seed(42)
    ground_truth = torch.randint(0, 256, (K, M // 2), dtype=torch.uint8, device=device)
    local_shard = ground_truth[
        :, rank * (M_per_rank // 2) : (rank + 1) * (M_per_rank // 2)
    ].contiguous()

    # Simulate NCCL all_gather dim-0 on local_shard [K, M_per_rank//2]
    gathered_parts = [torch.zeros_like(local_shard) for _ in range(world_size)]
    dist.all_gather(gathered_parts, local_shard)
    nccl_result = torch.cat(gathered_parts, dim=0)  # [K*W, M_per_rank//2] interleaved

    result = swap_first_dims(nccl_result, world_size)  # [K, M//2]

    assert result.shape == ground_truth.shape, (
        f"Expected {ground_truth.shape}, got {result.shape}"
    )
    torch.testing.assert_close(result, ground_truth, atol=0, rtol=0)

    # Also test 4-D scale tensor
    K_blocks = K // 128
    M_blocks_per_rank = max(1, M_per_rank // 64)
    scale_truth = torch.randint(
        0,
        256,
        (K_blocks, M_blocks_per_rank * world_size, 32, 16),
        dtype=torch.uint8,
        device=device,
    )
    scale_shard = scale_truth[
        :, rank * M_blocks_per_rank : (rank + 1) * M_blocks_per_rank, :, :
    ].contiguous()
    scale_parts = [torch.zeros_like(scale_shard) for _ in range(world_size)]
    dist.all_gather(scale_parts, scale_shard)
    scale_nccl = torch.cat(
        scale_parts, dim=0
    )  # [K_blocks*W, M_blocks_per_rank, 32, 16]
    scale_result = swap_first_dims(scale_nccl, world_size)
    assert scale_result.shape == scale_truth.shape, (
        f"Expected {scale_truth.shape}, got {scale_result.shape}"
    )
    torch.testing.assert_close(scale_result, scale_truth, atol=0, rtol=0)


@pytest.mark.skipif(not has_triton(), reason="unsupported without triton")
@pytest.mark.skipif(not is_sm_at_least_100(), reason="Requires SM100+")
@pytest.mark.skipif(
    not torch_version_at_least("2.10.0"), reason="torch.compile requires PyTorch 2.10+"
)
def test_column_single_rank_equivalence(distributed_env: DeviceMesh):
    """Verify the TP autograd function matches the single-GPU NVFP4 path at world_size=1."""
    from torchao.prototype.mx_formats.nvfp4_linear import nvfp4_mm_triton

    mesh = distributed_env
    device = mesh.device_type
    rank = dist.get_rank()
    pg = dist.new_group([0])
    M, K, N = 256, 256, 256
    if rank != 0:
        dist.barrier()
        return

    torch.manual_seed(7)
    x = torch.randn(M, K, dtype=torch.bfloat16, device=device)
    w = torch.randn(N, K, dtype=torch.bfloat16, device=device)
    bias = torch.randn(N, dtype=torch.bfloat16, device=device)
    sr_seed = torch.randint(-(2**63), 2**63 - 1, (1,), dtype=torch.int64, device=device)

    # Single-GPU reference
    sr_seed_ref = sr_seed.clone()
    y_ref = nvfp4_mm_triton.apply(
        x.clone(), w.clone(), bias.clone(), sr_seed_ref, _TP_RHT_SIGN_VECTOR
    )

    # Column-parallel with world_size=1 (no actual distributed calls needed,
    # but we use a trivial group with just rank 0)
    sr_seed_tp = sr_seed.clone()
    y_tp = nvfp4_col_parallel_mm.apply(
        x.clone(), w.clone(), bias.clone(), sr_seed_tp, pg, 1
    )

    torch.testing.assert_close(y_ref, y_tp, atol=1e-2, rtol=1e-2)
    dist.barrier()


@pytest.mark.skipif(not has_triton(), reason="unsupported without triton")
@pytest.mark.skipif(not is_sm_at_least_100(), reason="Requires SM100+")
@pytest.mark.skipif(
    not torch_version_at_least("2.10.0"), reason="torch.compile requires PyTorch 2.10+"
)
def test_column_forward(distributed_env: DeviceMesh):
    """Verify column-parallel forward output shape, dtype, and SQNR vs fp32 reference."""
    mesh = distributed_env
    device = mesh.device_type
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    tp_group = mesh.get_group()
    M, K, N = 512, 256, 512

    assert M % world_size == 0 and N % world_size == 0
    M_per_rank = M // world_size
    N_per_rank = N // world_size

    torch.manual_seed(11)
    x_full = torch.randn(M, K, dtype=torch.bfloat16, device=device)
    w_full = torch.randn(N, K, dtype=torch.bfloat16, device=device)
    bias_full = torch.randn(N, dtype=torch.bfloat16, device=device)

    x_local = x_full[rank * M_per_rank : (rank + 1) * M_per_rank, :]
    w_local = w_full[rank * N_per_rank : (rank + 1) * N_per_rank, :]
    bias_local = bias_full[rank * N_per_rank : (rank + 1) * N_per_rank]
    sr_seed = torch.randint(-(2**63), 2**63 - 1, (1,), dtype=torch.int64, device=device)

    y = nvfp4_col_parallel_mm.apply(
        x_local, w_local, bias_local, sr_seed, tp_group, world_size
    )

    assert y.shape == (
        M,
        N_per_rank,
    ), f"Rank {rank}: expected ({M}, {N_per_rank}), got {y.shape}"
    assert y.dtype == torch.bfloat16, f"Expected bfloat16, got {y.dtype}"
    assert not y.isnan().any(), "Output contains NaN"

    y_ref_full = x_full.float() @ w_full.float().t() + bias_full.float()
    y_ref_shard = y_ref_full[:, rank * N_per_rank : (rank + 1) * N_per_rank]
    sqnr = compute_error(y_ref_shard, y.float())
    SQNR_THRESHOLD = 15.0
    assert sqnr >= SQNR_THRESHOLD, f"Forward SQNR {sqnr:.2f} dB < {SQNR_THRESHOLD} dB"


@pytest.mark.skipif(not has_triton(), reason="unsupported without triton")
@pytest.mark.skipif(not is_sm_at_least_100(), reason="Requires SM100+")
@pytest.mark.skipif(
    not torch_version_at_least("2.10.0"), reason="torch.compile requires PyTorch 2.10+"
)
def test_column_backward(distributed_env: DeviceMesh):
    """Verify column-parallel backward gradient shapes and SQNR vs fp32 reference."""
    mesh = distributed_env
    device = mesh.device_type
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    tp_group = mesh.get_group()
    M, K, N = 512, 256, 512

    assert M % world_size == 0 and N % world_size == 0
    M_per_rank = M // world_size
    N_per_rank = N // world_size

    torch.manual_seed(5)
    x_full = torch.randn(M, K, dtype=torch.bfloat16, device=device)
    w_full = torch.randn(N, K, dtype=torch.bfloat16, device=device)
    bias_full = torch.randn(N, dtype=torch.bfloat16, device=device)
    dy_full = torch.randn(M, N, dtype=torch.bfloat16, device=device)

    x_local = (
        x_full[rank * M_per_rank : (rank + 1) * M_per_rank, :]
        .contiguous()
        .detach()
        .requires_grad_(True)
    )
    w_local = (
        w_full[rank * N_per_rank : (rank + 1) * N_per_rank, :]
        .contiguous()
        .detach()
        .requires_grad_(True)
    )
    bias_local = (
        bias_full[rank * N_per_rank : (rank + 1) * N_per_rank]
        .contiguous()
        .detach()
        .requires_grad_(True)
    )
    dy_local = dy_full[:, rank * N_per_rank : (rank + 1) * N_per_rank].contiguous()
    sr_seed = torch.randint(-(2**63), 2**63 - 1, (1,), dtype=torch.int64, device=device)

    y = nvfp4_col_parallel_mm.apply(
        x_local, w_local, bias_local, sr_seed, tp_group, world_size
    )
    y.backward(dy_local)

    assert x_local.grad is not None, "x_local.grad is None"
    assert w_local.grad is not None, "w_local.grad is None"
    assert bias_local.grad is not None, "bias_local.grad is None"

    assert x_local.grad.shape == (
        M_per_rank,
        K,
    ), f"Rank {rank}: dx shape expected ({M_per_rank}, {K}), got {x_local.grad.shape}"
    assert w_local.grad.shape == (
        N_per_rank,
        K,
    ), f"Rank {rank}: dw shape expected ({N_per_rank}, {K}), got {w_local.grad.shape}"
    assert bias_local.grad.shape == (N_per_rank,), (
        f"Rank {rank}: db shape expected ({N_per_rank},), got {bias_local.grad.shape}"
    )
    assert not x_local.grad.isnan().any(), "dx contains NaN"
    assert not w_local.grad.isnan().any(), "dw contains NaN"
    assert not bias_local.grad.isnan().any(), "db contains NaN"

    x_ref = x_full.float().detach().requires_grad_(True)
    w_ref = w_full.float().detach().requires_grad_(True)
    y_ref = x_ref @ w_ref.t()
    y_ref.backward(dy_full.float())

    dx_ref = x_ref.grad[rank * M_per_rank : (rank + 1) * M_per_rank, :]
    dw_ref = w_ref.grad[rank * N_per_rank : (rank + 1) * N_per_rank, :]
    db_ref = dy_local.sum(dim=0)

    dx_sqnr = compute_error(dx_ref, x_local.grad.float())
    dw_sqnr = compute_error(dw_ref, w_local.grad.float())
    DX_SQNR_THRESHOLD = 14.0
    DW_SQNR_THRESHOLD = 14.0
    assert dx_sqnr >= DX_SQNR_THRESHOLD, (
        f"dx SQNR {dx_sqnr:.2f} dB < {DX_SQNR_THRESHOLD} dB"
    )
    assert dw_sqnr >= DW_SQNR_THRESHOLD, (
        f"dw SQNR {dw_sqnr:.2f} dB < {DW_SQNR_THRESHOLD} dB"
    )
    torch.testing.assert_close(bias_local.grad, db_ref, atol=0, rtol=0)


@pytest.mark.skipif(not has_triton(), reason="unsupported without triton")
@pytest.mark.skipif(not is_sm_at_least_100(), reason="Requires SM100+")
@pytest.mark.skipif(
    not torch_version_at_least("2.10.0"), reason="torch.compile requires PyTorch 2.10+"
)
def test_column_parallelize_module(distributed_env: DeviceMesh):
    """Verify NVFP4ColwiseParallel works through parallelize_module."""
    mesh = distributed_env
    device = mesh.device_type
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    M, K, N = 512, 256, 512

    assert M % world_size == 0 and N % world_size == 0
    M_per_rank = M // world_size
    N_per_rank = N // world_size

    torch.manual_seed(29)
    module = NVFP4Linear(
        K,
        N,
        bias=True,
        kernel_preference=KernelPreference.TRITON,
        device=device,
        dtype=torch.bfloat16,
    )
    w_full = module.weight.detach().clone()
    bias_full = module.bias.detach().clone()
    module = parallelize_module(module, mesh, NVFP4ColwiseParallel())

    assert module.process_group is not None
    assert module.world_size == world_size
    assert module.tensor_parallel_style == "colwise"
    assert isinstance(module.weight, DTensor)
    assert isinstance(module.bias, DTensor)
    assert module.weight.placements == (Shard(0),)
    assert module.bias.placements == (Shard(0),)

    x_full = torch.randn(M, K, dtype=torch.bfloat16, device=device)
    dy_full = torch.randn(M, N, dtype=torch.bfloat16, device=device)
    x_local = (
        x_full[rank * M_per_rank : (rank + 1) * M_per_rank, :]
        .contiguous()
        .detach()
        .requires_grad_(True)
    )
    dy_local = dy_full[:, rank * N_per_rank : (rank + 1) * N_per_rank].contiguous()

    y = module(x_local)
    y.backward(dy_local)

    weight_grad = module.weight.grad.to_local()
    bias_grad = module.bias.grad.to_local()
    assert x_local.grad is not None, "x_local.grad is None"
    assert weight_grad is not None, "weight grad is None"
    assert bias_grad is not None, "bias grad is None"
    assert y.shape == (
        M,
        N_per_rank,
    ), f"Rank {rank}: expected ({M}, {N_per_rank}), got {y.shape}"
    assert x_local.grad.shape == (M_per_rank, K)
    assert weight_grad.shape == (N_per_rank, K)
    assert bias_grad.shape == (N_per_rank,)

    x_ref = x_full.float().detach().requires_grad_(True)
    w_ref = w_full.float().detach().requires_grad_(True)
    y_ref = x_ref @ w_ref.t() + bias_full.float()
    y_ref.backward(dy_full.float())

    y_ref_shard = y_ref[:, rank * N_per_rank : (rank + 1) * N_per_rank]
    dx_ref = x_ref.grad[rank * M_per_rank : (rank + 1) * M_per_rank, :]
    dw_ref = w_ref.grad[rank * N_per_rank : (rank + 1) * N_per_rank, :]
    db_ref = dy_local.sum(dim=0)

    SQNR_THRESHOLD = 15.0
    DX_SQNR_THRESHOLD = 14.0
    DW_SQNR_THRESHOLD = 14.0
    y_sqnr = compute_error(y_ref_shard, y.float())
    dx_sqnr = compute_error(dx_ref, x_local.grad.float())
    dw_sqnr = compute_error(dw_ref, weight_grad.float())
    assert y_sqnr >= SQNR_THRESHOLD, (
        f"Forward SQNR {y_sqnr:.2f} dB < {SQNR_THRESHOLD} dB"
    )
    assert dx_sqnr >= DX_SQNR_THRESHOLD, (
        f"dx SQNR {dx_sqnr:.2f} dB < {DX_SQNR_THRESHOLD} dB"
    )
    assert dw_sqnr >= DW_SQNR_THRESHOLD, (
        f"dw SQNR {dw_sqnr:.2f} dB < {DW_SQNR_THRESHOLD} dB"
    )
    torch.testing.assert_close(bias_grad, db_ref, atol=0, rtol=0)


@pytest.mark.skipif(not has_triton(), reason="unsupported without triton")
@pytest.mark.skipif(not is_sm_at_least_100(), reason="Requires SM100+")
@pytest.mark.skipif(
    not torch_version_at_least("2.10.0"), reason="torch.compile requires PyTorch 2.10+"
)
def test_row_single_rank_equivalence(distributed_env: DeviceMesh):
    """Verify the row-parallel autograd function matches the single-GPU NVFP4 path at world_size=1."""
    from torchao.prototype.mx_formats.nvfp4_linear import nvfp4_mm_triton

    mesh = distributed_env
    device = mesh.device_type
    rank = dist.get_rank()
    pg = dist.new_group([0])
    M, K, N = 256, 256, 256
    if rank != 0:
        dist.barrier()
        return

    torch.manual_seed(17)
    x = torch.randn(M, K, dtype=torch.bfloat16, device=device)
    w = torch.randn(N, K, dtype=torch.bfloat16, device=device)
    bias = torch.randn(N, dtype=torch.bfloat16, device=device)
    sr_seed = torch.randint(-(2**63), 2**63 - 1, (1,), dtype=torch.int64, device=device)

    sr_seed_ref = sr_seed.clone()
    y_ref = nvfp4_mm_triton.apply(
        x.clone(), w.clone(), bias.clone(), sr_seed_ref, _TP_RHT_SIGN_VECTOR
    )

    sr_seed_tp = sr_seed.clone()
    y_tp = nvfp4_row_parallel_mm.apply(
        x.clone(), w.clone(), bias.clone(), sr_seed_tp, pg, 1
    )

    torch.testing.assert_close(y_ref, y_tp, atol=1e-2, rtol=1e-2)
    dist.barrier()


@pytest.mark.skipif(not has_triton(), reason="unsupported without triton")
@pytest.mark.skipif(not is_sm_at_least_100(), reason="Requires SM100+")
@pytest.mark.skipif(
    not torch_version_at_least("2.10.0"), reason="torch.compile requires PyTorch 2.10+"
)
def test_row_forward(distributed_env: DeviceMesh):
    """Verify row-parallel forward output shape, dtype, and SQNR vs fp32 reference."""
    mesh = distributed_env
    device = mesh.device_type
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    tp_group = mesh.get_group()
    M, K, N = 512, 256, 512

    assert M % world_size == 0 and K % world_size == 0
    M_per_rank = M // world_size
    K_per_rank = K // world_size

    torch.manual_seed(19)
    x_full = torch.randn(M, K, dtype=torch.bfloat16, device=device)
    w_full = torch.randn(N, K, dtype=torch.bfloat16, device=device)
    bias = torch.randn(N, dtype=torch.bfloat16, device=device)

    x_local = x_full[:, rank * K_per_rank : (rank + 1) * K_per_rank].contiguous()
    w_local = w_full[:, rank * K_per_rank : (rank + 1) * K_per_rank].contiguous()
    sr_seed = torch.randint(-(2**63), 2**63 - 1, (1,), dtype=torch.int64, device=device)

    y = nvfp4_row_parallel_mm.apply(
        x_local, w_local, bias, sr_seed, tp_group, world_size
    )

    assert y.shape == (
        M_per_rank,
        N,
    ), f"Rank {rank}: expected ({M_per_rank}, {N}), got {y.shape}"
    assert y.dtype == torch.bfloat16, f"Expected bfloat16, got {y.dtype}"
    assert not y.isnan().any(), "Output contains NaN"

    y_ref_full = x_full.float() @ w_full.float().t() + bias.float()
    y_ref_shard = y_ref_full[rank * M_per_rank : (rank + 1) * M_per_rank, :]
    sqnr = compute_error(y_ref_shard, y.float())
    SQNR_THRESHOLD = 15.0
    assert sqnr >= SQNR_THRESHOLD, f"Forward SQNR {sqnr:.2f} dB < {SQNR_THRESHOLD} dB"


@pytest.mark.skipif(not has_triton(), reason="unsupported without triton")
@pytest.mark.skipif(not is_sm_at_least_100(), reason="Requires SM100+")
@pytest.mark.skipif(
    not torch_version_at_least("2.10.0"), reason="torch.compile requires PyTorch 2.10+"
)
def test_row_backward(distributed_env: DeviceMesh):
    """Verify row-parallel backward gradient shapes and SQNR vs fp32 reference."""
    mesh = distributed_env
    device = mesh.device_type
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    tp_group = mesh.get_group()
    M, K, N = 512, 256, 512

    assert M % world_size == 0 and K % world_size == 0
    M_per_rank = M // world_size
    K_per_rank = K // world_size

    torch.manual_seed(23)
    x_full = torch.randn(M, K, dtype=torch.bfloat16, device=device)
    w_full = torch.randn(N, K, dtype=torch.bfloat16, device=device)
    bias = torch.randn(N, dtype=torch.bfloat16, device=device, requires_grad=True)
    dy_full = torch.randn(M, N, dtype=torch.bfloat16, device=device)

    x_local = (
        x_full[:, rank * K_per_rank : (rank + 1) * K_per_rank]
        .contiguous()
        .detach()
        .requires_grad_(True)
    )
    w_local = (
        w_full[:, rank * K_per_rank : (rank + 1) * K_per_rank]
        .contiguous()
        .detach()
        .requires_grad_(True)
    )
    dy_local = dy_full[rank * M_per_rank : (rank + 1) * M_per_rank, :].contiguous()
    sr_seed = torch.randint(-(2**63), 2**63 - 1, (1,), dtype=torch.int64, device=device)

    y = nvfp4_row_parallel_mm.apply(
        x_local, w_local, bias, sr_seed, tp_group, world_size
    )
    y.backward(dy_local)

    assert x_local.grad is not None, "x_local.grad is None"
    assert w_local.grad is not None, "w_local.grad is None"
    assert bias.grad is not None, "bias.grad is None"

    assert x_local.grad.shape == (
        M,
        K_per_rank,
    ), f"Rank {rank}: dx shape expected ({M}, {K_per_rank}), got {x_local.grad.shape}"
    assert w_local.grad.shape == (
        N,
        K_per_rank,
    ), f"Rank {rank}: dw shape expected ({N}, {K_per_rank}), got {w_local.grad.shape}"
    assert bias.grad.shape == (N,), (
        f"Rank {rank}: db shape expected ({N},), got {bias.grad.shape}"
    )
    assert not x_local.grad.isnan().any(), "dx contains NaN"
    assert not w_local.grad.isnan().any(), "dw contains NaN"
    assert not bias.grad.isnan().any(), "db contains NaN"

    x_ref = x_full.float().detach().requires_grad_(True)
    w_ref = w_full.float().detach().requires_grad_(True)
    y_ref = x_ref @ w_ref.t() + bias.detach().float()
    y_ref.backward(dy_full.float())

    dx_ref = x_ref.grad[:, rank * K_per_rank : (rank + 1) * K_per_rank]
    dw_ref = w_ref.grad[:, rank * K_per_rank : (rank + 1) * K_per_rank]
    db_ref = dy_local.sum(dim=0)
    # Row-parallel bias is replicated, so each rank should see the full bias grad.
    dist.all_reduce(db_ref, op=dist.ReduceOp.SUM, group=tp_group)

    dx_sqnr = compute_error(dx_ref, x_local.grad.float())
    dw_sqnr = compute_error(dw_ref, w_local.grad.float())
    DX_SQNR_THRESHOLD = 14.0
    DW_SQNR_THRESHOLD = 14.0
    assert dx_sqnr >= DX_SQNR_THRESHOLD, (
        f"dx SQNR {dx_sqnr:.2f} dB < {DX_SQNR_THRESHOLD} dB"
    )
    assert dw_sqnr >= DW_SQNR_THRESHOLD, (
        f"dw SQNR {dw_sqnr:.2f} dB < {DW_SQNR_THRESHOLD} dB"
    )
    torch.testing.assert_close(bias.grad, db_ref, atol=0, rtol=0)


@pytest.mark.skipif(not has_triton(), reason="unsupported without triton")
@pytest.mark.skipif(not is_sm_at_least_100(), reason="Requires SM100+")
@pytest.mark.skipif(
    not torch_version_at_least("2.10.0"), reason="torch.compile requires PyTorch 2.10+"
)
def test_row_parallelize_module(distributed_env: DeviceMesh):
    """Verify NVFP4RowwiseParallel works through parallelize_module."""
    mesh = distributed_env
    device = mesh.device_type
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    tp_group = mesh.get_group()
    M, K, N = 512, 256, 512

    assert M % world_size == 0 and K % world_size == 0
    M_per_rank = M // world_size
    K_per_rank = K // world_size

    torch.manual_seed(31)
    module = NVFP4Linear(
        K,
        N,
        bias=True,
        kernel_preference=KernelPreference.TRITON,
        device=device,
        dtype=torch.bfloat16,
    )
    w_full = module.weight.detach().clone()
    bias_full = module.bias.detach().clone()
    module = parallelize_module(module, mesh, NVFP4RowwiseParallel())

    assert module.process_group is not None
    assert module.world_size == world_size
    assert module.tensor_parallel_style == "rowwise"
    assert isinstance(module.weight, DTensor)
    assert isinstance(module.bias, DTensor)
    assert module.weight.placements == (Shard(1),)
    assert module.bias.placements == (Replicate(),)

    x_full = torch.randn(M, K, dtype=torch.bfloat16, device=device)
    dy_full = torch.randn(M, N, dtype=torch.bfloat16, device=device)
    x_local = (
        x_full[:, rank * K_per_rank : (rank + 1) * K_per_rank]
        .contiguous()
        .detach()
        .requires_grad_(True)
    )
    dy_local = dy_full[rank * M_per_rank : (rank + 1) * M_per_rank, :].contiguous()

    y = module(x_local)
    y.backward(dy_local)

    weight_grad = module.weight.grad.to_local()
    bias_grad = module.bias.grad.to_local()
    assert x_local.grad is not None, "x_local.grad is None"
    assert weight_grad is not None, "weight grad is None"
    assert bias_grad is not None, "bias grad is None"
    assert y.shape == (
        M_per_rank,
        N,
    ), f"Rank {rank}: expected ({M_per_rank}, {N}), got {y.shape}"
    assert x_local.grad.shape == (M, K_per_rank)
    assert weight_grad.shape == (N, K_per_rank)
    assert bias_grad.shape == (N,)

    x_ref = x_full.float().detach().requires_grad_(True)
    w_ref = w_full.float().detach().requires_grad_(True)
    y_ref = x_ref @ w_ref.t() + bias_full.float()
    y_ref.backward(dy_full.float())

    y_ref_shard = y_ref[rank * M_per_rank : (rank + 1) * M_per_rank, :]
    dx_ref = x_ref.grad[:, rank * K_per_rank : (rank + 1) * K_per_rank]
    dw_ref = w_ref.grad[:, rank * K_per_rank : (rank + 1) * K_per_rank]
    db_ref = dy_local.sum(dim=0)
    dist.all_reduce(db_ref, op=dist.ReduceOp.SUM, group=tp_group)

    SQNR_THRESHOLD = 15.0
    DX_SQNR_THRESHOLD = 14.0
    DW_SQNR_THRESHOLD = 14.0
    y_sqnr = compute_error(y_ref_shard, y.float())
    dx_sqnr = compute_error(dx_ref, x_local.grad.float())
    dw_sqnr = compute_error(dw_ref, weight_grad.float())
    assert y_sqnr >= SQNR_THRESHOLD, (
        f"Forward SQNR {y_sqnr:.2f} dB < {SQNR_THRESHOLD} dB"
    )
    assert dx_sqnr >= DX_SQNR_THRESHOLD, (
        f"dx SQNR {dx_sqnr:.2f} dB < {DX_SQNR_THRESHOLD} dB"
    )
    assert dw_sqnr >= DW_SQNR_THRESHOLD, (
        f"dw SQNR {dw_sqnr:.2f} dB < {DW_SQNR_THRESHOLD} dB"
    )
    torch.testing.assert_close(bias_grad, db_ref, atol=0, rtol=0)


@pytest.mark.skipif(not has_triton(), reason="unsupported without triton")
@pytest.mark.skipif(not is_sm_at_least_100(), reason="Requires SM100+")
@pytest.mark.skipif(
    not torch_version_at_least("2.10.0"), reason="torch.compile requires PyTorch 2.10+"
)
def test_mlp_colwise_rowwise_parallelize_module(distributed_env: DeviceMesh):
    """Verify colwise-to-rowwise NVFP4 MLP composition through parallelize_module."""
    mesh = distributed_env
    device = mesh.device_type
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    M, K, H = 512, 256, 512

    assert M % world_size == 0 and H % world_size == 0
    M_per_rank = M // world_size
    H_per_rank = H // world_size

    torch.manual_seed(37)
    model = NVFP4MLP(K, H, device=device, dtype=torch.bfloat16)
    weights = {name: param.detach().clone() for name, param in model.named_parameters()}
    model = _parallelize_nvfp4_mlp(model, mesh)

    assert model.w1.tensor_parallel_style == "colwise"
    assert model.w2.tensor_parallel_style == "colwise"
    assert model.out_proj.tensor_parallel_style == "rowwise"
    assert model.w1.weight.placements == (Shard(0),)
    assert model.w2.weight.placements == (Shard(0),)
    assert model.out_proj.weight.placements == (Shard(1),)
    assert model.w1.bias.placements == (Shard(0),)
    assert model.w2.bias.placements == (Shard(0),)
    assert model.out_proj.bias.placements == (Replicate(),)

    x_full = torch.randn(M, K, dtype=torch.bfloat16, device=device)
    x_local = x_full[rank * M_per_rank : (rank + 1) * M_per_rank, :].contiguous()

    y = model(x_local)

    assert y.shape == (
        M_per_rank,
        K,
    ), f"Rank {rank}: expected ({M_per_rank}, {K}), got {y.shape}"
    dy = torch.randn_like(y)
    y.backward(dy)

    assert y.dtype == torch.bfloat16
    assert not y.isnan().any(), "MLP output contains NaN"

    local_grads = {
        "w1.weight": model.w1.weight.grad.to_local(),
        "w1.bias": model.w1.bias.grad.to_local(),
        "w2.weight": model.w2.weight.grad.to_local(),
        "w2.bias": model.w2.bias.grad.to_local(),
        "out_proj.weight": model.out_proj.weight.grad.to_local(),
        "out_proj.bias": model.out_proj.bias.grad.to_local(),
    }
    expected_grad_shapes = {
        "w1.weight": (H_per_rank, K),
        "w1.bias": (H_per_rank,),
        "w2.weight": (H_per_rank, K),
        "w2.bias": (H_per_rank,),
        "out_proj.weight": (K, H_per_rank),
        "out_proj.bias": (K,),
    }
    for name, grad in local_grads.items():
        assert grad is not None, f"{name}.grad is None"
        assert grad.shape == expected_grad_shapes[name]
        assert not grad.isnan().any(), f"{name}.grad contains NaN"

    y_ref = _fp32_mlp_reference(x_full, weights)
    y_ref_shard = y_ref[rank * M_per_rank : (rank + 1) * M_per_rank, :]
    sqnr = compute_error(y_ref_shard, y.float())
    SQNR_THRESHOLD = 10.0
    assert sqnr >= SQNR_THRESHOLD, f"MLP SQNR {sqnr:.2f} dB < {SQNR_THRESHOLD} dB"


@pytest.mark.skipif(not has_triton(), reason="unsupported without triton")
@pytest.mark.skipif(not is_sm_at_least_100(), reason="Requires SM100+")
@pytest.mark.skipif(
    not torch_version_at_least("2.10.0"), reason="torch.compile requires PyTorch 2.10+"
)
def test_mlp_colwise_rowwise_parallelize_module_cuda_graph_compile(
    distributed_env: DeviceMesh,
):
    """Verify composed NVFP4 TP MLP works under torch.compile CUDA graphs."""
    mesh = distributed_env
    device = mesh.device_type
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    M, K, H = 512, 256, 512

    assert M % world_size == 0
    M_per_rank = M // world_size

    prepare_for_cuda_graph(torch.device(device), sign_vectors=(_TP_RHT_SIGN_VECTOR,))
    torch.manual_seed(41)
    model = NVFP4MLP(K, H, device=device, dtype=torch.bfloat16)
    model = _parallelize_nvfp4_mlp(model, mesh)
    compiled_model = torch.compile(model, mode="reduce-overhead", fullgraph=True)
    compiled_bwd = torch.compile(fullgraph=True, mode="reduce-overhead")

    x_full = torch.randn(M, K, dtype=torch.bfloat16, device=device)
    x_local = (
        x_full[rank * M_per_rank : (rank + 1) * M_per_rank, :]
        .contiguous()
        .detach()
        .requires_grad_(True)
    )
    target = torch.randn(M_per_rank, K, dtype=torch.bfloat16, device=device)

    for _ in range(3):
        model.zero_grad(set_to_none=True)
        x_local.grad = None
        with torch._dynamo.compiled_autograd._enable(compiled_bwd):
            y = compiled_model(x_local)
            assert y.shape == (M_per_rank, K)
            loss = F.mse_loss(y.float(), target.float())
            loss.backward()

    assert x_local.grad is not None
    assert not x_local.grad.isnan().any(), "compiled MLP input grad contains NaN"
    for name, param in model.named_parameters():
        assert param.grad is not None, f"{name}.grad is None"
        grad = param.grad.to_local() if isinstance(param.grad, DTensor) else param.grad
        assert not grad.isnan().any(), f"{name}.grad contains NaN"
