import pytest
import torch
import torch.distributed as dist

from torchao.prototype.mx_formats.mx_tensor import MXTensor
from torchao.utils import is_sm_at_least_90, torch_version_at_least

if not torch_version_at_least("2.7.0"):
    pytest.skip("Unsupported PyTorch version", allow_module_level=True)


def setup_distributed():
    dist.init_process_group("nccl")
    # seed must be the same in all processes
    torch.manual_seed(42)
    local_rank = torch.distributed.get_rank()
    torch.cuda.set_device(local_rank)
    return local_rank


def _test_allgather(local_rank):
    golden_qdata = (
        torch.randint(0, 256, (256, 512), dtype=torch.uint8)
        .to(torch.float8_e5m2)
        .to(local_rank)
    )

    # Random scale factors (typically float32 or uint8 for e8m0)
    golden_scale = (
        torch.randint(0, 256, (256, 16), dtype=torch.uint8)
        .view(torch.float8_e8m0fnu)
        .to(local_rank)
    )

    # Create golden MXTensor
    golden_mx = MXTensor(
        golden_qdata,
        golden_scale,
        elem_dtype=torch.float8_e5m2,
        block_size=32,
        orig_dtype=torch.float32,
        kernel_preference=None,
        act_quant_kwargs=None,
        is_swizzled_scales=None,
    )

    local_rank = torch.distributed.get_rank()
    world_size = torch.distributed.get_world_size()

    # Each rank gets its shard (split along dim 0)
    shard_size = golden_qdata.shape[0] // world_size  # 2 rows per rank
    start_idx = local_rank * shard_size
    end_idx = (local_rank + 1) * shard_size

    # Create local MXTensor from shard
    local_mx = MXTensor(
        golden_qdata[start_idx:end_idx].clone().to(local_rank),
        golden_scale[start_idx:end_idx].clone().to(local_rank),
        elem_dtype=torch.float8_e5m2,
        block_size=32,
        orig_dtype=torch.float32,
        kernel_preference=None,
        act_quant_kwargs=None,
        is_swizzled_scales=None,
    )

    # Perform all_gather
    gathered_mx = torch.ops._c10d_functional.all_gather_into_tensor.default(
        local_mx,
        world_size,
        "0",
    )
    gathered_mx = torch.ops._c10d_functional.wait_tensor.default(gathered_mx)

    # Verify type
    assert isinstance(gathered_mx, MXTensor), (
        f"Expected MXTensor, got {type(gathered_mx)}"
    )

    # Verify shape
    assert gathered_mx.shape == golden_mx.shape, (
        f"Shape mismatch: {gathered_mx.shape} vs {golden_mx.shape}"
    )

    # Verify qdata matches golden exactly
    if not torch.equal(gathered_mx.qdata, golden_qdata):
        assert False, "qdata mismatch"

    # Verify scale matches golden exactly
    if not torch.equal(
        gathered_mx.scale.view(torch.uint8),
        golden_scale.view(torch.uint8),
    ):
        assert False, "scale mismatch"

    assert gathered_mx.block_size == 32


if __name__ == "__main__":
    local_rank = setup_distributed()

    assert is_sm_at_least_90() == True, "SM must be > 9.0"

    try:
        _test_allgather(local_rank)
    except Exception as e:
        raise e

    torch.distributed.destroy_process_group()
