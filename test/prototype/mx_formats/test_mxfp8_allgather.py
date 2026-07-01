import traceback

import torch
import torch.distributed as dist

from torchao.prototype.mx_formats.mx_tensor import MXTensor


def setup_distributed():
    device = torch.accelerator.current_accelerator()
    dist.init_process_group()
    # seed must be the same in all processes
    torch.manual_seed(42)
    local_rank = torch.distributed.get_rank()
    torch.accelerator.set_device_index(local_rank)
    return torch.device(device.type, local_rank)


def print_once(msg):
    if torch.distributed.get_rank() == 0:
        print(msg)


def _test_allgather(device):
    golden_qdata = (
        torch.randint(0, 256, (256, 512), dtype=torch.uint8)
        .to(torch.float8_e5m2)
        .to(device)
    )

    # Random scale factors (typically float32 or uint8 for e8m0)
    golden_scale = (
        torch.randint(0, 256, (256, 16), dtype=torch.uint8)
        .view(torch.float8_e8m0fnu)
        .to(device)
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
        is_swizzled_scales=False,
    )

    local_rank = torch.distributed.get_rank()
    world_size = torch.distributed.get_world_size()

    # Each rank gets its shard (split along dim 0)
    shard_size = golden_qdata.shape[0] // world_size  # 2 rows per rank
    start_idx = local_rank * shard_size
    end_idx = (local_rank + 1) * shard_size

    # Create local MXTensor from shard
    local_mx = MXTensor(
        golden_qdata[start_idx:end_idx].clone().to(device),
        golden_scale[start_idx:end_idx].clone().to(device),
        elem_dtype=torch.float8_e5m2,
        block_size=32,
        orig_dtype=torch.float32,
        kernel_preference=None,
        act_quant_kwargs=None,
        is_swizzled_scales=False,
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
    device = setup_distributed()
    tests = [
        _test_allgather,
    ]

    failed_cnt = 0
    for test in tests:
        try:
            test(device)
        except Exception as e:
            print_once(f"\033[31m\u274c FAILED {test.__name__}: {e}\033[0m")
            print_once(traceback.format_exc())
            failed_cnt += 1
        else:
            print_once(f"\033[32m\u2705 PASSED {test.__name__}\033[0m")

    print_once(f"FAILED: {failed_cnt} PASSED: {len(tests) - failed_cnt}")

    torch.distributed.destroy_process_group()
