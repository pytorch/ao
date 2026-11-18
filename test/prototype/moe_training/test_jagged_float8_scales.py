# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD 3-Clause license found in the
# LICENSE file in the root directory of this source tree.

import pytest
import torch

from torchao.utils import is_MI300, is_MI350, is_sm_at_least_89

if not (
    torch.cuda.is_available() and (is_sm_at_least_89() or is_MI300() or is_MI350())
):
    pytest.skip(
        "Requires FP8-capable GPU (CUDA SM89+, MI300, or MI350)",
        allow_module_level=True,
    )

pytest.importorskip("triton", reason="Triton required to run this test")

from torchao.float8.config import e4m3_dtype
from torchao.prototype.moe_training.kernels.jagged_float8_scales import (
    triton_fp8_per_group_colwise_scales,
)
from torchao.prototype.moe_training.utils import (
    _is_column_major,
    torch_to_float8_per_group_colwise,
)


@pytest.mark.parametrize("round_scales_to_power_of_2", [True, False])
def test_jagged_colwise_scales_uneven_and_dynamic_groups(
    round_scales_to_power_of_2: bool,
):
    device = "cuda"
    torch.manual_seed(0)

    # 1. Uneven group sizes where a group exceeds the mean group size (including
    # single-group, empty leading/middle groups, and >2048 fallback to loop kernel).
    cases = [
        (512, 32, [200, 512], [480]),
        (1024, 96, [10, 30, 700, 1024], [650, 1000]),
        (512, 64, [512], [500]),
        (2700, 64, [100, 2700], [2650]),
    ]
    for total_rows, cols, ends, spike_rows in cases:
        x = torch.randn(total_rows, cols, dtype=torch.bfloat16, device=device)
        x = x.t().contiguous().t()
        for r in spike_rows:
            x[r] = 32.0
        offs = torch.tensor(ends, dtype=torch.int32, device=device)
        ref_fp8, ref_scales = torch_to_float8_per_group_colwise(
            x,
            offs,
            target_dtype=e4m3_dtype,
            round_scales_to_power_of_2=round_scales_to_power_of_2,
        )
        ker_fp8, ker_scales = triton_fp8_per_group_colwise_scales(
            x,
            offs,
            output_dtype=e4m3_dtype,
            round_scales_to_power_of_2=round_scales_to_power_of_2,
        )
        assert _is_column_major(ker_fp8)
        if round_scales_to_power_of_2:
            assert torch.equal(ref_scales, ker_scales), (
                f"scales mismatch for ends={ends}"
            )
            assert torch.equal(ref_fp8, ker_fp8), f"fp8 mismatch for ends={ends}"
        else:
            torch.testing.assert_close(ker_scales, ref_scales, rtol=1e-6, atol=1e-6)
            torch.testing.assert_close(
                ker_fp8.float(), ref_fp8.float(), rtol=0.125, atol=1e-3
            )

    # Empty leading/middle group handling (0 rows in group 0 or group 1)
    x_empty = (
        torch.randn(512, 64, dtype=torch.bfloat16, device=device).t().contiguous().t()
    )
    x_empty[480] = 32.0
    ref_single_fp8, ref_single_scales = torch_to_float8_per_group_colwise(
        x_empty,
        torch.tensor([512], dtype=torch.int32, device=device),
        target_dtype=e4m3_dtype,
        round_scales_to_power_of_2=round_scales_to_power_of_2,
    )
    ker_empty_fp8, ker_empty_scales = triton_fp8_per_group_colwise_scales(
        x_empty,
        torch.tensor([0, 512], dtype=torch.int32, device=device),
        output_dtype=e4m3_dtype,
        round_scales_to_power_of_2=round_scales_to_power_of_2,
    )
    if round_scales_to_power_of_2:
        assert torch.equal(ker_empty_fp8, ref_single_fp8)
        assert torch.equal(ker_empty_scales[64:], ref_single_scales.reshape(-1))
    else:
        torch.testing.assert_close(
            ker_empty_scales[64:], ref_single_scales.reshape(-1), rtol=1e-6, atol=1e-6
        )
        torch.testing.assert_close(
            ker_empty_fp8.float(), ref_single_fp8.float(), rtol=0.125, atol=1e-3
        )

    # 2. Same K=1024 across calls with increasing max group size (256 -> 512 -> 960)
    # to ensure autotune does not reuse a stale smaller MAX_GROUP_SIZE bound.
    x_dyn = (
        torch.ones((1024, 32), dtype=torch.bfloat16, device=device).t().contiguous().t()
    )
    for ends, spikes in (
        ([256, 512, 768, 1024], []),
        ([512, 1024], [480, 992]),
        ([64, 1024], [992]),
    ):
        x_call = x_dyn.clone()
        for r in spikes:
            x_call[r] = 32.0
        offs = torch.tensor(ends, dtype=torch.int32, device=device)
        ref_fp8, ref_scales = torch_to_float8_per_group_colwise(
            x_call,
            offs,
            target_dtype=e4m3_dtype,
            round_scales_to_power_of_2=round_scales_to_power_of_2,
        )
        ker_fp8, ker_scales = triton_fp8_per_group_colwise_scales(
            x_call,
            offs,
            output_dtype=e4m3_dtype,
            round_scales_to_power_of_2=round_scales_to_power_of_2,
        )
        assert torch.equal(ref_fp8, ker_fp8)
        assert torch.equal(ref_scales, ker_scales)

    # 3. torch.compile(fullgraph=True) and CUDA graph capture compatibility
    x_comp = (
        torch.ones((512, 32), dtype=torch.bfloat16, device=device).t().contiguous().t()
    )
    x_comp[480] = 32.0
    offs_comp = torch.tensor([200, 512], dtype=torch.int32, device=device)
    eager_fp8, eager_scales = triton_fp8_per_group_colwise_scales(
        x_comp,
        offs_comp,
        output_dtype=e4m3_dtype,
        round_scales_to_power_of_2=round_scales_to_power_of_2,
    )
    compiled_fn = torch.compile(triton_fp8_per_group_colwise_scales, fullgraph=True)
    try:
        comp_fp8, comp_scales = compiled_fn(
            x_comp,
            offs_comp,
            output_dtype=e4m3_dtype,
            round_scales_to_power_of_2=round_scales_to_power_of_2,
        )
    finally:
        torch._dynamo.reset()
    assert torch.equal(eager_fp8, comp_fp8)
    assert torch.equal(eager_scales, comp_scales)

    g = torch.cuda.CUDAGraph()
    s = torch.cuda.Stream()
    with torch.cuda.stream(s):
        triton_fp8_per_group_colwise_scales(
            x_comp,
            offs_comp,
            output_dtype=e4m3_dtype,
            round_scales_to_power_of_2=round_scales_to_power_of_2,
        )
    torch.cuda.synchronize()
    with torch.cuda.graph(g, stream=s):
        graph_fp8, graph_scales = triton_fp8_per_group_colwise_scales(
            x_comp,
            offs_comp,
            output_dtype=e4m3_dtype,
            round_scales_to_power_of_2=round_scales_to_power_of_2,
        )
    g.replay()
    assert torch.equal(eager_fp8, graph_fp8)
    assert torch.equal(eager_scales, graph_scales)
