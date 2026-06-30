# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD 3-Clause license found in the
# LICENSE file in the root directory of this source tree.

import pytest
import torch
from torch.distributed.tensor import DTensor, Partial, Replicate, Shard
from torch.nn import functional as F
from torch.testing._internal.common_utils import run_tests
from torch.testing._internal.distributed._tensor.common_dtensor import (
    DTensorTestBase,
    with_comms,
)

from test.prototype.fp8_blockwise_distributed_utils import (
    BlockwiseFP8DTensorTestMixin,
)
from torchao.utils import is_MI300, is_MI350, is_sm_at_least_90

if not (
    torch.cuda.is_available() and (is_sm_at_least_90() or is_MI300() or is_MI350())
):
    pytest.skip(
        "Requires FP8-capable GPU (CUDA SM90+, MI300, or MI350)",
        allow_module_level=True,
    )

pytest.importorskip("triton", reason="Triton required to run this test")

from torchao.prototype.blockwise_fp8_training.grouped_kernels import (
    emulated_blockwise_scaled_grouped_mm,
    triton_fp8_blockwise_weight_quant_grouped_rhs,
    triton_fp8_blockwise_weight_quant_grouped_transposed_rhs,
)
from torchao.prototype.blockwise_fp8_training.kernels import (
    BLOCKWISE_1X128_SCALING_TYPE,
    BLOCKWISE_128X128_SCALING_TYPE,
    _scaling_type_value,
    triton_fp8_blockwise_act_quant_lhs,
    triton_fp8_blockwise_act_quant_rhs,
    triton_fp8_blockwise_act_quant_transposed_lhs,
)
from torchao.prototype.moe_training.blockwise_fp8.grouped_mm import (
    _to_fp8_blockwise_then_emulated_scaled_grouped_mm,
)
from torchao.quantization.utils import compute_error
from torchao.testing.utils import skip_if_rocm

try:
    from ._fp8_blockwise_distributed_test_utils import (
        assert_blockwise_grouped_experts_tp_applied,
        assert_close,
        assert_dtensor_parameter_grads_match,
        assert_dtensor_parameter_values_match,
        assert_parameters_are_training_wrappers,
        full_tensor,
        make_blockwise_grouped_experts_pair,
        parallelize_blockwise_grouped_experts_tensor_parallel,
    )
except ImportError:
    from _fp8_blockwise_distributed_test_utils import (  # type: ignore[no-redef]
        assert_blockwise_grouped_experts_tp_applied,
        assert_close,
        assert_dtensor_parameter_grads_match,
        assert_dtensor_parameter_values_match,
        assert_parameters_are_training_wrappers,
        full_tensor,
        make_blockwise_grouped_experts_pair,
        parallelize_blockwise_grouped_experts_tensor_parallel,
    )

torch._dynamo.config.cache_size_limit = 1000


def _make_column_major_weight_t(E: int, N: int, K: int) -> torch.Tensor:
    weight = torch.randn(E, N, K, dtype=torch.bfloat16, device="cuda")
    return weight.contiguous().transpose(-2, -1)


@skip_if_rocm("ROCm not supported")
@pytest.mark.parametrize(
    "offs,pad_token_groups_for_grouped_mm",
    [
        (torch.tensor([256, 512], dtype=torch.int32), False),
        (torch.tensor([129, 384, 500], dtype=torch.int32), True),
    ],
)
def test_fp8_blockwise_emulated_grouped_mm_fwd_bwd(
    offs, pad_token_groups_for_grouped_mm
):
    torch.manual_seed(0)
    offs = offs.cuda()
    E = offs.numel()
    M = int(offs[-1].item())
    K, N = 256, 256
    A = torch.randn(M, K, dtype=torch.bfloat16, device="cuda", requires_grad=True)
    B_t = _make_column_major_weight_t(E, N, K).requires_grad_(True)

    A_ref = A.detach().clone().requires_grad_(True)
    B_t_ref = B_t.detach().clone().requires_grad_(True)

    out = _to_fp8_blockwise_then_emulated_scaled_grouped_mm(
        A,
        B_t,
        offs,
        pad_token_groups_for_grouped_mm=pad_token_groups_for_grouped_mm,
    )
    ref = torch._grouped_mm(A_ref, B_t_ref, offs=offs, out_dtype=torch.bfloat16)

    assert out.shape == ref.shape
    assert out.dtype == torch.bfloat16
    assert compute_error(ref, out) >= 27.0

    out.float().square().mean().backward()
    ref.float().square().mean().backward()

    assert compute_error(A_ref.grad, A.grad) >= 26.0
    assert compute_error(B_t_ref.grad, B_t.grad) >= 26.0


@skip_if_rocm("ROCm not supported")
def test_fp8_blockwise_emulated_grouped_mm_compile_aligned_groups():
    E, M, K, N = 2, 256, 128, 128
    A = torch.randn(E * M, K, dtype=torch.bfloat16, device="cuda")
    B_t = _make_column_major_weight_t(E, N, K)
    offs = torch.arange(M, (E + 1) * M, M, device="cuda", dtype=torch.int32)

    compiled = torch.compile(
        _to_fp8_blockwise_then_emulated_scaled_grouped_mm, fullgraph=True
    )
    out = compiled(A, B_t, offs, pad_token_groups_for_grouped_mm=False)

    assert out.shape == (E * M, N)
    assert out.dtype == torch.bfloat16


@skip_if_rocm("ROCm not supported")
@pytest.mark.parametrize(
    "offs,pad_token_groups_for_grouped_mm",
    [
        (torch.tensor([128, 256], dtype=torch.int32), False),
        (torch.tensor([129, 256], dtype=torch.int32), True),
    ],
)
def test_fp8_blockwise_quantized_grouped_experts_production_path_fwd_bwd(
    offs,
    pad_token_groups_for_grouped_mm,
):
    device = torch.device("cuda")
    offs = offs.to(device)
    ref_model, model = make_blockwise_grouped_experts_pair(
        seed=123,
        device=device,
        pad_token_groups_for_grouped_mm=pad_token_groups_for_grouped_mm,
        quantize_ref_model=False,
    )
    assert_parameters_are_training_wrappers(model.parameters())
    ref_optim = torch.optim.SGD(ref_model.parameters(), lr=1e-2)
    optim = torch.optim.SGD(model.parameters(), lr=1e-2)

    torch.manual_seed(500)
    x = torch.randn(256, 256, dtype=torch.bfloat16, device=device).requires_grad_(True)
    x_ref = x.detach().clone().requires_grad_(True)
    target = torch.randn_like(x)

    ref_optim.zero_grad(set_to_none=True)
    optim.zero_grad(set_to_none=True)

    ref_out = ref_model(x_ref, offs)
    out = model(x, offs)
    assert_close(out, ref_out, min_sqnr=23.0)

    ref_loss = F.mse_loss(ref_out, target)
    loss = F.mse_loss(out, target)
    assert_close(loss, ref_loss, atol=1e-2, rtol=1e-2)

    ref_loss.backward()
    loss.backward()
    assert x_ref.grad is not None
    assert x.grad is not None
    assert_close(x.grad, x_ref.grad, min_sqnr=22.0)
    for ref_param, param in zip(
        ref_model.parameters(), model.parameters(), strict=True
    ):
        assert ref_param.grad is not None
        assert param.grad is not None
        assert_close(param.grad, ref_param.grad, min_sqnr=22.0)

    ref_optim.step()
    optim.step()
    for ref_param, param in zip(
        ref_model.parameters(), model.parameters(), strict=True
    ):
        assert_close(param, ref_param, min_sqnr=21.0)


@skip_if_rocm("ROCm not supported")
def test_fp8_blockwise_quantized_grouped_experts_compile_fullgraph_fwd_bwd():
    from torch._dynamo.testing import CompileCounterWithBackend

    device = torch.device("cuda")
    offs = torch.tensor([128, 256], dtype=torch.int32, device=device)
    ref_model, model = make_blockwise_grouped_experts_pair(
        seed=321,
        device=device,
        pad_token_groups_for_grouped_mm=False,
        quantize_ref_model=False,
    )
    assert_parameters_are_training_wrappers(model.parameters())
    ref_params = tuple(ref_model.parameters())
    model_params = tuple(model.parameters())

    torch._dynamo.reset()
    compiled_frame_counter = CompileCounterWithBackend("inductor")

    def step(x):
        out = model(x, offs)
        grads = torch.autograd.grad(out.float().square().mean(), (x, *model_params))
        return (out.detach(), *grads)

    with torch._dynamo.config.patch(trace_autograd_ops=True):
        compiled_step = torch.compile(
            step,
            backend=compiled_frame_counter,
            fullgraph=True,
        )

    def run_once(seed: int) -> None:
        torch.manual_seed(seed)
        x = torch.randn(
            256, 256, dtype=torch.bfloat16, device=device, requires_grad=True
        )
        x_ref = x.detach().clone().requires_grad_(True)

        with torch._dynamo.config.patch(trace_autograd_ops=True):
            out, x_grad, *param_grads = compiled_step(x)
        ref_out = ref_model(x_ref, offs)
        ref_grads = torch.autograd.grad(
            ref_out.float().square().mean(), (x_ref, *ref_params)
        )

        assert_close(out, ref_out, min_sqnr=23.0)
        assert_close(x_grad, ref_grads[0], min_sqnr=22.0)
        for grad, ref_grad in zip(param_grads, ref_grads[1:], strict=True):
            assert_close(grad, ref_grad, min_sqnr=22.0)

    run_once(900)
    assert compiled_frame_counter.frame_count == 1

    run_once(901)
    assert compiled_frame_counter.frame_count == 1


class TestFP8BlockwiseGroupedMMDTensor(BlockwiseFP8DTensorTestMixin, DTensorTestBase):
    world_size = 2
    block_size = 128

    @with_comms
    @skip_if_rocm("ROCm not supported")
    def test_grouped_weight_quant_op_sharding(self):
        mesh = self._build_cuda_mesh()
        device = torch.device(f"cuda:{self.rank % torch.cuda.device_count()}")
        torch.manual_seed(123)
        B_t = _make_column_major_weight_t(E=2, N=256, K=256).to(device)

        cases = (
            (
                triton_fp8_blockwise_weight_quant_grouped_transposed_rhs,
                (
                    (Replicate(), (Replicate(), Replicate())),
                    (Shard(1), (Shard(1), Shard(1))),
                    (Shard(2), (Shard(2), Shard(2))),
                ),
            ),
            (
                triton_fp8_blockwise_weight_quant_grouped_rhs,
                (
                    (Replicate(), (Replicate(), Replicate())),
                    (Shard(1), (Shard(2), Shard(2))),
                    (Shard(2), (Shard(1), Shard(1))),
                ),
            ),
        )

        for quant_op, placement_cases in cases:
            global_outputs = quant_op(B_t, block_size=self.block_size)
            for input_placement, expected_placements in placement_cases:
                with self.subTest(
                    quant_op=getattr(quant_op, "_name", repr(quant_op)),
                    input_placement=repr(input_placement),
                ):
                    local_B_t, dist_B_t = self._dtensor_from_global(
                        mesh,
                        B_t,
                        input_placement,
                        contiguous=False,
                    )
                    local_outputs = quant_op(local_B_t, block_size=self.block_size)
                    dist_outputs = quant_op(dist_B_t, block_size=self.block_size)
                    for dist_output, local_output, global_output, placement in zip(
                        dist_outputs,
                        local_outputs,
                        global_outputs,
                        expected_placements,
                        strict=True,
                    ):
                        self._assert_dtensor_matches(
                            dist_output,
                            local_output,
                            global_output,
                            placement,
                        )

    @with_comms
    @skip_if_rocm("ROCm not supported")
    def test_grouped_mm_tp_sharding(self):
        mesh = self._build_cuda_mesh()
        device = torch.device(f"cuda:{self.rank % torch.cuda.device_count()}")
        torch.manual_seed(321)
        offs = torch.tensor([128, 256], dtype=torch.int32, device=device)
        dist_offs = DTensor.from_local(
            offs,
            mesh,
            [Replicate()],
            run_check=False,
        )
        A = torch.randn(256, 256, dtype=torch.bfloat16, device=device)
        B_t = _make_column_major_weight_t(E=2, N=256, K=256).to(device)

        # Forward output-feature sharding: A @ B_t with B_t sharded on N.
        global_A_fp8, global_A_s = triton_fp8_blockwise_act_quant_lhs(A)
        global_B_fp8, global_B_s = (
            triton_fp8_blockwise_weight_quant_grouped_transposed_rhs(B_t)
        )
        expected_global = emulated_blockwise_scaled_grouped_mm(
            global_A_fp8,
            global_B_fp8,
            global_A_s,
            _scaling_type_value(BLOCKWISE_1X128_SCALING_TYPE),
            global_B_s,
            _scaling_type_value(BLOCKWISE_128X128_SCALING_TYPE),
            offs,
            torch.bfloat16,
            self.block_size,
        )

        local_A, dist_A = self._dtensor_from_global(
            mesh, A, Replicate(), contiguous=True
        )
        local_B_t, dist_B_t = self._dtensor_from_global(
            mesh, B_t, Shard(2), contiguous=False
        )
        dist_A_fp8, dist_A_s = triton_fp8_blockwise_act_quant_lhs(dist_A)
        dist_B_fp8, dist_B_s = triton_fp8_blockwise_weight_quant_grouped_transposed_rhs(
            dist_B_t
        )
        dist_out = emulated_blockwise_scaled_grouped_mm(
            dist_A_fp8,
            dist_B_fp8,
            dist_A_s,
            _scaling_type_value(BLOCKWISE_1X128_SCALING_TYPE),
            dist_B_s,
            _scaling_type_value(BLOCKWISE_128X128_SCALING_TYPE),
            dist_offs,
            torch.bfloat16,
            self.block_size,
        )
        local_A_fp8, local_A_s = triton_fp8_blockwise_act_quant_lhs(local_A)
        local_B_fp8, local_B_s = (
            triton_fp8_blockwise_weight_quant_grouped_transposed_rhs(local_B_t)
        )
        expected_local = emulated_blockwise_scaled_grouped_mm(
            local_A_fp8,
            local_B_fp8,
            local_A_s,
            _scaling_type_value(BLOCKWISE_1X128_SCALING_TYPE),
            local_B_s,
            _scaling_type_value(BLOCKWISE_128X128_SCALING_TYPE),
            offs,
            torch.bfloat16,
            self.block_size,
        )
        self._assert_dtensor_matches(
            dist_out,
            expected_local,
            expected_global,
            Shard(1),
        )

        # Dgrad contraction sharding: grad_out @ B with both operands sharded on N.
        grad_output = torch.randn(256, 256, dtype=torch.bfloat16, device=device)
        global_go_fp8, global_go_s = triton_fp8_blockwise_act_quant_lhs(grad_output)
        global_B_dgrad_fp8, global_B_dgrad_s = (
            triton_fp8_blockwise_weight_quant_grouped_rhs(B_t)
        )
        expected_global = emulated_blockwise_scaled_grouped_mm(
            global_go_fp8,
            global_B_dgrad_fp8,
            global_go_s,
            _scaling_type_value(BLOCKWISE_1X128_SCALING_TYPE),
            global_B_dgrad_s,
            _scaling_type_value(BLOCKWISE_128X128_SCALING_TYPE),
            offs,
            torch.bfloat16,
            self.block_size,
        )
        local_go, dist_go = self._dtensor_from_global(
            mesh, grad_output, Shard(1), contiguous=True
        )
        local_B_t, dist_B_t = self._dtensor_from_global(
            mesh, B_t, Shard(2), contiguous=False
        )
        dist_go_fp8, dist_go_s = triton_fp8_blockwise_act_quant_lhs(dist_go)
        dist_B_dgrad_fp8, dist_B_dgrad_s = (
            triton_fp8_blockwise_weight_quant_grouped_rhs(dist_B_t)
        )
        dist_out = emulated_blockwise_scaled_grouped_mm(
            dist_go_fp8,
            dist_B_dgrad_fp8,
            dist_go_s,
            _scaling_type_value(BLOCKWISE_1X128_SCALING_TYPE),
            dist_B_dgrad_s,
            _scaling_type_value(BLOCKWISE_128X128_SCALING_TYPE),
            dist_offs,
            torch.bfloat16,
            self.block_size,
        )
        local_go_fp8, local_go_s = triton_fp8_blockwise_act_quant_lhs(local_go)
        local_B_dgrad_fp8, local_B_dgrad_s = (
            triton_fp8_blockwise_weight_quant_grouped_rhs(local_B_t)
        )
        expected_local = emulated_blockwise_scaled_grouped_mm(
            local_go_fp8,
            local_B_dgrad_fp8,
            local_go_s,
            _scaling_type_value(BLOCKWISE_1X128_SCALING_TYPE),
            local_B_dgrad_s,
            _scaling_type_value(BLOCKWISE_128X128_SCALING_TYPE),
            offs,
            torch.bfloat16,
            self.block_size,
        )
        self._assert_dtensor_matches(
            dist_out,
            expected_local,
            expected_global,
            Partial(),
            min_global_sqnr=35.0,
        )

    @with_comms
    @skip_if_rocm("ROCm not supported")
    def test_grouped_wgrad_tp_sharding(self):
        mesh = self._build_cuda_mesh()
        device = torch.device(f"cuda:{self.rank % torch.cuda.device_count()}")
        torch.manual_seed(456)
        offs = torch.tensor([128, 256], dtype=torch.int32, device=device)
        dist_offs = DTensor.from_local(
            offs,
            mesh,
            [Replicate()],
            run_check=False,
        )
        grad_output = torch.randn(256, 256, dtype=torch.bfloat16, device=device)
        A = torch.randn(256, 256, dtype=torch.bfloat16, device=device)

        global_go_t_fp8, global_go_t_s = triton_fp8_blockwise_act_quant_transposed_lhs(
            grad_output
        )
        global_A_rhs_fp8, global_A_rhs_s = triton_fp8_blockwise_act_quant_rhs(A)
        expected_global = emulated_blockwise_scaled_grouped_mm(
            global_go_t_fp8,
            global_A_rhs_fp8,
            global_go_t_s,
            _scaling_type_value(BLOCKWISE_1X128_SCALING_TYPE),
            global_A_rhs_s,
            _scaling_type_value(BLOCKWISE_1X128_SCALING_TYPE),
            offs,
            torch.bfloat16,
            self.block_size,
        )

        # Output N sharding from grad_output's feature shard.
        local_go, dist_go = self._dtensor_from_global(
            mesh, grad_output, Shard(1), contiguous=True
        )
        local_A, dist_A = self._dtensor_from_global(
            mesh, A, Replicate(), contiguous=True
        )
        dist_go_t_fp8, dist_go_t_s = triton_fp8_blockwise_act_quant_transposed_lhs(
            dist_go
        )
        dist_A_rhs_fp8, dist_A_rhs_s = triton_fp8_blockwise_act_quant_rhs(dist_A)
        dist_out = emulated_blockwise_scaled_grouped_mm(
            dist_go_t_fp8,
            dist_A_rhs_fp8,
            dist_go_t_s,
            _scaling_type_value(BLOCKWISE_1X128_SCALING_TYPE),
            dist_A_rhs_s,
            _scaling_type_value(BLOCKWISE_1X128_SCALING_TYPE),
            dist_offs,
            torch.bfloat16,
            self.block_size,
        )
        local_go_t_fp8, local_go_t_s = triton_fp8_blockwise_act_quant_transposed_lhs(
            local_go
        )
        local_A_rhs_fp8, local_A_rhs_s = triton_fp8_blockwise_act_quant_rhs(local_A)
        expected_local = emulated_blockwise_scaled_grouped_mm(
            local_go_t_fp8,
            local_A_rhs_fp8,
            local_go_t_s,
            _scaling_type_value(BLOCKWISE_1X128_SCALING_TYPE),
            local_A_rhs_s,
            _scaling_type_value(BLOCKWISE_1X128_SCALING_TYPE),
            offs,
            torch.bfloat16,
            self.block_size,
        )
        self._assert_dtensor_matches(
            dist_out,
            expected_local,
            expected_global,
            Shard(1),
        )

        # Output K sharding from input activation feature shard.
        local_go, dist_go = self._dtensor_from_global(
            mesh, grad_output, Replicate(), contiguous=True
        )
        local_A, dist_A = self._dtensor_from_global(mesh, A, Shard(1), contiguous=True)
        dist_go_t_fp8, dist_go_t_s = triton_fp8_blockwise_act_quant_transposed_lhs(
            dist_go
        )
        dist_A_rhs_fp8, dist_A_rhs_s = triton_fp8_blockwise_act_quant_rhs(dist_A)
        dist_out = emulated_blockwise_scaled_grouped_mm(
            dist_go_t_fp8,
            dist_A_rhs_fp8,
            dist_go_t_s,
            _scaling_type_value(BLOCKWISE_1X128_SCALING_TYPE),
            dist_A_rhs_s,
            _scaling_type_value(BLOCKWISE_1X128_SCALING_TYPE),
            dist_offs,
            torch.bfloat16,
            self.block_size,
        )
        local_go_t_fp8, local_go_t_s = triton_fp8_blockwise_act_quant_transposed_lhs(
            local_go
        )
        local_A_rhs_fp8, local_A_rhs_s = triton_fp8_blockwise_act_quant_rhs(local_A)
        expected_local = emulated_blockwise_scaled_grouped_mm(
            local_go_t_fp8,
            local_A_rhs_fp8,
            local_go_t_s,
            _scaling_type_value(BLOCKWISE_1X128_SCALING_TYPE),
            local_A_rhs_s,
            _scaling_type_value(BLOCKWISE_1X128_SCALING_TYPE),
            offs,
            torch.bfloat16,
            self.block_size,
        )
        self._assert_dtensor_matches(
            dist_out,
            expected_local,
            expected_global,
            Shard(2),
        )

    @with_comms
    @skip_if_rocm("ROCm not supported")
    def test_grouped_mm_module_tp_parity(self):
        mesh = self._build_cuda_mesh()
        device = torch.device(f"cuda:{self.rank % torch.cuda.device_count()}")
        cases = (
            (False, torch.tensor([128, 256], dtype=torch.int32, device=device)),
            (True, torch.tensor([129, 256], dtype=torch.int32, device=device)),
        )

        for pad_token_groups_for_grouped_mm, offs in cases:
            with self.subTest(pad_token_groups=pad_token_groups_for_grouped_mm):
                ref_model, tp_model = make_blockwise_grouped_experts_pair(
                    seed=789,
                    device=device,
                    pad_token_groups_for_grouped_mm=pad_token_groups_for_grouped_mm,
                )
                tp_model = parallelize_blockwise_grouped_experts_tensor_parallel(
                    tp_model,
                    mesh,
                )
                assert_blockwise_grouped_experts_tp_applied(tp_model)

                ref_optim = torch.optim.SGD(ref_model.parameters(), lr=1e-2)
                tp_optim = torch.optim.SGD(tp_model.parameters(), lr=1e-2)

                for iter_idx in range(2):
                    torch.manual_seed(900 + iter_idx)
                    x = torch.randn(256, 256, device=device, dtype=torch.bfloat16)
                    target = torch.randn(256, 256, device=device, dtype=torch.bfloat16)

                    ref_optim.zero_grad(set_to_none=True)
                    tp_optim.zero_grad(set_to_none=True)

                    ref_out = ref_model(x, offs)
                    tp_out = tp_model(x, offs)
                    assert_close(tp_out, ref_out, min_sqnr=23.0)

                    ref_loss = F.mse_loss(ref_out, target)
                    tp_loss = F.mse_loss(full_tensor(tp_out), target)
                    assert_close(tp_loss, ref_loss, atol=1e-2, rtol=1e-2)

                    ref_loss.backward()
                    tp_loss.backward()
                    assert_dtensor_parameter_grads_match(
                        ref_model.parameters(),
                        tp_model.parameters(),
                        min_sqnr=20.0,
                    )

                    ref_optim.step()
                    tp_optim.step()
                    assert_dtensor_parameter_values_match(
                        ref_model.parameters(),
                        tp_model.parameters(),
                        min_sqnr=20.0,
                    )


if __name__ == "__main__":
    run_tests()
