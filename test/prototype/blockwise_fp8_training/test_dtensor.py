# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD 3-Clause license found in the
# LICENSE file in the root directory of this source tree.

import unittest

import pytest
import torch
from packaging import version
from torch.distributed._tensor import DTensor
from torch.distributed.tensor import Partial, Replicate, Shard
from torch.testing._internal.common_utils import run_tests
from torch.testing._internal.distributed._tensor.common_dtensor import (
    DTensorTestBase,
    with_comms,
)

triton = pytest.importorskip("triton", reason="Triton required to run this test")

from torchao.float8.config import e4m3_dtype
from torchao.prototype.blockwise_fp8_training.kernels import (
    triton_fp8_blockwise_act_quant_lhs,
    triton_fp8_blockwise_act_quant_rhs,
    triton_fp8_blockwise_act_quant_transposed_lhs,
    triton_fp8_blockwise_weight_quant_rhs,
    triton_fp8_blockwise_weight_quant_transposed_rhs,
    triton_fp8_gemm_1x128_128x1,
    triton_fp8_gemm_1x128_128x128,
)
from torchao.utils import is_MI300, is_MI350, is_ROCM, is_sm_at_least_90


QUANT_PRESERVE_PLACEMENTS = (
    (Replicate(), (Replicate(), Replicate())),
    (Shard(0), (Shard(0), Shard(0))),
    (Shard(1), (Shard(1), Shard(1))),
)

QUANT_TRANSPOSE_PLACEMENTS = (
    (Replicate(), (Replicate(), Replicate())),
    (Shard(0), (Shard(1), Shard(1))),
    (Shard(1), (Shard(0), Shard(0))),
)


def _quant_skip_reason() -> str | None:
    if not torch.cuda.is_available():
        return "Need CUDA available"
    if torch.cuda.device_count() < 2:
        return "Need at least 2 CUDA devices"
    if not (is_sm_at_least_90() or is_MI300() or is_MI350()):
        return "Requires FP8-capable GPU (CUDA SM90+, MI300, or MI350)"
    if version.parse(triton.__version__) < version.parse("3.3.0"):
        return "Triton version < 3.3.0"
    return None


def _gemm_skip_reason() -> str | None:
    if skip_reason := _quant_skip_reason():
        return skip_reason
    if is_ROCM():
        return "Blockwise FP8 GEMM has numerical issues on ROCm"
    return None


def _op_name(op) -> str:
    return getattr(op, "_name", repr(op))


QUANT_SKIP_REASON = _quant_skip_reason()
GEMM_SKIP_REASON = _gemm_skip_reason()


class TestBlockwiseFP8DTensorSharding(DTensorTestBase):
    world_size = 2
    block_size = 128

    def _build_cuda_mesh(self):
        torch.cuda.set_device(self.rank % torch.cuda.device_count())
        mesh = self.build_device_mesh()
        mesh._device_type = "cuda"
        return mesh

    def _local_shard(self, tensor: torch.Tensor, placement, *, contiguous: bool):
        if isinstance(placement, Replicate):
            return tensor.contiguous() if contiguous else tensor

        if not isinstance(placement, Shard):
            raise AssertionError(f"Unsupported placement: {placement}")

        self.assertEqual(tensor.size(placement.dim) % self.world_size, 0)
        chunk = tensor.size(placement.dim) // self.world_size
        local = tensor.narrow(placement.dim, self.rank * chunk, chunk)
        return local.contiguous() if contiguous else local

    def _dtensor_from_global(
        self, mesh, tensor: torch.Tensor, placement, *, contiguous: bool
    ):
        local_tensor = self._local_shard(tensor, placement, contiguous=contiguous)
        dist_tensor = DTensor.from_local(
            local_tensor, mesh, [placement], run_check=False
        )
        return local_tensor, dist_tensor

    def _assert_distributed_output(
        self,
        dist_output: DTensor,
        expected_local_output: torch.Tensor,
        expected_global_output: torch.Tensor,
        expected_placement,
        expected_global_shape,
        *,
        global_atol: float = 0.0,
        global_rtol: float = 0.0,
    ):
        self.assertIsInstance(dist_output, DTensor)
        self.assertEqual(dist_output.placements, (expected_placement,))
        self.assertEqual(tuple(dist_output.shape), tuple(expected_global_shape))
        torch.testing.assert_close(
            dist_output.to_local(), expected_local_output, atol=0, rtol=0
        )
        torch.testing.assert_close(
            dist_output.redistribute(placements=[Replicate()]).to_local(),
            expected_global_output,
            atol=global_atol,
            rtol=global_rtol,
        )

    def _assert_quant_outputs(
        self,
        dist_outputs,
        expected_local_outputs,
        expected_global_outputs,
        expected_placements,
        expected_global_shapes,
    ):
        for (
            dist_output,
            expected_local,
            expected_global,
            expected_placement,
            expected_global_shape,
        ) in zip(
            dist_outputs,
            expected_local_outputs,
            expected_global_outputs,
            expected_placements,
            expected_global_shapes,
            strict=True,
        ):
            self._assert_distributed_output(
                dist_output=dist_output,
                expected_local_output=expected_local,
                expected_global_output=expected_global,
                expected_placement=expected_placement,
                expected_global_shape=expected_global_shape,
            )

    def _assert_gemm_output(
        self,
        dist_output: DTensor,
        expected_local_output: torch.Tensor,
        expected_global_output: torch.Tensor,
        expected_placement,
        expected_global_shape,
    ):
        # because we do a distributed reduction for the check against global output
        # we have a slight tolerance to account for floating point noise
        atol, rtol = (1e-5, 1e-3) if isinstance(expected_placement, Partial) else (0, 0)
        self._assert_distributed_output(
            dist_output=dist_output,
            expected_local_output=expected_local_output,
            expected_global_output=expected_global_output,
            expected_placement=expected_placement,
            expected_global_shape=expected_global_shape,
            global_atol=atol,
            global_rtol=rtol,
        )

    @with_comms
    @unittest.skipIf(QUANT_SKIP_REASON is not None, QUANT_SKIP_REASON or "")
    def test_quant_op_sharding(self):
        mesh = self._build_cuda_mesh()
        device = torch.device(f"cuda:{self.rank % torch.cuda.device_count()}")
        torch.manual_seed(42)

        x = torch.randn(256, 512, dtype=torch.bfloat16, device=device)
        # case placements are in form of (input, (output))
        cases = (
            (triton_fp8_blockwise_act_quant_lhs, QUANT_PRESERVE_PLACEMENTS),
            (triton_fp8_blockwise_act_quant_rhs, QUANT_PRESERVE_PLACEMENTS),
            (triton_fp8_blockwise_act_quant_transposed_lhs, QUANT_TRANSPOSE_PLACEMENTS),
            (triton_fp8_blockwise_weight_quant_rhs, QUANT_PRESERVE_PLACEMENTS),
            (
                triton_fp8_blockwise_weight_quant_transposed_rhs,
                QUANT_TRANSPOSE_PLACEMENTS,
            ),
        )

        for quant_op, placement_map in cases:
            global_outputs = quant_op(x, block_size=self.block_size, dtype=e4m3_dtype)
            expected_global_shapes = tuple(output.shape for output in global_outputs)
            for input_placement, expected_placements in placement_map:
                with self.subTest(
                    quant_op=_op_name(quant_op), input_placement=repr(input_placement)
                ):
                    local_x, dist_x = self._dtensor_from_global(
                        mesh,
                        x,
                        input_placement,
                        contiguous=True,
                    )

                    dist_outputs = quant_op(
                        dist_x,
                        block_size=self.block_size,
                        dtype=e4m3_dtype,
                    )
                    expected_local_outputs = quant_op(
                        local_x,
                        block_size=self.block_size,
                        dtype=e4m3_dtype,
                    )

                    self._assert_quant_outputs(
                        dist_outputs=dist_outputs,
                        expected_local_outputs=expected_local_outputs,
                        expected_global_outputs=global_outputs,
                        expected_placements=expected_placements,
                        expected_global_shapes=expected_global_shapes,
                    )

    @with_comms
    @unittest.skipIf(GEMM_SKIP_REASON is not None, GEMM_SKIP_REASON or "")
    def test_gemm_op_sharding(self):
        mesh = self._build_cuda_mesh()
        device = torch.device(f"cuda:{self.rank % torch.cuda.device_count()}")
        torch.manual_seed(67)
        out_dtype = torch.float32

        gemm_cases = (
            (
                triton_fp8_gemm_1x128_128x128,
                triton_fp8_blockwise_act_quant_lhs,
                triton_fp8_blockwise_weight_quant_transposed_rhs,
                torch.randn(256, 512, dtype=torch.bfloat16, device=device),
                torch.randn(256, 512, dtype=torch.bfloat16, device=device),
                (256, 256),
                (
                    # lhs input, rhs input, and output placements
                    (Replicate(), Replicate(), Replicate()),
                    (Shard(0), Replicate(), Shard(0)),
                    (Replicate(), Shard(0), Shard(1)),
                    (Shard(1), Shard(1), Partial()),
                ),
            ),
            (
                triton_fp8_gemm_1x128_128x1,
                triton_fp8_blockwise_act_quant_transposed_lhs,
                triton_fp8_blockwise_act_quant_rhs,
                torch.randn(512, 256, dtype=torch.bfloat16, device=device),
                torch.randn(512, 256, dtype=torch.bfloat16, device=device),
                (256, 256),
                (
                    # lhs input, rhs input, and output placements
                    (Replicate(), Replicate(), Replicate()),
                    (Shard(1), Replicate(), Shard(0)),
                    (Replicate(), Shard(1), Shard(1)),
                    (Shard(0), Shard(0), Partial()),
                ),
            ),
        )

        for (
            gemm_op,
            lhs_quant_op,
            rhs_quant_op,
            lhs_input,
            rhs_input,
            expected_shape,
            placement_cases,
        ) in gemm_cases:
            global_a, global_a_s = lhs_quant_op(
                lhs_input,
                block_size=self.block_size,
                dtype=e4m3_dtype,
            )
            global_b, global_b_s = rhs_quant_op(
                rhs_input,
                block_size=self.block_size,
                dtype=e4m3_dtype,
            )
            expected_global_output = gemm_op(
                global_a,
                global_b,
                global_a_s,
                global_b_s,
                block_size=self.block_size,
                out_dtype=out_dtype,
            )

            for (
                lhs_input_placement,
                rhs_input_placement,
                expected_output_placement,
            ) in placement_cases:
                with self.subTest(
                    gemm_op=_op_name(gemm_op),
                    lhs_input_placement=repr(lhs_input_placement),
                    rhs_input_placement=repr(rhs_input_placement),
                ):
                    local_lhs_input, dist_lhs_input = self._dtensor_from_global(
                        mesh,
                        lhs_input,
                        lhs_input_placement,
                        contiguous=True,
                    )
                    local_rhs_input, dist_rhs_input = self._dtensor_from_global(
                        mesh,
                        rhs_input,
                        rhs_input_placement,
                        contiguous=True,
                    )

                    dist_a, dist_a_s = lhs_quant_op(
                        dist_lhs_input,
                        block_size=self.block_size,
                        dtype=e4m3_dtype,
                    )
                    dist_b, dist_b_s = rhs_quant_op(
                        dist_rhs_input,
                        block_size=self.block_size,
                        dtype=e4m3_dtype,
                    )
                    dist_output = gemm_op(
                        dist_a,
                        dist_b,
                        dist_a_s,
                        dist_b_s,
                        block_size=self.block_size,
                        out_dtype=out_dtype,
                    )

                    local_a, local_a_s = lhs_quant_op(
                        local_lhs_input,
                        block_size=self.block_size,
                        dtype=e4m3_dtype,
                    )
                    local_b, local_b_s = rhs_quant_op(
                        local_rhs_input,
                        block_size=self.block_size,
                        dtype=e4m3_dtype,
                    )
                    expected_local_output = gemm_op(
                        local_a,
                        local_b,
                        local_a_s,
                        local_b_s,
                        block_size=self.block_size,
                        out_dtype=out_dtype,
                    )

                    self._assert_gemm_output(
                        dist_output=dist_output,
                        expected_local_output=expected_local_output,
                        expected_global_output=expected_global_output,
                        expected_placement=expected_output_placement,
                        expected_global_shape=expected_shape,
                    )


if __name__ == "__main__":
    run_tests()
