# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD 3-Clause license found in the
# LICENSE file in the root directory of this source tree.

import pytest
import torch
from torch.distributed._composable.fsdp import fully_shard
from torch.nn import functional as F
from torch.testing._internal.common_distributed import skip_if_lt_x_gpu
from torch.testing._internal.common_fsdp import FSDPTest
from torch.testing._internal.common_utils import run_tests

triton = pytest.importorskip("triton", reason="Triton required to run this test")

from ._distributed_test_utils import (
    allreduce_reference_grads,
    assert_close,
    assert_dtensor_parameter_grads_match,
    assert_dtensor_parameter_values_match,
    assert_parameters_are_dtensors,
    get_blockwise_linear_skip_reason,
    get_replicated_local_batch,
    make_quantized_toy_model_pair,
)

if skip_reason := get_blockwise_linear_skip_reason(
    triton_module=triton,
    min_cuda_devices=2,
):
    pytest.skip(skip_reason, allow_module_level=True)


class TestBlockwiseFP8FSDP2(FSDPTest):
    @property
    def world_size(self) -> int:
        return 2

    @skip_if_lt_x_gpu(2)
    def test_fsdp2_parity(self):
        for use_triton in (False, True):
            with self.subTest(use_triton=use_triton):
                ref_model, fsdp_model = make_quantized_toy_model_pair(
                    use_triton=use_triton,
                    broadcast_weights=True,
                )

                fully_shard(fsdp_model.ffn.w1)
                fully_shard(fsdp_model.ffn.w2)
                fully_shard(fsdp_model.ffn.out_proj)
                fully_shard(fsdp_model)

                assert_parameters_are_dtensors(fsdp_model.parameters())

                ref_optim = torch.optim.SGD(ref_model.parameters(), lr=1e-2)
                fsdp_optim = torch.optim.SGD(fsdp_model.parameters(), lr=1e-2)

                for iter_idx in range(2):
                    local_input, local_target = get_replicated_local_batch(
                        replica_count=self.world_size,
                        replica_index=self.rank,
                        iter_idx=iter_idx,
                    )

                    ref_optim.zero_grad(set_to_none=True)
                    fsdp_optim.zero_grad(set_to_none=True)

                    ref_out = ref_model(local_input)
                    fsdp_out = fsdp_model(local_input)
                    assert_close(fsdp_out, ref_out)

                    ref_loss = F.mse_loss(ref_out, local_target)
                    fsdp_loss = F.mse_loss(fsdp_out, local_target)
                    assert_close(fsdp_loss, ref_loss, atol=1e-3, rtol=1e-3)

                    ref_loss.backward()
                    fsdp_loss.backward()

                    allreduce_reference_grads(ref_model, world_size=self.world_size)
                    assert_dtensor_parameter_grads_match(
                        ref_model.parameters(),
                        fsdp_model.parameters(),
                    )

                    ref_optim.step()
                    fsdp_optim.step()

                    assert_dtensor_parameter_values_match(
                        ref_model.parameters(),
                        fsdp_model.parameters(),
                    )


if __name__ == "__main__":
    run_tests()
