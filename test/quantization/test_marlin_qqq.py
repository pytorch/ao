# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD 3-Clause license found in the
# LICENSE file in the root directory of this source tree.
import copy
import unittest

import torch
from torch import nn
from torch.testing._internal.common_utils import TestCase, run_tests

from torchao.dtypes import MarlinQQQLayout
from torchao.quantization.marlin_qqq import (
    pack_to_marlin_qqq,
    unpack_from_marlin_qqq,
)
from torchao.quantization.quant_api import (
    int8_dynamic_activation_int4_weight,
    quantize_,
)
from torchao.quantization.quant_primitives import (
    MappingType,
    _choose_qparams_and_quantize_affine_qqq,
)
from torchao.testing.utils import skip_if_rocm
from torchao.utils import TORCH_VERSION_AT_LEAST_2_5


@skip_if_rocm("ROCm enablement in progress")
class TestMarlinQQQ(TestCase):
    def setUp(self):
        super().setUp()
        torch.manual_seed(0)

        self.input = torch.randn((64, 32, 8192), dtype=torch.float16, device="cuda")
        self.model = (
            nn.Sequential(
                nn.Linear(8192, 21504),
                nn.Linear(21504, 8192),
                nn.ReLU(),
                nn.Linear(8192, 21504),
                nn.Linear(21504, 8192),
            )
            .half()
            .cuda()
        )

    @unittest.skipIf(not torch.cuda.is_available(), "Need CUDA available")
    @skip_if_rocm("ROCm development in progress")
    def test_marlin_qqq(self):
        output_ref = self.model(self.input)
        for group_size in [-1, 128]:
            modelq = copy.deepcopy(self.model)
            quantize_(
                modelq,
                int8_dynamic_activation_int4_weight(
                    group_size=group_size,
                    mapping_type=MappingType.SYMMETRIC,
                    act_mapping_type=MappingType.SYMMETRIC,
                    layout=MarlinQQQLayout(),
                ),
            )
            output = modelq(self.input)

            assert torch.allclose(output, output_ref, atol=1e-1), (
                "Results are not close"
            )

    @unittest.skipIf(not TORCH_VERSION_AT_LEAST_2_5, "Needs PyTorch 2.5+")
    @unittest.skipIf(not torch.cuda.is_available(), "Need CUDA available")
    @skip_if_rocm("ROCm development in progress")
    def test_marlin_qqq_compile(self):
        model_copy = copy.deepcopy(self.model)
        model_copy.forward = torch.compile(model_copy.forward, fullgraph=True)
        output_ref = model_copy(self.input)

        for group_size in [-1, 128]:
            modelq = copy.deepcopy(self.model)
            quantize_(
                modelq,
                int8_dynamic_activation_int4_weight(
                    group_size=group_size,
                    mapping_type=MappingType.SYMMETRIC,
                    act_mapping_type=MappingType.SYMMETRIC,
                    layout=MarlinQQQLayout(),
                ),
            )
            modelq.forward = torch.compile(modelq.forward, fullgraph=True)
            output = modelq(self.input)

            assert torch.allclose(output, output_ref, atol=1e-1), (
                "Results are not close"
            )

    @unittest.skipIf(not torch.cuda.is_available(), "Need CUDA available")
    def test_pack_unpack_equivalence(self):
        num_bits = 4
        shape = (11008, 4096)

        w = torch.rand(shape, dtype=torch.float16, device="cuda")

        for group_size in [-1, 128]:
            # Quantize weights
            q_w, s_group, s_channel, _ = _choose_qparams_and_quantize_affine_qqq(
                w, num_bits, group_size
            )

            q_w = q_w.t()
            s_group = s_group.t()
            s_channel = s_channel.t()

            # Test pack/unpack equivalence
            q_w_comp, packed_s_group, packed_s_channel = pack_to_marlin_qqq(
                q_w, s_group, s_channel, num_bits, group_size
            )
            unpacked_q_w, unpacked_s_group, unpacked_s_channel = unpack_from_marlin_qqq(
                q_w_comp,
                packed_s_group,
                packed_s_channel,
                q_w.shape,
                num_bits,
                group_size,
            )

            assert torch.equal(q_w, unpacked_q_w), (
                "Unpacked weights do not match original weights"
            )
            assert torch.equal(s_channel, unpacked_s_channel), (
                "Unpacked s_channel do not match original s_channel"
            )
            assert torch.equal(s_group, unpacked_s_group), (
                "Unpacked s_group do not match original s_group"
            )


if __name__ == "__main__":
    run_tests()
