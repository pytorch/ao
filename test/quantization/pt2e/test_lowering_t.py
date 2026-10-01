# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD 3-Clause license found in the
# LICENSE file in the root directory of this source tree.

import pytest
import torch

from torchao.quantization.pt2e.lowering import lower_pt2e_quantized_to_x86


class Transpose(torch.nn.Module):
    def forward(self, x):
        return x.t() + 1


@pytest.mark.parametrize("shape", [(), (5,), (3, 4)])
def test_lower_t_preserves_scalar_vector_and_matrix(shape):
    model = Transpose().eval()
    x = torch.randn(shape)
    lowered = lower_pt2e_quantized_to_x86(model, (x,))
    torch.testing.assert_close(lowered(x), model(x))
