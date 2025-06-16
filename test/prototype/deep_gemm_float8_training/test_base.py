# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD 3-Clause license found in the
# LICENSE file in the root directory of this source tree.

import copy
import random

import pytest
import torch

from torchao.utils import (
    TORCH_VERSION_AT_LEAST_2_7,
)

if not TORCH_VERSION_AT_LEAST_2_7:
    pytest.skip("Unsupported PyTorch version", allow_module_level=True)


from torchao.float8.float8_utils import compute_error
from torchao.prototype.deep_gemm_float8_training.deep_gemm_utils import (
    scale_narrow_tiles,
    scale_square_tiles,
    scaled_mm_deep_gemm_128_1_128_1,
    scaled_mm_deep_gemm_128_1_128_128,
    unscale_narrow_tiles,
    unscale_square_tiles,
)
from torchao.prototype.deep_gemm_float8_training.linear import (
    DeepGemmFloat8Linear,
    DeepGemmFloat8LinearConfig,
)
from torchao.quantization import quantize_

random.seed(0)
torch.manual_seed(0)


class TestDeepGemmUtils:
    @pytest.mark.parametrize("mkn", [(128, 128, 128), (256, 512, 1024)])
    def test_128_1_128_128_gemm(self, mkn):
        M, K, N = mkn
        tile_size = 128
        x = torch.randn(M, K, dtype=torch.bfloat16, device="cuda")
        w = torch.randn(N, K, dtype=torch.bfloat16, device="cuda")
        xq, xs = scale_narrow_tiles(x, tile_size=tile_size)
        wq, ws = scale_square_tiles(w, tile_size=tile_size)
        y = scaled_mm_deep_gemm_128_1_128_128(xq, wq, 1.0 / xs, 1.0 / ws)
        y_ref = x @ w.T
        sqnr = compute_error(y_ref, y)
        assert sqnr > 26.0

    @pytest.mark.parametrize("mkn", [(128, 128, 128), (256, 512, 1024)])
    def test_128_1_128_1_gemm(self, mkn):
        M, K, N = mkn
        tile_size = 128
        x = torch.randn(M, K, dtype=torch.bfloat16, device="cuda")
        g = torch.randn(N, K, dtype=torch.bfloat16, device="cuda")
        xq, xs = scale_narrow_tiles(x, tile_size=tile_size)
        gq, gs = scale_narrow_tiles(g, tile_size=tile_size)
        gi = scaled_mm_deep_gemm_128_1_128_1(xq, gq, 1.0 / xs, 1.0 / gs)
        gi_ref = x @ g.T
        sqnr = compute_error(gi_ref, gi)
        assert sqnr > 27.0

    def test_scale_square_tiles(self):
        h, w = 8, 8
        tile_size = 4

        x = torch.arange(h * w, device="cuda").float().reshape(h, w)
        xq, s = scale_square_tiles(x, tile_size=tile_size)
        xqdq = unscale_square_tiles(xq, s, tile_size=tile_size)
        sqnr = compute_error(x, xqdq)
        assert sqnr >= 25.0

    def test_scale_narrow_tiles(self):
        h, w = 8, 16
        tile_size = 4

        x = torch.arange(h * w, device="cuda").float().reshape(h, w)
        xq, s = scale_narrow_tiles(x, tile_size=tile_size)
        xqdq = unscale_narrow_tiles(xq, s, tile_size=tile_size)
        sqnr = compute_error(x, xqdq)
        assert sqnr >= 32.0


class TestDeepGemmLinear:
    @pytest.mark.parametrize("x_rank", [2, 3])
    def test_hello_world(self, x_rank):
        M, K, N = 128, 256, 512

        x_ref = torch.randn(M, K, dtype=torch.bfloat16, device="cuda")
        while len(x_ref.shape) < x_rank:
            x_ref = x_ref.unsqueeze(0)
        x_ref.requires_grad_()

        m_ref = torch.nn.Linear(K, N, bias=False).bfloat16().cuda()
        go_ref = torch.randn(M, N, dtype=torch.bfloat16, device="cuda")
        while len(go_ref.shape) < x_rank:
            go_ref = go_ref.unsqueeze(0)

        x = copy.deepcopy(x_ref).requires_grad_()
        m = copy.deepcopy(m_ref)
        go = copy.deepcopy(go_ref)

        m = DeepGemmFloat8Linear.from_float(m)

        y_ref = m_ref(x_ref)
        y_ref.backward(go_ref)
        y = m(x)
        y.backward(go)

        sqnr_y = compute_error(y_ref, y)
        sqnr_gi = compute_error(x_ref.grad, x.grad)
        sqnr_gw = compute_error(m_ref.weight.grad, m.weight.grad)
        assert sqnr_y >= 25.0
        assert sqnr_gi >= 25.0
        assert sqnr_gw >= 25.0

    def test_api(self):
        m = torch.nn.Sequential(torch.nn.Linear(128, 128, bias=False))
        quantize_(m, config=DeepGemmFloat8LinearConfig())
        assert type(m[0]) == DeepGemmFloat8Linear


if __name__ == "__main__":
    pytest.main([__file__])
