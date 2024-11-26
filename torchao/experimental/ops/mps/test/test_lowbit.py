# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import os
import sys
import torch
import unittest

from parameterized import parameterized

libname = "libtorchao_ops_mps_linear_fp_act_xbit_weight_aten.dylib"
libpath = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "../cmake-out/lib/", libname)
)

try:
    for nbit in range(1, 8):
        getattr(torch.ops.torchao, f"_linear_fp_act_{nbit}bit_weight")
        getattr(torch.ops.torchao, f"_pack_weight_{nbit}bit")
except AttributeError:
    try:
        torch.ops.load_library(libpath)
    except:
        raise RuntimeError(f"Failed to load library {libpath}")
    else:
        try:
            for nbit in range(1, 8):
                getattr(torch.ops.torchao, f"_linear_fp_act_{nbit}bit_weight")
                getattr(torch.ops.torchao, f"_pack_weight_{nbit}bit")
        except AttributeError as e:
            raise e


class TestLowBitQuantWeightsLinear(unittest.TestCase):
    CASES = [
        (nbit, *param)
        for nbit in range(1, 8)
        for param in [
            (1, 8, 1, 32),
            (1, 32, 1, 32),
            (1, 32, 1, 64),
            (1, 56, 1, 64),
            (1, 64, 1, 64),
            (1, 72, 1, 64),
            (1, 1000, 1, 64),
            (3, 64, 5, 64),
            (7, 64, 23, 64),
            (17, 120, 23, 128),
            (17, 128, 23, 128),
            (41, 144, 23, 128),
            (41, 128, 23, 128),
            (81, 8, 1, 256),
            (19, 256, 17, 256),
            (1, 1000, 81, 256),
        ]
    ]

    def _init_tensors(self, group_size, M, K, N, nbit, device="mps"):
        ceil_K_group_size = (K + group_size - 1) // group_size
        A = torch.rand(M, K, dtype=torch.float32, device=device)
        W = torch.randint(0, 1 << nbit, (N, K), dtype=torch.uint8, device=device)
        S = torch.rand(ceil_K_group_size, N, dtype=torch.float32, device=device) + 0.01
        Z = torch.randint(
            0,
            1 << nbit,
            (ceil_K_group_size, N),
            dtype=torch.float32,
            device=device,
        )
        Z = -Z * S
        return A, W, S, Z

    def _reference_linear_lowbit_quant_weights(self, A, W, group_size, S, Z, nbit):
        # A is (M, K)
        # W is (N, K)
        # S is (K // group_size, N)
        # Z is (K // group_size, N)
        N = W.shape[0]
        K = W.shape[1]
        W = W.to(torch.float32)
        scales = S.t().unsqueeze(2).repeat(1, 1, group_size).view(N, -1)[:, :K]
        zeros = Z.t().unsqueeze(2).repeat(1, 1, group_size).view(N, -1)[:, :K]
        W = scales * W + zeros
        return torch.mm(A, W.t())

    @parameterized.expand(CASES)
    def test_linear(self, nbit, M=1, K=32, N=32, group_size=32):
        print(f"nbit: {nbit}, M: {M}, K: {K}, N: {N}, group_size: {group_size}")
        A, W, S, Z = self._init_tensors(group_size, M, K, N, nbit=nbit)
        packing_op = getattr(torch.ops.torchao, f"_pack_weight_{nbit}bit")
        linear_op = getattr(torch.ops.torchao, f"_linear_fp_act_{nbit}bit_weight")
        B = packing_op(W.cpu()).to("mps")
        result = linear_op(A, B, group_size, S, Z).cpu()
        expected = self._reference_linear_lowbit_quant_weights(
            A.cpu(), W.cpu(), group_size, S.cpu(), Z.cpu(), nbit=nbit
        )
        torch.testing.assert_close(result, expected, rtol=0.001, atol=0.001)


if __name__ == "__main__":
    unittest.main()
