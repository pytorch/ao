# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import unittest

import torch
from torch_custom_op import (
    linear_a8sz_w_lowbit_reference_impl,
    replace_linear_with_quantized_linear,
)
import copy

class TestTorchCustomOp(unittest.TestCase):
    def test_accuracy(self):
        group_size = 128
        m = 1
        n = 1071
        k = 4096
        activations = torch.randn(m, k, dtype=torch.float32)
        model = torch.nn.Sequential(*[torch.nn.Linear(k, n, bias=False)])

        for nbit in [2, 3, 4, 5]:
            for has_weight_zeros in [False, True]:
                quantized_model = copy.deepcopy(model)
                replace_linear_with_quantized_linear(
                    quantized_model,
                        kwargs={
                            "group_size": group_size,
                            "nbit": nbit,
                            "has_weight_zeros": has_weight_zeros,
                        },
                )

                with torch.no_grad():
                    result = quantized_model(activations)
                    expected_result = linear_a8sz_w_lowbit_reference_impl(
                        model[0].weight, activations, group_size, nbit, has_weight_zeros
                    )
        
                num_mismatch_at_low_tol = 0
                num_total = result.reshape(-1).shape[0]
                for i in range(num_total):
                    actual_val = result.reshape(-1)[i]
                    expected_val = expected_result.reshape(-1)[i]
                    self.assertTrue(torch.allclose(actual_val, expected_val, atol=1e-6))
                    if not torch.allclose(actual_val, expected_val):
                        num_mismatch_at_low_tol += 1

                # Assert at most 5% of entries are not close at a low tolerance
                self.assertTrue(num_mismatch_at_low_tol / num_total <= 0.05)        
 
if __name__ == '__main__':
    unittest.main()
