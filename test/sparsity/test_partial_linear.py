# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD 3-Clause license found in the
# LICENSE file in the root directory of this source tree.
import unittest

import torch

from torchao.prototype.sparsity.PartialLinear.partial_linear import PartialLinear


class TestPartialLinear(unittest.TestCase):
    def test_partial_linear(self):
        in_features = 20
        out_features = 10
        top_k = 5
        m = PartialLinear(in_features, out_features, top_k=top_k)
        
        self.assertEqual(m.in_features, in_features)
        self.assertEqual(m.out_features, out_features)
        self.assertEqual(m.top_k, top_k)
        
        nonzero_per_row = m.mask.sum(dim=1)
        self.assertTrue(torch.all(nonzero_per_row == top_k))
        
        x = torch.randn(32, in_features)
        output = m(x)
        self.assertEqual(output.shape, (32, out_features))
        
        x_single = torch.randn(in_features)
        output_single = m(x_single)
        self.assertEqual(output_single.shape, (out_features,))
    
    def test_invalid_k(self):
        with self.assertRaises(ValueError):
            PartialLinear(20, 10, top_k=0)
        
        with self.assertRaises(ValueError):
            PartialLinear(20, 10, top_k=21)


if __name__ == "__main__":
    unittest.main()
