# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD 3-Clause license found in the
# LICENSE file in the root directory of this source tree.

import unittest

import torch
from torch.testing._internal import common_utils

from torchao.prototype.pat import make_m_by_n_mask


class TestMakeMByNMask(common_utils.TestCase):
    def test_2_4(self):
        weight = torch.tensor(
            [
                [1.0, 8.0, 3.0, 2.0],
                [-5.0, -4.0, 0.0, -9.0],
            ]
        )
        original = weight.clone()

        mask = make_m_by_n_mask(weight, m_by_n=(2, 4))

        self.assertEqual(mask.dtype, torch.bool)
        self.assertEqual(mask.shape, weight.shape)
        self.assertEqual(
            mask.tolist(),
            [
                [False, True, True, False],
                [True, False, False, True],
            ],
        )
        self.assertEqual(mask.sum(dim=1).tolist(), [2, 2])
        self.assertEqual(weight, original)

    @common_utils.parametrize(
        "m_by_n,expected",
        [
            ((1, 4), [[False, False, False, True]]),
            ((3, 4), [[False, True, True, True]]),
            ((2, 3), [[False, True, True, True]]),
            ((1, 1), [[True, True, True, True]]),
        ],
    )
    def test_m_by_n_combinations(self, m_by_n, expected):
        weight = torch.tensor([[1.0, -2.0, 3.0, -4.0]])

        mask = make_m_by_n_mask(weight, m_by_n=m_by_n, remainder_policy="pad")

        self.assertEqual(mask.tolist(), expected)

    def test_negative_values(self):
        weight = torch.tensor(
            [
                [-1.0, 2.0, -10.0, 0.5],
                [3.0, -4.0, 0.0, -7.0],
            ]
        )

        mask = make_m_by_n_mask(weight, m_by_n=(2, 4))

        self.assertEqual(
            mask.tolist(),
            [
                [False, True, True, False],
                [False, True, False, True],
            ],
        )

    def test_existing_zeros(self):
        weight = torch.tensor(
            [
                [0.0, 5.0, 0.0, 1.0],
                [0.0, 0.0, 3.0, 2.0],
                [0.0, 0.0, 0.0, 0.0],
            ]
        )

        mask = make_m_by_n_mask(weight, m_by_n=(2, 4))

        self.assertEqual(
            mask.tolist(),
            [
                [False, True, False, True],
                [False, False, True, True],
                [True, True, False, False],
            ],
        )
        self.assertEqual(mask.sum(dim=1).tolist(), [2, 2, 2])

    def test_equal_magnitudes_keep_lower_indices(self):
        weight = torch.tensor(
            [
                [2.0, 2.0, 2.0, 2.0],
                [-1.0, 1.0, -1.0, 1.0],
                [1.0, 1.0, -1.0, -1.0],
            ]
        )

        mask = make_m_by_n_mask(weight, m_by_n=(2, 4))

        self.assertEqual(
            mask.tolist(),
            [
                [True, True, False, False],
                [True, True, False, False],
                [True, True, False, False],
            ],
        )

    def test_partial_group_pad(self):
        weight = torch.tensor(
            [
                [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
                [6.0, 5.0, 4.0, 3.0, 2.0, 1.0],
            ]
        )

        mask = make_m_by_n_mask(weight, m_by_n=(2, 4), remainder_policy="pad")

        self.assertEqual(mask.shape, weight.shape)
        self.assertEqual(
            mask.tolist(),
            [
                [False, False, True, True, True, True],
                [True, True, False, False, True, True],
            ],
        )

    def test_partial_group_pad_keeps_available(self):
        weight = torch.tensor([[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]])

        mask = make_m_by_n_mask(weight, m_by_n=(3, 4), remainder_policy="pad")

        self.assertEqual(mask.tolist(), [[False, True, True, True, True, True]])
        self.assertEqual(mask.sum().item(), 5)

    def test_partial_group_error(self):
        weight = torch.randn(2, 6)

        with self.assertRaisesRegex(ValueError, "not divisible"):
            make_m_by_n_mask(weight, m_by_n=(2, 4), remainder_policy="error")

        exact = torch.arange(8.0).reshape(1, 8)
        mask = make_m_by_n_mask(exact, m_by_n=(2, 4), remainder_policy="error")
        self.assertEqual(mask.sum(dim=1).tolist(), [4])

    def test_invalid_m_by_n(self):
        invalid = [
            (0, 4),
            (2, 0),
            (3, 2),
            (-1, 4),
            (2,),
            (2, 4, 8),
            (2.0, 4),
            (2, 4.0),
            (True, 4),
            (None, 4),
        ]
        weight = torch.ones(2, 4)

        for m_by_n in invalid:
            with self.subTest(m_by_n=m_by_n):
                with self.assertRaisesRegex(ValueError, "m_by_n"):
                    make_m_by_n_mask(weight, m_by_n=m_by_n)

    def test_invalid_remainder_policy(self):
        weight = torch.ones(2, 4)

        for policy in ("ignore", "", 1):
            with self.subTest(policy=policy):
                with self.assertRaisesRegex(ValueError, "remainder_policy"):
                    make_m_by_n_mask(weight, m_by_n=(2, 4), remainder_policy=policy)

    def test_invalid_weight(self):
        with self.assertRaisesRegex(ValueError, "2-D"):
            make_m_by_n_mask(torch.ones(4), m_by_n=(2, 4))
        with self.assertRaisesRegex(ValueError, "2-D"):
            make_m_by_n_mask(torch.ones(2, 2, 4), m_by_n=(2, 4))
        with self.assertRaisesRegex(TypeError, "torch.Tensor"):
            make_m_by_n_mask([[1.0, 2.0, 3.0, 4.0]], m_by_n=(2, 4))

    def test_mask_is_not_a_parameter(self):
        weight = torch.randn(2, 4, requires_grad=True)

        mask = make_m_by_n_mask(weight, m_by_n=(2, 4))

        self.assertFalse(mask.requires_grad)
        self.assertEqual(mask.dtype, torch.bool)


common_utils.instantiate_parametrized_tests(TestMakeMByNMask)


if __name__ == "__main__":
    unittest.main()
