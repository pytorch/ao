# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD 3-Clause license found in the
# LICENSE file in the root directory of this source tree.
import unittest
from unittest.mock import patch

import torch
from torch.utils._python_dispatch import return_and_correct_aliasing

from torchao.utils import TorchAOBaseTensor, torch_version_at_least


class TestTorchVersionAtLeast(unittest.TestCase):
    def test_torch_version_at_least(self):
        test_cases = [
            ("2.5.0a0+git9f17037", "2.5.0", True),
            ("2.5.0a0+git9f17037", "2.4.0", True),
            ("2.5.0.dev20240708+cu121", "2.5.0", True),
            ("2.5.0.dev20240708+cu121", "2.4.0", True),
            ("2.5.0", "2.4.0", True),
            ("2.5.0", "2.5.0", True),
            ("2.4.0", "2.4.0", True),
            ("2.4.0", "2.5.0", False),
        ]

        for torch_version, compare_version, expected_result in test_cases:
            with patch("torch.__version__", torch_version):
                result = torch_version_at_least(compare_version)

                self.assertEqual(
                    result,
                    expected_result,
                    f"Failed for torch.__version__={torch_version}, comparing with {compare_version}",
                )


class TestTorchAOBaseTensor(unittest.TestCase):
    def test_print_arg_types(self):
        class MyTensor(TorchAOBaseTensor):
            def __new__(cls, data):
                shape = data.shape
                return torch.Tensor._make_wrapper_subclass(cls, shape)  # type: ignore[attr-defined]

            def __init__(self, data):
                self.data = data

        l = torch.nn.Linear(10, 10)
        with self.assertRaisesRegex(NotImplementedError, "arg_types"):
            l.weight = torch.nn.Parameter(MyTensor(l.weight))

    def test_default_impls(self):
        """Making sure some common functions has default implementations, such as
        __tensor_unflatten__, __tensor_flatten__, _apply_fn_to_data, __repr__, to
        """

        class MyTensor(TorchAOBaseTensor):
            tensor_data_names = ["qdata"]
            tensor_attribute_names = ["attr"]

            def __new__(cls, qdata, attr):
                shape = qdata.shape
                return torch.Tensor._make_wrapper_subclass(cls, shape)  # type: ignore[attr-defined]

            def __init__(self, qdata, attr):
                self.qdata = qdata
                self.attr = attr

        implements = MyTensor.implements

        @implements(torch.ops.aten.detach.default)
        def _(func, types, args, kwargs):
            return return_and_correct_aliasing(
                func, args, kwargs, args[0]._apply_fn_to_data(torch.detach)
            )

        l = torch.nn.Linear(1, 1)
        l.weight = torch.nn.Parameter(MyTensor(l.weight, "attr"))
        lp_tensor = l.weight
        tensor_data_name_dict, tensor_attributes = lp_tensor.__tensor_flatten__()
        tensor_data_dict = {
            name: getattr(lp_tensor, name) for name in tensor_data_name_dict
        }
        outer_size = lp_tensor.size()
        outer_stride = lp_tensor.stride()
        reconstructed = type(lp_tensor).__tensor_unflatten__(
            tensor_data_dict, tensor_attributes, outer_size, outer_stride
        )
        self.assertTrue(torch.equal(lp_tensor.qdata, reconstructed.qdata))
        self.assertEqual(lp_tensor.attr, reconstructed.attr)
        print(lp_tensor)


if __name__ == "__main__":
    unittest.main()
