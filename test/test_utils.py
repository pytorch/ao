# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD 3-Clause license found in the
# LICENSE file in the root directory of this source tree.
import unittest
import warnings
from unittest.mock import patch

import torch

from torchao.testing.utils import skip_if_no_cuda
from torchao.utils import TorchAOBaseTensor, torch_version_at_least


class TestTorchVersion(unittest.TestCase):
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

    def test_torch_version_deprecation(self):
        """
        Test that TORCH_VERSION_AT_LEAST* and TORCH_VERSION_AFTER*
        trigger deprecation warnings on use, not on import.
        """
        # Reset deprecation warning state, otherwise we won't log warnings here
        warnings.resetwarnings()

        # Importing and referencing should not trigger deprecation warning
        with warnings.catch_warnings(record=True) as _warnings:
            from torchao.utils import (
                TORCH_VERSION_AFTER_2_2,
                TORCH_VERSION_AFTER_2_3,
                TORCH_VERSION_AFTER_2_4,
                TORCH_VERSION_AFTER_2_5,
                TORCH_VERSION_AT_LEAST_2_2,
                TORCH_VERSION_AT_LEAST_2_3,
                TORCH_VERSION_AT_LEAST_2_4,
                TORCH_VERSION_AT_LEAST_2_5,
                TORCH_VERSION_AT_LEAST_2_6,
                TORCH_VERSION_AT_LEAST_2_7,
                TORCH_VERSION_AT_LEAST_2_8,
            )

            deprecated_api_to_name = [
                (TORCH_VERSION_AT_LEAST_2_8, "TORCH_VERSION_AT_LEAST_2_8"),
                (TORCH_VERSION_AT_LEAST_2_7, "TORCH_VERSION_AT_LEAST_2_7"),
                (TORCH_VERSION_AT_LEAST_2_6, "TORCH_VERSION_AT_LEAST_2_6"),
                (TORCH_VERSION_AT_LEAST_2_5, "TORCH_VERSION_AT_LEAST_2_5"),
                (TORCH_VERSION_AT_LEAST_2_4, "TORCH_VERSION_AT_LEAST_2_4"),
                (TORCH_VERSION_AT_LEAST_2_3, "TORCH_VERSION_AT_LEAST_2_3"),
                (TORCH_VERSION_AT_LEAST_2_2, "TORCH_VERSION_AT_LEAST_2_2"),
                (TORCH_VERSION_AFTER_2_5, "TORCH_VERSION_AFTER_2_5"),
                (TORCH_VERSION_AFTER_2_4, "TORCH_VERSION_AFTER_2_4"),
                (TORCH_VERSION_AFTER_2_3, "TORCH_VERSION_AFTER_2_3"),
                (TORCH_VERSION_AFTER_2_2, "TORCH_VERSION_AFTER_2_2"),
            ]
            self.assertEqual(len(_warnings), 0)

        # Accessing the boolean value should trigger deprecation warning
        with warnings.catch_warnings(record=True) as _warnings:
            for api, name in deprecated_api_to_name:
                num_warnings_before = len(_warnings)
                if api:
                    pass
                regex = f"{name} is deprecated and will be removed"
                self.assertEqual(len(_warnings), num_warnings_before + 1)
                self.assertIn(regex, str(_warnings[-1].message))


class TestTorchAOBaseTensor(unittest.TestCase):
    def test_print_arg_types(self):
        class MyTensor(TorchAOBaseTensor):
            def __new__(cls, data):
                shape = data.shape
                return torch.Tensor._make_wrapper_subclass(cls, shape)  # type: ignore[attr-defined]

            def __init__(self, data):
                self.data = data

        l = torch.nn.Linear(10, 10)
        # since we did not define `tensor_data_names` and `tensor_attribute_names` for MyTensor
        # the following call will error out because `detach` is defined in `TorchAOBaseTensor`
        # but would rely on `tensor_data_names` and `tensor_attribute_names` being defined for it to work
        # user could either specify `tensor_data_names` and `tensor_attribute_names` or manually implement
        # detach op
        with self.assertRaisesRegex(NotImplementedError, "arg_types"):
            l.weight = torch.nn.Parameter(MyTensor(l.weight))

    def _test_default_impls_helper(self, lp_tensor, lp_tensor_for_copy):
        # test __tensor_flatten__ and __tensor_unflatten__
        tensor_data_names, tensor_attributes = lp_tensor.__tensor_flatten__()
        tensor_data_dict = {
            name: getattr(lp_tensor, name) for name in tensor_data_names
        }
        outer_size = lp_tensor.size()
        outer_stride = lp_tensor.stride()
        reconstructed = type(lp_tensor).__tensor_unflatten__(
            tensor_data_dict, tensor_attributes, outer_size, outer_stride
        )
        self.assertTrue(torch.equal(lp_tensor.qdata, reconstructed.qdata))
        self.assertEqual(lp_tensor.attr, reconstructed.attr)

        # `to` / `_to_copy`
        original_device = lp_tensor.device
        lp_tensor = lp_tensor.to("cuda")
        self.assertEqual(lp_tensor.device.type, "cuda")
        lp_tensor = lp_tensor.to(original_device)
        self.assertEqual(lp_tensor.device, original_device)

        # __repr__
        _ = str(lp_tensor)

        # other ops
        lp_tensor = lp_tensor.detach()
        # explicitly testing aten.alias
        lp_tensor = torch.ops.aten.alias(lp_tensor)
        lp_tensor = lp_tensor.clone()
        # get all tensor_data_names for both
        # non optional and valid optional tensors
        tensor_data_names = lp_tensor.tensor_data_names.copy()
        if hasattr(lp_tensor, "optional_tensor_data_names"):
            for tensor_data_name in lp_tensor.optional_tensor_data_names:
                if getattr(lp_tensor, tensor_data_name) is not None:
                    tensor_data_names.append(tensor_data_name)

        # for each of the tensor data, we try to
        # make it non-contiguous and then use
        # lp_tensor.contiguous() call to make sure
        # contiguous() works
        for tensor_data_name in tensor_data_names:
            tensor = getattr(lp_tensor, tensor_data_name)
            # making qdata not contiguous
            tensor = tensor.transpose(0, 1).contiguous()
            tensor = tensor.transpose(0, 1)
            setattr(lp_tensor, tensor_data_name, tensor)
            self.assertFalse(getattr(lp_tensor, tensor_data_name).is_contiguous())
            lp_tensor = lp_tensor.contiguous()
            # making sure contiguous call works
            self.assertTrue(getattr(lp_tensor, tensor_data_name).is_contiguous())

        # copy_
        # making sure that initially tensor values are not the same so we can test copy_
        self.assertNotEqual(lp_tensor.qdata[0][0], lp_tensor_for_copy.qdata[0][0])
        # copy_ requires the attributes to be the same
        for tensor_attr_name in lp_tensor.tensor_attribute_names:
            self.assertEqual(
                getattr(lp_tensor, tensor_attr_name),
                getattr(lp_tensor_for_copy, tensor_attr_name),
            )
        lp_tensor.copy_(lp_tensor_for_copy)
        # after copy_, the tensor values should match
        for tensor_data_name in tensor_data_names:
            self.assertTrue(
                torch.equal(
                    getattr(lp_tensor, tensor_data_name),
                    getattr(lp_tensor_for_copy, tensor_data_name),
                )
            )

    @skip_if_no_cuda()
    def test_default_impls(self):
        """Making sure some common functions has default implementations, such as
        __tensor_unflatten__, __tensor_flatten__, _apply_fn_to_data, __repr__, to
        """

        class MyTensor(TorchAOBaseTensor):
            tensor_data_names = ["qdata"]
            tensor_attribute_names = ["attr", "device"]

            def __new__(cls, qdata, attr, device=None):
                shape = qdata.shape
                if device is None:
                    device = qdata.device
                kwargs = {"device": device}
                return torch.Tensor._make_wrapper_subclass(cls, shape, **kwargs)  # type: ignore[attr-defined]

            def __init__(self, qdata, attr, device=None):
                self.qdata = qdata
                self.attr = attr

        l = torch.nn.Linear(2, 3)
        l.weight = torch.nn.Parameter(MyTensor(l.weight, "attr"))
        lp_tensor = l.weight

        another_tensor = torch.nn.Linear(2, 3).weight
        # attribute has to be the same
        lp_tensor_for_copy = MyTensor(another_tensor, "attr")
        self._test_default_impls_helper(lp_tensor, lp_tensor_for_copy)

    @skip_if_no_cuda()
    def test_default_impls_with_optional_data(self):
        class MyTensorWithOptionalData(TorchAOBaseTensor):
            tensor_data_names = ["qdata"]
            optional_tensor_data_names = ["zero_point"]
            tensor_attribute_names = ["attr", "device"]

            def __new__(cls, qdata, zero_point=None, attr=1.0, device=None):
                shape = qdata.shape
                if device is None:
                    device = qdata.device
                kwargs = {"device": device}
                return torch.Tensor._make_wrapper_subclass(cls, shape, **kwargs)  # type: ignore[attr-defined]

            def __init__(self, qdata, zero_point=None, attr=1.0, device=None):
                self.qdata = qdata
                self.zero_point = zero_point
                self.attr = attr

        # test both the optional Tensor is None
        # and not None
        l = torch.nn.Linear(2, 3)
        lp_tensor = MyTensorWithOptionalData(l.weight, None, "attr")
        l = torch.nn.Linear(2, 3)
        lp_tensor_for_copy = MyTensorWithOptionalData(l.weight, None, "attr")
        self._test_default_impls_helper(lp_tensor, lp_tensor_for_copy)

        l = torch.nn.Linear(2, 3)
        lp_tensor = MyTensorWithOptionalData(
            l.weight, torch.zeros_like(l.weight), "attr"
        )
        l = torch.nn.Linear(2, 3)
        lp_tensor_for_copy = MyTensorWithOptionalData(
            l.weight, torch.zeros_like(l.weight), "attr"
        )
        self._test_default_impls_helper(lp_tensor, lp_tensor_for_copy)


if __name__ == "__main__":
    unittest.main()
