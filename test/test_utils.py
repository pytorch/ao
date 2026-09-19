# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD 3-Clause license found in the
# LICENSE file in the root directory of this source tree.
import unittest
from unittest.mock import patch

import torch
import torch.nn.functional as F

from torchao.quantization import Int8Tensor
from torchao.utils import TorchAOBaseTensor, torch_version_at_least


class TestTorchVersion(unittest.TestCase):
    def test_torch_version_at_least(self):
        test_cases = [
            ("2.5.0a0+git9f17037", "2.5.0", False),  # [2, 5, -1] < [2, 5, 0]
            ("2.5.0a0+git9f17037", "2.4.0", True),  # [2, 5, -1] > [2, 4, 0]
            ("2.5.0.dev20240708+cu121", "2.5.0", False),  # [2, 5, -1] < [2, 5, 0]
            ("2.5.0.dev20240708+cu121", "2.4.0", True),  # [2, 5, -1] > [2, 4, 0]
            ("2.5.0", "2.4.0", True),  # [2, 5, 0] > [2, 4, 0]
            ("2.5.0", "2.5.0", True),  # [2, 5, 0] >= [2, 5, 0]
            ("2.4.0", "2.4.0", True),  # [2, 4, 0] >= [2, 4, 0]
            ("2.4.0", "2.5.0", False),  # [2, 4, 0] < [2, 5, 0]
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
        # since we did not define `tensor_data_names` and `tensor_attribute_names` for MyTensor
        # the following call will error out because `detach` is defined in `TorchAOBaseTensor`
        # but would rely on `tensor_data_names` and `tensor_attribute_names` being defined for it to work
        # user could either specify `tensor_data_names` and `tensor_attribute_names` or manually implement
        # detach op
        with self.assertRaisesRegex(NotImplementedError, "arg_types"):
            l.weight = torch.nn.Parameter(MyTensor(l.weight))

    def test_get_to_kwargs_non_blocking(self):
        """Verify _get_to_kwargs parses and returns the non_blocking flag."""

        class MyTensor(TorchAOBaseTensor):
            tensor_data_names = ["qdata"]
            tensor_attribute_names = ["attr", "device"]

            def __new__(cls, qdata, attr="attr", device=None):
                if device is None:
                    device = qdata.device
                kwargs = {"device": device, "dtype": qdata.dtype}
                r = torch.Tensor._make_wrapper_subclass(cls, qdata.shape, **kwargs)
                r.qdata = qdata
                r.attr = attr
                return r

            def __init__(self, qdata, attr="attr", device=None):
                pass

        t = MyTensor(torch.randn(4, 4))

        # non_blocking=True is preserved
        kwargs = t._get_to_kwargs(device="cpu", non_blocking=True)
        self.assertTrue(kwargs["non_blocking"])

        # non_blocking=False (explicit) is preserved
        kwargs = t._get_to_kwargs(device="cpu", non_blocking=False)
        self.assertFalse(kwargs["non_blocking"])

        # default (not specified) → False
        kwargs = t._get_to_kwargs(device="cpu")
        self.assertFalse(kwargs["non_blocking"])

    def _test_default_impls_helper(self, lp_tensor, lp_tensor_for_copy):
        # get `all_tensor_data_names` and `all_tensor_attribute_names`
        all_tensor_data_names = lp_tensor.tensor_data_names.copy()
        if hasattr(lp_tensor, "optional_tensor_data_names"):
            for tensor_data_name in lp_tensor.optional_tensor_data_names:
                if getattr(lp_tensor, tensor_data_name) is not None:
                    all_tensor_data_names.append(tensor_data_name)
        all_tensor_attribute_names = lp_tensor.tensor_attribute_names.copy()
        if hasattr(lp_tensor, "optional_tensor_attribute_names"):
            for tensor_attribute_name in lp_tensor.optional_tensor_attribute_names:
                if getattr(lp_tensor, tensor_attribute_name) is not None:
                    all_tensor_attribute_names.append(tensor_attribute_name)

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
        for tensor_data_name in all_tensor_data_names:
            self.assertTrue(
                torch.equal(
                    getattr(lp_tensor, tensor_data_name),
                    getattr(reconstructed, tensor_data_name),
                )
            )
        for tensor_attribute_name in all_tensor_attribute_names:
            self.assertEqual(
                getattr(lp_tensor, tensor_attribute_name),
                getattr(reconstructed, tensor_attribute_name),
            )

        self.assertTrue(torch.equal(lp_tensor.qdata, reconstructed.qdata))
        self.assertEqual(lp_tensor.attr, reconstructed.attr)

        device = torch.accelerator.current_accelerator()
        # test _get_to_kwargs
        _ = lp_tensor._get_to_kwargs(torch.strided, device=device)
        _ = lp_tensor._get_to_kwargs(layout=torch.strided, device=device)

        # `to` / `_to_copy`
        original_device = lp_tensor.device
        lp_tensor = lp_tensor.to(device)
        self.assertEqual(lp_tensor.device.type, device.type)
        lp_tensor = lp_tensor.to(original_device)
        self.assertEqual(lp_tensor.device, original_device)

        # __repr__
        _ = str(lp_tensor)

        # op test: detach
        lp_tensor = lp_tensor.detach()
        # op test: alias
        lp_tensor = torch.ops.aten.alias(lp_tensor)

        # op test: clone
        lp_tensor_clone = lp_tensor.clone()

        for tensor_data_name in all_tensor_data_names:
            self.assertTrue(
                torch.equal(
                    getattr(lp_tensor_clone, tensor_data_name),
                    getattr(lp_tensor, tensor_data_name),
                )
            )
        for tensor_attribute_name in all_tensor_attribute_names:
            self.assertEqual(
                getattr(lp_tensor_clone, tensor_attribute_name),
                getattr(lp_tensor, tensor_attribute_name),
            )

        # op test: transpose
        # non optional and valid optional tensors

        # for each of the tensor data, we try to
        # make it non-contiguous and then use
        # lp_tensor.contiguous() call to make sure
        # contiguous() works
        for tensor_data_name in all_tensor_data_names:
            tensor = getattr(lp_tensor, tensor_data_name)
            # making qdata not contiguous
            tensor = tensor.transpose(0, 1).contiguous()
            tensor = tensor.transpose(0, 1)
            setattr(lp_tensor, tensor_data_name, tensor)
            self.assertFalse(getattr(lp_tensor, tensor_data_name).is_contiguous())

        lp_tensor_t = lp_tensor.contiguous()

        # making sure contiguous call works
        for tensor_data_name in all_tensor_data_names:
            self.assertTrue(getattr(lp_tensor_t, tensor_data_name).is_contiguous())

        # making sure transpose does not change attributes
        for tensor_attribute_name in all_tensor_attribute_names:
            self.assertEqual(
                getattr(lp_tensor_t, tensor_attribute_name),
                getattr(lp_tensor, tensor_attribute_name),
            )

        # op test: copy_
        # making sure that initially tensor values are not the same so we can test copy_
        self.assertNotEqual(lp_tensor.qdata[0][0], lp_tensor_for_copy.qdata[0][0])
        # copy_ requires the attributes to be the same
        for tensor_attribute_name in all_tensor_attribute_names:
            self.assertEqual(
                getattr(lp_tensor_for_copy, tensor_attribute_name),
                getattr(lp_tensor, tensor_attribute_name),
            )

        lp_tensor.copy_(lp_tensor_for_copy)
        # after copy_, the tensor values should match
        for tensor_data_name in all_tensor_data_names:
            self.assertTrue(
                torch.equal(
                    getattr(lp_tensor, tensor_data_name),
                    getattr(lp_tensor_for_copy, tensor_data_name),
                )
            )
        # after copy_, the tensor attributes still matches
        # copy_ requires the attributes to be the same
        for tensor_attribute_name in all_tensor_attribute_names:
            self.assertEqual(
                getattr(lp_tensor_for_copy, tensor_attribute_name),
                getattr(lp_tensor, tensor_attribute_name),
            )

    @unittest.skipIf(not torch.accelerator.is_available(), "Need CUDA available")
    def test_default_impls(self):
        """Making sure some common functions has default implementations, such as
        __tensor_unflatten__, __tensor_flatten__, _apply_fn_to_data, __repr__, to
        """

        class MyTensor(TorchAOBaseTensor):
            tensor_data_names = ["qdata"]
            tensor_attribute_names = ["attr", "device"]

            def __new__(cls, qdata, attr, device):
                shape = qdata.shape
                if device is None:
                    device = qdata.device
                kwargs = {"device": device}
                return torch.Tensor._make_wrapper_subclass(cls, shape, **kwargs)  # type: ignore[attr-defined]

            def __init__(self, qdata, attr, device):
                self.qdata = qdata
                self.attr = attr

        l = torch.nn.Linear(2, 3)
        l.weight = torch.nn.Parameter(MyTensor(l.weight, "attr", None))
        lp_tensor = l.weight

        another_tensor = torch.nn.Linear(2, 3).weight
        # attribute has to be the same
        lp_tensor_for_copy = MyTensor(another_tensor, "attr", None)
        self._test_default_impls_helper(lp_tensor, lp_tensor_for_copy)

    @unittest.skipIf(not torch.accelerator.is_available(), "Need GPU available")
    def test_default_impls_with_optional_data(self):
        class MyTensorWithOptionalData(TorchAOBaseTensor):
            tensor_data_names = ["qdata"]
            tensor_attribute_names = ["attr", "device"]
            optional_tensor_data_names = ["zero_point"]

            def __new__(cls, qdata, attr, device, zero_point=None):
                shape = qdata.shape
                if device is None:
                    device = qdata.device
                kwargs = {"device": device}
                return torch.Tensor._make_wrapper_subclass(cls, shape, **kwargs)  # type: ignore[attr-defined]

            def __init__(self, qdata, attr, device, zero_point=None):
                self.qdata = qdata
                self.attr = attr
                self.zero_point = zero_point

        # test both the optional Tensor is None
        # and not None
        l = torch.nn.Linear(2, 3)
        lp_tensor = MyTensorWithOptionalData(l.weight, "attr", None, None)
        l = torch.nn.Linear(2, 3)
        lp_tensor_for_copy = MyTensorWithOptionalData(l.weight, "attr", None, None)
        self._test_default_impls_helper(lp_tensor, lp_tensor_for_copy)

        l = torch.nn.Linear(2, 3)
        lp_tensor = MyTensorWithOptionalData(
            l.weight, "attr", None, torch.zeros_like(l.weight)
        )
        l = torch.nn.Linear(2, 3)
        lp_tensor_for_copy = MyTensorWithOptionalData(
            l.weight, "attr", None, torch.zeros_like(l.weight)
        )
        self._test_default_impls_helper(lp_tensor, lp_tensor_for_copy)

    @unittest.skipIf(not torch.accelerator.is_available(), "Need GPU available")
    def test_default_impls_with_optional_attr(self):
        class MyTensorWithOptionalData(TorchAOBaseTensor):
            tensor_data_names = ["qdata"]
            tensor_attribute_names = ["attr", "device"]
            optional_tensor_data_names = ["zero_point"]
            optional_tensor_attribute_names = ["optional_attr"]

            def __new__(cls, qdata, attr, device, zero_point=None, optional_attr=None):
                shape = qdata.shape
                if device is None:
                    device = qdata.device
                kwargs = {"device": device}
                return torch.Tensor._make_wrapper_subclass(cls, shape, **kwargs)  # type: ignore[attr-defined]

            def __init__(
                self, qdata, attr, device, zero_point=None, optional_attr=None
            ):
                self.qdata = qdata
                self.attr = attr
                self.zero_point = zero_point
                self.optional_attr = optional_attr

        # test both the optional Tensor is None
        # and not None
        l = torch.nn.Linear(2, 3)
        lp_tensor = MyTensorWithOptionalData(l.weight, "attr", None, zero_point=None)
        l = torch.nn.Linear(2, 3)
        lp_tensor_for_copy = MyTensorWithOptionalData(
            l.weight, "attr", None, zero_point=None
        )
        self._test_default_impls_helper(lp_tensor, lp_tensor_for_copy)

        l = torch.nn.Linear(2, 3)
        lp_tensor = MyTensorWithOptionalData(
            l.weight, "attr", None, zero_point=None, optional_attr="value"
        )
        l = torch.nn.Linear(2, 3)
        lp_tensor_for_copy = MyTensorWithOptionalData(
            l.weight, "attr", None, zero_point=None, optional_attr="value"
        )
        self._test_default_impls_helper(lp_tensor, lp_tensor_for_copy)

    def _get_copy_test_tensor_class(self):
        class MyTensor(TorchAOBaseTensor):
            tensor_data_names = ["qdata"]
            tensor_attribute_names = ["attr", "device"]
            optional_tensor_data_names = ["zero_point"]
            optional_tensor_attribute_names = ["optional_attr"]

            def __new__(
                cls,
                qdata,
                attr="attr",
                device=None,
                zero_point=None,
                optional_attr=None,
            ):
                if device is None:
                    device = qdata.device
                kwargs = {"device": device}
                r = torch.Tensor._make_wrapper_subclass(cls, qdata.shape, **kwargs)  # type: ignore[attr-defined]
                r.qdata = qdata
                r.attr = attr
                r.zero_point = zero_point
                r.optional_attr = optional_attr
                return r

            def __init__(
                self,
                qdata,
                attr="attr",
                device=None,
                zero_point=None,
                optional_attr=None,
            ):
                pass

        return MyTensor

    def test_copy__attribute_mismatch_error_names_field(self):
        """copy_ error should name the mismatched attribute and both values,
        instead of dumping both tensors' full reprs (#3046)"""
        MyTensor = self._get_copy_test_tensor_class()
        lp_tensor = MyTensor(torch.randn(2, 3), "attr1")
        lp_tensor_for_copy = MyTensor(torch.randn(2, 3), "attr2")
        with self.assertRaisesRegex(ValueError, r"attr: 'attr1' vs 'attr2'") as context:
            lp_tensor.copy_(lp_tensor_for_copy)
        # the message should not contain tensor data dumps
        self.assertNotIn("tensor(", str(context.exception))

    def test_copy__type_mismatch_error_names_types(self):
        MyTensor = self._get_copy_test_tensor_class()

        class OtherTensor(TorchAOBaseTensor):
            tensor_data_names = ["data"]
            tensor_attribute_names = ["device"]

            def __new__(cls, data, device=None):
                if device is None:
                    device = data.device
                kwargs = {"device": device}
                r = torch.Tensor._make_wrapper_subclass(cls, data.shape, **kwargs)  # type: ignore[attr-defined]
                r.data = data
                return r

            def __init__(self, data, device=None):
                pass

        lp_tensor = MyTensor(torch.randn(2, 3))
        other_tensor = OtherTensor(torch.randn(2, 3))
        with self.assertRaisesRegex(ValueError, r"type: MyTensor vs OtherTensor"):
            lp_tensor.copy_(other_tensor)

    def _get_packed_copy_test_tensor_class(self):
        """Outer shape is decoupled from the inner tensor shape, as in packed formats"""

        class MyPackedTensor(TorchAOBaseTensor):
            tensor_data_names = ["qdata"]
            tensor_attribute_names = ["device"]

            def __new__(cls, qdata, shape, device=None):
                if device is None:
                    device = qdata.device
                kwargs = {"device": device}
                r = torch.Tensor._make_wrapper_subclass(cls, shape, **kwargs)  # type: ignore[attr-defined]
                r.qdata = qdata
                return r

            def __init__(self, qdata, shape, device=None):
                pass

        return MyPackedTensor

    def test_copy__tensor_shape_mismatch_error_names_tensor(self):
        """Inner tensor shape mismatch should be reported by name even when
        the outer shapes match (e.g. packed formats)"""
        MyPackedTensor = self._get_packed_copy_test_tensor_class()
        lp_tensor = MyPackedTensor(torch.randn(2), torch.Size([4]))
        lp_tensor_for_copy = MyPackedTensor(torch.randn(4), torch.Size([4]))
        with self.assertRaisesRegex(
            ValueError, r"qdata\.shape: torch\.Size\(\[2\]\) vs torch\.Size\(\[4\]\)"
        ):
            lp_tensor.copy_(lp_tensor_for_copy)

    def test_copy__outer_shape_mismatch_reported_on_its_own(self):
        """An outer shape mismatch is reported by itself when the inner tensors agree.

        Packed formats decouple the two, so `shape` and `qdata.shape` are independent
        signals and the message must not conflate them.
        """
        MyPackedTensor = self._get_packed_copy_test_tensor_class()
        lp_tensor = MyPackedTensor(torch.randn(4), torch.Size([4]))
        lp_tensor_for_copy = MyPackedTensor(torch.randn(4), torch.Size([2, 2]))
        with self.assertRaisesRegex(
            ValueError,
            r"metadata mismatch: shape: torch\.Size\(\[4\]\) vs torch\.Size\(\[2, 2\]\)$",
        ):
            lp_tensor.copy_(lp_tensor_for_copy)

    def test_copy__optional_tensor_presence_mismatch(self):
        """A None vs non-None optional tensor should be reported as a metadata
        mismatch (in both directions) instead of crashing"""
        MyTensor = self._get_copy_test_tensor_class()
        lp_tensor = MyTensor(torch.randn(2, 3))
        lp_tensor_for_copy = MyTensor(torch.randn(2, 3), zero_point=torch.zeros(3))
        with self.assertRaisesRegex(ValueError, r"zero_point: None vs Tensor"):
            lp_tensor.copy_(lp_tensor_for_copy)
        with self.assertRaisesRegex(ValueError, r"zero_point: Tensor vs None"):
            lp_tensor_for_copy.copy_(lp_tensor)

    def test_copy__multiple_mismatches_all_reported(self):
        MyTensor = self._get_copy_test_tensor_class()
        lp_tensor = MyTensor(torch.randn(2, 3), "attr1", optional_attr="value1")
        lp_tensor_for_copy = MyTensor(
            torch.randn(2, 3), "attr2", optional_attr="value2"
        )
        with self.assertRaisesRegex(
            ValueError, r"attr: 'attr1' vs 'attr2'.*optional_attr: 'value1' vs 'value2'"
        ):
            lp_tensor.copy_(lp_tensor_for_copy)

    def test_copy__with_matching_metadata_succeeds(self):
        MyTensor = self._get_copy_test_tensor_class()
        lp_tensor = MyTensor(torch.randn(2, 3), zero_point=torch.zeros(3))
        lp_tensor_for_copy = MyTensor(torch.randn(2, 3), zero_point=torch.ones(3))
        lp_tensor.copy_(lp_tensor_for_copy)
        self.assertTrue(torch.equal(lp_tensor.qdata, lp_tensor_for_copy.qdata))
        self.assertTrue(
            torch.equal(lp_tensor.zero_point, lp_tensor_for_copy.zero_point)
        )

    def test_copy__real_tensor_subclass_names_the_single_field(self):
        """End-to-end on a shipped subclass rather than a test-local one: the
        message names the one field that differs instead of dumping both
        tensors' full reprs (#3046)"""

        def make_int8_tensor(block_size):
            return Int8Tensor(
                torch.randint(-8, 7, (4, 8), dtype=torch.int8),
                torch.randn(4, 1),
                block_size,
                torch.bfloat16,
            )

        lp_tensor = make_int8_tensor([1, 8])
        lp_tensor_for_copy = make_int8_tensor([1, 4])
        with self.assertRaisesRegex(
            ValueError, r"block_size: \[1, 8\] vs \[1, 4\]"
        ) as context:
            lp_tensor.copy_(lp_tensor_for_copy)
        # the entire point of #3046: no tensor data, no wall of text
        self.assertNotIn("tensor(", str(context.exception))
        self.assertLess(len(str(context.exception)), 200)

    def test_implements_and_torch_function_together(self):
        """Ensure a function decorated with both @_implements and @_implements_torch_function works."""
        counter = {"calls": 0}

        class MyTensor(TorchAOBaseTensor):
            tensor_data_names = ["qdata"]
            tensor_attribute_names = ["attr", "device"]

            def __new__(cls, qdata: torch.Tensor, attr: str = "attr", device=None):
                kwargs = {}
                if device is None:
                    device = qdata.device
                kwargs["device"] = device
                kwargs["dtype"] = qdata.dtype
                r = torch.Tensor._make_wrapper_subclass(cls, qdata.shape, **kwargs)
                r.qdata = qdata
                r.attr = attr
                return r

            def __init__(self, qdata: torch.Tensor, attr: str = "attr", device=None):
                pass

        implements = MyTensor.implements
        implements_torch_function = MyTensor.implements_torch_function

        @implements([torch.ops.aten.t.default])
        @implements_torch_function([F.linear])
        def fake_linear(func, types, args, kwargs):
            counter["calls"] += 1

        l = torch.nn.Linear(2, 3)
        l.weight = torch.nn.Parameter(MyTensor(l.weight.detach(), "attr", None))
        x = torch.randn(4, 2)

        # Torch function path
        F.linear(x, l.weight, l.bias)
        self.assertEqual(
            counter["calls"], 1, "Expected fake_linear to be called via F.linear"
        )

        # ATen path
        mt = MyTensor(torch.randn(3, 4))
        torch.ops.aten.t.default(mt)
        self.assertEqual(
            counter["calls"], 2, "Expected fake_linear to be called via aten.t.default"
        )

    def test_subclassing(self):
        counters = {"parent": 0}

        class Parent(TorchAOBaseTensor):
            tensor_data_names = ["qdata"]
            tensor_attribute_names = ["attr"]

            def __new__(cls, qdata, attr):
                r = torch.Tensor._make_wrapper_subclass(cls, qdata.shape)
                r.qdata = qdata
                r.attr = attr
                return r

            def __init__(self, qdata, attr):
                pass

        @Parent.implements([torch.ops.aten.cat.default])
        def parent_cat(func, types, args, kwargs):
            counters["parent"] += 1

        class Child(Parent):
            tensor_data_names = ["qdata"]
            tensor_attribute_names = ["attr"]

        # Call via Child
        t1 = Child(torch.randn(2, 3), "a")
        t2 = Child(torch.randn(2, 3), "b")
        torch.ops.aten.cat.default([t1, t2], 0)

        self.assertEqual(counters["parent"], 1)

        # Add new op to Parent after Child exists
        @Parent.implements([torch.ops.aten.relu.default])
        def parent_relu(func, types, args, kwargs):
            pass

        # Child should not see it
        with self.assertRaises(RuntimeError):
            torch.ops.aten.relu.default(t1)

    def test_subclassing_with_real_op(self):
        counter = {"calls": 0}

        class Parent(TorchAOBaseTensor):
            tensor_data_names = ["qdata"]
            tensor_attribute_names = ["attr"]

            def __new__(cls, qdata, attr):
                r = torch.Tensor._make_wrapper_subclass(cls, qdata.shape)
                r.qdata = qdata
                r.attr = attr
                return r

            def __init__(self, qdata, attr):
                pass

        @Parent.implements([torch.ops.aten.cat.default])
        def parent_cat(func, types, args, kwargs):
            counter["calls"] += 1

        class Child(Parent):
            tensor_data_names = ["qdata"]
            tensor_attribute_names = ["attr"]

        t1 = Child(torch.randn(2, 3), "a")
        t2 = Child(torch.randn(2, 3), "b")

        torch.ops.aten.cat.default([t1, t2], 0)

        self.assertEqual(counter["calls"], 1)

    def test_op_overwrite(self):
        counters = {"parent": 0, "child": 0}

        class Parent(TorchAOBaseTensor):
            tensor_data_names = ["qdata"]
            tensor_attribute_names = ["attr"]

            def __new__(cls, qdata, attr):
                r = torch.Tensor._make_wrapper_subclass(cls, qdata.shape)
                r.qdata = qdata
                r.attr = attr
                return r

            def __init__(self, qdata, attr):
                pass

        @Parent.implements([torch.ops.aten.cat.default])
        def parent_cat(func, types, args, kwargs):
            counters["parent"] += 1

        class Child(Parent):
            tensor_data_names = ["qdata"]
            tensor_attribute_names = ["attr"]

        @Child.implements([torch.ops.aten.cat.default])
        def child_cat(func, types, args, kwargs):
            counters["child"] += 1

        # Parent instance should call parent implementation
        p1 = Parent(torch.randn(2, 3), "a")
        p2 = Parent(torch.randn(2, 3), "b")
        torch.ops.aten.cat.default([p1, p2], 0)

        self.assertEqual(counters["parent"], 1)
        self.assertEqual(counters["child"], 0)

        # Child instance should call child implementation
        c1 = Child(torch.randn(2, 3), "a")
        c2 = Child(torch.randn(2, 3), "b")
        torch.ops.aten.cat.default([c1, c2], 0)

        self.assertEqual(counters["parent"], 1)
        self.assertEqual(counters["child"], 1)

    def test_multiple_inheritance(self):
        counters = {"A": 0, "B": 0}

        class A(TorchAOBaseTensor):
            tensor_data_names = ["a"]
            tensor_attribute_names = ["b"]

            def __new__(cls, a, b):
                r = torch.Tensor._make_wrapper_subclass(cls, a.shape)
                r.a = a
                r.b = b
                return r

            def __init__(self, a, b):
                pass

        @A.implements([torch.ops.aten.neg.default])
        def a_neg(func, types, args, kwargs):
            counters["A"] += 1

        class B(TorchAOBaseTensor):
            tensor_data_names = ["a"]
            tensor_attribute_names = ["b"]

            def __new__(cls, a, b):
                r = torch.Tensor._make_wrapper_subclass(cls, a.shape)
                r.a = a
                r.b = b
                return r

            def __init__(self, a, b):
                pass

        @B.implements([torch.ops.aten.neg.default])
        def b_neg(func, types, args, kwargs):
            counters["B"] += 1

        class C(A, B):
            tensor_data_names = ["a"]
            tensor_attribute_names = ["b"]

        t = C(torch.randn(3), "x")
        torch.ops.aten.neg.default(t)

        self.assertEqual(counters["A"], 0)
        self.assertEqual(counters["B"], 1)

    def test_multiple_inheritance_with_child_override(self):
        counters = {"A": 0, "B": 0, "C": 0}

        class A(TorchAOBaseTensor):
            tensor_data_names = ["a"]
            tensor_attribute_names = ["b"]

            def __new__(cls, a, b):
                r = torch.Tensor._make_wrapper_subclass(cls, a.shape)
                r.a = a
                r.b = b
                return r

            def __init__(self, a, b):
                pass

        @A.implements([torch.ops.aten.neg.default])
        def a_neg(func, types, args, kwargs):
            counters["A"] += 1

        class B(TorchAOBaseTensor):
            tensor_data_names = ["a"]
            tensor_attribute_names = ["b"]

            def __new__(cls, a, b):
                r = torch.Tensor._make_wrapper_subclass(cls, a.shape)
                r.a = a
                r.b = b
                return r

            def __init__(self, a, b):
                pass

        @B.implements([torch.ops.aten.neg.default])
        def b_neg(func, types, args, kwargs):
            counters["B"] += 1

        class C(A, B):
            tensor_data_names = ["a"]
            tensor_attribute_names = ["b"]

        @C.implements([torch.ops.aten.neg.default])
        def c_neg(func, types, args, kwargs):
            counters["C"] += 1

        # Instance of A, A impl
        a = A(torch.randn(3), "x")
        torch.ops.aten.neg.default(a)

        self.assertEqual(counters["A"], 1)
        self.assertEqual(counters["B"], 0)
        self.assertEqual(counters["C"], 0)

        # Instance of B, B impl
        b = B(torch.randn(3), "x")
        torch.ops.aten.neg.default(b)

        self.assertEqual(counters["A"], 1)
        self.assertEqual(counters["B"], 1)
        self.assertEqual(counters["C"], 0)

        # Instance of C, C impl only
        c = C(torch.randn(3), "x")
        torch.ops.aten.neg.default(c)

        self.assertEqual(counters["A"], 1)
        self.assertEqual(counters["B"], 1)
        self.assertEqual(counters["C"], 1)


if __name__ == "__main__":
    unittest.main()
