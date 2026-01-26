# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD 3-Clause license found in the
# LICENSE file in the root directory of this source tree.

# Owner(s): ["oncall: cpu inductor"]
import copy
import functools
import sys
import unittest
from typing import NamedTuple

import torch
from torch._dynamo.testing import make_test_cls_with_patches
from torch._inductor import config
from torch._inductor.test_case import TestCase as InductorTestCase
from torch._inductor.utils import (
    run_and_get_cpp_code,
)
from torch.testing._internal.common_device_type import (
    get_desired_device_type_test_bases,
)
from torch.testing._internal.common_utils import (
    IS_MACOS,
    IS_WINDOWS,
    slowTest,
)
from torch.testing._internal.inductor_utils import HAS_CPU

from torchao.utils import torch_version_at_least

try:
    try:
        from . import (
            test_x86inductor_fusion,
        )
    except ImportError:
        import test_x86inductor_fusion
except unittest.SkipTest:
    if __name__ == "__main__":
        sys.exit(0)
    raise


_desired_test_bases = get_desired_device_type_test_bases()
RUN_CPU = (
    HAS_CPU
    and any(getattr(x, "device_type", "") == "cpu" for x in _desired_test_bases)
    and not IS_MACOS
)


class CppWrapperTemplate:
    pass


class TestCppWrapper(InductorTestCase):
    device = "cpu"


class DynamicShapesCppWrapperCpuTests(InductorTestCase):
    device = "cpu"


test_failures_cpp_wrapper = {}


def make_test_case(
    name,
    device,
    tests,
    condition=True,
    slow=False,
    func_inputs=None,
    code_string_count=None,
    test_build_separate=False,
):
    test_name = f"{name}_{device}" if device else name
    if code_string_count is None:
        code_string_count = {}

    func = getattr(tests, test_name)
    assert callable(func), "not a callable"
    func = slowTest(func) if slow else func
    new_test_name = f"{test_name}_separate" if test_build_separate else test_name
    patches = {"cpp_wrapper": True}
    if torch_version_at_least("2.8.0"):
        patches.update({"cpp_wrapper_build_separate": test_build_separate})

    @config.patch(**patches)
    def fn(self):
        tests.setUpClass()
        tests.setUp()
        try:
            with torch._C._PreserveDispatchKeyGuard():
                torch._C._dispatch_tls_set_dispatch_key_included(
                    torch._C.DispatchKey.Dense, True
                )

                _, code = run_and_get_cpp_code(
                    func, *func_inputs if func_inputs else []
                )
                # If a test generates no code, skip the remaining checks.  This can
                # happen for tests validating build-dependent features (e.g. datatypes
                # that are available on some platforms and not others).
                if code:
                    if test_build_separate:
                        self.assertIn("kernel_src", code)
                    self.assertIn("CppWrapperCodeCache", code)
                    self.assertTrue(
                        all(
                            code.count(string) == code_string_count[string]
                            for string in code_string_count
                        )
                    )
        finally:
            tests.tearDown()
            tests.tearDownClass()

    fn.__name__ = new_test_name

    fn.__dict__ = copy.deepcopy(func.__dict__)
    if condition:
        setattr(
            CppWrapperTemplate,
            new_test_name,
            fn,
        )


def copy_tests(my_cls, other_cls, suffix, test_failures=None, xfail_prop=None):  # noqa: B902
    for name, value in my_cls.__dict__.items():
        if name.startswith("test_"):
            # You cannot copy functions in Python, so we use closures here to
            # create objects with different ids. Otherwise, unittest.skip
            # would modify all methods sharing the same object id. Also, by
            # using a default argument, we create a copy instead of a
            # reference. Otherwise, we would lose access to the value.

            @functools.wraps(value)
            def new_test(self, value=value):
                return value(self)

            # Copy __dict__ which may contain test metadata
            new_test.__dict__ = copy.deepcopy(value.__dict__)

            if xfail_prop is not None and hasattr(value, xfail_prop):
                new_test = unittest.expectedFailure(new_test)

            tf = test_failures and test_failures.get(name)
            if tf and suffix in tf.suffixes:
                skip_func = (
                    unittest.skip("Skipped!")
                    if tf.is_skip
                    else unittest.expectedFailure
                )
                new_test = skip_func(new_test)

            setattr(other_cls, f"{name}_{suffix}", new_test)

    # Special case convenience routine
    if hasattr(my_cls, "is_dtype_supported"):
        other_cls.is_dtype_supported = my_cls.is_dtype_supported


def make_dynamic_cls(cls, xfail_prop="_expected_failure_dynamic"):
    return make_test_cls_with_patches(
        cls,
        "DynamicShapes",
        "_dynamic_shapes",
        (torch._dynamo.config, "assume_static_by_default", False),
        xfail_prop=xfail_prop,
    )


if RUN_CPU:

    class BaseTest(NamedTuple):
        name: str
        device: str = "cpu"
        tests: InductorTestCase = InductorTestCase()
        condition: bool = True
        slow: bool = False
        func_inputs: list = None
        code_string_count: dict = {}
        test_build_separate: bool = False

    for item in [
        BaseTest(
            "test_qconv2d",
            "cpu",
            test_x86inductor_fusion.TestPatternMatcher(),
            condition=torch.backends.mkldnn.is_available() and not IS_WINDOWS,
        ),
        BaseTest(
            "test_qconv2d_relu",
            "cpu",
            test_x86inductor_fusion.TestPatternMatcher(),
            condition=torch.backends.mkldnn.is_available() and not IS_WINDOWS,
        ),
        BaseTest(
            "test_qconv2d_add",
            "cpu",
            test_x86inductor_fusion.TestPatternMatcher(),
            condition=torch.backends.mkldnn.is_available() and not IS_WINDOWS,
        ),
        BaseTest(
            "test_qconv2d_add_relu",
            "cpu",
            test_x86inductor_fusion.TestPatternMatcher(),
            condition=torch.backends.mkldnn.is_available() and not IS_WINDOWS,
        ),
        BaseTest(
            "test_qconv2d_dequant_promotion",
            "cpu",
            test_x86inductor_fusion.TestPatternMatcher(),
            condition=torch.backends.mkldnn.is_available() and not IS_WINDOWS,
        ),
        BaseTest(
            "test_qconv2d_maxpool2d_linear_dynamic",
            "cpu",
            test_x86inductor_fusion.TestDynamicPatternMatcher(),
            condition=torch.backends.mkldnn.is_available() and not IS_WINDOWS,
            func_inputs=[
                [
                    "aoti_torch_cpu__qconv_pointwise_tensor",
                    "torch.ops.quantized.max_pool2d",
                    "aoti_torch_cpu__qlinear_pointwise_tensor",
                ]
            ],
        ),
        *[
            BaseTest(
                func,
                "",
                test_x86inductor_fusion.TestPatternMatcher(),
                condition=torch.backends.mkldnn.is_available() and not IS_WINDOWS,
            )
            for func in dir(test_x86inductor_fusion.TestPatternMatcher())
            if func.startswith("test_qlinear")
        ],
        BaseTest(
            "test_qconv2d_with_concat",
            "cpu",
            test_x86inductor_fusion.TestPatternMatcher(),
            condition=torch.backends.mkldnn.is_available() and not IS_WINDOWS,
        ),
        BaseTest(
            "test_dynamic_qlinear",
            "cpu",
            test_x86inductor_fusion.TestPatternMatcher(),
            condition=torch.backends.mkldnn.is_available() and not IS_WINDOWS,
        ),
        BaseTest(
            "test_dynamic_qlinear_qat",
            "cpu",
            test_x86inductor_fusion.TestPatternMatcher(),
            condition=torch.backends.mkldnn.is_available() and not IS_WINDOWS,
        ),
    ]:
        make_test_case(
            item.name,
            item.device,
            item.tests,
            item.condition,
            item.slow,
            item.func_inputs,
            item.code_string_count,
            item.test_build_separate,
        )

    copy_tests(
        CppWrapperTemplate,
        TestCppWrapper,
        "cpp_wrapper",
        test_failures_cpp_wrapper,
    )

    DynamicShapesCppWrapperTemplate = make_dynamic_cls(CppWrapperTemplate)

    copy_tests(
        DynamicShapesCppWrapperTemplate,
        DynamicShapesCppWrapperCpuTests,
        "cpp_wrapper",
        test_failures_cpp_wrapper,
        xfail_prop="_expected_failure_dynamic_wrapper",
    )


if __name__ == "__main__":
    from torch._inductor.test_case import run_tests

    if RUN_CPU:
        run_tests(needs="filelock")
