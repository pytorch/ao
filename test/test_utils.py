import functools
import unittest
from unittest.mock import patch

import pytest
import torch

from torchao.utils import TorchAOBaseTensor, torch_version_at_least


def skip_if_rocm(message=None):
    """Decorator to skip tests on ROCm platform with custom message.

    Args:
        message (str, optional): Additional information about why the test is skipped.
    """

    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            if torch.version.hip is not None:
                skip_message = "Skipping the test in ROCm"
                if message:
                    skip_message += f": {message}"
                pytest.skip(skip_message)
            return func(*args, **kwargs)

        return wrapper

    # Handle both @skip_if_rocm and @skip_if_rocm() syntax
    if callable(message):
        func = message
        message = None
        return decorator(func)
    return decorator


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


if __name__ == "__main__":
    unittest.main()
