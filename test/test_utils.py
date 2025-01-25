import unittest
from unittest.mock import patch

import torch

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


if __name__ == "__main__":
    unittest.main()
